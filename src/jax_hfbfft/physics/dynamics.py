"""
Module for Time-Dependent Hartree-Fock (TDHF)

The static solver relaxes to the ground state by damped gradient descent/
    imaginary-time propagation, decaying excited components away

This module does full time propogation (i hbar d(psi)/dt = h[rho(t)] psi)
Orthonormality and energy conservation are automatically conserved
"""

import dataclasses
from dataclasses import dataclass, field
from functools import partial
from typing import Optional, Tuple, Callable

import jax
import jax.numpy as jnp

from jax_hfbfft.jax_config import get_dtypes
from jax_hfbfft.core.grid import Grid
from jax_hfbfft.core.force import Force
from jax_hfbfft.physics.densities import Densities, compute_densities
from jax_hfbfft.physics.meanfield import (
    Meanfield, compute_skyrme_meanfield, apply_hamiltonian,
)
from jax_hfbfft.physics.energies import compute_integrated_energy
from jax_hfbfft.physics.coulomb import CoulombSolver, solve_poisson


# hbar*c in MeV fm.
HBARC = 197.3269804


@jax.tree_util.register_dataclass
@dataclass
class TDConfig:
    """Configuration for a TDHF run."""
    dt: float = 0.2                  # time step in fm/c
    n_steps: int = 1000              # number of steps
    taylor_order: int = field(default=6, metadata=dict(static=True))  
    predictor_corrector: bool = field(default=True, metadata=dict(static=True))
    use_coulomb: bool = field(default=True, metadata=dict(static=True))
    diag_interval: int = field(default=10, metadata=dict(static=True))
    verbose: bool = True


@jax.tree_util.register_dataclass
@dataclass
class TDState:
    """
    TDHF state
    """
    psi: jax.Array            # (nstates, 2, nx, ny, nz) complex
    wocc: jax.Array           # (nstates,) fixed occupations
    isospin: jax.Array        # (nstates,)
    densities: Densities
    meanfield: Meanfield
    coulomb_solver: CoulombSolver
    wcoul: jax.Array
    time: float               # fm/c
    step: int

    @property
    def nstates(self) -> int:
        return self.psi.shape[0]


@partial(jax.jit, static_argnums=(5,))
def propagate_taylor(
    psi: jax.Array,           # (nstates, 2, nx, ny, nz)
    meanfield: Meanfield,
    isospin: jax.Array,
    grid: Grid,
    dt: float,
    order: int,               # static
) -> jax.Array:
    """
    Advance psi by exp(-i h dt / hbar), truncated at `order` in the Taylor series.
    """
    phase = -1j * dt / HBARC

    def h_on_all(p):
        return jax.vmap(lambda pp, iq: apply_hamiltonian(pp, meanfield, iq, grid))(
            p, isospin
        )

    term = psi
    out = psi
    for n in range(1, order + 1):
        term = (phase / n) * h_on_all(term)
        out = out + term
    return out


def build_fields(
    psi: jax.Array,
    wocc: jax.Array,
    isospin: jax.Array,
    grid: Grid,
    force: Force,
    coulomb_solver: CoulombSolver,
    use_coulomb: bool,
) -> Tuple[Densities, Meanfield, jax.Array]:
    """
    Densities and mean field from the current wavefunctions.
    """
    zeros = jnp.zeros_like(wocc)
    densities = compute_densities(
        psi, wocc, zeros, jnp.ones_like(wocc), jnp.ones_like(wocc), isospin, grid
    )

    wcoul = jnp.zeros((grid.nx, grid.ny, grid.nz), dtype=densities.rho.dtype)
    if use_coulomb:
        wcoul = solve_poisson(densities.rho[1], coulomb_solver, grid)

    meanfield = compute_skyrme_meanfield(
        densities, force, grid,
        coulomb_potential=wcoul if use_coulomb else None,
        use_coulomb=use_coulomb,
    )
    return densities, meanfield, wcoul


def tdhf_step(
    state: TDState,
    grid: Grid,
    force: Force,
    config: TDConfig,
) -> TDState:
    """
    Time-dependent time step

    Two options: predictor-corrector vs naive step (using h[rho(t)] over whole interval)
    """
    if config.predictor_corrector: 
        # Full dt with the old Hamiltonian,
        # then average t and t+dt to estimate rho(t + dt/2)
        # Scheme taken from Sky3d

        # Predicted psi and density
        psi_pred = propagate_taylor(
            state.psi, state.meanfield, state.isospin, grid,
            config.dt, max(1, config.taylor_order // 2),
        ) 
        dens_pred, _, _ = build_fields(
            psi_pred, state.wocc, state.isospin, grid, force,
            state.coulomb_solver, False,
        )
        # Find half-step fields to use for propogation
        dens_mid = jax.tree_util.tree_map(
            lambda a, b: 0.5 * (a + b), state.densities, dens_pred
        )
        wcoul_mid = jnp.zeros_like(state.wcoul)
        if config.use_coulomb:
            wcoul_mid = solve_poisson(dens_mid.rho[1], state.coulomb_solver, grid)
        mf_mid = compute_skyrme_meanfield(
            dens_mid, force, grid,
            coulomb_potential=wcoul_mid if config.use_coulomb else None,
            use_coulomb=config.use_coulomb,
        )
    else:
        mf_mid = state.meanfield

    psi_new = propagate_taylor(
        state.psi, mf_mid, state.isospin, grid, config.dt, config.taylor_order,
    )

    densities, meanfield, wcoul = build_fields(
        psi_new, state.wocc, state.isospin, grid, force,
        state.coulomb_solver, config.use_coulomb,
    )

    return dataclasses.replace(
        state,
        psi=psi_new,
        densities=densities,
        meanfield=meanfield,
        wcoul=wcoul,
        time=state.time + config.dt,
        step=state.step + 1,
    )


# Initial Conditions
def prepare_tdhf_state(
    static_state,
    grid: Grid,
    force: Force,
    use_coulomb: bool = True,
    occupation_tol: float = 1e-6,
    stationarity_tol: float = 1.0,
) -> TDState:
    """
    Convert a converged static SolverState into a TDHF starting state.

    `force` must be the force the state was relaxed with
    """
    wocc = jnp.asarray(static_state.wocc)
    frac = jnp.minimum(jnp.abs(wocc), jnp.abs(1.0 - wocc))
    worst = float(jnp.max(frac))
    if worst > occupation_tol:
        n_frac = int(jnp.sum(frac > occupation_tol))
        raise ValueError(
            f"TDHF requires sharp occupations, but {n_frac} states are "
            f"fractional (worst deviation {worst:.3e}).  Run the static "
            f"calculation with ipair=0; paired states need TDHFB."
        )
    wocc = jnp.round(wocc)

    # The s-channel time-odd couplings (c_s, c_ds, c_sDs) are off by default, as
    # in Sky3D; they are unconstrained by the fit and can be unstable, so warn.
    if force.has_time_odd:
        rho_crit = _spin_stability_density(force)
        msg = (f"  [warning] s-channel time-odd couplings are active "
               f"(c_s0={force.c_s0:.1f}, c_s1={force.c_s1:.1f}).\n"
               f"            Sky3D omits these entirely.")
        if rho_crit is not None:
            msg += (f"\n            C^s_0 goes NEGATIVE below rho = {rho_crit:.5f} "
                    f"fm^-3 ({rho_crit/0.16*100:.1f}% of saturation), which makes "
                    f"s=0\n            an energy MAXIMUM in the surface region: "
                    f"spin density grows exponentially there.")
        print(msg)

    coulomb_solver = static_state.coulomb_solver
    densities, meanfield, wcoul = build_fields(
        static_state.psi, wocc, static_state.isospin, grid, force,
        coulomb_solver, use_coulomb,
    )
    state = TDState(
        psi=static_state.psi,
        wocc=wocc,
        isospin=static_state.isospin,
        densities=densities,
        meanfield=meanfield,
        coulomb_solver=coulomb_solver,
        wcoul=wcoul,
        time=0.0,
        step=0,
    )

    # A state relaxed in a different Hamiltonian (wrong force, or residual spin
    # density the static solver zeroed) is not stationary and drifts in TDHF.
    resid = stationarity_residual(state, grid, force)
    if resid > stationarity_tol:
        smax = float(jnp.max(jnp.abs(densities.sdens)))
        print(f"  [warning] handover state is not an eigenstate of the TDHF "
              f"Hamiltonian: ||(h-<h>)psi|| = {resid:.3e} MeV, max|s| = {smax:.2e} fm^-3.\n"
              f"            Small values are normal; a large one means the state "
              f"was relaxed with a different force.")
    return state


def _spin_stability_density(force: Force):
    """
    Returns density below which the isoscalar spin coupling C^s_0(rho) turns negative
    """
    if force.c_ds0 == 0.0 or force.c_s0 >= 0.0:
        return None
    return (abs(force.c_s0) / force.c_ds0) ** (1.0 / force.power)


def stationarity_residual(state: "TDState", grid: Grid, force: Force) -> float:
    """
    Calculate stationarity residual summed over states, in MeV.

    Zero for an exact eigenstate of h.
    """
    def one(p, iq):
        hp = apply_hamiltonian(p, state.meanfield, iq, grid)
        e = jnp.real(jnp.sum(jnp.conj(p) * hp)) * grid.wxyz
        r = hp - e * p
        return jnp.sqrt(jnp.real(jnp.sum(jnp.conj(r) * r)) * grid.wxyz)

    res = jax.vmap(one)(state.psi, state.isospin)
    return float(jnp.max(res * state.wocc))


def apply_boost(psi: jax.Array, grid: Grid, k: Tuple[float, float, float]) -> jax.Array:
    """
    Translational boost: psi -> exp(i k.r) psi, giving the nucleus momentum
    hbar*k per nucleon. Excites centre-of-mass mode at zero excitation energy.
    """
    X, Y, Z = grid.get_meshgrid()
    ph = jnp.exp(1j * (k[0] * X + k[1] * Y + k[2] * Z))  # momentum boost keeps +i
    return psi * ph[None, None, :, :, :]


def apply_multipole_boost(
    psi: jax.Array, grid: Grid, operator: jax.Array, eta: float,
) -> jax.Array:
    """
    Small-amplitude excitation: psi -> exp(i eta Q) psi for a one-body operator Q.
    Useful for strength functions, as the kick excites every mode simultaneously
    """
    # Sign matches Sky3D's extboost: psi *= exp(-i * extfield).
    ph = jnp.exp(-1j * eta * operator)
    return psi * ph[None, None, :, :, :]


def reset_cm_velocity(state: TDState, grid: Grid) -> TDState:
    """
    Remove any net centre-of-mass momentum (avoids boundary)
    """
    # current has shape (isospin, component, nx, ny, nz); sum over isospin and space
    P = jnp.sum(state.densities.current, axis=(0, 2, 3, 4)) * grid.wxyz   # (3,)
    A = jnp.sum(state.densities.rho) * grid.wxyz
    k_corr = -P / jnp.maximum(A, 1e-12)
    psi_new = apply_boost(state.psi, grid,
                          (float(k_corr[0]), float(k_corr[1]), float(k_corr[2])))
    return dataclasses.replace(state, psi=psi_new)


# Diagnostics

def orthonormality_error(psi: jax.Array, isospin: jax.Array, wxyz: float) -> float:
    """
    Max deviation of the overlap matrix from the identity within each isospin.
    """
    worst = 0.0
    for iq in (0, 1):
        idx = jnp.nonzero(isospin == iq)[0]
        if idx.shape[0] == 0:
            continue
        blk = psi[idx].reshape(idx.shape[0], -1)
        S = jnp.dot(jnp.conj(blk), blk.T) * wxyz
        worst = max(worst, float(jnp.max(jnp.abs(S - jnp.eye(idx.shape[0], dtype=S.dtype)))))
    return worst


def compute_moments(densities: Densities, grid: Grid) -> dict:
    """Centre of mass and the low multipole moments, tracked over time."""
    rho = densities.rho[0] + densities.rho[1]
    X, Y, Z = grid.get_meshgrid()
    w = grid.wxyz
    A = jnp.sum(rho) * w
    r2 = X**2 + Y**2 + Z**2
    return {
        'A': float(A),
        'cm_x': float(jnp.sum(rho * X) * w / A),
        'cm_y': float(jnp.sum(rho * Y) * w / A),
        'cm_z': float(jnp.sum(rho * Z) * w / A),
        'r2': float(jnp.sum(rho * r2) * w / A),
        'Q20': float(jnp.sum(rho * (2 * Z**2 - X**2 - Y**2)) * w),
        # Radius about the center of mass, not the origin
        'r2_int': float(jnp.sum(rho * r2) * w / A
                        - ((jnp.sum(rho * X) * w / A) ** 2
                           + (jnp.sum(rho * Y) * w / A) ** 2
                           + (jnp.sum(rho * Z) * w / A) ** 2)),
        'Q22': float(jnp.sum(rho * (X**2 - Y**2)) * w),
    }


def total_energy(state: TDState, grid: Grid, force: Force,
                 use_coulomb: bool = True) -> float:
    """Total energy from the density functional."""
    e = compute_integrated_energy(
        state.densities, force, grid,
        coulomb_potential=state.wcoul,
        pairing_energy=jnp.zeros(2),
        mass_number=int(round(float(jnp.sum(state.wocc)))),
        use_coulomb=use_coulomb,
    )
    return float(e.ehfint)


def run_tdhf(
    initial: TDState,
    grid: Grid,
    force: Force,
    config: Optional[TDConfig] = None,
    callback: Optional[Callable[[TDState, dict], None]] = None,
) -> Tuple[TDState, list]:
    """
    Run TDHF, returns the final state and the diagnostic history.
    """
    if config is None:
        config = TDConfig()

    state = initial
    E0 = total_energy(state, grid, force, config.use_coulomb)
    history = []

    def record(st: TDState) -> dict:
        E = total_energy(st, grid, force, config.use_coulomb)
        rec = dict(
            step=st.step, time=st.time, E=E,
            dE_rel=(E - E0) / abs(E0),
            ortho=orthonormality_error(st.psi, st.isospin, grid.wxyz),
            **compute_moments(st.densities, grid),
        )
        history.append(rec)
        return rec

    rec = record(state)
    if config.verbose:
        print(f"  t=0  E={E0:.6f} MeV  A={rec['A']:.4f}")

    for _ in range(config.n_steps):
        state = tdhf_step(state, grid, force, config)
        if state.step % config.diag_interval == 0 or state.step == config.n_steps:
            rec = record(state)
            if callback is not None:
                callback(state, rec)
            if config.verbose:
                print(f"  t={state.time:8.2f} fm/c  E={rec['E']:14.6f}  "
                      f"dE/E={rec['dE_rel']:+.3e}  ortho={rec['ortho']:.2e}  "
                      f"A={rec['A']:.4f}")

    return state, history


# strength functions, taken from Sky3d

def strength_function(times, q_of_t, eta, e_max=60.0, n_e=600, window=True):
    """
    Strength distribution S(E) from the time signal of a one-body observable.
    Args:
        times: sample times in fm/c
        q_of_t: <Q>(t) at those times
        eta: the boost amplitude used
        e_max, n_e: energy grid
        window: apply the cos^2 filter

    Returns:
        (energies in MeV, strength)
    """
    import numpy as np
    t = np.asarray(times, dtype=float)
    q = np.asarray(q_of_t, dtype=float)
    dq = q - q[0]

    if window and t[-1] > 0:
        dq = dq * np.cos(np.pi * t / (2.0 * t[-1])) ** 2

    E = np.linspace(0.0, e_max, n_e)
    # trapezoidal in t for each E
    phase = np.outer(E / HBARC, t)                 # (n_e, n_t)
    integrand = np.sin(phase) * dq[None, :]
    S = np.trapezoid(integrand, t, axis=1) / (np.pi * eta)
    return E, S


def strength_moments(E, S, e_min=0.0):
    """
    m0, m1 and the centroid m1/m0 of a strength distribution.
    """
    import numpy as np
    m = E >= e_min
    Em, Sm = E[m], np.abs(S[m])
    m0 = np.trapezoid(Sm, Em)
    m1 = np.trapezoid(Em * Sm, Em)
    return float(m0), float(m1), float(m1 / m0) if m0 > 0 else float('nan')
