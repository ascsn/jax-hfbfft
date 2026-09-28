"""
Time-dependent Hartree-Fock (TDHF).

Real-time propagation of a Slater determinant,

    i hbar d(psi)/dt = h[rho(t)] psi,

starting from a converged static solution (run_hfb with ipair=0).

The exact evolution is unitary and conserves the total energy, so the numerical
drift of orthonormality and energy is the main accuracy diagnostic. Pairing is
not supported: prepare_tdhf_state rejects fractionally occupied states.

Units: energies in MeV, lengths in fm, times in fm/c.
"""

import dataclasses
import logging
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp

from jax_hfbfft.core.grid import Grid
from jax_hfbfft.core.force import Force
from jax_hfbfft.physics.densities import Densities, compute_densities
from jax_hfbfft.physics.meanfield import (
    Meanfield, compute_skyrme_meanfield, apply_hamiltonian,
)
from jax_hfbfft.physics.energies import compute_integrated_energy
from jax_hfbfft.physics.coulomb import CoulombSolver, solve_poisson

logger = logging.getLogger(__name__)

# hbar*c in MeV fm. With h in MeV and dt in fm/c, h*dt/HBARC is dimensionless.
HBARC = 197.3269804


@dataclass(frozen=True)
class TDConfig:
    """
    Configuration for a TDHF run.

    Frozen (hashable) so it can be passed to jitted functions as a static
    argument; changing any field triggers one recompilation.
    """
    dt: float = 0.2                    # time step (fm/c)
    n_steps: int = 1000                # steps taken by run_tdhf
    taylor_order: int = 6              # order of the Taylor propagator
    predictor_corrector: bool = True   # evaluate h at the mid-step density
    use_coulomb: bool = True
    diag_interval: int = 10            # steps between recorded diagnostics
    density_chunk: int = 0             # see compute_densities(chunk=...)


@jax.tree_util.register_dataclass
@dataclass
class TDState:
    """State of a TDHF calculation. Occupations are fixed for the whole run."""
    psi: jax.Array            # (nstates, 2, nx, ny, nz) complex
    wocc: jax.Array           # (nstates,) occupations, all 0 or 1
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


# ── Propagator ───────────────────────────────────────────────────────────────

@partial(jax.jit, static_argnums=(5,))
def propagate_taylor(
    psi: jax.Array,
    meanfield: Meanfield,
    isospin: jax.Array,
    grid: Grid,
    dt: float,
    order: int,
) -> jax.Array:
    """
    Advance psi by exp(-i h dt / hbar), truncated at `order` in the Taylor series.

    Uses term_n = (-i dt/hbar / n) h term_{n-1}, i.e. `order` applications of h.
    The truncation makes each step non-unitary at O(dt^(order+1)).
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
    density_chunk: int = 0,
) -> Tuple[Densities, Meanfield, jax.Array]:
    """
    Densities, mean field and Coulomb potential from the current wavefunctions.

    Unlike the static solver, the time-odd densities (current, spin) are kept:
    a moving nucleus has them and they act on the dynamics.
    """
    ones = jnp.ones_like(wocc)
    densities = compute_densities(
        psi, wocc, jnp.zeros_like(wocc), ones, ones, isospin, grid,
        chunk=density_chunk,
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
    Advance the state by one time step of config.dt.

    With predictor_corrector (the default, following Sky3D), a full step with
    h[rho(t)] predicts rho(t + dt); the average of rho(t) and the prediction
    gives the mid-step field h[rho(t + dt/2)], which is then used for the real
    step. This makes the scheme second-order accurate in dt. Without it, the
    whole step uses h[rho(t)] and is first-order accurate.

    This function is pure and can be jitted; see advance() for the jitted,
    multi-step form.
    """
    if config.predictor_corrector:
        # The predictor only locates the mid-step field, so half the order suffices.
        psi_pred = propagate_taylor(
            state.psi, state.meanfield, state.isospin, grid,
            config.dt, max(1, config.taylor_order // 2),
        )
        dens_pred, _, _ = build_fields(
            psi_pred, state.wocc, state.isospin, grid, force,
            state.coulomb_solver, False, config.density_chunk,
        )
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
        state.coulomb_solver, config.use_coulomb, config.density_chunk,
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


@partial(jax.jit, static_argnames=("config", "n_steps"))
def _advance_jit(state, grid, force, config, n_steps):
    return jax.lax.fori_loop(
        0, n_steps, lambda _, s: tdhf_step(s, grid, force, config), state
    )


def advance(
    state: TDState,
    grid: Grid,
    force: Force,
    config: TDConfig,
    n_steps: int,
) -> TDState:
    """
    Take n_steps time steps in a single compiled call.

    Several times faster than calling tdhf_step in a Python loop for small
    systems, where per-call dispatch overhead dominates. Each distinct n_steps
    compiles once.
    """
    state = dataclasses.replace(
        state,
        time=jnp.asarray(state.time, dtype=state.densities.rho.dtype),
        step=jnp.asarray(state.step, dtype=jnp.int32),
    )
    # Run length and diagnostic spacing do not affect a step; dropping them from
    # the static config avoids recompiling when only they change.
    config = dataclasses.replace(config, n_steps=0, diag_interval=0)
    return _advance_jit(state, grid, force, config, int(n_steps))


# ── Initial conditions ───────────────────────────────────────────────────────

def prepare_tdhf_state(
    static_state,
    grid: Grid,
    force: Optional[Force] = None,
    use_coulomb: bool = True,
    occupation_tol: float = 1e-6,
    stationarity_tol: float = 1.0,
    drop_unoccupied: bool = True,
    density_chunk: int = 0,
) -> TDState:
    """
    Convert a converged static SolverState into a TDHF starting state.

    Args:
        static_state: Result of run_hfb (ipair=0).
        grid: The grid the state lives on.
        force: Hamiltonian to propagate with. Defaults to static_state.force,
            the force run_hfb relaxed the state in; a different force makes the
            starting state non-stationary.
        use_coulomb: Include the Coulomb interaction.
        occupation_tol: Largest allowed deviation of an occupation from 0 or 1.
        stationarity_tol: Warn if ||(h - <h>) psi|| exceeds this (MeV).
        drop_unoccupied: Propagate only occupied orbitals. Exact (empty states
            contribute nothing) and saves the cost of the empty ones.
        density_chunk: see compute_densities(chunk=...).

    Raises:
        ValueError: if occupations are fractional (a paired state needs TDHFB)
            or no force is available.
    """
    if force is None:
        force = getattr(static_state, "force", None)
    if force is None:
        raise ValueError(
            "No force given and the static state does not record one; pass the "
            "force the state was relaxed in."
        )

    wocc = jnp.asarray(static_state.wocc)
    frac = jnp.minimum(jnp.abs(wocc), jnp.abs(1.0 - wocc))
    worst = float(jnp.max(frac))
    if worst > occupation_tol:
        n_frac = int(jnp.sum(frac > occupation_tol))
        raise ValueError(
            f"TDHF requires sharp occupations, but {n_frac} states are "
            f"fractional (worst deviation {worst:.3e}). Run the static "
            f"calculation with ipair=0; paired states need TDHFB."
        )
    wocc = jnp.round(wocc)
    psi, isospin = static_state.psi, static_state.isospin
    if drop_unoccupied:
        keep = jnp.nonzero(wocc > 0.5)[0]
        psi, wocc, isospin = psi[keep], wocc[keep], isospin[keep]

    if force.has_time_odd:
        msg = (f"s-channel time-odd couplings are active (c_s0={force.c_s0:.1f}, "
               f"c_s1={force.c_s1:.1f}); Sky3D omits these terms.")
        rho_crit = _spin_stability_density(force)
        if rho_crit is not None:
            msg += (f" C^s_0(rho) is negative below rho = {rho_crit:.5f} fm^-3, "
                    f"where spin density is unstable and can grow exponentially.")
        warnings.warn(msg, stacklevel=2)

    coulomb_solver = static_state.coulomb_solver
    densities, meanfield, wcoul = build_fields(
        psi, wocc, isospin, grid, force, coulomb_solver, use_coulomb, density_chunk,
    )
    state = TDState(
        psi=psi,
        wocc=wocc,
        isospin=isospin,
        densities=densities,
        meanfield=meanfield,
        coulomb_solver=coulomb_solver,
        wcoul=wcoul,
        time=0.0,
        step=0,
    )

    # A state relaxed in a different Hamiltonian is not stationary and drifts.
    resid = stationarity_residual(state, grid)
    if resid > stationarity_tol:
        warnings.warn(
            f"Starting state is not an eigenstate of the TDHF Hamiltonian: "
            f"max ||(h - <h>) psi|| = {resid:.3e} MeV. This usually means it was "
            f"relaxed with a different force.",
            stacklevel=2,
        )
    return state


def _spin_stability_density(force: Force) -> Optional[float]:
    """Density below which C^s_0(rho) = c_s0 + c_ds0 rho^alpha is negative, or None."""
    if force.c_ds0 == 0.0 or force.c_s0 >= 0.0:
        return None
    return (abs(force.c_s0) / force.c_ds0) ** (1.0 / force.power)


def stationarity_residual(state: TDState, grid: Grid) -> float:
    """
    Largest ||(h - <h>) psi|| over the occupied states, in MeV.

    Zero for a state built from eigenstates of the current mean field.
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
    Translational boost psi -> exp(i k.r) psi: momentum hbar*k per nucleon.

    On a periodic box, exp(i k.r) is continuous across the boundary only if
    k is a multiple of 2 pi / L; otherwise the wavefunctions must be
    negligible at the box edge or the boost adds spurious kinetic energy.
    """
    X, Y, Z = grid.get_meshgrid()
    ph = jnp.exp(1j * (k[0] * X + k[1] * Y + k[2] * Z))
    return psi * ph[None, None, :, :, :]


def apply_multipole_boost(
    psi: jax.Array, grid: Grid, operator: jax.Array, eta: float,
) -> jax.Array:
    """
    Small-amplitude excitation psi -> exp(-i eta Q) psi for a one-body operator Q
    (sign as in Sky3D). Excites all modes of Q's symmetry at once; the response
    <Q>(t) gives the strength function. Keep eta small enough that the
    response is linear (halving eta should halve it).
    """
    ph = jnp.exp(-1j * eta * operator)
    return psi * ph[None, None, :, :, :]


def apply_isovector_dipole_boost(
    psi: jax.Array, isospin: jax.Array, grid: Grid, eta: float, Z: int, N: int,
) -> jax.Array:
    """
    Isovector dipole kick along z: protons and neutrons pushed in opposite
    directions, with weights +N/A (protons) and -Z/A (neutrons) so that the
    centre of mass is not excited.
    """
    A = Z + N
    _, _, ZC = grid.get_meshgrid()
    w = jnp.where(isospin == 1, N / A, -Z / A)
    return psi * jnp.exp(-1j * eta * w[:, None, None, None, None] * ZC[None, None])


def reset_cm_velocity(state: TDState, grid: Grid) -> TDState:
    """
    Remove any net centre-of-mass momentum (Sky3D's resetcm).

    Uses P / hbar = integral of j over space and boosts by k = -P / (hbar A).
    Densities and fields are not rebuilt: the boost changes only the current.
    """
    P = jnp.sum(state.densities.current, axis=(0, 2, 3, 4)) * grid.wxyz   # (3,)
    A = jnp.sum(state.densities.rho) * grid.wxyz
    k_corr = -P / jnp.maximum(A, 1e-12)
    psi_new = apply_boost(state.psi, grid,
                          (float(k_corr[0]), float(k_corr[1]), float(k_corr[2])))
    return dataclasses.replace(state, psi=psi_new)


# ── Diagnostics ──────────────────────────────────────────────────────────────

@jax.jit
def _orthonormality_error(psi, isospin, wxyz):
    flat = psi.reshape(psi.shape[0], -1)
    S = (jnp.conj(flat) @ flat.T) * wxyz
    same = isospin[:, None] == isospin[None, :]
    dev = jnp.where(same, S - jnp.eye(psi.shape[0], dtype=S.dtype), 0.0)
    return jnp.max(jnp.abs(dev))


def orthonormality_error(psi: jax.Array, isospin: jax.Array, wxyz: float) -> float:
    """
    Largest deviation of <psi_i|psi_j> from delta_ij within each isospin.

    Neutron-proton overlaps are excluded: the two species need not be
    orthogonal.
    """
    return float(_orthonormality_error(psi, isospin, wxyz))


@jax.jit
def _moments(densities: Densities, grid: Grid) -> dict:
    rho = densities.rho[0] + densities.rho[1]
    X, Y, Z = grid.get_meshgrid()
    w = grid.wxyz
    A = jnp.sum(rho) * w
    cm = jnp.array([jnp.sum(rho * X), jnp.sum(rho * Y), jnp.sum(rho * Z)]) * w / A
    r2 = jnp.sum(rho * (X**2 + Y**2 + Z**2)) * w / A
    return {
        'A': A,
        'cm_x': cm[0], 'cm_y': cm[1], 'cm_z': cm[2],
        'r2': r2,
        'r2_int': r2 - jnp.sum(cm**2),   # about the centre of mass
        'Q20': jnp.sum(rho * (2 * Z**2 - X**2 - Y**2)) * w,
        'Q22': jnp.sum(rho * (X**2 - Y**2)) * w,
    }


def compute_moments(densities: Densities, grid: Grid) -> dict:
    """
    Particle number, centre of mass, <r^2> (about the origin and about the
    centre of mass, 'r2_int'), and the Q20/Q22 quadrupole moments.
    """
    return {k: float(v) for k, v in jax.device_get(_moments(densities, grid)).items()}


@partial(jax.jit, static_argnames=("mass_number", "use_coulomb"))
def _energy(densities, wcoul, force, grid, mass_number, use_coulomb):
    return compute_integrated_energy(
        densities, force, grid,
        coulomb_potential=wcoul,
        pairing_energy=jnp.zeros(2, dtype=densities.rho.dtype),
        mass_number=mass_number,
        use_coulomb=use_coulomb,
    ).ehfint


def total_energy(state: TDState, grid: Grid, force: Force,
                 use_coulomb: bool = True) -> float:
    """Total energy from the density functional (the conserved quantity), MeV."""
    mass_number = int(round(float(jnp.sum(state.wocc))))
    return float(_energy(state.densities, state.wcoul, force, grid,
                         mass_number, use_coulomb))


@partial(jax.jit, static_argnames=("mass_number", "use_coulomb"))
def _observables(state, grid, force, mass_number, use_coulomb):
    obs = _moments(state.densities, grid)
    obs['E'] = _energy(state.densities, state.wcoul, force, grid,
                       mass_number, use_coulomb)
    obs['ortho'] = _orthonormality_error(state.psi, state.isospin, grid.wxyz)
    obs['time'] = jnp.asarray(state.time)
    obs['step'] = jnp.asarray(state.step)
    return obs


def observables(state: TDState, grid: Grid, force: Force,
                use_coulomb: bool = True) -> dict:
    """
    All standard diagnostics in one compiled call and one device-to-host copy:
    time, step, E, ortho, A, cm_x/y/z, r2, r2_int, Q20, Q22.
    """
    mass_number = int(round(float(jnp.sum(state.wocc))))
    obs = jax.device_get(_observables(state, grid, force, mass_number, use_coulomb))
    out = {k: float(v) for k, v in obs.items()}
    out['step'] = int(out['step'])
    return out


def run_tdhf(
    initial: TDState,
    grid: Grid,
    force: Force,
    config: Optional[TDConfig] = None,
    callback: Optional[Callable[[TDState, dict], None]] = None,
) -> Tuple[TDState, list]:
    """
    Propagate for config.n_steps steps, recording observables() every
    config.diag_interval steps (and at the end).

    Each record also carries dE_rel = (E - E0) / |E0|; energy drift is the
    main accuracy measure. Returns (final state, list of records).
    """
    if config is None:
        config = TDConfig()

    state = initial
    history = []
    E0 = None

    def record(st):
        nonlocal E0
        rec = observables(st, grid, force, config.use_coulomb)
        if E0 is None:
            E0 = rec['E']
        rec['dE_rel'] = (rec['E'] - E0) / abs(E0)
        history.append(rec)
        if callback is not None:
            callback(st, rec)
        logger.info("t=%8.2f fm/c  E=%14.6f MeV  dE/E=%+.3e  ortho=%.2e  A=%.4f",
                    rec['time'], rec['E'], rec['dE_rel'], rec['ortho'], rec['A'])
        return rec

    record(state)
    done = 0
    while done < config.n_steps:
        n = min(config.diag_interval, config.n_steps - done)
        state = advance(state, grid, force, config, n)
        done += n
        record(state)

    return state, history


# ── Strength functions (as in Sky3D) ─────────────────────────────────────────

def strength_function(times, q_of_t, eta, e_max=60.0, n_e=600, window=True):
    """
    Strength distribution S(E) from the time signal of a one-body observable
    after a kick exp(-i eta Q):

        S(E) = (1 / pi eta) * integral dt  delta<Q>(t) sin(E t / hbar)

    The energy resolution is about 2 pi hbar c / T for a run of length T
    (1 MeV needs T ~ 1240 fm/c). The optional cos^2 window suppresses the
    ringing caused by the abrupt end of the signal.

    Args:
        times: sample times in fm/c
        q_of_t: <Q>(t) at those times
        eta: the kick amplitude used
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
    phase = np.outer(E / HBARC, t)                 # (n_e, n_t)
    integrand = np.sin(phase) * dq[None, :]
    S = np.trapezoid(integrand, t, axis=1) / (np.pi * eta)
    return E, S


def strength_moments(E, S, e_min=0.0):
    """Moments m0, m1 and the centroid m1/m0 of |S(E)| above e_min."""
    import numpy as np
    m = E >= e_min
    Em, Sm = E[m], np.abs(S[m])
    m0 = np.trapezoid(Sm, Em)
    m1 = np.trapezoid(Em * Sm, Em)
    return float(m0), float(m1), float(m1 / m0) if m0 > 0 else float('nan')
