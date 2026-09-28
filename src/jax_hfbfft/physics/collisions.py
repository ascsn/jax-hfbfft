"""
Initial states and diagnostics for TDHF heavy-ion collisions.

Two fragments, each a converged static (ipair=0) solution, are placed in one
box, boosted toward each other along z, orthonormalized, and propagated as a
single Slater determinant.

Hamiltonian consistency: for forces with the (A-1)/A centre-of-mass correction
(zpe=0), the composite of mass A1 + A2 is propagated with
apply_cm_correction(force, A1 + A2). Relax each fragment with that same force,
not with its own (A-1)/A, or the fragments start out of equilibrium.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import jax.numpy as jnp

from jax_hfbfft.core.grid import Grid
from jax_hfbfft.core.force import Force
from jax_hfbfft.physics.coulomb import CoulombSolver
from jax_hfbfft.physics.densities import Densities, compute_densities
from jax_hfbfft.physics.dynamics import TDState, apply_boost, build_fields


@dataclass
class Fragment:
    """The occupied orbitals of one converged nucleus."""
    psi: jnp.ndarray       # (n_occupied, 2, nx, ny, nz)
    isospin: jnp.ndarray   # (n_occupied,)
    Z: int
    N: int

    @property
    def A(self) -> int:
        return self.Z + self.N

    @classmethod
    def from_static(cls, state, tol: float = 1e-6) -> "Fragment":
        """Take the occupied orbitals of a run_hfb result (ipair=0)."""
        psi, isospin = occupied_orbitals(state.psi, state.wocc, state.isospin, tol)
        iso = np.asarray(isospin)
        return cls(psi=psi, isospin=isospin,
                   Z=int((iso == 1).sum()), N=int((iso == 0).sum()))


def occupied_orbitals(psi, wocc, isospin, tol: float = 1e-6):
    """
    The orbitals with occupation 1, as (psi, isospin).

    Raises:
        ValueError: if any occupation is fractional (paired states need TDHFB).
    """
    w = np.asarray(wocc)
    frac = np.minimum(np.abs(w), np.abs(1.0 - w))
    if frac.max() > tol:
        raise ValueError(
            f"{int((frac > tol).sum())} states are fractionally occupied "
            f"(worst {frac.max():.3e}); TDHF needs a static solution with ipair=0."
        )
    keep = np.where(np.round(w) > 0.5)[0]
    return psi[keep], isospin[keep]


def translate(psi, grid: Grid, dx_fm: float = 0.0, dz_fm: float = 0.0):
    """
    Translate by whole grid points along x and z (a periodic roll, so exact).

    Returns (psi, actual_dx, actual_dz): the requested shifts are rounded to
    the grid spacing. The wrap-around is harmless only if the fragment density
    is negligible at the box edge; check boundary_density.
    """
    nx_shift = int(round(dx_fm / grid.dx))
    nz_shift = int(round(dz_fm / grid.dz))
    out = jnp.roll(psi, nx_shift, axis=2)
    out = jnp.roll(out, nz_shift, axis=4)
    return out, nx_shift * grid.dx, nz_shift * grid.dz


def lowdin_orthonormalize(psi, isospin, wxyz: float):
    """
    Symmetric (Loewdin) orthonormalization within each isospin block.

    Symmetric, so neither fragment is favoured. Neutron-proton overlaps are
    left alone: the two species need not be orthogonal.

    Returns (psi, max |S - I| before orthonormalization).

    Raises:
        ValueError: if an overlap matrix is singular (overlapping fragments).
    """
    psi = np.asarray(psi)
    iso = np.asarray(isospin)
    out = np.array(psi)
    worst = 0.0
    for iq in (0, 1):
        idx = np.where(iso == iq)[0]
        if len(idx) == 0:
            continue
        flat = psi[idx].reshape(len(idx), -1)
        S = (flat.conj() @ flat.T) * wxyz
        worst = max(worst, float(np.abs(S - np.eye(len(idx))).max()))
        ev, U = np.linalg.eigh(S)
        if ev.min() <= 0:
            raise ValueError(
                f"Overlap matrix for isospin {iq} is singular (min eigenvalue "
                f"{ev.min():.2e}); the fragments overlap. Increase the separation."
            )
        s_inv_half = (U * ev**-0.5) @ U.conj().T
        # phi_i = sum_j psi_j (S^-1/2)_ji; with states as rows this is the transpose.
        out[idx] = (s_inv_half.T @ flat).reshape(psi[idx].shape)
    return jnp.asarray(out), worst


def collision_boosts(frag1: Fragment, frag2: Fragment, force: Force,
                     e_cm: float) -> Tuple[float, float]:
    """
    Per-nucleon wave numbers (k1, k2) in fm^-1 giving kinetic energy e_cm
    (MeV) in the centre-of-mass frame.

    Zero total momentum requires A1 k1 = A2 k2, and
    E_cm = k1^2 S1 + k2^2 S2 with S_i = N_i h2m_n + Z_i h2m_p.
    """
    h2m = np.asarray(force.h2m)
    S1 = frag1.N * h2m[0] + frag1.Z * h2m[1]
    S2 = frag2.N * h2m[0] + frag2.Z * h2m[1]
    ratio = frag1.A / frag2.A
    k1 = float(np.sqrt(e_cm / (S1 + ratio**2 * S2)))
    return k1, ratio * k1


def make_collision_state(
    frag1: Fragment,
    frag2: Fragment,
    grid: Grid,
    force: Force,
    e_cm: float,
    separation: float,
    impact_parameter: float = 0.0,
    coulomb_solver: Optional[CoulombSolver] = None,
    use_coulomb: bool = True,
    boost: bool = True,
) -> Tuple[TDState, dict]:
    """
    Place two fragments on the z axis and boost them toward each other.

    Fragment 1 is placed at negative z and moves toward +z. Offsets are
    weighted by the partner's mass so that the centre of mass sits at the
    origin; positions are rounded to the grid.

    Args:
        frag1, frag2: The fragments (see Fragment.from_static).
        grid: The collision box.
        force: The composite Hamiltonian (see the module docstring).
        e_cm: Centre-of-mass kinetic energy, MeV.
        separation: Initial distance between the centres along z, fm.
        impact_parameter: Offset along x, fm.
        coulomb_solver: Defaults to CoulombSolver.create(grid).
        use_coulomb: Include the Coulomb interaction.
        boost: If False, build the same configuration at rest (useful for
            checking the injected kinetic energy).

    Returns:
        (state, info) where info holds the actual positions, k1, k2 and the
        largest fragment-fragment overlap before orthonormalization.
    """
    a_tot = frag1.A + frag2.A
    psi1, x1, z1 = translate(frag1.psi, grid,
                             dx_fm=-impact_parameter * frag2.A / a_tot,
                             dz_fm=-separation * frag2.A / a_tot)
    psi2, x2, z2 = translate(frag2.psi, grid,
                             dx_fm=+impact_parameter * frag1.A / a_tot,
                             dz_fm=+separation * frag1.A / a_tot)
    k1, k2 = collision_boosts(frag1, frag2, force, e_cm)
    if boost:
        psi1 = apply_boost(psi1, grid, (0.0, 0.0, +k1))
        psi2 = apply_boost(psi2, grid, (0.0, 0.0, -k2))

    psi = jnp.concatenate([psi1, psi2], axis=0)
    isospin = jnp.concatenate([frag1.isospin, frag2.isospin])
    wocc = jnp.ones(psi.shape[0])
    psi, overlap = lowdin_orthonormalize(psi, isospin, grid.wxyz)

    if coulomb_solver is None:
        coulomb_solver = CoulombSolver.create(grid)
    densities, meanfield, wcoul = build_fields(
        psi, wocc, isospin, grid, force, coulomb_solver, use_coulomb,
    )
    state = TDState(psi=psi, wocc=wocc, isospin=isospin, densities=densities,
                    meanfield=meanfield, coulomb_solver=coulomb_solver,
                    wcoul=wcoul, time=0.0, step=0)
    info = dict(x1=x1, z1=z1, x2=x2, z2=z2, k1=k1, k2=k2, max_overlap=overlap)
    return state, info


# ── Diagnostics ──────────────────────────────────────────────────────────────

def fragment_separation(densities: Densities, grid: Grid) -> float:
    """
    Distance between the density centroids of the z < 0 and z > 0 halves.

    Defined at every instant without fragment finding; small when the system
    is one object. It is measured along the beam axis, so after appreciable
    rotation it is the projection of the true distance.
    """
    rho = densities.rho[0] + densities.rho[1]
    _, _, Z = grid.get_meshgrid()
    hi = Z > 0
    m_hi = jnp.sum(rho * hi)
    m_lo = jnp.sum(rho * ~hi)
    z_hi = jnp.sum(rho * Z * hi) / m_hi
    z_lo = jnp.sum(rho * Z * ~hi) / m_lo
    return float(z_hi - z_lo)


def boundary_density(densities: Densities, grid: Grid) -> float:
    """Largest density on the six faces of the box (fm^-3)."""
    rho = np.asarray(densities.rho[0] + densities.rho[1])
    faces = [rho[0], rho[-1], rho[:, 0], rho[:, -1], rho[:, :, 0], rho[:, :, -1]]
    return max(float(np.abs(f).max()) for f in faces)


def boundness(psi, isospin, grid: Grid, mass_number: int,
              radius_factor: float = 2.5) -> dict:
    """
    Check that a static fragment is bound before using it.

    An over-iterated static solution can lower its energy by moving a fraction
    of a nucleon into unbound box states. Such a state is stationary, so the
    usual convergence measures do not flag it; the density far outside the
    nucleus does.

    Returns a dict with peak_rho (fm^-3), a_outside (nucleons beyond
    radius_factor * 1.2 A^(1/3) fm), face (largest density on the box faces),
    and bound (False if a_outside > 0.05, face > 1e-4 or peak_rho > 0.20).
    """
    ones = jnp.ones(psi.shape[0])
    dens = compute_densities(psi, ones, jnp.zeros_like(ones), ones, ones, isospin, grid)
    rho = np.asarray(dens.rho[0] + dens.rho[1])
    X, Y, Z = (np.asarray(a) for a in grid.get_meshgrid())
    radius = radius_factor * 1.2 * mass_number ** (1.0 / 3.0)
    far = np.sqrt(X**2 + Y**2 + Z**2) > radius
    a_out = float((rho * far).sum() * grid.wxyz)
    face = boundary_density(dens, grid)
    peak = float(rho.max())
    return dict(peak_rho=peak, a_outside=a_out, radius=radius, face=face,
                bound=not (a_out > 0.05 or face > 1e-4 or peak > 0.20))
