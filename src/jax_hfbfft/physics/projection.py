"""
Particle-number projection after variation (PAV) of a converged even-even static
solution.

In the canonical basis the gauge-rotated state e^{i phi N}|Phi> has, with
z = e^{2 i phi} and a product over Kramers pairs,

    x(phi) = <Phi|e^{i phi N}|Phi> = prod_k (u_k^2 + v_k^2 z),

and transition densities that are the orbital sums with complex weights

    rho_k = v_k^2 z / d_k,   kappa_k = u_k v_k z / d_k,   kappabar_k* = u_k v_k / d_k,
    d_k = u_k^2 + v_k^2 z.

The projected energy follows the mixed-density prescription,

    E_NZ = sum_lm y_n(l) y_p(m) E[phi_l, phi_m] / (sum_l y_n(l) sum_m y_p(m)),
    y_q(l) = e^{-i phi_l N_q} x_q(phi_l),

on the Fomenko points phi_l = pi l / L, exact for components |N' - N| < 2L.
The density-dependent terms take the transition density to the power alpha
(principal branch); with them the projected energy is not free of the
self-interaction poles at v^2 = 1/2 (Dobaczewski et al., PRC 76, 054315), so
check the result against L.
"""

import jax
import jax.numpy as jnp
import numpy as np

from jax_hfbfft.physics.coulomb import solve_poisson
from jax_hfbfft.physics.densities import compute_densities
from jax_hfbfft.physics.energies import compute_integrated_energy


def _log_overlap(v2, phi):
    """log(u^2 + v^2 e^{2 i phi}), continuous in phi on [0, pi] (the circle winds around 0 when v^2 > u^2)."""
    u2, z = 1.0 - v2, jnp.exp(2j * phi)
    return jnp.where(v2 > u2, 2j * phi + jnp.log(v2 + u2 / z), jnp.log(u2 + v2 * z))


def project_particle_number(state, grid, force, n_gauge: int = 11, use_coulomb: bool = True) -> dict:
    """
    Number-projected energy of a static solution (state from run_hfb), with both
    isospins projected on their particle numbers on n_gauge (odd) Fomenko points.

    Returns dict(energy, unprojected, correction, imag, distribution): `energy` and
    `unprojected` are the functional with and without projection (the Lipkin-Nogami
    term left out; zpe as in the iterations), `correction` their difference, to be
    added to the static total energy (minus its LN energy), `imag` the imaginary part
    of the projected energy (zero up to the Fomenko error), and distribution[q] the
    probabilities {N': |<N'|Phi>|^2} for N' = N_q + 2j.
    """
    if n_gauge % 2 == 0:
        raise ValueError("n_gauge must be odd (an even L puts a gauge point on the v^2 = 1/2 pole)")
    if float(jnp.sum(jnp.abs(getattr(state.pairing, 'entropy', 0.0)))) > 0:
        raise NotImplementedError("projection of a finite-temperature state")
    iso = np.asarray(state.isospin)
    v2 = jnp.clip(state.wocc, 0.0, 1.0)
    w = state.wstates
    N = [float(jnp.sum(jnp.where(state.isospin == q, w * v2, 0.0))) for q in (0, 1)]
    if any(abs(n - 2 * round(n / 2)) > 1e-3 for n in N):
        raise NotImplementedError("projection of odd particle numbers (blocked states)")
    N = [int(round(n)) for n in N]
    phis = np.pi * np.arange(1, n_gauge + 1) / n_gauge
    psi2 = jnp.sum(jnp.abs(state.psi) ** 2, axis=1)                      # (n, nx, ny, nz)
    pw = 0.5 * w * state.pairwg * state.wguv
    tensor = state.densities.jsym is not None

    def rotated(phi):
        """Transition densities at gauge angle phi (all states rotated) and per-isospin pair densities."""
        z = jnp.exp(2j * phi)
        d = 1.0 - v2 + v2 * z
        rw = v2 * z / d
        dens = [compute_densities(state.psi, part(rw), 0.0 * v2, state.pairwg, w, state.isospin, grid,
                                  tensor=tensor) for part in (jnp.real, jnp.imag)]
        dens = jax.tree_util.tree_map(lambda a, b: a + 1j * b, *dens)
        chi = lambda c: jnp.stack([jnp.einsum('k,kxyz->xyz', jnp.where(iso == q, c, 0.0), psi2) for q in (0, 1)])
        return dens, chi(pw * z / d), chi(pw / d)

    def log_x(phi, q):
        return jnp.sum(jnp.where(iso == q, 0.5 * w * _log_overlap(v2, phi), 0.0))

    def coulomb(rho_p):
        if not use_coulomb:
            return None
        V = lambda r: solve_poisson(r, state.coulomb_solver, grid)
        return V(jnp.real(rho_p)) + 1j * V(jnp.imag(rho_p))

    A = sum(N)
    g_pair = {5: lambda rho: 1.0, 6: lambda rho: 1.0 - rho / force.rho0pr}.get(force.ipair, lambda rho: 0.0)

    @jax.jit
    def energy(rn, rp, vc):
        """E at the neutron and proton gauge angles of rotated() results rn, rp."""
        dens = jax.tree_util.tree_map(lambda a, b: jnp.stack([a[0], b[1]]), rn[0], rp[0])
        g = g_pair(dens.rho[0] + dens.rho[1])
        epair = jnp.stack([force.v0neut * grid.wxyz * jnp.sum(g * rn[1][0] * rn[2][0]),
                           force.v0prot * grid.wxyz * jnp.sum(g * rp[1][1] * rp[2][1])])
        return compute_integrated_energy(dens, force, grid, coulomb_potential=vc, pairing_energy=epair,
                                         mass_number=A, use_coulomb=use_coulomb).ehfint

    rot = [rotated(phi) for phi in phis]
    vcs = [coulomb(r[0].rho[1]) for r in rot]
    y = np.array([[np.exp(-1j * phi * N[q] + complex(log_x(phi, q))) for phi in phis] for q in (0, 1)])
    E = np.array([[complex(energy(rn, rp, vc)) for rp, vc in zip(rot, vcs)] for rn in rot])
    proj = (y[0] @ E @ y[1]) / (y[0].sum() * y[1].sum())

    d0 = rotated(0.0)
    e0 = float(jnp.real(energy(d0, d0, coulomb(d0[0].rho[1]))))
    js = np.arange(-(n_gauge // 2), n_gauge // 2 + 1)
    dist = [{N[q] + 2 * j: float(np.real(np.mean(y[q] * np.exp(-2j * phis * j)))) for j in js if N[q] + 2 * j >= 0}
            for q in (0, 1)]
    return dict(energy=float(proj.real), unprojected=e0, correction=float(proj.real) - e0,
                imag=float(proj.imag), distribution=dict(n=dist[0], p=dist[1]))
