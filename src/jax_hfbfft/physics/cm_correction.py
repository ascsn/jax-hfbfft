"""
Microscopic centre-of-mass correction E_cm = <P^2> / (2 m A), canonical basis
(Bender, Rutz, Reinhard, Maruhn, EPJA 7, 467 (2000), Eq. 14):

    <P^2> = sum_k v_k^2 <k|p^2|k> - sum_{kl} v_k v_l (v_k v_l + u_k u_l) |<k|p|l>|^2

the exchange sum running over states of the same isospin.  This is what SkyAx
subtracts for zpe = 1 (zpe_cm), and hence what the Fayans functionals were fitted
with.  Host convention: force.zpe = 3 (see energies.py / solver.py).
"""
import jax.numpy as jnp


def _grad_psi(psi, grid):
    """d/dx, d/dy, d/dz of spinor orbitals psi (nst, 2, nx, ny, nz)."""
    gx = jnp.einsum('ij,abjkl->abikl', grid.der1x, psi)
    gy = jnp.einsum('jl,abilk->abijk', grid.der1y, psi)
    gz = jnp.einsum('kl,abijl->abijk', grid.der1z, psi)
    return gx, gy, gz


def microscopic_cm_energy(psi, wocc, wstates, isospin, grid, h2m):
    """E_cm in MeV.  wocc = v^2 per canonical state; wstates = state weights."""
    w = grid.wxyz
    v2 = wocc * wstates
    v = jnp.sqrt(jnp.clip(wocc, 0.0, 1.0)) * jnp.sqrt(wstates)
    u = jnp.sqrt(jnp.clip(1.0 - wocc, 0.0, 1.0)) * jnp.sqrt(wstates)
    grads = _grad_psi(psi, grid)
    one_body = sum(w * jnp.sum(v2[:, None, None, None, None] * jnp.abs(g) ** 2) for g in grads)
    two_body = 0.0
    same = (isospin[:, None] == isospin[None, :])
    fac = v[:, None] * v[None, :] * (v[:, None] * v[None, :] + u[:, None] * u[None, :]) * same
    for g in grads:
        pkl = w * jnp.einsum('kaxyz,laxyz->kl', jnp.conjugate(psi), g)   # <k|d|l>
        two_body = two_body + jnp.sum(fac * jnp.abs(pkl) ** 2)
    A = jnp.sum(v2)
    h2m_av = 0.5 * (h2m[0] + h2m[1])
    return h2m_av * (one_body - two_body) / A
