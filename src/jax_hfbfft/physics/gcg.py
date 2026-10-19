"""
Generalized conjugate-gradient (GCG) orbital step for the lowest eigenvectors of
the mean-field Hamiltonian, after Sala and Colo, arXiv:2606.17652.

For each isospin block with orbitals F (orthonormal, the current approximation to
the lowest n eigenvectors of h), one step builds the subspace

    W = (h - e_s)^-1 r,   r_k = (h - e_k) phi_k     (inverse-Hamiltonian direction)
    P = F - F_prev <F_prev|F>                       (conjugate direction)
    F                                               (current orbitals)

with e_k = <phi_k|h|phi_k> and the shift e_s = e_0 - |e_0|/100 below the lowest
level, and replaces F by the n lowest Ritz vectors of h in span{W, P, F}.  The
inverse is applied approximately, by n_inner conjugate-gradient steps with the
kinetic preconditioner (h2m k^2 + |e_0|/100)^-1 (Sala and Colo write W as the
solution of (h - e_s) W_k = (e_k - e_s) phi_k, which is phi_k minus our W).

Only the eigenvectors of h are sought, so this replaces the gradient step of HF
and BCS (tbcs) runs, where the orbitals are eigenstates of h; full HFB needs the
two-basis method on top (HFB solved in the space of these orbitals).
"""

from functools import partial

import jax
import jax.numpy as jnp

from jax_hfbfft.physics.meanfield import apply_hamiltonian


def _block_step(F, F_prev, mf, q: int, grid, n_inner: int, h2m: float):
    """One GCG step for the orbitals F (n, 2, nx, ny, nz) of isospin q. Returns (F, e, hF)."""
    n, w = F.shape[0], grid.wxyz
    h = lambda x: jax.vmap(lambda p: apply_hamiltonian(p, mf, q, grid))(x)
    ip = lambda a, b: w * jnp.einsum('ksxyz,lsxyz->kl', jnp.conj(a), b)

    hF = h(F)
    e = jnp.real(jnp.einsum('ksxyz,ksxyz->k', jnp.conj(F), hF)) * w
    sigma = jnp.abs(jnp.min(e)) / 100.0
    shift = jnp.min(e) - sigma
    r = hF - e[:, None, None, None, None] * F

    # W = (h - shift)^-1 r by preconditioned CG (fixed n_inner steps, all orbitals at once)
    fft = lambda x: jnp.fft.fftn(x, axes=(-3, -2, -1))
    ifft = lambda x: jnp.fft.ifftn(x, axes=(-3, -2, -1))
    Minv = lambda x: ifft(fft(x) / (h2m * grid.k2 + sigma))
    A = lambda x: h(x) - shift * x
    dot = lambda a, b: jnp.real(jnp.einsum('ksxyz,ksxyz->k', jnp.conj(a), b))[:, None, None, None, None]

    def cg(_, st):
        x, res, z, p, rz = st
        Ap = A(p)
        alpha = rz / jnp.where(dot(p, Ap) == 0, 1.0, dot(p, Ap))
        x, res = x + alpha * p, res - alpha * Ap
        z = Minv(res)
        rz_new = dot(res, z)
        return x, res, z, z + rz_new / jnp.where(rz == 0, 1.0, rz) * p, rz_new

    z0 = Minv(r)
    W, *_ = jax.lax.fori_loop(0, n_inner, cg, (jnp.zeros_like(r), r, z0, z0, dot(r, z0)))

    P = F - jnp.einsum('lk,lsxyz->ksxyz', ip(F_prev, F), F_prev)
    # W and P orthogonal to F and of unit norm: near convergence they are as small as the
    # residual, and unnormalised they fall below the linear-dependence cut and the step stalls
    def prepare(X):
        X = X - jnp.einsum('lk,lsxyz->ksxyz', ip(F, X), F)
        norm = jnp.sqrt(jnp.real(jnp.einsum('ksxyz,ksxyz->k', jnp.conj(X), X)) * w)
        return X / jnp.where(norm > 1e-14, norm, 1.0)[:, None, None, None, None]
    W, P = prepare(W), prepare(P)
    V = jnp.concatenate([W, P, F])                                        # (3n, ...)
    hV = jnp.concatenate([h(W), h(P), hF])

    # Rayleigh-Ritz with S-orthonormalisation; genuinely dependent directions (P = 0 at
    # the first step, W or P inside the rest) are removed by giving them a large Ritz value
    S = ip(V, V)
    s, U = jnp.linalg.eigh(0.5 * (S + jnp.conj(S.T)))
    keep = s > 1e-10 * jnp.max(s)
    X = U * jnp.where(keep, 1.0 / jnp.sqrt(jnp.where(keep, s, 1.0)), 0.0)[None, :]
    Hr = jnp.conj(X.T) @ ip(V, hV) @ X
    Hr = 0.5 * (Hr + jnp.conj(Hr.T)) + jnp.diag(jnp.where(keep, 0.0, 1e8))
    theta, C = jnp.linalg.eigh(Hr)
    coef = X @ C[:, :n]                                                   # (3n, n)
    comb = lambda x: jnp.einsum('jk,jsxyz->ksxyz', coef, x)
    return comb(V), theta[:n], comb(hV)


@partial(jax.jit, static_argnames=('npsi_n', 'n_inner'))
def gcg_step(psi, psi_prev, meanfield, grid, npsi_n: int, n_inner: int = 8, h2m: float = 20.73553):
    """
    One GCG step for both isospins (neutrons are psi[:npsi_n]); psi_prev are the
    orbitals of the previous step (or psi itself at the first one).

    Returns (psi, sp_energy, hpsi): the n lowest Ritz vectors of h per isospin, their
    Ritz values, and h applied to them.
    """
    out = [_block_step(psi[sl], psi_prev[sl], meanfield, q, grid, n_inner, h2m)
           for q, sl in ((0, slice(0, npsi_n)), (1, slice(npsi_n, None)))]
    return tuple(jnp.concatenate(x) for x in zip(*out))
