"""
Two-basis HFB (SolverConfig.orbital_solver = 'two_basis'; Gall et al., Z. Phys. A 348, 183 (1994);
Terasaki et al., Nucl. Phys. A 600, 371 (1996)).

Each iteration
  1. refines the HF basis, the lowest n eigenvectors of the mean-field Hamiltonian h, by one
     generalized conjugate-gradient (GCG) step (Sala and Colo, arXiv:2606.17652; gcg_step), and
  2. solves the HFB problem exactly in that basis (two_basis_hfb), returning canonical orbitals
     and occupations for the canonical-basis density code.

HF eigenstates cannot drift or collapse the way free canonical orbitals can, so the iteration has
a true fixed point. Pairing acts in a soft window on the HF energies, f_k = 1/(1 + exp((e_k -
lam - E_c)/(w E_c))) with E_c = Force.state_cutoff and w = Force.softcut_range; the window must
lie well inside the basis (e_max - lam > ~1.7 E_c), or the iteration can oscillate.
"""

from functools import partial

import jax
import jax.numpy as jnp

from jax_hfbfft.physics.meanfield import apply_hamiltonian


def _block_step(F, F_prev, mf, q: int, grid, n_inner: int, h2m: float):
    """
    One GCG step for the orbitals F (n, 2, nx, ny, nz) of isospin q: the n lowest Ritz vectors of h
    in span{W, P, F}, with W = (h - e_s)^-1 r (r_k = (h - e_k) phi_k, e_s just below the lowest
    level; n_inner preconditioned CG steps) and P = F - F_prev <F_prev|F>. Returns (F, e, hF).
    """
    n, w = F.shape[0], grid.wxyz
    h = lambda x: jax.vmap(lambda p: apply_hamiltonian(p, mf, q, grid))(x)
    ip = lambda a, b: w * jnp.einsum('ksxyz,lsxyz->kl', jnp.conj(a), b)
    dot = lambda a, b: jnp.real(jnp.einsum('ksxyz,ksxyz->k', jnp.conj(a), b))[:, None, None, None, None]

    hF = h(F)
    e = jnp.real(jnp.einsum('ksxyz,ksxyz->k', jnp.conj(F), hF)) * w
    sigma = jnp.abs(jnp.min(e)) / 100.0
    shift = jnp.min(e) - sigma
    r = hF - e[:, None, None, None, None] * F

    fft = lambda x: jnp.fft.fftn(x, axes=(-3, -2, -1))
    Minv = lambda x: jnp.fft.ifftn(fft(x) / (h2m * grid.k2 + sigma), axes=(-3, -2, -1))

    def cg(_, st):
        x, res, z, p, rz = st
        Ap = h(p) - shift * p
        pAp = dot(p, Ap)
        alpha = rz / jnp.where(pAp == 0, 1.0, pAp)
        x, res = x + alpha * p, res - alpha * Ap
        z = Minv(res)
        rz_new = dot(res, z)
        return x, res, z, z + rz_new / jnp.where(rz == 0, 1.0, rz) * p, rz_new

    z0 = Minv(r)
    W = jax.lax.fori_loop(0, n_inner, cg, (jnp.zeros_like(r), r, z0, z0, dot(r, z0)))[0]
    P = F - jnp.einsum('lk,lsxyz->ksxyz', ip(F_prev, F), F_prev)

    # W and P orthogonal to F and normalised: near convergence they are as small as the residual,
    # and unnormalised they fall below the linear-dependence cut and the step stalls
    def prepare(X):
        X = X - jnp.einsum('lk,lsxyz->ksxyz', ip(F, X), F)
        norm = jnp.sqrt(dot(X, X) * w)
        return X / jnp.where(norm > 1e-14, norm, 1.0)
    W, P = prepare(W), prepare(P)
    V = jnp.concatenate([W, P, F])
    hV = jnp.concatenate([h(W), h(P), hF])

    # Rayleigh-Ritz; dependent directions (P = 0 at the first step) get a large Ritz value
    s, U = jnp.linalg.eigh(ip(V, V))
    keep = s > 1e-10 * jnp.max(s)
    X = U * jnp.where(keep, 1.0 / jnp.sqrt(jnp.where(keep, s, 1.0)), 0.0)[None, :]
    Hr = jnp.conj(X.T) @ ip(V, hV) @ X
    theta, C = jnp.linalg.eigh(0.5 * (Hr + jnp.conj(Hr.T)) + jnp.diag(jnp.where(keep, 0.0, 1e8)))
    coef = X @ C[:, :n]
    comb = lambda x: jnp.einsum('jk,jsxyz->ksxyz', coef, x)
    return comb(V), theta[:n], comb(hV)


@partial(jax.jit, static_argnames=('npsi_n', 'n_inner'))
def gcg_step(psi, psi_prev, meanfield, grid, npsi_n: int, n_inner: int = 2, h2m: float = 20.73553):
    """One GCG step for both isospins (neutrons first). Returns (psi, ritz_values, h psi)."""
    out = [_block_step(psi[sl], psi_prev[sl], meanfield, q, grid, n_inner, h2m)
           for q, sl in ((0, slice(0, npsi_n)), (1, slice(npsi_n, None)))]
    return tuple(jnp.concatenate(x) for x in zip(*out))


def _hfb_block(h, D, n_target, lam_prev, smallp, blocked, window: float = 10.0, n_bisect: int = 36):
    """
    HFB in the HF basis of one isospin: [[h - lam, D], [D^H, -(h - lam)]] with lam by bisection in
    lam_prev +- window. EFA: the HF states with blocked = 1 (a Kramers doublet) leave the pairing
    problem and hold v2 = 1/2 each, uv = 0. Returns (lam, v2, uv, C, rho, kappa): occupations, uv,
    the canonical basis C (columns, in the HF basis) and rho~, kappa~.
    """
    n = h.shape[0]
    free = 1.0 - blocked
    h = h + jnp.diag(1.0e4 * blocked).astype(h.dtype)      # out of the Fermi search
    D = D * free[:, None] * free[None, :]
    n_target = n_target - 0.5 * jnp.sum(blocked)

    def solve(lam):
        a = (h - lam * jnp.eye(n)).astype(D.dtype)
        Q = jnp.linalg.eigh(jnp.block([[a, D], [jnp.conj(D.T), -a]]))[1]
        U, V = Q[:n, n:], Q[n:, n:]                          # positive quasiparticle energies
        return V @ jnp.conj(V.T), U @ jnp.conj(V.T)

    def bisect(_, lo_hi):
        lo, hi = lo_hi
        mid = 0.5 * (lo + hi)
        low = jnp.real(jnp.trace(solve(mid)[0])) < n_target
        return jnp.where(low, mid, lo), jnp.where(low, hi, mid)

    lo, hi = jax.lax.fori_loop(0, n_bisect, bisect, (lam_prev - window, lam_prev + window))
    lam = 0.5 * (lo + hi)
    rho, kap = solve(lam)
    rho = rho + jnp.diag(0.5 * blocked).astype(rho.dtype)
    v2, C = jnp.linalg.eigh(0.5 * (rho + jnp.conj(rho.T)))
    v2 = jnp.clip(v2, 1e-10, 1.0 - 1e-10)
    in_doublet = jnp.sum(blocked[:, None] * jnp.abs(C) ** 2, axis=0) > 0.5
    uv = jnp.where(in_doublet, 0.0, jnp.sqrt(jnp.maximum(v2 * (1.0 - v2), smallp)))
    return lam, v2, uv, C, rho, kap


@partial(jax.jit, static_argnames=('npsi_n',))
def two_basis_hfb(psi, e, v_pair, grid, npsi_n: int, targets, lam_prev, smallp, cutoff, width, blocked=None, hmat=None):
    """
    HFB in the HF basis psi (Ritz values e) with D_kl = f_k f_l <phi_k|v_pair|phi_l> (the window f,
    module docstring). The pair density is built from the windowed canonical orbitals
    sum_k C_ki f_k phi_k, so that E_pair = 1/2 sum uv <Delta> is its variational value.
    blocked: (nstates,) EFA doublet mask or None; hmat: h in the basis psi when it is not diagonal.

    Returns (psi_can, sp_energy, deltaf, wocc, wguv, stats, weight, psi_pair): canonical orbitals,
    <h> and <Delta> in them, v2, uv, pairing statistics (dict of (2,) arrays, as solve_pairing),
    per-HF-orbital weights rho_kk + |kappa_kk| and the windowed canonical orbitals.
    """
    blocked = jnp.zeros(e.shape[0]) if blocked is None else blocked
    out = []
    for q, sl in ((0, slice(0, npsi_n)), (1, slice(npsi_n, None))):
        phi, eq = psi[sl], e[sl]
        f = jnp.where(cutoff[q] > 0, 1.0 / (1.0 + jnp.exp((eq - lam_prev[q] - cutoff[q])
                                                           / jnp.maximum(width * cutoff[q], 1e-6))), 1.0)
        D = grid.wxyz * jnp.einsum('ksxyz,lsxyz->kl', jnp.conj(phi), v_pair[q][None, None] * phi)
        D = f[:, None] * D * f[None, :]
        hq = jnp.diag(eq).astype(D.dtype) if hmat is None else hmat[sl, sl]
        lam, v2, uv, C, rho, kap = _hfb_block(hq, D, targets[q], lam_prev[q], smallp, blocked[sl])
        spe = jnp.real(jnp.einsum('ki,kl,li->i', jnp.conj(C), hq, C))
        dlt = jnp.real(jnp.einsum('ki,kl,li->i', jnp.conj(C), D, C))
        half = 0.5 * uv
        epair, nuv = jnp.sum(half * dlt), jnp.maximum(jnp.sum(half), 1e-20)
        stats = dict(eferm=lam, epair=epair, avdelt=epair / nuv, avg=epair / nuv ** 2,
                     avdeltv2=jnp.sum(dlt * v2) / jnp.maximum(jnp.sum(v2), 1e-20))
        out.append((jnp.einsum('ki,ksxyz->isxyz', C, phi), spe, dlt, v2, uv, stats,
                    jnp.real(jnp.diag(rho)) + jnp.abs(jnp.diag(kap)),
                    jnp.einsum('ki,ksxyz->isxyz', f[:, None] * C, phi)))
    cat = lambda i: jnp.concatenate([out[0][i], out[1][i]])
    stats = {k: jnp.stack([out[0][5][k], out[1][5][k]]) for k in out[0][5]}
    return cat(0), cat(1), cat(2), cat(3), cat(4), stats, cat(6), cat(7)
