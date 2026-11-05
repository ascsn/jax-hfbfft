"""
Collective inertia of a constrained static solution: the ATDHFB perturbative
cranking mass and the GCM metric, as in HFBTHO (hfbtho_collective.f90).

With the energy moments of the constraint operators Q_i,

    M^(k)_ij = sum_{k,l} Re(Q^i_kl Q^j_kl*) (u_k v_l + u_l v_k)^2 / (E_k + E_l)^k,

over pairs of canonical orbitals of the same isospin (u, v from the
occupations, E_k = sqrt((e_k - lambda)^2 + Delta_k^2)), the ATDHFB mass is
M = hbar^2 [M^(1)]^-1 M^(3) [M^(1)]^-1, the GCM metric G = 1/2 [M^(1)]^-1
M^(2) [M^(1)]^-1, and the zero-point energy E0 = 1/2 Tr(G M^-1).  The
canonical basis makes this the BCS form of HFBTHO's quasiparticle formula.

The sum needs the unoccupied states up to the 2 hbar omega region and beyond;
in a box these are many continuum states.  With complement=True the states
outside the orbital basis are summed implicitly (Sternheimer): treating them as
unpaired (v = 0, E_l = e_l - lambda), their contribution for orbital k is
2 v_k^2 <g|[(1-P)(h - lambda + E_k)(1-P)]^-n|g>, g = (1-P) Q phi_k, obtained by
conjugate-gradient solves with h.  The explicit sum then keeps only the orbitals
with v^2 > 1e-4, rotated to diagonalise h within groups of occupations equal to
1e-4 (the canonical basis is undefined there: with weak pairing the solver leaves
the occupied orbitals as mixtures, and the nearly empty ones unconverged); the
rest are summed exactly by the complement as empty states.  This assumes the kept
orbitals cover the pairing window.
"""

from functools import partial
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np


def fermi_levels(state) -> np.ndarray:
    """lambda per isospin: the pairing Fermi energy, or the middle of the gap when unpaired."""
    iso, v2, e = (np.asarray(x) for x in (state.isospin, state.wocc, state.sp_energy))
    lam = np.asarray(state.pairing.eferm, dtype=float).copy()
    for q in (0, 1):
        occ, emp = (iso == q) & (v2 > 0.5), (iso == q) & (v2 <= 0.5)
        sharp = np.all((v2[iso == q] < 1e-6) | (v2[iso == q] > 1 - 1e-6))
        if sharp and occ.any() and emp.any():
            lam[q] = 0.5 * (e[occ].max() + e[emp].min())
    return lam


def _as_function(op):
    """A one-body operator as a function on a spinor (2, nx, ny, nz); arrays multiply."""
    return op if callable(op) else (lambda phi: op * phi)


VMIN = 1e-4      # orbitals below this occupation are left to the complement (as v = 0)


def _kept_basis(state, grid, keep, tol: float = 1e-4):
    """
    (psi, e, delta) of the orbitals `keep`, rotated to diagonalise h within each group
    of occupations equal to within tol (such a rotation changes the density by O(tol));
    Delta_k is recomputed for the rotated orbitals.
    """
    psi, e = np.array(state.psi[keep]), np.array(state.sp_energy[keep], dtype=float)
    mf = getattr(state, 'meanfield', None)
    if mf is None:                            # no mean field given: take the orbitals as they are
        return jnp.asarray(psi), jnp.asarray(e), state.deltaf[keep]
    from jax_hfbfft.physics.meanfield import apply_hamiltonian
    iso, v2 = np.asarray(state.isospin)[keep], np.clip(np.asarray(state.wocc)[keep], 0, 1)
    for q in (0, 1):
        sub = np.nonzero(iso == q)[0]
        sub = sub[np.argsort(v2[sub])]
        for idx in np.split(sub, np.nonzero(np.diff(v2[sub]) > tol)[0] + 1):
            if len(idx) < 2:
                continue
            hpsi = jax.vmap(lambda p: apply_hamiltonian(p, mf, q, grid))(jnp.asarray(psi[idx]))
            H = np.asarray(grid.wxyz * jnp.einsum('ksxyz,lsxyz->kl', jnp.conj(psi[idx]), hpsi))
            e[idx], U = np.linalg.eigh(0.5 * (H + H.conj().T))
            psi[idx] = np.einsum('kl,ksxyz->lsxyz', U, psi[idx])
    psi = jnp.asarray(psi)
    dens = jnp.sum(jnp.abs(psi) ** 2, axis=1)
    delta = state.pairwg[keep] * grid.wxyz * jnp.sum(mf.v_pair[jnp.asarray(iso)] * dens, axis=(1, 2, 3))
    return psi, jnp.asarray(e), delta


def energy_moments(state, grid, operators, time_odd: bool = False, complement: bool = True,
                   cg_iters: int = 300):
    """
    M^(1), M^(2), M^(3) (each (m, m)) for operators given on the grid or as functions on a
    spinor. Time-odd operators (e.g. angular momentum) take (u_k v_l - u_l v_k)^2.
    """
    ops = [_as_function(op) for op in operators]
    keep = np.nonzero(np.asarray(state.wocc) > VMIN)[0] if complement else np.arange(len(state.wocc))
    iso = jnp.asarray(np.asarray(state.isospin)[keep])
    v2 = jnp.clip(state.wocc[keep], 0.0, 1.0)
    u, v = jnp.sqrt(1.0 - v2), jnp.sqrt(v2)
    lam = jnp.asarray(fermi_levels(state))[iso]
    psi, e, delta = _kept_basis(state, grid, keep)
    E = jnp.sqrt((e - lam) ** 2 + delta ** 2)
    Q = jnp.stack([grid.wxyz * jnp.einsum('ksxyz,lsxyz->kl', jnp.conj(psi), jax.vmap(op)(psi))
                   for op in ops])                                            # (m, n, n)
    same = iso[:, None] == iso[None, :]
    sign = -1.0 if time_odd else 1.0
    w = jnp.where(same, (u[:, None] * v[None, :] + sign * v[:, None] * u[None, :]) ** 2, 0.0)
    S = jnp.where(w > 0, E[:, None] + E[None, :], 1.0)
    QQ = jnp.real(jnp.einsum('ikl,jkl->ijkl', Q, jnp.conj(Q)))
    M = [np.asarray(jnp.sum(QQ * (w / S**k), axis=(2, 3))) for k in (1, 2, 3)]
    if complement:   # out-of-basis states are empty: both pair factors reduce to v_k^2
        M = [a + b for a, b in zip(M, _complement_moments(psi, iso, state.meanfield, grid, ops, lam, E, v2, cg_iters))]
    return M


def cranking_inertia(state, grid, operators: Sequence[jnp.ndarray], complement: bool = True,
                     cg_iters: int = 300) -> dict:
    """
    ATDHFB cranking mass tensor (hbar^2 MeV^-1 per unit of Q^2), GCM metric and
    zero-point energy (MeV) for the operators Q_i(r) given on the grid
    (e.g. the constraint operators of the collective path).

    complement: add the states outside the orbital basis implicitly (see the
        module docstring); cg_iters: conjugate-gradient iterations per solve.

    Returns dict(M1, M2, M3, mass, metric, zpe); M1-M3 and mass are (m, m).
    """
    M1, M2, M3 = energy_moments(state, grid, operators, complement=complement, cg_iters=cg_iters)
    M1inv = np.linalg.inv(M1)
    mass = M1inv @ M3 @ M1inv
    metric = 0.5 * M1inv @ M2 @ M1inv
    return dict(M1=M1, M2=M2, M3=M3, mass=mass, metric=metric,
                zpe=0.5 * float(np.trace(metric @ np.linalg.inv(mass))))


def angular_momentum_operator(grid, axis: str = 'x'):
    """J_axis = L + S as a function on a spinor (hbar = 1): -i (r x grad)_axis + sigma_axis / 2."""
    X, Y, Z = grid.get_meshgrid()
    d = {'x': (grid.der1x, 'ij,sjkl->sikl'), 'y': (grid.der1y, 'ij,skjl->skil'),
         'z': (grid.der1z, 'ij,sklj->skli')}
    grad = lambda c, phi: jnp.einsum(d[c][1], d[c][0], phi)
    sig = {'x': lambda p: jnp.stack([p[1], p[0]]), 'y': lambda p: jnp.stack([-1j * p[1], 1j * p[0]]),
           'z': lambda p: jnp.stack([p[0], -p[1]])}
    (a, A), (b, B) = {'x': (('y', Y), ('z', Z)), 'y': (('z', Z), ('x', X)), 'z': (('x', X), ('y', Y))}[axis]
    return lambda phi: -1j * (A * grad(b, phi) - B * grad(a, phi)) + 0.5 * sig[axis](phi)


def moment_of_inertia(state, grid, axis: str = 'x', complement: bool = True, cg_iters: int = 300) -> dict:
    """
    Inglis-Belyaev (cranking) moment of inertia about `axis`, in hbar^2 MeV^-1:
    J = sum_kl |<k|J|l>|^2 (u_k v_l - u_l v_k)^2 / (E_k + E_l), with the rigid-body value
    J_rigid = Int rho (r^2 - r_axis^2) / (2 h2m) for comparison.
    """
    M1, _, _ = energy_moments(state, grid, [angular_momentum_operator(grid, axis)], time_odd=True,
                              complement=complement, cg_iters=cg_iters)
    R = dict(zip('xyz', grid.get_meshgrid()))
    perp = sum(R[c] ** 2 for c in 'xyz' if c != axis)
    h2m = np.asarray(state.force.h2m if getattr(state, 'force', None) is not None else [20.73553] * 2)
    rigid = sum(float(jnp.sum(state.densities.rho[q] * perp) * grid.wxyz) / (2 * float(h2m[q])) for q in (0, 1))
    return dict(J=float(M1[0, 0]), J_rigid=rigid)


def _complement_moments(psi, iso, mf, grid, operators, lam, E, v2, cg_iters, tol=1e-10):
    """2 sum_k v_k^2 (<g_i|y_j>, <y_i|y_j>, <y_i|z_j>) with A y = g, A z = y, A = (1-P)(h - lam_k + E_k)(1-P),
    P the projector on the orbitals psi."""
    from jax_hfbfft.physics.meanfield import apply_hamiltonian
    m = len(operators)
    h2m = float(np.max(np.asarray(getattr(mf, 'bmass', 20.7))))

    def project(x, q):                     # (1 - P) x within isospin q
        basis = jnp.where((iso == q)[:, None, None, None, None], psi, 0.0)
        c = grid.wxyz * jnp.einsum('lsxyz,sxyz->l', jnp.conj(basis), x)
        return x - jnp.einsum('l,lsxyz->sxyz', c, basis)

    def solve(b, q, shift):                # preconditioned CG for A y = b
        A = lambda x: project(apply_hamiltonian(project(x, q), mf, q, grid) + shift * x, q)
        Minv = lambda r: jnp.fft.ifftn(jnp.fft.fftn(r, axes=(-3, -2, -1)) / (h2m * grid.k2 + shift + 20.0),
                                       axes=(-3, -2, -1))
        dot = lambda a, c: jnp.real(jnp.sum(jnp.conj(a) * c))

        def body(st):
            it, y, r, z, p, rz = st
            Ap = A(p)
            alpha = rz / dot(p, Ap)
            y, r = y + alpha * p, r - alpha * Ap
            z = project(Minv(r), q)
            rz_new = dot(r, z)
            return it + 1, y, r, z, z + (rz_new / rz) * p, rz_new

        z0 = project(Minv(b), q)
        rz0 = dot(b, z0)
        go = lambda st: (st[0] < cg_iters) & (st[5] > (tol**2) * rz0)
        _, y, *_ = jax.lax.while_loop(go, body, (0, jnp.zeros_like(b), b, z0, z0, rz0))
        return y

    @partial(jax.jit, static_argnums=1)
    def per_state(phi, q, shift):
        g = jnp.stack([project(op(phi), q) for op in operators])
        y = jnp.stack([solve(gi, q, shift) for gi in g])
        z = jnp.stack([solve(yi, q, shift) for yi in y])
        ip = lambda a, b: grid.wxyz * jnp.real(jnp.einsum('isxyz,jsxyz->ij', jnp.conj(a), b))
        return ip(g, y), ip(y, y), ip(y, z)

    C = np.zeros((3, m, m))
    for k in range(len(v2)):
        q = int(iso[k])
        shift = float(E[k]) - float(lam[k])            # h - lambda + E_k = h + shift
        C += 2.0 * float(v2[k]) * np.asarray(jnp.stack(per_state(psi[k], q, shift)))
    return C[0], C[1], C[2]
