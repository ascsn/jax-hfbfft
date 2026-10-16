"""
Time-dependent Hartree-Fock-Bogoliubov: the global in-span matrix scheme.

The state is a set of orbitals phi_m and, per isospin, the full generalized
density in their span,

    R = [[rho~, kap~], [kap~^+, 1 - rho~^T]],
    rho(r s, r' s')   = sum_mn rho~_mn phi_m(r s) phi_n*(r' s'),
    kappa(r s, r' s') = sum_mn kap~_mn phi_m(r s) phi_n(r' s'),

with rho~ Hermitian and kap~ antisymmetric (n x n, n = number of orbitals).
The pairing density is xi_q(r) = 1/2 sum_mn kap~_mn [phi_m phi_n]_singlet and
Delta_q(r) = v0_q (1 - rho/rho0pr) xi_q(r) for DDDI pairing.

One step (`make_step`) is a predictor-corrector around midpoint fields. Each
half does:

  1. out-of-span pair force: the orbitals turn into the complement of their
     span under B = -(i dt/hbar) Q F rho~^-1, F_m = dE_pair/d<phi_m|, applied
     as the exact unitary exp(B Phi^+ - Phi B^+) in n_comp exponential-midpoint
     sub-rotations. rho~^-1 is regularised by eps; half of the force is
     applied before and half after steps 2-3;
  2. in-span pairing: R -> U R U^+ with U = exp(-i dt H~/hbar),
     H~ = [[0, Dt], [Dt^+, 0]], Dt_mn = -<phi_m|Delta|eps phi_n*>, Delta taken
     self-consistently at the half step; split dt/2 before and after step 3
     (Strang);
  3. mean field: phi -> exp(-i h dt/hbar) phi (Taylor series).

The step is second order: the energy error falls 4.0x per halving of dt
(U238 fission state, 2 fm/c). The earlier first-order variant (Euler
sub-rotations, pair dt then h dt) is strang=False, midpoint=False.

The step ends with a Loewdin re-orthonormalisation and a re-diagonalisation of
rho~ inside each isospin, which changes the description, not the state. With a
complete orbital basis this is exact TDHFB; with a truncated one it converges
to it as orbitals are added (20O against exact TDHFB and against LISE TD-SLDA).

Recipe used for every validated result:
  * odd grid sizes (the kinetic operator is then exactly variational, see
    meanfield._kinetic_term);
  * a static solved with state_cutoff = 0, so every state weight is 1 (a pair
    with soft-cutoff weight 0 < w < 1 is never stationary: its residual is
    -Delta (1 - w)); `from_static` refuses anything else;
  * pair amplitudes rebuilt from the occupations, kappa = sqrt(v2 (1 - v2))
    (`uv_floor=0`), not the static solver's floored sqrt(max(v2(1-v2), 1e-6));
  * the static relaxed with `relax` (see below), then `purify` after any kick
    or collision setup;
  * eps = 1e-7, pred_n_comp = 1, and the second-order step (strang, midpoint,
    n_comp = 1): dt = 0.1 fm/c, or 0.2 on 1.0 fm grids (U238 fission, 10 fm/c
    against dt 0.025: density 6e-7, E_pair 2 keV, energy 9e-7 MeV, at 11.6
    s/fm/c on an H200; the first-order step at dt 0.1 gave 1e-5, 35 keV,
    1.8e-3 MeV at 16 s/fm/c). The 20O, Sn120 and LISE validations predate it
    and used the first-order step at dt 0.05, n_comp 4.

Statics. The canonical-basis static solver keeps only the diagonal gaps and
floors the preconditioner weights, so its nearly empty orbitals never converge:
its result is not the minimum of this functional in its own orbital space, and
TDHFB at rest relaxes it (a slow growth of the pairing energy). `relax` solves
HFB in the representation of the dynamics; from its result 20O at rest keeps
E_pair to 0.02% over 60 fm/c (5.9% unrelaxed). `stationarity` measures the
per-orbital out-of-span velocity that this removes.

`ebata_state` / `make_ebata_step` implement the canonical-basis Cb-TDHFB of
Ebata et al., PRC 82, 034306 (2010) with contact pairing, the comparison arm:
diagonal gap only, orbitals under h - eta_k, amplitudes by his Eqs. (23)/(27).
"""

import dataclasses
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from jax_hfbfft.core.force import Force
from jax_hfbfft.core.grid import Grid
from jax_hfbfft.physics.coulomb import CoulombSolver, solve_poisson
from jax_hfbfft.physics.densities import Densities, compute_densities
from jax_hfbfft.physics.dynamics import HBARC, propagate_taylor
from jax_hfbfft.physics.energies import compute_integrated_energy
from jax_hfbfft.physics.meanfield import apply_hamiltonian, compute_skyrme_meanfield


@dataclass
class GState:
    """Orbitals and the in-span matrices of the global scheme."""
    psi: jax.Array      # (n, 2, nx, ny, nz)
    rho: jax.Array      # (n, n) Hermitian
    kap: jax.Array      # (n, n) antisymmetric
    isospin: jax.Array  # (n,)
    time: float = 0.0

    @property
    def nstates(self) -> int:
        return self.psi.shape[0]


@dataclass
class EbataState:
    """Canonical (diagonal) state of Ebata's Cb-TDHFB, with fixed pair partners."""
    psi: jax.Array       # (n, 2, nx, ny, nz)
    occ: jax.Array       # (n,) rho_k
    kappa: jax.Array     # (n,) complex kappa_k
    partners: jax.Array  # (n,) Kramers partner of each orbital, fixed at t = 0
    isospin: jax.Array
    time: float = 0.0

    @property
    def nstates(self) -> int:
        return self.psi.shape[0]


# ── Pairing primitives ───────────────────────────────────────────────────────

def _same_iso(isospin):
    return isospin[:, None] == isospin[None, :]


def _eps_spin(p):
    """eps phi: (up, dn) -> (dn, -up). The time reverse of phi is -eps conj(phi)."""
    return jnp.stack([p[:, 1], -p[:, 0]], axis=1)


def _singlet(p, q):
    """[p q]_singlet(r) = p_up q_dn - p_dn q_up; equals |p|^2 for q = T p."""
    return p[:, 0] * q[:, 1] - p[:, 1] * q[:, 0]


def _by_isospin(contrib, isospin):
    """Sum per-orbital fields (n, nx, ny, nz) into (2, nx, ny, nz)."""
    is_n = (isospin == 0)[:, None, None, None]
    return jnp.stack([jnp.sum(jnp.where(is_n, contrib, 0.0), 0),
                      jnp.sum(jnp.where(is_n, 0.0, contrib), 0)])


def pair_density(psi, kap, isospin):
    """xi_q(r) = 1/2 sum_mn kap~_mn [phi_m phi_n]_singlet, same isospin."""
    chi = jnp.tensordot(jnp.where(_same_iso(isospin), kap, 0.0), psi, axes=([0], [0]))
    return _by_isospin(0.5 * _singlet(chi, psi), isospin)


def pair_density_canonical(psi, kappa, isospin):
    """xi_q(r) = 1/2 sum_k kappa_k |phi_k|^2: each orbital paired with its own time reverse."""
    dens = jnp.sum(jnp.real(psi * jnp.conjugate(psi)), axis=1)
    return _by_isospin(0.5 * kappa[:, None, None, None] * dens, isospin)


def pair_field(xi, rho_tot, force: Force):
    """Complex pair field Delta_q(r): v0_q xi_q, times 1 - rho/rho0pr for DDDI (ipair 6)."""
    density_factor = jnp.where(force.ipair == 6, 1.0 - rho_tot / force.rho0pr, 1.0)
    v0 = jnp.stack([jnp.full_like(rho_tot, force.v0neut), jnp.full_like(rho_tot, force.v0prot)])
    active = jnp.where((force.ipair == 5) | (force.ipair == 6), 1.0, 0.0)
    return active * v0 * xi * density_factor[None, ...]


def pair_view(psi, grid: Grid, force: Force):
    """
    The orbitals as the pair channel sees them: low-pass filtered with
    f(k) = 1 / (1 + exp((|k| - pair_kcut) / pair_kwidth)) (identity when
    force.pair_kcut = 0). f is real and even, so the filter commutes with time
    reversal; pairing built from the filtered orbitals is a momentum-cutoff
    regularisation of the zero-range interaction.
    """
    if not force.pair_kcut:
        return psi
    k = jnp.sqrt(grid.kx[:, None, None] ** 2 + grid.ky[None, :, None] ** 2 + grid.kz[None, None, :] ** 2)
    f = 1.0 / (1.0 + jnp.exp((k - force.pair_kcut) / force.pair_kwidth))
    return jnp.fft.ifftn(jnp.fft.fftn(psi, axes=(-3, -2, -1)) * f, axes=(-3, -2, -1))


def anomalous_matrix(psi, delta, isospin, grid: Grid):
    """A_mn = <phi_m| Delta |eps conj(phi_n)>, same isospin (antisymmetric)."""
    n = psi.shape[0]
    dq = jnp.where((isospin == 0)[:, None, None, None], delta[0][None], delta[1][None])
    w = (jnp.conjugate(psi) * dq[:, None]).reshape(n, -1)
    t = _eps_spin(jnp.conjugate(psi)).reshape(n, -1)
    return jnp.where(_same_iso(isospin), grid.wxyz * (w @ t.T), 0.0)


def _eigh(H, site: str):
    """
    jnp.linalg.eigh with a host-LAPACK fallback. On the GPU the jitted eigh of
    the tied matrices of a collapsed-pairing state (rho~ with occupations 0/1,
    the identity-like overlap, the ~1e-18 complement Gram matrix) intermittently
    returns NaN (Ca48, 2026-10-12); a result that is not finite, orthonormal and
    reconstructing H to 1e-8 is recomputed by numpy (and reported).
    """
    lam, U = jnp.linalg.eigh(H)
    eye = jnp.eye(H.shape[0], dtype=U.dtype)
    orth = jnp.abs(jnp.conjugate(U.T) @ U - eye).max()
    rec = jnp.abs((U * lam[None, :]) @ jnp.conjugate(U.T) - H).max() / jnp.maximum(jnp.abs(H).max(), 1e-300)
    ok = jnp.isfinite(lam).all() & jnp.isfinite(U).all() & (orth < 1e-8) & (rec < 1e-8)

    def host(H):
        jax.debug.print("tdhfb: GPU eigh failed in " + site + " (orth {o:.1e}, rec {r:.1e}); host LAPACK", o=orth, r=rec)
        shapes = (jax.ShapeDtypeStruct(lam.shape, lam.dtype), jax.ShapeDtypeStruct(U.shape, U.dtype))
        return jax.pure_callback(lambda h: tuple(a.astype(t.dtype) for a, t in zip(np.linalg.eigh(np.asarray(h)), shapes)),
                                 shapes, H)

    return jax.lax.cond(ok, lambda H: (lam, U), host, H)


# ── Fields and energy ────────────────────────────────────────────────────────

def _densities_diag(psi, occ, isospin, grid: Grid) -> Densities:
    zeros, ones = jnp.zeros_like(occ), jnp.ones_like(occ)
    return compute_densities(psi, occ, zeros, ones, ones, isospin, grid)


def _densities_matrix(psi, rho, isospin, grid: Grid) -> Densities:
    """All Skyrme densities of rho~, from its eigen-orbitals with signed weights."""
    shift = 10.0                      # separates the isospins inside one eigh
    lam, V = _eigh(rho + jnp.diag(jnp.where(isospin == 1, shift, 0.0)).astype(rho.dtype), 'densities')
    iso_k = jnp.where(lam > 0.5 * shift, 1, 0).astype(isospin.dtype)
    lam = lam - jnp.where(iso_k == 1, shift, 0.0)
    chi = jnp.tensordot(V, psi, axes=([0], [0]))
    return _densities_diag(chi, lam, iso_k, grid)


def _complete_fields(dens, xi, grid, force, cs, use_coulomb, external=None):
    """Coulomb, mean field and pair field. |xi| enters the DDDI rearrangement term.
    external(dens) -> (2, nx, ny, nz) adds a potential (a constraint) to h."""
    dens = dataclasses.replace(dens, chi=jnp.abs(xi))
    wcoul = jnp.zeros(grid.shape, dtype=dens.rho.dtype)
    if use_coulomb:
        wcoul = solve_poisson(dens.rho[1], cs, grid)
    mf = compute_skyrme_meanfield(dens, force, grid,
                                  coulomb_potential=wcoul if use_coulomb else None,
                                  constraint_potential=None if external is None else external(dens),
                                  use_coulomb=use_coulomb)
    delta = pair_field(xi, dens.rho[0] + dens.rho[1], force)
    return dens, mf, wcoul, delta


def fields(S, grid: Grid, force: Force, cs: CoulombSolver, use_coulomb: bool = True, external=None):
    """(densities, meanfield, wcoul, delta, xi) of a GState or EbataState (external: see _complete_fields)."""
    if isinstance(S, EbataState):
        dens = _densities_diag(S.psi, S.occ, S.isospin, grid)
        xi = pair_density_canonical(pair_view(S.psi, grid, force), S.kappa, S.isospin)
    else:
        dens = _densities_matrix(S.psi, S.rho, S.isospin, grid)
        xi = pair_density(pair_view(S.psi, grid, force), S.kap, S.isospin)
    return _complete_fields(dens, xi, grid, force, cs, use_coulomb, external) + (xi,)


def energy(S, grid: Grid, force: Force, cs: CoulombSolver, use_coulomb: bool = True):
    """Total energy and the pairing energy per isospin, E_pair,q = Int Delta_q xi_q*."""
    return energy_of_fields(fields(S, grid, force, cs, use_coulomb), grid, force, use_coulomb)


def energy_of_fields(flds, grid: Grid, force: Force, use_coulomb: bool = True):
    """energy() from the output of fields()."""
    dens, mf, wcoul, delta, xi = flds
    ep = grid.wxyz * jnp.real(jnp.sum(delta * jnp.conjugate(xi), axis=(-3, -2, -1)))
    e = compute_integrated_energy(
        dens, force, grid, coulomb_potential=wcoul if use_coulomb else None,
        pairing_energy=ep, mass_number=int(round(float(grid.wxyz * jnp.sum(dens.rho)))),
        use_coulomb=use_coulomb, ecorrp=mf.ecorrp)
    return float(e.ehfint), np.asarray(ep)


# ── Initial states ───────────────────────────────────────────────────────────

def kramers_basis(psi, occ, kappa, isospin, wxyz: float, tol: float = 1e-7, max_defect: float = 1e-6):
    """
    Rotate each block of equal (occ, kappa) orbitals into an exact Kramers basis,
    phi_abar = T phi_a. Densities are unchanged (a unitary inside a block of
    equal occupations). Must be built from time-even (unboosted) orbitals: time
    reversal flips momentum, so a moving nucleus has no partners to find.
    Raises if a block is not closed under T to within max_defect.

    Returns (psi, partners).
    """
    P = np.asarray(psi)
    shape = P.shape[1:]
    out = P.copy()
    partners = -np.ones(P.shape[0], dtype=np.int32)
    occ_n, kap_n, iso = np.asarray(occ), np.asarray(kappa), np.asarray(isospin)
    worst = 0.0

    def T(v):
        v = v.reshape(shape)
        return np.stack([-np.conj(v[1]), np.conj(v[0])]).reshape(-1)

    for q in (0, 1):
        idx = np.where(iso == q)[0]
        if not len(idx):
            continue
        idx = idx[np.argsort(occ_n[idx], kind='stable')]
        groups, cur = [], [idx[0]]
        for a, b in zip(idx[:-1], idx[1:]):
            if (abs(occ_n[b] - occ_n[a]) <= tol * max(1.0, abs(occ_n[a]))
                    and abs(kap_n[b] - kap_n[a]) <= tol):
                cur.append(b)
            else:
                groups.append(cur)
                cur = [b]
        groups.append(cur)
        for G in groups:
            if len(G) % 2:
                raise ValueError(f"kramers_basis: odd block of {len(G)} equal-occupation orbitals "
                                 f"(isospin {q}, occ {occ_n[G[0]]:.6e})")
            B = P[G].reshape(len(G), -1) * np.sqrt(wxyz)          # orthonormal rows
            k = 0
            while B.shape[0]:
                u = B[0]
                vin = (np.conj(B) @ T(u)) @ B                     # T u projected on the block
                worst = max(worst, 1.0 - float(np.real(np.vdot(vin, vin))))
                v = vin - np.vdot(u, vin) * u
                v = v / np.linalg.norm(v)
                out[G[k]] = (u / np.sqrt(wxyz)).reshape(shape)
                out[G[k + 1]] = (v / np.sqrt(wxyz)).reshape(shape)
                partners[G[k]], partners[G[k + 1]] = G[k + 1], G[k]
                k += 2
                # deflate by <u|b> u = B @ conj(u), and the same for v
                M = B - np.outer(B @ np.conj(u), u) - np.outer(B @ np.conj(v), v)
                if M.shape[0] <= 2:
                    break
                B = np.linalg.svd(M, full_matrices=False)[2][:M.shape[0] - 2]
    if worst > max_defect:
        raise ValueError(f"kramers_basis: a block is not closed under time reversal "
                         f"(defect {worst:.3e})")
    return jnp.asarray(out), jnp.asarray(partners)


def _canonical(psi, wocc, wguv, isospin, grid, wstates=None, pairwg=None, uv_floor=0.0, max_defect=1e-6,
               kramers_tol=1e-7):
    """Static -> (Kramers-basis psi, occ, kappa, partners)."""
    occ = jnp.asarray(wocc)
    if wstates is not None and np.max(np.abs(np.asarray(wstates) - 1.0)) > 1e-8:
        raise ValueError("the static has state weights != 1 (a soft state cutoff); TDHFB needs "
                         "a static solved with state_cutoff = 0")
    wguv = jnp.asarray(wguv)
    if uv_floor is None:
        kappa = wguv
    else:
        kappa = jnp.sqrt(jnp.maximum(occ * (1.0 - occ), uv_floor)) * jnp.sign(wguv)
    if pairwg is not None:
        kappa = kappa * jnp.asarray(pairwg)
    kappa = kappa.astype(jnp.complex128)
    psi, partners = kramers_basis(psi, occ, kappa, isospin, float(grid.wxyz), tol=kramers_tol,
                                   max_defect=max_defect)
    return psi, occ, kappa, partners


def from_static(psi, wocc, wguv, isospin, grid: Grid, wstates=None, pairwg=None,
                uv_floor: Optional[float] = 0.0, max_defect: float = 1e-6, kramers_tol: float = 1e-7) -> GState:
    """
    GState of a converged static (canonical) HFB solution.

    The orbitals are rotated into a Kramers basis and each pair (a, abar) gets
    kap~[a, abar] = kappa_a g_a, with the unit phase g_a fixing [phi_a phi_abar]
    to |phi_a|^2. uv_floor=0 rebuilds kappa = sqrt(v2 (1 - v2)) from the
    occupations; None keeps the static's wguv. max_defect: tolerance on the
    time-reversal symmetry of the static (raise it for a state relax will
    converge anyway, e.g. one resampled onto a new grid). kramers_tol: how close
    occupations must be to share a Kramers block (raise it with max_defect for a
    loosely converged static, whose partners split by ~1e-5).
    """
    psi, occ, kappa, partners = _canonical(psi, wocc, wguv, isospin, grid, wstates, pairwg, uv_floor,
                                           max_defect, kramers_tol)
    craw = grid.wxyz * jnp.sum(_singlet(psi, psi[partners]), axis=(1, 2, 3))
    g = jnp.conjugate(craw) / jnp.abs(craw)
    n = occ.shape[0]
    idx = jnp.arange(n)
    val = 0.5 * (kappa * g - kappa[partners] * g[partners])
    kap = jnp.zeros((n, n), dtype=kappa.dtype).at[idx, partners].set(val)
    return GState(psi=psi, rho=jnp.diag(occ).astype(kappa.dtype), kap=kap,
                  isospin=jnp.asarray(isospin))


def ebata_state(psi, wocc, wguv, isospin, grid: Grid, wstates=None, pairwg=None,
                uv_floor: Optional[float] = 0.0) -> EbataState:
    """EbataState of a converged static, with rest-frame Kramers partners (see from_static)."""
    psi, occ, kappa, partners = _canonical(psi, wocc, wguv, isospin, grid, wstates, pairwg, uv_floor)
    return EbataState(psi=psi, occ=occ, kappa=kappa, partners=partners, isospin=jnp.asarray(isospin))


def kick(S, operator, eta: float):
    """Apply exp(i eta Q(r)) to every orbital (Q given on the grid)."""
    return dataclasses.replace(S, psi=S.psi * jnp.exp(1j * eta * operator)[None, None])


# ── Global scheme: one step ──────────────────────────────────────────────────

def _reg_inv(M, eps):
    """Regularised inverse of a Hermitian matrix, U diag(l / (l^2 + eps^2)) U^+."""
    lam, U = _eigh(0.5 * (M + jnp.conjugate(M.T)), 'reg_inv')
    return (U * (lam / (lam ** 2 + eps ** 2))[None, :]) @ jnp.conjugate(U.T)


def _complement_increment(psi, kap, isospin, same, Rinv, delta, grid, dt, view=lambda p: p):
    """B = -(i dt/hbar) Q F rho~^-1 with F_m = -P Delta sum_n kap~*_mn eps (P phi_n)* (P = view, the pair filter)."""
    M = Rinv.T @ jnp.conjugate(jnp.where(same, kap, 0.0))
    dq = jnp.where((isospin == 0)[:, None, None, None], delta[0][None], delta[1][None])
    X = view(-dq[:, None] * jnp.tensordot(M, _eps_spin(jnp.conjugate(view(psi))), axes=([1], [0])))
    return (-1j * dt / HBARC) * _project_out(psi, X, same, grid)


def _project_out(psi, X, same, grid):
    """Q X: each column of X with its components along the same-isospin orbitals removed."""
    n = psi.shape[0]
    ov = jnp.where(same, grid.wxyz * (jnp.conjugate(psi).reshape(n, -1) @ X.reshape(n, -1).T), 0.0)
    return X - jnp.tensordot(ov, psi, axes=([0], [0]))


def _complement_rotate(psi, B, same, grid):
    """exp(B Phi^+ - Phi B^+) Phi = Phi cos(sqrt G) + B sinc(sqrt G), G = B^+ B (B in the complement)."""
    n = psi.shape[0]
    b = B.reshape(n, -1)
    G = jnp.where(same, grid.wxyz * (jnp.conjugate(b) @ b.T), 0.0)
    lam, U = _eigh(0.5 * (G + jnp.conjugate(G.T)), 'complement_rotate')
    sq = jnp.sqrt(jnp.maximum(lam, 0.0))
    C = (U * jnp.cos(sq)[None, :]) @ jnp.conjugate(U.T)
    sinc = jnp.where(sq > 1e-12, jnp.sin(sq) / jnp.where(sq > 1e-12, sq, 1.0), 1.0 - lam / 6.0)
    Fs = (U * sinc[None, :]) @ jnp.conjugate(U.T)
    return jnp.tensordot(C.T, psi, axes=([1], [0])) + jnp.tensordot(Fs.T, B, axes=([1], [0]))


def _exp_update(rho, kap, Dt, dt):
    """R -> U R U^+, U = exp(-i dt H~/hbar), H~ = [[0, Dt], [Dt^+, 0]]; purity is exact."""
    n = rho.shape[0]
    H = jnp.block([[jnp.zeros_like(Dt), Dt], [jnp.conjugate(Dt.T), jnp.zeros_like(Dt)]])
    lam, V = _eigh(0.5 * (H + jnp.conjugate(H.T)), 'exp_update')
    U = (V * jnp.exp(-1j * lam * dt / HBARC)[None, :]) @ jnp.conjugate(V.T)
    R = jnp.block([[rho, kap], [jnp.conjugate(kap.T), jnp.eye(n, dtype=rho.dtype) - rho.T]])
    R = U @ R @ jnp.conjugate(U.T)
    return R[:n, :n], R[:n, n:]


def _lowdin(psi, same, grid):
    n = psi.shape[0]
    p = psi.reshape(n, -1)
    Sm = jnp.where(same, grid.wxyz * (jnp.conjugate(p) @ p.T), 0.0)
    lam, U = _eigh(0.5 * (Sm + jnp.conjugate(Sm.T)), 'lowdin')
    return jnp.tensordot(((U * lam ** -0.5) @ jnp.conjugate(U.T)).T, psi, axes=([1], [0]))


def absorbing_potential(grid: Grid, width: float, strength: float):
    """
    W(r) = strength * sum_d s_d^2, s_d the depth into a layer of `width` fm at
    each face of the box (0 inside, 1 at the face): the imaginary part of an
    absorbing boundary, -i W(r) in h.
    """
    W = jnp.zeros((grid.nx, grid.ny, grid.nz))
    for ax, c in enumerate((grid.x, grid.y, grid.z)):
        c = jnp.asarray(c)
        edge = jnp.max(jnp.abs(c))
        s = jnp.clip((jnp.abs(c) - (edge - width)) / width, 0.0, 1.0)
        shape = [1, 1, 1]
        shape[ax] = -1
        W = W + (s ** 2).reshape(shape)
    return strength * W


def _absorb(psi, rho, kap, W, same, grid, dt):
    """
    Absorbing boundary: damp the orbitals by exp(-W dt/hbar), re-orthonormalise
    them, and carry the norm loss into the in-span matrices, rho~ -> S^1/2 rho~
    S^1/2 and kap~ -> S^1/2 kap~ S^1/2^T (S the overlap of the damped orbitals),
    so that the represented rho(r, r') and kappa(r, r') are exactly the damped
    ones. The absorbed particles leave; the state becomes slightly mixed.
    """
    p = psi * jnp.exp(-W * dt / HBARC)[None, None]
    n = p.shape[0]
    q = p.reshape(n, -1)
    Sm = jnp.where(same, grid.wxyz * (jnp.conjugate(q) @ q.T), 0.0)
    lam, U = _eigh(0.5 * (Sm + jnp.conjugate(Sm.T)), 'absorb')
    lam = jnp.maximum(lam, 1e-300)
    s_half = (U * jnp.sqrt(lam)[None, :]) @ jnp.conjugate(U.T)
    s_mhalf = (U * lam ** -0.5) @ jnp.conjugate(U.T)
    psi = jnp.tensordot(s_mhalf.T, p, axes=([1], [0]))
    return psi, s_half @ rho @ s_half, s_half @ kap @ s_half.T


def _canon_perm(isospin):
    """Column order of the shifted eigh in _recanon: each isospin's columns back into its own slots."""
    lab = np.asarray(isospin)
    perm = np.empty(len(lab), dtype=int)
    perm[np.argsort(lab, kind='stable')] = np.arange(len(lab))
    return jnp.asarray(perm)


def _canon_rotation(rho, isospin, perm):
    """V diagonalising rho~ inside each isospin, columns kept in their isospin's slots."""
    same = _same_iso(isospin)
    Hm = jnp.where(same, rho, 0.0)
    Hm = 0.5 * (Hm + jnp.conjugate(Hm.T)) + jnp.diag(10.0 * isospin).astype(Hm.dtype)
    _, V = _eigh(Hm, 'canon_rotation')
    return jnp.where(same, V[:, perm], 0.0)


def _recanon(psi, rho, kap, isospin, perm):
    """Diagonalise rho~ inside each isospin and rotate orbitals and kap~ with it (a gauge change)."""
    V = _canon_rotation(rho, isospin, perm)
    return (jnp.tensordot(V.T, psi, axes=([1], [0])), jnp.conjugate(V.T) @ rho @ V,
            jnp.conjugate(V.T) @ kap @ jnp.conjugate(V))


def make_step(S: GState, grid: Grid, force: Force, cs: CoulombSolver, eps: float = 1e-7,
              n_comp: int = 1, pred_n_comp: Optional[int] = 1, taylor_order: int = 6,
              use_coulomb: bool = True, strang: bool = True, midpoint: bool = True, absorb=None):
    """
    Jitted step(S, dt) -> S of the global scheme for the orbital set of S.

    n_comp sub-rotations apply the complement force; the predictor, whose only
    job is the midpoint fields, uses pred_n_comp (None = n_comp, the exact
    two-pass scheme; 1 changes rho by ~6e-10 over 10 fm/c and is 1.5x faster).
    Rebuild the step if the number or isospin of the orbitals changes.
    strang: split the in-span pairing update symmetrically around the
    mean-field propagation (pair dt/2, h dt, pair dt/2) instead of pair dt, h dt.
    midpoint: each complement sub-rotation uses the generator at its midpoint,
    projected back onto the complement of the starting orbitals (exponential
    midpoint, local error O(h^3)), instead of the generator at its start
    (forward Euler, an energy error O(dt / n_comp)). Both together make the
    step second order, and then n_comp = 1 is enough (4 changes the error by
    15%); either alone leaves it first order.
    absorb: W(r) in MeV (absorbing_potential) for absorbing boundaries, applied
    after each step (see _absorb); None for a reflecting box.
    """
    iso = S.isospin
    same = _same_iso(iso)
    perm = _canon_perm(iso)
    pnc = n_comp if pred_n_comp is None else pred_n_comp
    pv = lambda p: pair_view(p, grid, force)

    def flds(psi, rho, kap):
        dens = _densities_matrix(psi, rho, iso, grid)
        dens, mf, _, delta = _complete_fields(dens, pair_density(pv(psi), kap, iso), grid, force, cs, use_coulomb)
        return dens, mf, delta

    def Dt_of(psi, kap, rho_tot):
        p = pv(psi)
        delta = pair_field(pair_density(p, kap, iso), rho_tot, force)
        return -anomalous_matrix(p, delta, iso, grid)

    def complement(psi, rho, kap, delta, dt, nc):
        Rinv = _reg_inv(jnp.where(same, rho, 0.0), eps)     # rho~ is fixed during the rotations
        B = lambda p: _complement_increment(p, kap, iso, same, Rinv, delta, grid, 0.5 * dt / nc, pv)
        for _ in range(nc):
            if midpoint:
                # Phi + h f(Phi_h) + O(h^3): the in-span part that Q removes cancels the rotation's cos term
                Bh = B(_complement_rotate(psi, 0.5 * B(psi), same, grid))
                psi = _complement_rotate(psi, _project_out(psi, Bh, same, grid), same, grid)
            else:
                psi = _complement_rotate(psi, B(psi), same, grid)
        return psi

    def pair_update(psi, rho, kap, rho_tot, h):
        # in-span pairing: exact unitary with a self-consistent midpoint pair field
        _, kh = _exp_update(rho, kap, Dt_of(psi, kap, rho_tot), 0.5 * h)
        return _exp_update(rho, kap, Dt_of(psi, kh, rho_tot), h)

    def substep(psi, rho, kap, mf, delta, rho_tot, dt, nc):
        psi = complement(psi, rho, kap, delta, dt, nc)
        if strang:
            rho, kap = pair_update(psi, rho, kap, rho_tot, 0.5 * dt)
            psi = propagate_taylor(psi, mf, iso, grid, dt, taylor_order)
            rho, kap = pair_update(psi, rho, kap, rho_tot, 0.5 * dt)
        else:
            rho, kap = pair_update(psi, rho, kap, rho_tot, dt)
            psi = propagate_taylor(psi, mf, iso, grid, dt, taylor_order)
        psi = complement(psi, rho, kap, delta, dt, nc)
        return psi, rho, kap

    @jax.jit
    def core(psi, rho, kap, dt):
        dn0, mf0, d0 = flds(psi, rho, kap)
        rt0 = dn0.rho[0] + dn0.rho[1]
        pp, rp, kp = substep(psi, rho, kap, mf0, d0, rt0, dt, pnc)
        dn1, mf1, d1 = flds(pp, rp, kp)
        mfm = jax.tree_util.tree_map(lambda a, b: 0.5 * (a + b), mf0, mf1)
        psi, rho, kap = substep(psi, rho, kap, mfm, 0.5 * (d0 + d1),
                                0.5 * (rt0 + dn1.rho[0] + dn1.rho[1]), dt, n_comp)
        return _recanon(_lowdin(psi, same, grid), rho, kap, iso, perm)

    if absorb is not None:
        Wabs = jnp.asarray(absorb)
        absorb_step = jax.jit(lambda psi, rho, kap, dt: _absorb(psi, rho, kap, Wabs, same, grid, dt))

    def step(Sx: GState, dt: float) -> GState:
        psi, rho, kap = core(Sx.psi, Sx.rho, Sx.kap, dt)
        if absorb is not None:
            psi, rho, kap = absorb_step(psi, rho, kap, dt)
        return dataclasses.replace(Sx, psi=psi, rho=rho, kap=kap, time=Sx.time + dt)
    return step


# ── Global scheme: state maintenance ─────────────────────────────────────────

def purity(S: GState) -> float:
    """|rho~ - rho~^2 - kap~ kap~^+|: zero for a quasiparticle vacuum."""
    return float(jnp.linalg.norm(S.rho - S.rho @ S.rho - S.kap @ jnp.conjugate(S.kap.T)))


def purify(S: GState, tol: float = 1e-14, max_iter: int = 60) -> GState:
    """
    Make R an exact projector (McWeeny, R <- 3R^2 - 2R^3, per isospin), then
    re-canonicalise.

    Statics carry an absolute impurity ~1e-6 that the scheme preserves exactly;
    in orbitals with occupation ~1e-5 that is an O(1) relative error in
    kappa/rho, which the complement force amplifies.
    """
    rho, kap, iso = np.array(S.rho), np.array(S.kap), np.asarray(S.isospin)
    for q in (0, 1):
        m = np.where(iso == q)[0]
        if not len(m):
            continue
        n = len(m)
        r, k = rho[np.ix_(m, m)], kap[np.ix_(m, m)]
        R = np.block([[r, k], [k.conj().T, np.eye(n) - r.T]])
        R = 0.5 * (R + R.conj().T)
        for _ in range(max_iter):
            R2 = R @ R
            if np.linalg.norm(R2 - R) < tol:
                break
            R = 3.0 * R2 - 2.0 * R2 @ R
            R = 0.5 * (R + R.conj().T)
        rho[np.ix_(m, m)] = R[:n, :n]
        kap[np.ix_(m, m)] = 0.5 * (R[:n, n:] - R[:n, n:].T)
    psi, rho, kap = _recanon(S.psi, jnp.asarray(rho), jnp.asarray(kap), S.isospin, _canon_perm(S.isospin))
    return dataclasses.replace(S, psi=psi, rho=rho, kap=kap)


def drop_empty(S: GState, below: float):
    """
    Remove canonical orbitals with occupation below `below`, then purify.
    Returns (state, number removed, particles removed). Rebuild the step after.
    """
    psi, rho, kap = _recanon(S.psi, S.rho, S.kap, S.isospin, _canon_perm(S.isospin))
    occ = np.real(np.asarray(jnp.diagonal(rho)))
    keep = np.where(occ >= below)[0]
    if len(keep) == len(occ):
        return dataclasses.replace(S, psi=psi, rho=rho, kap=kap), 0, 0.0
    S2 = dataclasses.replace(S, psi=psi[keep], rho=rho[np.ix_(keep, keep)],
                             kap=kap[np.ix_(keep, keep)], isospin=S.isospin[keep])
    return purify(S2), len(occ) - len(keep), float(occ.sum() - occ[keep].sum())


def stationarity(S: GState, grid: Grid, force: Force, cs: CoulombSolver, eps: float = 1e-7,
                 use_coulomb: bool = True, external=None):
    """
    Per-orbital out-of-span velocity of the global scheme, hbar |d phi_m/dt|_out
    = |Q (h phi_m + F_m / rho_m)| in MeV, on the canonical state, with 1/rho
    regularised by eps as in make_step. Zero for a stationary state. Unlike the
    static solver's residual it does not scale with the occupation, so it
    exposes unconverged nearly empty orbitals.

    Returns (occupations, residuals) of the canonical orbitals.
    """
    psi, rho, kap = _recanon(S.psi, S.rho, S.kap, S.isospin, _canon_perm(S.isospin))
    S = dataclasses.replace(S, psi=psi, rho=rho, kap=kap)
    _, mf, _, delta, _ = fields(S, grid, force, cs, use_coulomb, external)
    iso, same, n = S.isospin, _same_iso(S.isospin), psi.shape[0]
    M = _reg_inv(jnp.where(same, rho, 0.0), eps).T @ jnp.conjugate(jnp.where(same, kap, 0.0))
    dq = jnp.where((iso == 0)[:, None, None, None], delta[0][None], delta[1][None])
    X = (jax.vmap(lambda p, q: apply_hamiltonian(p, mf, q, grid))(psi, iso)
         - pair_view(dq[:, None] * jnp.tensordot(M, _eps_spin(jnp.conjugate(pair_view(psi, grid, force))),
                                                 axes=([1], [0])), grid, force))
    ov = jnp.where(same, grid.wxyz * (jnp.conjugate(psi).reshape(n, -1) @ X.reshape(n, -1).T), 0.0)
    QX = X - jnp.tensordot(ov, psi, axes=([0], [0]))
    r = jnp.sqrt(grid.wxyz * jnp.sum(jnp.abs(QX) ** 2, axis=(1, 2, 3, 4)))
    return np.real(np.asarray(jnp.diagonal(rho))), np.asarray(r)


def mean_gaps(S: GState, grid: Grid, force: Force, cs: CoulombSolver, use_coulomb: bool = True):
    """|kap|-weighted mean pairing gap per isospin, |Tr(Dt kap~^+)| / sum_m |kap~_m| (MeV);
    for a canonical BCS state this is the static solver's sum uv Delta / sum uv."""
    delta = fields(S, grid, force, cs, use_coulomb)[3]
    Dt = -anomalous_matrix(pair_view(S.psi, grid, force), delta, S.isospin, grid)
    out = []
    for q in (0, 1):
        m = np.where(np.asarray(S.isospin) == q)[0]
        ix = np.ix_(m, m)
        num = abs(complex(jnp.sum(Dt[ix] * jnp.conjugate(S.kap[ix]))))
        den = float(jnp.sum(jnp.sqrt(jnp.sum(jnp.abs(S.kap[ix]) ** 2, axis=1))))
        out.append(num / den if den > 1e-12 else 0.0)
    return out


@jax.jit
def _span_projector(hmat, Dt, lam):
    """R of the negative-energy eigenvectors of [[h - lam, Dt], [Dt^+, -(h - lam)^*]]: (rho, kap, Tr rho)."""
    n = hmat.shape[0]
    h = hmat - lam * jnp.eye(n, dtype=hmat.dtype)
    H = jnp.block([[h, Dt], [jnp.conjugate(Dt.T), -jnp.conjugate(h)]])
    _, V = jnp.linalg.eigh(0.5 * (H + jnp.conjugate(H.T)))
    W = V[:, :n]
    rho = W[:n] @ jnp.conjugate(W[:n].T)
    kap = W[:n] @ jnp.conjugate(W[n:].T)
    return rho, 0.5 * (kap - kap.T), jnp.real(jnp.trace(rho))


def _span_hfb(hmat, Dt, N, lam0=None, tol=1e-11):
    """
    HFB inside the span of one isospin: R = projector on the negative-energy
    eigenvectors of [[h - lam, Dt], [Dt^+, -(h - lam)^*]], lam found so that
    Tr rho = N (Illinois regula falsi, bracketed around lam0 when given).
    Returns (rho, kap, lam).
    """
    def f(lam):
        return float(_span_projector(hmat, Dt, lam)[2]) - N

    if lam0 is None:
        e = np.linalg.eigvalsh(np.asarray(hmat))
        lo, hi = float(e[0]) - 50.0, float(e[-1]) + 50.0
    else:                                   # warm start: lam moves little between iterations
        lo, hi = lam0 - 0.02, lam0 + 0.02
    flo, fhi = f(lo), f(hi)
    while flo > 0:
        lo -= 2.0 * (hi - lo)
        flo = f(lo)
    while fhi < 0:
        hi += 2.0 * (hi - lo)
        fhi = f(hi)
    side = 0                                # Illinois regula falsi on the monotone N(lam)
    lam = lo if abs(flo) < abs(fhi) else hi
    for _ in range(100):
        if min(abs(flo), abs(fhi)) < tol * max(N, 1.0):   # e.g. unpaired: N(lam) is a step
            lam = lo if abs(flo) < abs(fhi) else hi
            break
        lam = (lo * fhi - hi * flo) / (fhi - flo) if fhi != flo else 0.5 * (lo + hi)
        if not lo < lam < hi:
            lam = 0.5 * (lo + hi)
        fl = f(lam)
        if abs(fl) < tol * max(N, 1.0) or hi - lo < 1e-13:
            break
        if fl < 0:
            lo, flo = lam, fl
            if side == -1:
                fhi *= 0.5
            side = -1
        else:
            hi, fhi = lam, fl
            if side == 1:
                flo *= 0.5
            side = 1
    rho, kap, _ = _span_projector(hmat, Dt, lam)
    return rho, kap, lam


def _relax_stages(S: GState, grid: Grid, force: Force, cs: CoulombSolver, step: float, e0: float,
                  use_coulomb: bool, external):
    """The compiled halves of a relax iteration for the orbital set of S: (stage1, stage2, blocks)."""
    iso = S.isospin
    same = _same_iso(iso)
    perm = _canon_perm(iso)
    h2m = force.h2m[jnp.asarray(iso)]          # per orbital; T(k) is built inside the jit (not a 300 MB constant)
    n = S.nstates
    blocks = [(q, np.ix_(m, m)) for q in (0, 1) for m in [np.where(np.asarray(iso) == q)[0]] if len(m)]
    # block-diagonal assembly in isospin-sorted order, then a gather back (XLA scatters are slow on GPU)
    order = np.concatenate([np.where(np.asarray(iso) == q)[0] for q, _ in blocks])
    inv = np.empty_like(order)
    inv[order] = np.arange(n)

    def assemble(bl):
        return jax.scipy.linalg.block_diag(*bl)[inv][:, inv]

    @jax.jit
    def stage1(psi, rho, kap, mf_prev, delta_prev, w):
        psi, rho, kap = _recanon(psi, rho, kap, iso, perm)
        dens = _densities_matrix(psi, rho, iso, grid)
        _, mf, _, delta = _complete_fields(dens, pair_density(pair_view(psi, grid, force), kap, iso), grid, force,
                                           cs, use_coulomb, external)
        mf = jax.tree_util.tree_map(lambda a, b: w * a + (1 - w) * b, mf, mf_prev)
        delta = w * delta + (1 - w) * delta_prev
        hpsi = jax.vmap(lambda p, q: apply_hamiltonian(p, mf, q, grid))(psi, iso)
        Dt = -anomalous_matrix(pair_view(psi, grid, force), delta, iso, grid)
        hm = grid.wxyz * (jnp.conjugate(psi).reshape(n, -1) @ hpsi.reshape(n, -1).T)
        hm = 0.5 * (hm + jnp.conjugate(hm.T))
        return psi, mf, delta, hpsi, [hm[ix] for _, ix in blocks], [Dt[ix] for _, ix in blocks]

    @jax.jit
    def stage2(psi, hpsi, delta, rblocks, kblocks):
        rho_new, kap_new = assemble(rblocks), assemble(kblocks)
        # out-of-span descent on the canonical orbitals of the new R~ (h psi rotated with psi)
        V = _canon_rotation(rho_new, iso, perm)
        psi = jnp.tensordot(V.T, psi, axes=([1], [0]))
        hpsi = jnp.tensordot(V.T, hpsi, axes=([1], [0]))
        rho = jnp.conjugate(V.T) @ rho_new @ V
        kap = jnp.conjugate(V.T) @ kap_new @ jnp.conjugate(V)
        occ = jnp.real(jnp.diagonal(rho))
        kmag = jnp.sqrt(jnp.sum(jnp.abs(jnp.where(same, kap, 0.0)) ** 2, axis=1))
        dq = jnp.where((iso == 0)[:, None, None, None], delta[0][None], delta[1][None])
        F = -pair_view(dq[:, None] * jnp.tensordot(jnp.conjugate(jnp.where(same, kap, 0.0)),
                                                    _eps_spin(jnp.conjugate(pair_view(psi, grid, force))),
                                                    axes=([1], [0])), grid, force)
        G = occ[:, None, None, None, None] * hpsi + F
        ov = jnp.where(same, grid.wxyz * (jnp.conjugate(psi).reshape(n, -1) @ G.reshape(n, -1).T), 0.0)
        G = G - jnp.tensordot(ov, psi, axes=([0], [0]))
        dmax = jnp.array([jnp.max(jnp.abs(delta[0])), jnp.max(jnp.abs(delta[1]))])[iso]
        Tk = h2m[:, None, None, None] * grid.k2[None]
        den = (occ[:, None, None, None] * (e0 + Tk) + (kmag * dmax)[:, None, None, None]) + 1e-30
        Gp = jnp.fft.ifftn(jnp.fft.fftn(G, axes=(-3, -2, -1)) / den[:, None], axes=(-3, -2, -1))
        return _lowdin(psi - step * Gp, same, grid), rho, kap

    return stage1, stage2, blocks


def relax(S: GState, grid: Grid, force: Force, cs: CoulombSolver, n_iter: int = 200,
          step: float = 1.0, e0: float = 100.0, mix: float = 0.5, use_coulomb: bool = True,
          verbose: int = 0, external=None, callback=None):
    """
    Static HFB in the representation of the global scheme: the minimum of the
    same functional over the orbitals and the in-span generalized density, so
    the result is stationary under make_step (up to the gauge rotation at lam).

    Each iteration, with the mean field and Delta(r) mixed linearly with the
    previous iteration's (mix): (1) in span, R~ becomes the projector of the
    full HFB matrix [[h~, Dt], [Dt^+, -h~*]] (lam per isospin from the particle
    number), so purity and N hold exactly; (2) out of span, each canonical
    orbital takes a preconditioned descent step on Q (rho_m h phi_m + F_m),
    with the preconditioner rho_m (e0 + T) + |kap_m| Delta_max, bounded for
    every occupation (the FORTRAN-type solver floors the weights instead, which
    leaves nearly empty orbitals unconverged); (3) Loewdin.

    external(dens) -> (2, nx, ny, nz) adds a potential to h, e.g. a constraint
    held at fixed multipliers; the energies printed exclude it. callback(it, S)
    is called every `verbose` iterations (e.g. to checkpoint): one long call
    compiles the iteration once, repeated short calls recompile it each time.

    Returns (state, history of (E, max residual by occupation band)).
    """
    stage1, stage2, blocks = _relax_stages(S, grid, force, cs, step, e0, use_coulomb, external)
    N = {q: float(jnp.real(jnp.trace(S.rho[ix]))) for q, ix in blocks}
    lams = {q: None for q, _ in blocks}
    hist = []
    _, mf_prev, _, delta_prev, _ = fields(S, grid, force, cs, use_coulomb, external)
    import time as _time
    t_last, it_last = _time.time(), -1
    for it in range(n_iter):
        psi, mf_prev, delta_prev, hpsi, hb, Db = stage1(S.psi, S.rho, S.kap, mf_prev, delta_prev, mix)
        rb, kb = [], []
        for (q, _), hq, Dq in zip(blocks, hb, Db):
            r, k, lams[q] = _span_hfb(hq, Dq, N[q], lams[q])
            rb.append(r)
            kb.append(k)
        psi, rho, kap = stage2(psi, hpsi, delta_prev, rb, kb)
        S = dataclasses.replace(S, psi=psi, rho=rho, kap=kap)
        if verbose and (it % verbose == 0 or it == n_iter - 1):
            E, ep = energy(S, grid, force, cs, use_coulomb)
            o, r = stationarity(S, grid, force, cs, use_coulomb=use_coulomb, external=external)
            b = np.minimum(o, 1 - o)
            bands = [float(r[(b >= lo) & (b < hi)].max()) if np.any((b >= lo) & (b < hi)) else 0.0
                     for lo, hi in ((1e-3, 1.0), (1e-5, 1e-3), (0.0, 1e-5))]
            hist.append((it, E, *ep, *bands))
            rate = (_time.time() - t_last) / max(it - it_last, 1)
            t_last, it_last = _time.time(), it
            print(f"  relax {it:5d}  E {E:.6f}  Epair {ep[0]:.4f} {ep[1]:.4f}  residual (MeV) "
                  f"occ>1e-3 {bands[0]:.2e}  1e-5..1e-3 {bands[1]:.2e}  <1e-5 {bands[2]:.2e}  ({rate:.2f} s/it)",
                  flush=True)
            if callback is not None:
                callback(it, S)
    return S, hist


# ── Collisions ───────────────────────────────────────────────────────────────

def smooth_boost_phase(z, z_c, L, band: float = 6.0, soft: float = 1.0):
    """
    f(z) with f' = 1 over the fragment centred at z_c and a compensating slope
    in a band at the far side of the periodic box, so exp(i k f(z)) is a
    Galilean boost of the bound orbitals for any k (not just lattice momenta).
    """
    z = np.asarray(z)
    dz = float(z[1] - z[0])
    u = (z - z_c + 0.5 * L) % L - 0.5 * L
    S = 0.5 * (1.0 + np.tanh((np.abs(u) - (0.5 * L - band)) / soft))
    g = (1.0 - S) - np.sum(1.0 - S) / max(np.sum(S), 1e-12) * S
    order = np.argsort(u)
    f = np.empty_like(z)
    f[order] = np.concatenate([[0.0], np.cumsum(0.5 * (g[order][1:] + g[order][:-1]) * dz)])
    i_c = int(np.argmin(np.abs(u)))
    return f - f[i_c] + u[i_c]


def place_fragments(psiA, psiB, grid: Grid, ZA: int, AA: int, ZB: int, AB: int,
                    sep: float, e_asym: float, band: float = 6.0):
    """
    Translate fragment A to -sep/2 and B to +sep/2 along z (whole grid points)
    and boost them toward each other with equal and opposite momenta, so that
    the asymptotic c.m. energy is e_asym: the kinetic energy at the starting
    separation is e_asym minus the point Coulomb energy there.

    Returns (psiA, psiB, info).
    """
    shift = int(round(0.5 * sep / grid.dz))
    zA, zB = -shift * grid.dz, shift * grid.dz
    vc = 1.44 * ZA * ZB / (zB - zA)
    ekin = e_asym - vc
    if ekin <= 0:
        raise ValueError(f"no kinetic energy left at {zB - zA:.2f} fm (Coulomb {vc:.1f} MeV)")
    P = np.sqrt(2.0 * 938.9 * AA * AB / (AA + AB) * ekin)
    kA, kB = P / (AA * 197.3), P / (AB * 197.3)
    L = grid.nz * grid.dz
    fA = jnp.asarray(smooth_boost_phase(grid.z, zA, L, band))
    fB = jnp.asarray(smooth_boost_phase(grid.z, zB, L, band))
    psiA = jnp.roll(psiA, -shift, axis=-1) * jnp.exp(1j * kA * fA)[None, None, None, None]
    psiB = jnp.roll(psiB, shift, axis=-1) * jnp.exp(-1j * kB * fB)[None, None, None, None]
    return psiA, psiB, dict(zA=zA, zB=zB, kA=kA, kB=kB, e_kin=ekin, e_coul=vc)


def _isospin_order(isoA, isoB):
    iso = jnp.concatenate([isoA, isoB])
    order = np.asarray(jnp.argsort(iso, stable=True))
    return order, iso[order]


def make_collision(GA: GState, GB: GState, grid: Grid, ZA: int, AA: int, ZB: int, AB: int,
                   sep: float, e_asym: float, phase: float = 0.0, band: float = 6.0):
    """
    Composite GState of two rest-frame fragments (already in the collision box):
    placed and boosted by place_fragments, relative pairing phase `phase` on
    fragment B, block-diagonal matrices sorted by isospin, Loewdin with the
    matrices carried so the composite is the product state. Purify it next.

    Returns (state, info).
    """
    psiA, psiB, info = place_fragments(GA.psi, GB.psi, grid, ZA, AA, ZB, AB, sep, e_asym, band)
    Z0 = jnp.zeros((GA.nstates, GB.nstates), dtype=GA.rho.dtype)
    rho = jnp.block([[GA.rho, Z0], [Z0.T, GB.rho]])
    kap = jnp.block([[GA.kap, Z0], [Z0.T, GB.kap * jnp.exp(1j * phase)]])
    order, iso = _isospin_order(GA.isospin, GB.isospin)
    psi, rho, kap = jnp.concatenate([psiA, psiB])[order], rho[order][:, order], kap[order][:, order]
    p = psi.reshape(psi.shape[0], -1)
    Sm = jnp.where(_same_iso(iso), grid.wxyz * (jnp.conjugate(p) @ p.T), 0.0)
    lam, U = jnp.linalg.eigh(0.5 * (Sm + jnp.conjugate(Sm.T)))
    Shalf = (U * lam ** 0.5) @ jnp.conjugate(U.T)
    psi = jnp.tensordot(((U * lam ** -0.5) @ jnp.conjugate(U.T)).T, psi, axes=([1], [0]))
    return GState(psi=psi, rho=Shalf @ rho @ Shalf, kap=Shalf @ kap @ Shalf.T, isospin=iso), info


def make_ebata_collision(EA: EbataState, EB: EbataState, grid: Grid, ZA: int, AA: int, ZB: int,
                         AB: int, sep: float, e_asym: float, phase: float = 0.0, band: float = 6.0):
    """The EbataState counterpart of make_collision (same placement, orbitals and order)."""
    psiA, psiB, info = place_fragments(EA.psi, EB.psi, grid, ZA, AA, ZB, AB, sep, e_asym, band)
    order, iso = _isospin_order(EA.isospin, EB.isospin)
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    pt = np.concatenate([np.asarray(EA.partners), np.asarray(EB.partners) + EA.nstates])
    psi = _lowdin(jnp.concatenate([psiA, psiB])[order], _same_iso(iso), grid)
    return EbataState(psi=psi, occ=jnp.concatenate([EA.occ, EB.occ])[order],
                      kappa=jnp.concatenate([EA.kappa, EB.kappa * jnp.exp(1j * phase)])[order],
                      partners=jnp.asarray(inv[pt[order]]), isospin=iso), info


# ── Ebata's Cb-TDHFB (comparison arm) ────────────────────────────────────────

def make_ebata_step(E: EbataState, grid: Grid, force: Force, cs: CoulombSolver,
                    taylor_order: int = 6, occ_floor: float = 1e-12, use_coulomb: bool = True):
    """
    Jitted step(E, dt) -> E of Ebata's Cb-TDHFB with contact pairing:

        i hbar rho_k'   = kappa_k Delta_k* - kappa_k* Delta_k
        i hbar kappa_k' = (eta_k + eta_kbar - 2 lambda) kappa_k + Delta_k (2 rho_k - 1)
        i hbar phi_k'   = (h - eta_k) phi_k,      eta_k = <phi_k|h|phi_k>

    with Delta_k = Int Delta(r) |phi_k|^2 (the diagonal gap: the off-diagonal
    pair field is discarded, his Eq. 28) and 2 lambda the |kappa|-weighted mean
    of eta_k + eta_kbar. Fields at the midpoint (predictor-corrector); the
    amplitudes advance by explicit Euler, as in every comparison run. Partners
    are the stored rest-frame ones: re-finding them by time reversal in a
    moving frame picks the counter-moving orbital.
    """
    iso, pt = E.isospin, E.partners
    same = _same_iso(iso)

    def flds(psi, occ, kappa, coulomb):
        dens = _densities_diag(psi, occ, iso, grid)
        return _complete_fields(dens, pair_density_canonical(pair_view(psi, grid, force), kappa, iso), grid, force,
                                cs, coulomb)

    @jax.jit
    def core(psi, occ, kappa, dt):
        dn0, mf0, _, d0 = flds(psi, occ, kappa, use_coulomb)
        psi_p = propagate_taylor(psi, mf0, iso, grid, dt, max(1, taylor_order // 2))
        dn_p, _, _, d_p = flds(psi_p, occ, kappa, False)
        dn_m = jax.tree_util.tree_map(lambda a, b: 0.5 * (a + b), dn0, dn_p)
        wc_m = solve_poisson(dn_m.rho[1], cs, grid) if use_coulomb else None
        mf_m = compute_skyrme_meanfield(dn_m, force, grid, coulomb_potential=wc_m, use_coulomb=use_coulomb)
        d_m = 0.5 * (d0 + d_p)
        hpsi = jax.vmap(lambda p, q: apply_hamiltonian(p, mf_m, q, grid))(psi, iso)
        eta = grid.wxyz * jnp.real(jnp.sum(jnp.conjugate(psi) * hpsi, axis=(1, 2, 3, 4)))
        eta_sum = eta + eta[pt]
        psi_new = propagate_taylor(psi, mf_m, iso, grid, dt, taylor_order)
        psi_new = psi_new * jnp.exp(1j * eta * dt / HBARC)[:, None, None, None, None]
        dq = jnp.where((iso == 0)[:, None, None, None], d_m[0][None], d_m[1][None])
        pp = pair_view(psi, grid, force)
        gap = grid.wxyz * jnp.sum(dq[:, None] * jnp.real(pp * jnp.conjugate(pp)), axis=(1, 2, 3, 4))
        two_lam = jnp.zeros_like(eta_sum)
        for q in (0, 1):
            wq = jnp.where(iso == q, jnp.abs(kappa), 0.0)
            two_lam = jnp.where(iso == q, jnp.sum(wq * eta_sum) / jnp.maximum(jnp.sum(wq), 1e-30), two_lam)
        drho = -(2.0 / HBARC) * jnp.imag(gap * jnp.conjugate(kappa))
        dkap = ((eta_sum - two_lam) * kappa + gap * (2.0 * occ - 1.0)) / (1j * HBARC)
        occ_new = jnp.clip(occ + dt * drho, occ_floor, 1.0 - occ_floor)
        return _lowdin(psi_new, same, grid), occ_new, kappa + dt * dkap

    def step(Ex: EbataState, dt: float) -> EbataState:
        psi, occ, kappa = core(Ex.psi, Ex.occ, Ex.kappa, dt)
        return dataclasses.replace(Ex, psi=psi, occ=occ, kappa=kappa, time=Ex.time + dt)
    return step


# ── Diagnostics ──────────────────────────────────────────────────────────────

def particle_numbers(S):
    """(N, Z): the trace of rho~ (or the sum of occupations) per isospin."""
    occ = S.occ if isinstance(S, EbataState) else jnp.real(jnp.diagonal(S.rho))
    return (float(jnp.sum(jnp.where(S.isospin == 0, occ, 0.0))),
            float(jnp.sum(jnp.where(S.isospin == 1, occ, 0.0))))


def _transport(S: GState, mf, grid):
    """-div j = (2/hbar) sum_k lam_k Im(chi_k^+ h chi_k) over the eigen-orbitals of rho~."""
    shift = 10.0
    lam, U = jnp.linalg.eigh(jnp.where(_same_iso(S.isospin), S.rho, 0.0)
                             + jnp.diag(jnp.where(S.isospin == 1, shift, 0.0)))
    iso_k = jnp.where(lam > 0.5 * shift, 1, 0)
    lam = lam - jnp.where(iso_k == 1, shift, 0.0)
    chi = jnp.tensordot(U, S.psi, axes=([0], [0]))
    hc = jax.vmap(lambda p, q: apply_hamiltonian(p, mf, q, grid))(chi, iso_k)
    r = (2.0 / HBARC) * jnp.imag(jnp.sum(jnp.conjugate(chi) * hc, axis=1)) * lam[:, None, None, None]
    return _by_isospin(r, iso_k)


def region_number_moments(S: GState, regions, grid: Grid):
    """
    Exact first and second moments of the particle number in spatial regions
    (Wick's theorem for the HFB state; a mixed state from absorbing boundaries
    too). regions: list of (nx, ny, nz) weights Theta_a(r) in [0, 1].

    For each isospin q returns dict(mean=(n_reg,), cov=(n_reg, n_reg)) with
      <N_a>          = Tr(rho~ O_a),
      <dN_a dN_b>    = Tr(rho~ O_ab) - Tr(rho~ O_a rho~ O_b) + Tr(kap~ O_b^T kap~^+ O_a),
    O_a,mn = <phi_m|Theta_a|phi_n> and O_ab the same with Theta_a Theta_b. Over
    all space the variance is 2 Tr(kap~ kap~^+) = 4 sum_k>0 u_k^2 v_k^2.
    """
    out = []
    w = grid.wxyz
    for q in (0, 1):
        m = np.asarray(S.isospin) == q
        p = jnp.asarray(S.psi)[m].reshape(int(m.sum()), 2, -1)
        rho, kap = jnp.asarray(S.rho)[np.ix_(m, m)], jnp.asarray(S.kap)[np.ix_(m, m)]
        th = [jnp.asarray(r).reshape(-1) for r in regions]
        O = [w * jnp.einsum('msr,r,nsr->mn', jnp.conjugate(p), t, p) for t in th]
        mean = np.array([float(jnp.real(jnp.trace(rho @ Oa))) for Oa in O])
        cov = np.zeros((len(th), len(th)))
        for a in range(len(th)):
            for b in range(a, len(th)):
                Oab = w * jnp.einsum('msr,r,nsr->mn', jnp.conjugate(p), th[a] * th[b], p)
                c = (jnp.trace(rho @ Oab) - jnp.trace(rho @ O[a] @ rho @ O[b])
                     + jnp.trace(kap @ O[b].T @ jnp.conjugate(kap.T) @ O[a]))
                cov[a, b] = cov[b, a] = float(jnp.real(c))
        out.append(dict(mean=mean, cov=cov))
    return out


def canonical_levels(psi, rho, isospin, mf, grid: Grid):
    """
    Canonical levels: the eigen-orbitals chi_k of rho~ within each isospin, with
    their occupations v2_k and single-particle energies <chi_k|h|chi_k> (MeV).
    Returns (isospin_k, v2_k, e_k), each (n,). Jit with grid closed over.
    """
    shift = 10.0
    lam, U = jnp.linalg.eigh(jnp.where(_same_iso(isospin), rho, 0.0) + jnp.diag(jnp.where(isospin == 1, shift, 0.0)))
    iso_k = jnp.where(lam > 0.5 * shift, 1, 0)
    lam = lam - jnp.where(iso_k == 1, shift, 0.0)
    chi = jnp.tensordot(U, psi, axes=([0], [0]))
    hc = jax.vmap(lambda p, q: apply_hamiltonian(p, mf, q, grid))(chi, iso_k)
    num = jnp.real(jnp.sum(jnp.conjugate(chi) * hc, axis=(1, 2, 3, 4)))
    return iso_k, lam, num / jnp.real(jnp.sum(jnp.conjugate(chi) * chi, axis=(1, 2, 3, 4)))


def continuity_defect(S: GState, ref_step, grid: Grid, force: Force, cs: CoulombSolver):
    """
    Local continuity: D = d rho/dt + div j, as the fraction Int|D| / Int|d rho/dt|
    per isospin, for the scheme and for its floor (same orbitals, kap~ = 0:
    pairing-free motion is exactly local, so the floor is the noise of the
    test). d rho/dt by Richardson extrapolation of two short steps of ref_step
    (a make_step with pred_n_comp=None). At rest d rho/dt ~ 0 and the fraction
    means nothing.

    Returns {'n': .., 'p': .., 'floor_n': .., 'floor_p': ..}.
    """
    def frac(g):
        dn0, mf0 = fields(g, grid, force, cs)[:2]
        rates = [(fields(ref_step(g, h), grid, force, cs)[0].rho - dn0.rho) / h for h in (0.01, 0.005)]
        rdot = 2.0 * rates[1] - rates[0]
        D = rdot - _transport(g, mf0, grid)
        return [float(jnp.sum(jnp.abs(D[q])) / jnp.maximum(jnp.sum(jnp.abs(rdot[q])), 1e-300)) for q in (0, 1)]
    s = frac(S)
    f = frac(dataclasses.replace(S, kap=jnp.zeros_like(S.kap)))
    return dict(n=s[0], p=s[1], floor_n=f[0], floor_p=f[1])
