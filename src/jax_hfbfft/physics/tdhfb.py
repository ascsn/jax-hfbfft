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
     as the exact unitary exp(B Phi^+ - Phi B^+). rho~^-1 is regularised by
     eps; half of the force is applied before and half after steps 2-3;
  2. in-span pairing: R -> U R U^+ with U = exp(-i dt H~/hbar),
     H~ = [[0, Dt], [Dt^+, 0]], Dt_mn = -<phi_m|Delta|eps phi_n*>, Delta taken
     self-consistently at the half step;
  3. mean field: phi -> exp(-i h dt/hbar) phi (Taylor series).

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
  * `purify` after any kick or collision setup;
  * dt = 0.05 fm/c, eps = 1e-7, n_comp = 4, pred_n_comp = 1.

Known limit: orbitals with occupation below ~1e-5 carry unconverged tails that
the 1/rho of step 1 amplifies (U238 at rest pumps neutron pairing). `drop_empty`
removes them; the long-time behaviour in that regime is still open.

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


def anomalous_matrix(psi, delta, isospin, grid: Grid):
    """A_mn = <phi_m| Delta |eps conj(phi_n)>, same isospin (antisymmetric)."""
    n = psi.shape[0]
    dq = jnp.where((isospin == 0)[:, None, None, None], delta[0][None], delta[1][None])
    w = (jnp.conjugate(psi) * dq[:, None]).reshape(n, -1)
    t = _eps_spin(jnp.conjugate(psi)).reshape(n, -1)
    return jnp.where(_same_iso(isospin), grid.wxyz * (w @ t.T), 0.0)


# ── Fields and energy ────────────────────────────────────────────────────────

def _densities_diag(psi, occ, isospin, grid: Grid) -> Densities:
    zeros, ones = jnp.zeros_like(occ), jnp.ones_like(occ)
    return compute_densities(psi, occ, zeros, ones, ones, isospin, grid)


def _densities_matrix(psi, rho, isospin, grid: Grid) -> Densities:
    """All Skyrme densities of rho~, from its eigen-orbitals with signed weights."""
    shift = 10.0                      # separates the isospins inside one eigh
    lam, V = jnp.linalg.eigh(rho + jnp.diag(jnp.where(isospin == 1, shift, 0.0)).astype(rho.dtype))
    iso_k = jnp.where(lam > 0.5 * shift, 1, 0).astype(isospin.dtype)
    lam = lam - jnp.where(iso_k == 1, shift, 0.0)
    chi = jnp.tensordot(V, psi, axes=([0], [0]))
    return _densities_diag(chi, lam, iso_k, grid)


def _complete_fields(dens, xi, grid, force, cs, use_coulomb):
    """Coulomb, mean field and pair field. |xi| enters the DDDI rearrangement term."""
    dens = dataclasses.replace(dens, chi=jnp.abs(xi))
    wcoul = jnp.zeros(grid.shape, dtype=dens.rho.dtype)
    if use_coulomb:
        wcoul = solve_poisson(dens.rho[1], cs, grid)
    mf = compute_skyrme_meanfield(dens, force, grid,
                                  coulomb_potential=wcoul if use_coulomb else None,
                                  use_coulomb=use_coulomb)
    delta = pair_field(xi, dens.rho[0] + dens.rho[1], force)
    return dens, mf, wcoul, delta


def fields(S, grid: Grid, force: Force, cs: CoulombSolver, use_coulomb: bool = True):
    """(densities, meanfield, wcoul, delta, xi) of a GState or EbataState."""
    if isinstance(S, EbataState):
        dens = _densities_diag(S.psi, S.occ, S.isospin, grid)
        xi = pair_density_canonical(S.psi, S.kappa, S.isospin)
    else:
        dens = _densities_matrix(S.psi, S.rho, S.isospin, grid)
        xi = pair_density(S.psi, S.kap, S.isospin)
    return _complete_fields(dens, xi, grid, force, cs, use_coulomb) + (xi,)


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

def kramers_basis(psi, occ, kappa, isospin, wxyz: float, tol: float = 1e-7):
    """
    Rotate each block of equal (occ, kappa) orbitals into an exact Kramers basis,
    phi_abar = T phi_a. Densities are unchanged (a unitary inside a block of
    equal occupations). Must be built from time-even (unboosted) orbitals: time
    reversal flips momentum, so a moving nucleus has no partners to find.

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
    if worst > 1e-6:
        raise ValueError(f"kramers_basis: a block is not closed under time reversal "
                         f"(defect {worst:.3e})")
    return jnp.asarray(out), jnp.asarray(partners)


def _canonical(psi, wocc, wguv, isospin, grid, wstates=None, pairwg=None, uv_floor=0.0):
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
    psi, partners = kramers_basis(psi, occ, kappa, isospin, float(grid.wxyz))
    return psi, occ, kappa, partners


def from_static(psi, wocc, wguv, isospin, grid: Grid, wstates=None, pairwg=None,
                uv_floor: Optional[float] = 0.0) -> GState:
    """
    GState of a converged static (canonical) HFB solution.

    The orbitals are rotated into a Kramers basis and each pair (a, abar) gets
    kap~[a, abar] = kappa_a g_a, with the unit phase g_a fixing [phi_a phi_abar]
    to |phi_a|^2. uv_floor=0 rebuilds kappa = sqrt(v2 (1 - v2)) from the
    occupations; None keeps the static's wguv.
    """
    psi, occ, kappa, partners = _canonical(psi, wocc, wguv, isospin, grid, wstates, pairwg, uv_floor)
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
    lam, U = jnp.linalg.eigh(0.5 * (M + jnp.conjugate(M.T)))
    return (U * (lam / (lam ** 2 + eps ** 2))[None, :]) @ jnp.conjugate(U.T)


def _complement_increment(psi, kap, isospin, same, Rinv, delta, grid, dt):
    """B = -(i dt/hbar) Q F rho~^-1 with F_m = -Delta sum_n kap~*_mn eps phi_n*."""
    M = Rinv.T @ jnp.conjugate(jnp.where(same, kap, 0.0))
    dq = jnp.where((isospin == 0)[:, None, None, None], delta[0][None], delta[1][None])
    X = -dq[:, None] * jnp.tensordot(M, _eps_spin(jnp.conjugate(psi)), axes=([1], [0]))
    n = psi.shape[0]
    ov = jnp.where(same, grid.wxyz * (jnp.conjugate(psi).reshape(n, -1) @ X.reshape(n, -1).T), 0.0)
    QX = X - jnp.tensordot(ov, psi, axes=([0], [0]))
    return (-1j * dt / HBARC) * QX


def _complement_rotate(psi, B, same, grid):
    """exp(B Phi^+ - Phi B^+) Phi = Phi cos(sqrt G) + B sinc(sqrt G), G = B^+ B (B in the complement)."""
    n = psi.shape[0]
    b = B.reshape(n, -1)
    G = jnp.where(same, grid.wxyz * (jnp.conjugate(b) @ b.T), 0.0)
    lam, U = jnp.linalg.eigh(0.5 * (G + jnp.conjugate(G.T)))
    sq = jnp.sqrt(jnp.maximum(lam, 0.0))
    C = (U * jnp.cos(sq)[None, :]) @ jnp.conjugate(U.T)
    sinc = jnp.where(sq > 1e-12, jnp.sin(sq) / jnp.where(sq > 1e-12, sq, 1.0), 1.0 - lam / 6.0)
    Fs = (U * sinc[None, :]) @ jnp.conjugate(U.T)
    return jnp.tensordot(C.T, psi, axes=([1], [0])) + jnp.tensordot(Fs.T, B, axes=([1], [0]))


def _exp_update(rho, kap, Dt, dt):
    """R -> U R U^+, U = exp(-i dt H~/hbar), H~ = [[0, Dt], [Dt^+, 0]]; purity is exact."""
    n = rho.shape[0]
    H = jnp.block([[jnp.zeros_like(Dt), Dt], [jnp.conjugate(Dt.T), jnp.zeros_like(Dt)]])
    lam, V = jnp.linalg.eigh(0.5 * (H + jnp.conjugate(H.T)))
    U = (V * jnp.exp(-1j * lam * dt / HBARC)[None, :]) @ jnp.conjugate(V.T)
    R = jnp.block([[rho, kap], [jnp.conjugate(kap.T), jnp.eye(n, dtype=rho.dtype) - rho.T]])
    R = U @ R @ jnp.conjugate(U.T)
    return R[:n, :n], R[:n, n:]


def _lowdin(psi, same, grid):
    n = psi.shape[0]
    p = psi.reshape(n, -1)
    Sm = jnp.where(same, grid.wxyz * (jnp.conjugate(p) @ p.T), 0.0)
    lam, U = jnp.linalg.eigh(0.5 * (Sm + jnp.conjugate(Sm.T)))
    return jnp.tensordot(((U * lam ** -0.5) @ jnp.conjugate(U.T)).T, psi, axes=([1], [0]))


def _canon_perm(isospin):
    """Column order of the shifted eigh in _recanon: each isospin's columns back into its own slots."""
    lab = np.asarray(isospin)
    perm = np.empty(len(lab), dtype=int)
    perm[np.argsort(lab, kind='stable')] = np.arange(len(lab))
    return jnp.asarray(perm)


def _recanon(psi, rho, kap, isospin, perm):
    """Diagonalise rho~ inside each isospin and rotate orbitals and kap~ with it (a gauge change)."""
    same = _same_iso(isospin)
    Hm = jnp.where(same, rho, 0.0)
    Hm = 0.5 * (Hm + jnp.conjugate(Hm.T)) + jnp.diag(10.0 * isospin).astype(Hm.dtype)
    _, V = jnp.linalg.eigh(Hm)
    V = jnp.where(same, V[:, perm], 0.0)
    return (jnp.tensordot(V.T, psi, axes=([1], [0])), jnp.conjugate(V.T) @ rho @ V,
            jnp.conjugate(V.T) @ kap @ jnp.conjugate(V))


def make_step(S: GState, grid: Grid, force: Force, cs: CoulombSolver, eps: float = 1e-7,
              n_comp: int = 4, pred_n_comp: Optional[int] = 1, taylor_order: int = 6,
              use_coulomb: bool = True):
    """
    Jitted step(S, dt) -> S of the global scheme for the orbital set of S.

    n_comp sub-rotations apply the complement force; the predictor, whose only
    job is the midpoint fields, uses pred_n_comp (None = n_comp, the exact
    two-pass scheme; 1 changes rho by ~6e-10 over 10 fm/c and is 1.5x faster).
    Rebuild the step if the number or isospin of the orbitals changes.
    """
    iso = S.isospin
    same = _same_iso(iso)
    perm = _canon_perm(iso)
    pnc = n_comp if pred_n_comp is None else pred_n_comp

    def flds(psi, rho, kap):
        dens = _densities_matrix(psi, rho, iso, grid)
        dens, mf, _, delta = _complete_fields(dens, pair_density(psi, kap, iso), grid, force, cs, use_coulomb)
        return dens, mf, delta

    def Dt_of(psi, kap, rho_tot):
        delta = pair_field(pair_density(psi, kap, iso), rho_tot, force)
        return -anomalous_matrix(psi, delta, iso, grid)

    def complement(psi, rho, kap, delta, dt, nc):
        Rinv = _reg_inv(jnp.where(same, rho, 0.0), eps)     # rho~ is fixed during the rotations
        for _ in range(nc):
            psi = _complement_rotate(
                psi, 0.5 / nc * _complement_increment(psi, kap, iso, same, Rinv, delta, grid, dt), same, grid)
        return psi

    def substep(psi, rho, kap, mf, delta, rho_tot, dt, nc):
        psi = complement(psi, rho, kap, delta, dt, nc)
        # in-span pairing: exact unitary with a self-consistent midpoint pair field
        _, kh = _exp_update(rho, kap, Dt_of(psi, kap, rho_tot), 0.5 * dt)
        rho, kap = _exp_update(rho, kap, Dt_of(psi, kh, rho_tot), dt)
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

    def step(Sx: GState, dt: float) -> GState:
        psi, rho, kap = core(Sx.psi, Sx.rho, Sx.kap, dt)
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
        return _complete_fields(dens, pair_density_canonical(psi, kappa, iso), grid, force, cs, coulomb)

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
        gap = grid.wxyz * jnp.sum(dq[:, None] * jnp.real(psi * jnp.conjugate(psi)), axis=(1, 2, 3, 4))
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
