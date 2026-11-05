"""
Linear response of a converged two-basis HFB state (SolverConfig.orbital_solver = 'two_basis').

One operator, the QRPA (linearised TDHFB) Liouvillian L of the ground state R0, gives

  * strength functions: (omega - L) dR = [F, R0] at omega = E + i Gamma/2, S(E) = -Im Tr(F dR) / pi, a
    Lorentzian of full width Gamma per state: from one Lanczos chain for all E (Response.lanczos and
    Response.spectrum) or by GMRES per E (Response.strength);
  * static polarizabilities m_-1 (Response.polarizability; alpha_D = 2 e^2 m_-1 for the dipole);
  * exact first derivatives of the ground state with respect to the parameters theta of the
    functional: L dR = -[dH/dtheta, R0] at omega = 0 with N, Z (and optional one-body constraints)
    held fixed, per parameter (Response.derivatives) or per observable for all parameters at once
    (adjoint, Response.sensitivities). dE/dtheta itself is Hellmann-Feynman (energy_gradient).

Representation. The HF basis {phi_k} of the state (the Ritz vectors in the span of its orbitals, n
per isospin) carries the Bogoliubov problem without time-reversal symmetry, H = [[h - lam, D],
[-D*, -(h - lam)*]] with the pairing window F = sum_k f(e_k - lam) |phi_k><phi_k| of
gcg.two_basis_hfb; R0 = projector on its negative-energy eigenvectors, taken as the exact partners (V*; U*) of
the positive ones. dR has 2qp blocks X (a > 0, b < 0) and Y (a < 0, b > 0), antisymmetric on physical
vectors. The rest of the single-particle space is reached by orbital amplitudes: for each
quasiparticle a with hole component v_a = sum_k V*_ka phi_k (energy E_a > 0), dR gains
|x_a><v_a| + |v_a><conj(w_a)|, x_a and conj(w_a) orthogonal to the basis (Sternheimer), with

    omega x_a = (h - lam + E_a) x_a + Q dh v_a,    omega w_a = -conj[(h - lam + E_a) conj(w_a) + Q dh^+ v_a].

No pairing reaches them: the window is zero outside the basis.

Moving window (static problems of paired states, Response(window='moving'), the default). The solver's
window follows h and lam, so its fixed point is not a stationary point of one energy: with F held fixed
(window='frozen', the QRPA proper) paired derivatives differ from finite differences of the solver at
the 1% level (0.5% even in dE, which is then not Hellmann-Feynman). The moving window adds
dF = Gamma o (dh - dlam) inside the basis, Gamma_kl = (f_k - f_l)/(e_k - e_l) (Daleckii-Krein; f' on ties,
dlam the number multiplier), and f_k (|y_k><k| + |k><y~_k|) outside it, y_k = Q (e_k - h)^-1 Q dh phi_k
(and y~_k with dh^+) for the windowed states: unknowns (y, conj(y~), Gamma o dh) join the static system,
dF changes the pair density of F kappa F^T and the pair field F Delta F^T, whose rows outside the basis
drive the x_a. The operator is then not K-symmetric: sensitivities() falls back to forward solves.
Validated against finite differences of the re-converged solver (O20 SLy4: dE 2e-7, radii 1e-5).
uv_floor > 0 statics (the default 1e-6 floors u v at 1e-3) are slightly non-variational: use uv_floor = 0.

All field derivatives are forward-mode
derivatives (jax.jvp) of the mean field, so every term of the functional enters exactly; the pair
field is g kappa~ with kappa~ = chi + i zeta complex (chi the real pair density of the statics).

Structure. With eta = +1 on (X, x) and -1 on (Y, w) and weight 1/2 on the doubled 2qp blocks, K = eta L
is Hermitian, and positive for a stable state: L is self-adjoint in <u, K v>, so the static problem
and its adjoint coincide (Response.sensitivities) and a three-term Lanczos recursion applies. On a
grid the translational mode is slightly unstable (K < 0 along it): Lanczos chains are kept in one
parity sector, and static dipole responses fix the c.m. (constraints).
"""

import dataclasses
from functools import partial
from typing import Optional, Sequence

import jax
import jax.scipy.linalg
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from jax_hfbfft.physics.coulomb import solve_poisson
from jax_hfbfft.physics.densities import compute_densities
from jax_hfbfft.physics.edf import FY_KEYS
from jax_hfbfft.physics.energies import compute_integrated_energy
from jax_hfbfft.physics.meanfield import apply_hamiltonian, compute_skyrme_meanfield
from jax_hfbfft.physics.constraints import compute_constraint_potential, penalty_qcorr

# Skyrme parameters the derivatives are taken with respect to (CJ0/CJ1 added for J^2 functionals)
SKYRME_PARAMS = ('t0', 't1', 't2', 't3', 'x0', 'x1', 'x2', 'x3', 't4', 'b4p', 'power', 'v0neut', 'v0prot')
# the same functional in coupling-constant form (Force parametrization 'C')
C_PARAMS_DEFAULT = ('Crho0', 'Crho1', 'Crho0D', 'Crho1D', 'Ctau0', 'Ctau1', 'Cdrho0', 'Cdrho1', 'CdJ0', 'CdJ1',
                    'power', 'v0neut', 'v0prot')
# Fayans volume, surface and pairing parameters (functional 'fayans'); its spin-orbit rides on t4/b4p
FAYANS_PARAMS = tuple('fy.' + k for k in ('avp', 'h1vp', 'h2vp', 'avm', 'h1vm', 'h2vm', 'asp', 'hgradp',
                                          'fxi_n', 'fxi_p', 'h1xi_n', 'h1xi_p', 'hgradxi_n', 'hgradxi_p'))


def _meanfield(dens, force, grid, csolver, use_coulomb):
    wc = solve_poisson(dens.rho[1], csolver, grid) if use_coulomb else None
    return compute_skyrme_meanfield(dens, force, grid, coulomb_potential=wc, use_coulomb=use_coulomb)


def with_params(force, theta: dict):
    """
    The force with the parameters theta replaced and its derived couplings recomputed. Keys are Force fields
    (t0, Crho0, ...), 'fy.<name>' for a Fayans parameter (physics/edf.FY_KEYS) or 'extra.<name>' for a parameter
    of a user-defined functional (Force.extra).
    """
    fy_keys = [k for k in theta if k.startswith('fy.')]
    plain = {k: v for k, v in theta.items() if not k.startswith(('fy.', 'extra.'))}
    ex_keys = [k for k in theta if k.startswith('extra.')]
    if ex_keys:
        plain['extra'] = {**(force.extra or {}), **{k[6:]: theta[k] for k in ex_keys}}
    if fy_keys:
        fy = force.fy
        for k in fy_keys:
            fy = fy.at[FY_KEYS.index(k[3:])].set(theta[k])
        plain['fy'] = fy
    return dataclasses.replace(force, **plain)._calculate_derived_coefficients()


def total_energy(dens, force, grid, csolver, mass_number: int, use_coulomb: bool = True):
    """E[densities] (MeV), the pairing energy as int v_pair chi (gcg.two_basis_hfb)."""
    mf = _meanfield(dens, force, grid, csolver, use_coulomb)
    epair = grid.wxyz * jnp.sum(mf.v_pair * dens.chi, axis=(1, 2, 3))
    wc = solve_poisson(dens.rho[1], csolver, grid) if use_coulomb else None
    return compute_integrated_energy(dens, force, grid, coulomb_potential=wc, pairing_energy=epair,
                                     mass_number=mass_number, use_coulomb=use_coulomb).ehfint


def default_params(force) -> tuple:
    """The parameters derivatives are taken with respect to by default: those primary for this force."""
    if getattr(force, 'functional', 'skyrme') == 'fayans':
        return FAYANS_PARAMS + ('b4p', 't4')
    base = C_PARAMS_DEFAULT if getattr(force, 'parametrization', 'tx') == 'C' else SKYRME_PARAMS
    return base + (('CJ0', 'CJ1') if force.tensor else ())


def params_of(force, names: Optional[Sequence[str]] = None) -> dict:
    names = tuple(names or default_params(force))
    out = {}
    for k in names:
        if k.startswith('fy.'):
            out[k] = jnp.asarray(float(force.fy[FY_KEYS.index(k[3:])]))
        elif k.startswith('extra.'):
            out[k] = jnp.asarray(float(force.extra[k[6:]]))
        else:
            out[k] = jnp.asarray(float(getattr(force, k)))
    return out


def energy_gradient(state, force, grid, names: Optional[Sequence[str]] = None, use_coulomb: bool = True) -> dict:
    """dE/dtheta at a converged state: the explicit derivative at fixed densities (Hellmann-Feynman)."""
    theta = params_of(force, names)
    A = int(state.nneut + state.nprot)
    f = lambda th: total_energy(state.densities, with_params(force, th), grid, state.coulomb_solver, A, use_coulomb)
    return {k: float(v) for k, v in jax.grad(f)(theta).items()}


# ── response machinery ────────────────────────────────────────────────────────────────────────────

@jax.tree_util.register_dataclass
@dataclasses.dataclass
class _Block:
    """One isospin: HF basis, windowed basis, Bogoliubov eigenvectors, hole components."""
    phi: jax.Array      # (n, 2, nx, ny, nz)
    fphi: jax.Array     # f_k phi_k
    e: jax.Array        # (n,) Ritz energies
    lam: jax.Array      # () Fermi energy
    W: jax.Array        # (2n, 2n) eigenvectors of H, ascending energies
    E: jax.Array        # (2n,)
    v: jax.Array        # (m, 2, nx, ny, nz) hole components v_a of the kept quasiparticles
    Eq: jax.Array       # (m,) their energies E_a > 0


def _apply(psi, mf, q, grid):
    return jax.vmap(lambda p: apply_hamiltonian(p, mf, q, grid))(psi)


def _ip(a, b, w):
    return w * jnp.einsum('ksxyz,lsxyz->kl', jnp.conj(a), b)


def _Q(b, x, w):
    return x - jnp.einsum('km,ksxyz->msxyz', _ip(b.phi, x, w), b.phi)


def _Qc(b, w, wgt):
    """The projector for the w amplitudes: conj(w) orthogonal to the basis."""
    return jnp.conj(_Q(b, jnp.conj(w), wgt))


def _cplx(tree_re, tree_im):
    return jax.tree_util.tree_map(lambda a, c: a + 1j * c, tree_re, tree_im)


def _conj_tree(t):
    return jax.tree_util.tree_map(jnp.conj, t)


def _pair_matrix(field, fphi, w, conj):
    """A - A^T, A_kl = int field (f phi_k up)(f phi_l dn), conjugated orbitals if conj."""
    p = jnp.conj(fphi) if conj else fphi
    A = w * jnp.einsum('kxyz,lxyz->kl', p[:, 0], field[None] * p[:, 1])
    return A - A.T


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class _Data:
    blocks: tuple
    dens0: object
    g: jax.Array            # (2, nx, ny, nz) pair coupling d v_pair / d chi
    csolver: object


def _densities(data, Z, grid, tensor):
    """Complex density perturbation (Densities, zeta) of the response vector Z."""
    w = grid.wxyz
    pairs, iso_p = [], []
    dchi, dzeta = [], []
    for q, (b, (X, Y, x, wv)) in enumerate(zip(data.blocks, Z)):
        n = b.e.shape[0]
        B = jnp.zeros((2 * n, 2 * n), X.dtype).at[n:, :n].set(X).at[:n, n:].set(Y)
        R = b.W @ B @ jnp.conj(b.W.T)
        drho, dkap, dkap2 = R[:n, :n], R[:n, n:], -jnp.conj(R[n:, :n])
        x, wv = _Q(b, x, w), _Qc(b, wv, w)         # amplitudes live outside the basis
        # |alpha><beta| pairs: drho = sum_k |phi_k><sum_l drho_kl^* phi_l| inside the basis (linear in Z, no
        # eigen-decomposition: a GPU eigh of the tied spectrum of drho can return NaN), then the amplitudes
        pairs.append((jnp.concatenate([b.phi, x, b.v]),
                      jnp.concatenate([jnp.einsum('kl,lsxyz->ksxyz', jnp.conj(drho), b.phi), b.v, jnp.conj(wv)])))
        iso_p.append(jnp.full(n + 2 * x.shape[0], q))
        kt = lambda K: -jnp.einsum('kxyz,kxyz->xyz', b.fphi[:, 0], jnp.einsum('kl,lxyz->kxyz', K, b.fphi[:, 1]))
        k1, k2 = kt(dkap), kt(dkap2)
        dchi.append((k1 + jnp.conj(k2)) / 2)
        dzeta.append((k1 - jnp.conj(k2)) / 2j)

    def dens(psi, wocc, iso):
        z = jnp.zeros_like(wocc)
        return compute_densities(psi, wocc, z, jnp.ones_like(wocc), jnp.ones_like(wocc), iso, grid, chunk=256,
                                 tensor=tensor)

    cat = lambda L, i: jnp.concatenate([t[i] for t in L])
    iso_p = jnp.concatenate(iso_p)
    alpha, beta = cat(pairs, 0), cat(pairs, 1)
    total = None
    for k in range(4):   # |alpha><beta| = 1/4 sum_k i^k |alpha + i^k beta><alpha + i^k beta|
        d = dens(alpha + (1j ** k) * beta, jnp.ones(alpha.shape[0]), iso_p)
        total = (jax.tree_util.tree_map(lambda s: (1j ** k) / 4 * s, d) if total is None
                 else jax.tree_util.tree_map(lambda t, s: t + (1j ** k) / 4 * s, total, d))
    total = dataclasses.replace(total, chi=jnp.stack(dchi))
    return total, jnp.stack(dzeta)


def _fields(data, force, grid, ddens, dzeta, use_coulomb):
    """Complex mean-field perturbation and the two pair fields (upper-right, lower-left) of dH."""
    f = lambda d: _meanfield(d, force, grid, data.csolver, use_coulomb)
    re = jax.tree_util.tree_map(jnp.real, ddens)
    im = jax.tree_util.tree_map(jnp.imag, ddens)
    dmf = _cplx(jax.jvp(f, (data.dens0,), (re,))[1], jax.jvp(f, (data.dens0,), (im,))[1])
    return dmf, dmf.v_pair + 1j * data.g * dzeta, dmf.v_pair - 1j * data.g * dzeta


def _drive(data, grid, dmf, dpair, dpairb):
    """The [dH, R0] part of L: right-hand sides of all blocks for the field perturbation dH."""
    w = grid.wxyz
    out = []
    for q, b in enumerate(data.blocks):
        n = b.e.shape[0]
        dh = _ip(b.phi, _apply(b.phi, dmf, q, grid), w)
        dH = jnp.block([[dh, _pair_matrix(dpair[q], b.fphi, w, True)],
                        [-_pair_matrix(dpairb[q], b.fphi, w, False), -dh.T]])
        G = jnp.conj(b.W.T) @ dH @ b.W
        x = _Q(b, _apply(b.v, dmf, q, grid), w)
        wv = -jnp.conj(_Q(b, _apply(b.v, _conj_tree(dmf), q, grid), w))
        out.append((G[n:, :n], -G[:n, n:], x, wv))
    return tuple(out)


def _diag(data, Z, mf0, grid):
    """The [H0, dR] part of L."""
    w = grid.wxyz
    out = []
    for q, (b, (X, Y, x, wv)) in enumerate(zip(data.blocks, Z)):
        n = b.e.shape[0]
        Ep, Em = b.E[n:], b.E[:n]
        h = lambda y: _Q(b, _apply(_Q(b, y, w), mf0, q, grid) + (b.Eq - b.lam)[:, None, None, None, None] * y, w)
        out.append(((Ep[:, None] - Em[None, :]) * X, (Em[:, None] - Ep[None, :]) * Y,
                    h(x), -jnp.conj(h(jnp.conj(wv)))))
    return tuple(out)


def _gmres(A, b, M, x0=None, tol=1e-8, restart=40, maxiter=40, host=None):
    """
    Restarted GMRES with right preconditioning on pytrees, A and M applied as black boxes (the
    response operator is complex-linear but built from real-valued density pieces, so
    lax.custom_linear_solve cannot transpose it). Returns (x, relative residual). host: keep the Krylov
    basis in host memory (default: when one vector exceeds 256 MB).
    """
    bf, unravel = ravel_pytree(b)
    Af = lambda v: ravel_pytree(A(unravel(jnp.asarray(v))))[0]
    Mf = lambda v: ravel_pytree(M(unravel(jnp.asarray(v))))[0]
    # large problems (208Pb: ~1 GB a vector) keep the Krylov basis in host memory
    big = bf.size * bf.dtype.itemsize > 2 ** 28 if host is None else host
    keep = (lambda v: np.asarray(v)) if big else (lambda v: v)
    dot = lambda a, c: complex(np.vdot(a, c)) if isinstance(a, np.ndarray) else complex(jnp.vdot(a, c))
    x = jnp.zeros_like(bf) if x0 is None else ravel_pytree(x0)[0]
    bn = float(jnp.linalg.norm(bf))
    rel = 1.0
    for _ in range(maxiter):
        r = bf - Af(x)
        beta = float(jnp.linalg.norm(r))
        rel = beta / bn
        if rel < tol:
            break
        V, Zs = [keep(r / beta)], []
        H = np.zeros((restart + 1, restart), complex)
        for j in range(restart):
            z = Mf(V[j])
            Zs.append(keep(z))
            u = keep(Af(z))
            for i in range(j + 1):
                H[i, j] = dot(V[i], u)
                u = u - H[i, j] * V[i]
            H[j + 1, j] = float(np.linalg.norm(u))
            V.append(u / H[j + 1, j])
            e1 = np.zeros(j + 2, complex); e1[0] = beta
            y = np.linalg.lstsq(H[:j + 2, :j + 1], e1, rcond=None)[0]
            est = np.linalg.norm(H[:j + 2, :j + 1] @ y - e1) / bn
            if est < tol:
                break
        x = x + jnp.asarray(sum(complex(c) * zc for c, zc in zip(y, Zs)))
    rel = float(jnp.linalg.norm(bf - Af(x))) / bn
    return unravel(x), rel


@partial(jax.jit, static_argnames=('use_coulomb',))
def _apply_L(data, mf0, force, grid, Z, use_coulomb: bool):
    """L Z = [H0, dR] + [dH[dR], R0] (module level: one compilation per shape, shared by all Response objects)."""
    d, z = _densities(data, Z, grid, force.tensor)
    return _add(_diag(data, Z, mf0, grid), _drive(data, grid, *_fields(data, force, grid, d, z, use_coulomb)))


def _add(a, b, s=1.0):
    return jax.tree_util.tree_map(lambda u, v: u + s * v, a, b)


def _scale(a, s):
    return jax.tree_util.tree_map(lambda u: s * u, a)


# ── moving pairing window (static problems) ───────────────────────────────────────────────────────

@jax.tree_util.register_dataclass
@dataclasses.dataclass
class _Win:
    """One isospin's window F = sum_k f(e_k - lam) |phi_k><phi_k|, which the solver's fixed point moves with h."""
    gam: jax.Array      # (n, n) divided differences (f_k - f_l)/(e_k - e_l), f'(e_k) on ties (Daleckii-Krein)
    idx: jax.Array      # (nw,) windowed HF states, f_k > fmin: they get out-of-basis amplitudes y_k
    fw: jax.Array       # (nw,) their f_k
    kap: jax.Array      # (n, n) kappa0 = R0[:n, n:]
    c: jax.Array        # (nw, m) (D0 V_low)_ka of the kept quasiparticles: couples y_k to the x_a rows


def _drive_win(data, win, grid, dmf):
    """
    Window rows of a field perturbation dH: Q dH phi_k and conj(Q dH^+ phi_k) for the windowed states (the
    motion of F out of the basis) and -Gamma o <phi|dH|phi> (its motion inside the basis; lambda enters as
    the -dlam N field of the number constraint).
    """
    w = grid.wxyz
    out = []
    for q, (b, wn) in enumerate(zip(data.blocks, win)):
        hphi = _apply(b.phi, dmf, q, grid)
        dh = _ip(b.phi, hphi, w)
        yr = _Q(b, hphi[wn.idx], w)
        ycr = jnp.conj(_Q(b, _apply(b.phi[wn.idx], _conj_tree(dmf), q, grid), w))
        out.append((yr, ycr, -wn.gam * dh))
    return tuple(out)


def _diag_win(data, win, Yw, mf0, grid):
    """Q (h0 - e_k) Q on y_k (conjugated on yc) and the identity on m."""
    w = grid.wxyz
    out = []
    for q, (b, wn, (y, yc, m)) in enumerate(zip(data.blocks, win, Yw)):
        ew = b.e[wn.idx][:, None, None, None, None]
        h = lambda u: _Q(b, _apply(_Q(b, u, w), mf0, q, grid) - ew * _Q(b, u, w), w)
        out.append((h(y), jnp.conj(h(jnp.conj(yc))), m))
    return tuple(out)


def _win_terms(data, win, v_pair, Yw, grid):
    """
    What the window motion dF = sum_kl m_kl |k><l| + sum_k f_k (|y_k><k| + |k><y~_k|) does: the change of the
    pair density of F kappa0 F^T (dchi, dzeta) and the 2qp / orbital rows of dF Delta F^T + F Delta dF^T.
    Yw = (y, yc, m) with yc = conj(y~), so every term is complex-linear, as the rest of L.
    """
    w = grid.wxyz
    kt2 = lambda K, p1, p2: -jnp.einsum('kxyz,kxyz->xyz', p1[:, 0], jnp.einsum('kl,lxyz->kxyz', K, p2[:, 1]))

    def bil(field, p1, p2):
        A = w * jnp.einsum('kxyz,lxyz->kl', p1[:, 0], field[None] * p2[:, 1])
        return A - A.T

    dchi, dzeta, rows = [], [], []
    for q, (b, wn, (y, yc, m)) in enumerate(zip(data.blocks, win, Yw)):
        n = b.e.shape[0]
        fw = wn.fw[:, None, None, None, None]
        pd = jnp.einsum('kl,ksxyz->lsxyz', m, b.phi).at[wn.idx].add(fw * y)                   # dF phi_l
        cpd = jnp.einsum('lk,ksxyz->lsxyz', m, jnp.conj(b.phi)).at[wn.idx].add(fw * yc)       # conj(dF[dH^+] phi_l)
        psi, cpsi = b.fphi, jnp.conj(b.fphi)
        k1 = kt2(wn.kap, pd, psi) + kt2(wn.kap, psi, pd)
        ck2 = kt2(jnp.conj(wn.kap), cpd, cpsi) + kt2(jnp.conj(wn.kap), cpsi, cpd)
        dchi.append((k1 + ck2) / 2)
        dzeta.append((k1 - ck2) / 2j)
        v0 = v_pair[q]
        dH = (jnp.zeros((2 * n, 2 * n), complex)
              .at[:n, n:].set(bil(v0, cpd, cpsi) + bil(v0, cpsi, cpd))
              .at[n:, :n].set(-(bil(v0, pd, psi) + bil(v0, psi, pd))))
        G = jnp.conj(b.W.T) @ dH @ b.W
        xs = _Q(b, jnp.einsum('ka,ksxyz->asxyz', wn.c, y), w)
        ws = -jnp.conj(_Q(b, jnp.einsum('ka,ksxyz->asxyz', wn.c, jnp.conj(yc)), w))
        rows.append((G[n:, :n], -G[:n, n:], xs, ws))
    return jnp.stack(dchi), jnp.stack(dzeta), tuple(rows)


@partial(jax.jit, static_argnames=('use_coulomb',))
def _apply_Lw(data, win, mf0, force, grid, U, use_coulomb: bool):
    """L of the moving-window statics on U = (Z, Yw): ([H0, dR] + [dH, R0] + window rows, window equations)."""
    Z, Yw = U
    d, z = _densities(data, Z, grid, force.tensor)
    dchi, dzeta, zw = _win_terms(data, win, mf0.v_pair, Yw, grid)
    d = dataclasses.replace(d, chi=d.chi + dchi)
    dmf, dp, dpb = _fields(data, force, grid, d, z + dzeta, use_coulomb)
    LZ = _add(_add(_diag(data, Z, mf0, grid), _drive(data, grid, dmf, dp, dpb)), zw)
    LW = _add(_diag_win(data, win, Yw, mf0, grid), _drive_win(data, win, grid, dmf))
    return LZ, LW


@jax.jit
def _ext_w(data, win, grid, dmf, dp, dpb):
    """Right-hand side of an external field perturbation in the moving-window statics."""
    return _drive(data, grid, dmf, dp, dpb), _drive_win(data, win, grid, dmf)


@jax.jit
def _total_densities(data, win, mf0, U, grid, force):
    """Density perturbation of U = (Z, Yw), the window's pair-density motion included."""
    Z, Yw = U
    d, z = _densities(data, Z, grid, force.tensor)
    dchi, dzeta, _ = _win_terms(data, win, mf0.v_pair, Yw, grid)
    return dataclasses.replace(d, chi=d.chi + dchi), z + dzeta


class Response:
    """
    QRPA response of a converged two-basis state (module docstring).

        R = Response(state, force, grid)
        E, S = R.strength(field, energies, gamma=1.0)      # field: (2, nx, ny, nz) one-body operator
        d = R.derivatives(['t0', 't3'])                   # ground-state derivatives

    vmin: quasiparticles whose hole component has norm^2 below it get no continuum amplitudes.
    window: 'moving' (default) gives the static problems (derivatives, polarizability, sensitivities) of a
    paired state the solver's own fixed point, whose pairing window follows the HF energies and lambda;
    'frozen' holds the window fixed (the variational problem; differs from finite differences of the solver
    at the 1% level). Strength functions always use the frozen window. fmin: windowed states (f_k > fmin)
    get out-of-basis window amplitudes.
    constraint_potential: for a constrained state, the one-body potential (2, nx, ny, nz) its h contains
    (sum_i -(lambda_i + qcorr_i) Q_i); 'auto' rebuilds it from state.constraint_state as the frozen-multiplier
    augmented-Lagrangian solve (solver.run_constrained_scan) applies it, penalty shift included. Hold the
    constrained moments fixed in the static problems with derivatives(constraints=...).
    """

    def __init__(self, state, force, grid, use_coulomb: bool = True, vmin: float = 1e-8, tol: float = 1e-8,
                 window: str = 'moving', fmin: float = 1e-6, constraint_potential=None):
        if window not in ('moving', 'frozen'):
            raise ValueError(f"window must be 'moving' or 'frozen', not {window!r}")
        self.force, self.grid, self.use_coulomb, self.tol = force, grid, use_coulomb, tol
        self.state = state
        dens0, w = state.densities, grid.wxyz
        mf0 = _meanfield(dens0, force, grid, state.coulomb_solver, use_coulomb)
        if isinstance(constraint_potential, str) and constraint_potential == 'auto':
            cs = state.constraint_state
            cs = dataclasses.replace(cs, qcorr=penalty_qcorr(cs, dens0, grid))
            constraint_potential = compute_constraint_potential(cs, grid)
        if constraint_potential is not None:
            mf0 = dataclasses.replace(mf0, upot=mf0.upot + jnp.asarray(constraint_potential))
        ones = dataclasses.replace(jax.tree_util.tree_map(jnp.zeros_like, dens0), chi=jnp.ones_like(dens0.chi))
        g = jax.jvp(lambda d: _meanfield(d, force, grid, state.coulomb_solver, use_coulomb).v_pair,
                    (dens0,), (ones,))[1]
        npsi_n = int(np.sum(np.asarray(state.isospin) == 0))
        paired = force.ipair in (5, 6) and float(jnp.max(jnp.abs(dens0.chi))) > 0.0
        blocks, report, parity, wins, keeps = [], [], [], [], []
        for q, sl in ((0, slice(0, npsi_n)), (1, slice(npsi_n, None))):
            psi = state.psi[sl]
            hm = _ip(psi, _apply(psi, mf0, q, grid), w)
            e, C = jnp.linalg.eigh(0.5 * (hm + jnp.conj(hm.T)))
            phi = jnp.einsum('ki,ksxyz->isxyz', C, psi)
            n = e.shape[0]
            if paired:
                lam = state.pairing.eferm[q]
                cut = force.state_cutoff[q]
                f = 1.0 / (1.0 + jnp.exp((e - lam - cut) / (force.softcut_range * cut)))
            else:   # HF: occupations of the Ritz vectors from the state's, lam mid-gap
                filled = np.asarray(jnp.diag(jnp.conj(C).T @ jnp.diag(state.wocc[sl]) @ C).real) > 0.5
                eo, eu = np.asarray(e)[filled], np.asarray(e)[~filled]
                lam = jnp.asarray(0.5 * (eo.max() + eu.min()) if len(eu) else eo.max() + 1.0)
                f = jnp.zeros(n)
            fphi = f[:, None, None, None, None] * phi
            D = _pair_matrix(mf0.v_pair[q], fphi, w, True) if paired else jnp.zeros((n, n), complex)
            a = jnp.diag(e - lam).astype(complex)
            Ev, Wv = jnp.linalg.eigh(jnp.block([[a, D], [-jnp.conj(D), -jnp.conj(a)]]))
            # negative-energy columns as the exact partners (V*; U*) of the positive ones, so that the 2qp
            # amplitudes X_mu,nubar carry the fermion antisymmetry in mu, nu (see _physical)
            Wp, Ep = Wv[:, n:], Ev[n:]
            Wv = jnp.concatenate([jnp.concatenate([jnp.conj(Wp[n:]), jnp.conj(Wp[:n])]), Wp], axis=1)
            Ev = jnp.concatenate([-Ep, Ep])
            vs = jnp.einsum('ka,ksxyz->asxyz', Wv[:n, :n], phi)       # hole components of E < 0 columns
            keep = np.nonzero(np.asarray(jnp.sum(jnp.abs(Wv[:n, :n]) ** 2, axis=0)) > vmin)[0]
            keeps.append(keep)
            rho = Wv[:n, :n] @ jnp.conj(Wv[:n, :n].T)
            res = jnp.sqrt(w * jnp.sum(jnp.abs(_Q(_Block(phi, fphi, e, lam, Wv, Ev, phi[:0], e[:0]),
                                                  _apply(phi, mf0, q, grid), w)) ** 2, axis=(1, 2, 3, 4)))
            # parity in the quasiparticle basis (for symmetry-projected Lanczos chains)
            P = _ip(phi, phi[:, :, ::-1, ::-1, ::-1], w)
            Pqp = jnp.conj(Wv.T) @ jax.scipy.linalg.block_diag(P, jnp.conj(P)) @ Wv
            parity.append((Pqp, Pqp[:n, :n][np.ix_(keep, keep)]))
            blocks.append(_Block(phi=phi, fphi=fphi, e=e, lam=jnp.asarray(lam), W=Wv, E=Ev,
                                 v=vs[keep], Eq=-Ev[:n][keep]))
            report.append(dict(n=n, kept=len(keep), N=float(jnp.real(jnp.trace(rho))),
                               basis_residual=float(jnp.max(res)), lam=float(lam)))
            if paired and window == 'moving':
                s = force.softcut_range * cut
                fp = -f * (1.0 - f) / s
                de = e[:, None] - e[None, :]
                tie = jnp.abs(de) < 1e-6
                gam = jnp.where(tie, 0.5 * (fp[:, None] + fp[None, :]), (f[:, None] - f[None, :]) / jnp.where(tie, 1.0, de))
                idx = np.nonzero(np.asarray(f) > fmin)[0]
                kap = Wv[:n, :n] @ jnp.conj(Wv[n:, :n].T)
                wins.append(_Win(gam=gam, idx=jnp.asarray(idx), fw=f[idx], kap=kap, c=(D @ Wv[n:, :n])[idx][:, keep]))
                report[-1].update(windowed=len(idx), basis_top=float(e[-1] - lam))
        self.report, self.parity, self.keeps = report, parity, keeps
        self.win = tuple(wins) if wins else None
        self.mf0 = mf0
        self.data = _Data(blocks=tuple(blocks), dens0=dens0, g=g, csolver=state.coulomb_solver)
        self.h2m = jnp.asarray(force.h2m)

    def ground_densities(self):
        """Densities of R0 itself (rho and chi), to check the representation against the state's."""
        rho, chi = [], []
        for b in self.data.blocks:
            n = b.e.shape[0]
            r = b.W[:n, :n] @ jnp.conj(b.W[:n, :n].T)
            k = b.W[:n, :n] @ jnp.conj(b.W[n:, :n].T)                       # V* U^T
            m, C = jnp.linalg.eigh(0.5 * (r + jnp.conj(r.T)))
            orb = jnp.einsum('ki,ksxyz->isxyz', C, b.phi)
            rho.append(jnp.einsum('i,isxyz->xyz', m, jnp.abs(orb) ** 2))
            chi.append(-jnp.real(jnp.einsum('kxyz,kxyz->xyz', b.fphi[:, 0],
                                            jnp.einsum('kl,lxyz->kxyz', k, b.fphi[:, 1]))))
        return jnp.stack(rho), jnp.stack(chi)

    # -- operator pieces ------------------------------------------------------------------------
    def apply_L(self, data, mf0, force, Z):
        return _apply_L(data, mf0, force, self.grid, Z, self.use_coulomb)

    def _precond(self, omega, sigma=10.0):
        k2 = self.grid.k2
        out = []
        for q, b in enumerate(self.data.blocks):
            n = b.e.shape[0]
            Ep, Em = b.E[n:], b.E[:n]
            kin = self.h2m[q] * k2 + sigma
            px = lambda y, kin=kin: -jnp.fft.ifftn(jnp.fft.fftn(y, axes=(-3, -2, -1)) / kin, axes=(-3, -2, -1))
            out.append((Ep, Em, px, b))
        def M(Z):
            res = []
            w = self.grid.wxyz
            for (Ep, Em, px, b), (X, Y, x, wv) in zip(out, Z):
                res.append((X / (omega - (Ep[:, None] - Em[None, :])), Y / (omega - (Em[:, None] - Ep[None, :])),
                            _Q(b, px(_Q(b, x, w)), w), -_Qc(b, px(_Qc(b, wv, w)), w)))
            return tuple(res)
        return M

    # -- strength functions -----------------------------------------------------------------------
    def drive_field(self, field):
        """Right-hand side [F, R0] of a one-body local operator F (field: (2, nx, ny, nz), per isospin)."""
        zero = jax.tree_util.tree_map(jnp.zeros_like, self.mf0)
        dmf = dataclasses.replace(_cplx(zero, zero), upot=field.astype(complex))
        zp = jnp.zeros_like(field, dtype=complex)
        return _drive(self.data, self.grid, dmf, zp, zp)

    def solve(self, omega, rhs, x0=None, restart=40, maxiter=40):
        """Z with (omega - L) Z = rhs, by preconditioned GMRES."""
        A = lambda Z: _add(_scale(Z, omega), self.apply_L(self.data, self.mf0, self.force, Z), -1.0)
        return _gmres(A, rhs, self._precond(omega), x0=x0, tol=self.tol, restart=restart, maxiter=maxiter)

    def strength(self, field, energies, gamma: float = 1.0, verbose: bool = True):
        """S(E) = -Im Tr(F dR(E + i gamma/2)) / pi (fm^2L / MeV for an r^L operator)."""
        rhs = self.drive_field(field)
        S, Z, rel = [], None, []
        w = self.grid.wxyz
        for E in energies:
            Z, r = self.solve(E + 0.5j * gamma, rhs, x0=Z)
            d, _ = _densities(self.data, Z, self.grid, self.force.tensor)
            S.append(-float(jnp.imag(w * jnp.sum(field * d.rho))) / np.pi)
            rel.append(r)
            if verbose:
                print(f"  FAM E {float(E):7.3f}  S {S[-1]:12.5g}  residual {r:.1e}", flush=True)
        return np.asarray(energies), np.asarray(S), np.asarray(rel)

    def project_parity(self, Z, sign: int = 1):
        """(Z + sign * Pi Z) / 2 with Pi dR Pi^+ the parity transform of the response vector."""
        out = []
        for (Pqp, M), b, (X, Y, x, wv) in zip(self.parity, self.data.blocks, Z):
            n = b.e.shape[0]
            B = jnp.zeros((2 * n, 2 * n), X.dtype).at[n:, :n].set(X).at[:n, n:].set(Y)
            Bp = Pqp @ B @ jnp.conj(Pqp.T)
            flip = lambda y: y[:, :, ::-1, ::-1, ::-1]
            xp = jnp.einsum('ba,asxyz->bsxyz', jnp.conj(M), flip(x))
            wp = jnp.einsum('ba,asxyz->bsxyz', M, flip(wv))
            out.append(((X + sign * Bp[n:, :n]) / 2, (Y + sign * Bp[:n, n:]) / 2, (x + sign * xp) / 2, (wv + sign * wp) / 2))
        return tuple(out)

    def _ipK(self, A, B):
        """<A, eta B> with weight 1/2 on the doubled 2qp blocks: K = eta L is Hermitian in it."""
        w, s = self.grid.wxyz, 0j
        for (X, Y, x, wv), (X2, Y2, x2, w2) in zip(A, B):
            s += 0.5 * (jnp.vdot(X, X2) - jnp.vdot(Y, Y2)) + w * (jnp.vdot(x, x2) - jnp.vdot(wv, w2))
        return complex(s)

    def lanczos(self, field, steps: int = 1000, verbose: bool = True, restart=40, maxiter=40, parity: Optional[int] = None,
                constraints=()):
        """
        Lanczos chain of L for the operator F. On physical vectors (2qp blocks antisymmetric) K = eta L is
        Hermitian and positive for a stable state, so L is self-adjoint in <u, K v> and a three-term
        recursion with one L per step gives the whole strength function (Response.spectrum).

        Tr(F dR(Z)) = <eta b, Z> = <u0, Z>_K with b the drive of F and u0 = L^-1 b, so the chain starts
        from u0 (one omega = 0 solve, number-conserving: the spurious zero modes drop out) and
        S(omega) = <u0, (omega - L)^-1 L u0>_K is a Gauss quadrature over its Ritz pairs.
        parity = +1/-1 keeps every vector in that parity sector: rounding otherwise feeds the opposite
        sector, where the grid's slightly unstable translational mode (K < 0) ends the chain.
        Returns dict(alpha, beta, beta0, m_minus1): L ~ tridiag(alpha, beta).
        """
        proj = (lambda Z: self.project_parity(Z, parity)) if parity else (lambda Z: Z)
        L = lambda Z: self.apply_L(self.data, self.mf0, self.force, Z)
        op, M = self._static_system(constraints, frozen=True)
        (b, _), rel = _gmres(op, (self.drive_field(field), jnp.zeros(2 + len(constraints), complex)), M, tol=self.tol,
                             restart=restart, maxiter=maxiter)
        if verbose:
            print(f"  Lanczos start u0 = L^-1 [F, R0]: residual {rel:.1e}", flush=True)
        b = proj(b)
        Lb = L(b)
        beta0 = np.sqrt(self._ipK(b, Lb).real)
        v, Lv = _scale(b, 1 / beta0), _scale(Lb, 1 / beta0)
        v_prev, beta = None, 0.0
        alpha, betas = [], []
        for j in range(steps):
            a = self._ipK(Lv, Lv)                     # <v, L v>_K = <L v, eta L v>
            alpha.append(a.real)
            u = _add(Lv, v, -a.real)
            if v_prev is not None:
                u = _add(u, v_prev, -beta)
            u = proj(u)
            Lu = L(u)
            nb = self._ipK(u, Lu).real
            if nb <= 0:
                break
            beta = np.sqrt(nb)
            betas.append(beta)
            v_prev, v, Lv = v, _scale(u, 1 / beta), _scale(Lu, 1 / beta)
            if verbose and (j + 1) % 100 == 0:
                print(f"  Lanczos {j + 1:5d}  alpha {a.real:10.3f}  beta {beta:10.3f}  |Im alpha| {abs(a.imag):.1e}", flush=True)
        return dict(alpha=np.asarray(alpha), beta=np.asarray(betas[:len(alpha) - 1]), beta0=beta0,
                    m_minus1=beta0 ** 2 / 2)                # <u0, L u0>_K = 2 m_-1 (static polarizability)

    @staticmethod
    def spectrum(chain, energies, gamma: float = 1.0, steps: Optional[int] = None):
        """
        S(E) = -Im Tr(F (E + i gamma/2 - L)^-1 [F, R0]) / pi from a Lanczos chain (first `steps` steps).

        The chain starts from u0 = L^-1 b (Response.lanczos), so S(omega) = beta0^2 sum_k S_0k^2 theta_k /
        (omega - theta_k) over its Ritz pairs (theta_k, S_k): a Gauss quadrature, insensitive to the ghost
        copies that loss of orthogonality produces.
        """
        m = len(chain['alpha']) if steps is None else steps
        T = np.diag(chain['alpha'][:m]) + np.diag(chain['beta'][:m - 1], 1) + np.diag(chain['beta'][:m - 1], -1)
        th, S = np.linalg.eigh(T)
        c = chain['beta0'] ** 2 * S[0] ** 2 * th
        z = np.asarray(energies)[:, None] + 0.5j * gamma
        return -np.imag((c[None, :] / (z - th[None, :])).sum(1)) / np.pi

    # -- ground-state derivatives -----------------------------------------------------------------
    def derivatives(self, names: Optional[Sequence[str]] = None, restart=40, maxiter=40, verbose: bool = True,
                    constraints=()):
        """
        d(ground state)/d(theta) for each parameter: the density perturbation (Densities, real part) and
        derived observables (dE total and explicit, d r_n, d r_p, d Q20), by the omega = 0 response with
        N and Z held fixed, and the expectation values of the one-body fields in `constraints` (e.g. the constrained
        moments of a run_constrained_scan state, with constraint_potential="auto").
        """
        force, grid, w = self.force, self.grid, self.grid.wxyz
        theta = params_of(force, names)
        dens0, A = self.data.dens0, int(self.state.nneut + self.state.nprot)
        Ef = lambda d, th: total_energy(d, with_params(force, th), grid, self.data.csolver, A, self.use_coulomb)
        dE_expl = jax.grad(lambda th: Ef(dens0, th))(theta)
        X, Y, Zc = grid.get_meshgrid()
        r2, q20 = X ** 2 + Y ** 2 + Zc ** 2, 2 * Zc ** 2 - X ** 2 - Y ** 2
        rq = [float(jnp.sqrt(jnp.sum(r2 * dens0.rho[q]) / jnp.sum(dens0.rho[q]))) for q in (0, 1)]
        Nq = [float(w * jnp.sum(dens0.rho[q])) for q in (0, 1)]
        op, M = self._static_system(constraints)
        out = {}
        for k, b in self._param_drives(theta).items():
            U, rel = _gmres(op, self._rhs(b, 2 + len(constraints)), M, tol=self.tol, restart=restart, maxiter=maxiter)
            d, _ = self._dens_of(U)
            dr = jax.tree_util.tree_map(jnp.real, d)
            dE_impl = jax.jvp(lambda dd: Ef(dd, theta), (dens0,), (dr,))[1]
            out[k] = dict(dE=float(dE_expl[k] + dE_impl), dE_explicit=float(dE_expl[k]), dE_implicit=float(dE_impl),
                          d_rn=float(jnp.sum(r2 * dr.rho[0]) * w / (2 * rq[0] * Nq[0])),
                          d_rp=float(jnp.sum(r2 * dr.rho[1]) * w / (2 * rq[1] * Nq[1])),
                          d_Q20=float(w * jnp.sum(q20 * (dr.rho[0] + dr.rho[1]))),
                          dlam=[float(jnp.real(x)) for x in U[-1]], residual=rel,
                          imag=float(jnp.max(jnp.abs(jnp.imag(d.rho)))), densities=dr)
            if verbose:
                o = out[k]
                print(f"  d/d{k:6s} dE {o['dE']:+.6e} (explicit {o['dE_explicit']:+.6e})  d r_n {o['d_rn']:+.4e}  "
                      f"d r_p {o['d_rp']:+.4e}  dQ20 {o['d_Q20']:+.4e}  residual {o['residual']:.1e}", flush=True)
        return out

    def _param_drives(self, theta: dict) -> dict:
        """[dH/dtheta, R0] for each parameter: the mean-field derivative at fixed densities (jvp in the force)."""
        mf_of = lambda th: _meanfield(self.data.dens0, with_params(self.force, th), self.grid, self.data.csolver,
                                      self.use_coulomb)
        out = {}
        for k in theta:
            tan = {j: jnp.asarray(1.0 if j == k else 0.0) for j in theta}
            dmf = jax.jvp(mf_of, (theta,), (tan,))[1]
            dmf = _cplx(dmf, jax.tree_util.tree_map(jnp.zeros_like, dmf))
            out[k] = (_drive(self.data, self.grid, dmf, dmf.v_pair, dmf.v_pair) if self.win is None
                      else _ext_w(self.data, self.win, self.grid, dmf, dmf.v_pair, dmf.v_pair))
        return out

    def _static_system(self, constraints=(), frozen: bool = False):
        """
        (op, M) of the omega = 0 problem L Z - sum_c dlam_c d_c = rhs, <C_c>(Z) = 0: N and Z, and the
        one-body fields C_c in `constraints` ((2, nx, ny, nz) each, e.g. the c.m. coordinate, which removes
        the spurious translational mode), held fixed; d_c = drive of C_c and <C_c>(Z) = <eta d_c, Z>.
        """
        one = jnp.ones_like(self.grid.k2)
        fields = [jnp.stack([one * (p == q) for p in (0, 1)]) for q in (0, 1)] + [jnp.asarray(c, float) for c in constraints]
        if self.win is not None and not frozen:
            return self._static_system_w(fields)
        drives = [self.drive_field(c) for c in fields]

        def op(U):
            Z, dl = U
            LZ = self.apply_L(self.data, self.mf0, self.force, Z)
            for c, d in enumerate(drives):
                LZ = _add(LZ, d, -dl[c])
            return LZ, jnp.asarray([self._ipK(d, Z) for d in drives])

        Mz = self._precond(0.0)
        return op, (lambda U: (_scale(Mz(U[0]), -1.0), U[1]))

    # -- moving-window statics ---------------------------------------------------------------------
    def _ext_field(self, field):
        """(Z rows, window rows) of a one-body local field (2, nx, ny, nz)."""
        zero = jax.tree_util.tree_map(jnp.zeros_like, self.mf0)
        dmf = dataclasses.replace(_cplx(zero, zero), upot=jnp.asarray(field).astype(complex))
        zp = jnp.zeros(dmf.upot.shape, complex)
        return _ext_w(self.data, self.win, self.grid, dmf, zp, zp)

    def _precond_win(self, sigma=10.0):
        k2, w = self.grid.k2, self.grid.wxyz

        def M(Yw):
            out = []
            for q, (b, (y, yc, m)) in enumerate(zip(self.data.blocks, Yw)):
                kin = self.h2m[q] * k2 + sigma
                px = lambda u: jnp.fft.ifftn(jnp.fft.fftn(u, axes=(-3, -2, -1)) / kin, axes=(-3, -2, -1))
                out.append((_Q(b, px(_Q(b, y, w)), w), _Qc(b, px(_Qc(b, yc, w)), w), m))
            return tuple(out)
        return M

    def _static_system_w(self, fields):
        """_static_system with the moving window: unknowns (Z, Yw, dlam), Yw = per isospin (y, yc, m)."""
        ext = [self._ext_field(c) for c in fields]

        def op(U):
            Z, Yw, dl = U
            LZ, LW = _apply_Lw(self.data, self.win, self.mf0, self.force, self.grid, (Z, Yw), self.use_coulomb)
            for c, (ez, ew) in enumerate(ext):
                LZ, LW = _add(LZ, ez, -dl[c]), _add(LW, ew, -dl[c])
            return LZ, LW, jnp.asarray([self._ipK(ez, Z) for ez, _ in ext])

        Mz, Mw = self._precond(0.0), self._precond_win()
        return op, (lambda U: (_scale(Mz(U[0]), -1.0), Mw(U[1]), U[2]))

    def _rhs(self, b, nc, sign=-1.0):
        """Static right-hand side sign * b (b = Z rows, or (Z rows, window rows) with the moving window)."""
        if self.win is None:
            return _scale(b, sign), jnp.zeros(nc, complex)
        return _scale(b[0], sign), _scale(b[1], sign), jnp.zeros(nc, complex)

    def _dens_of(self, U):
        """Density perturbation (Densities, zeta) of a static solution U."""
        if self.win is None:
            return _densities(self.data, U[0], self.grid, self.force.tensor)
        return _total_densities(self.data, self.win, self.mf0, (U[0], U[1]), self.grid, self.force)

    def _field_drive(self, field):
        return self.drive_field(jnp.asarray(field, float)) if self.win is None else self._ext_field(field)

    def polarizability(self, field, constraints=(), restart=40, maxiter=60):
        """
        Static polarizability of the one-body operator F at fixed N, Z (and fixed <C_c>): returns
        (m_-1, residual) with m_-1 = sum_n |<n|F|0>|^2 / E_n = <u0, eta b> / 2, u0 = L^-1 [F, R0].
        For the isovector dipole D_z = (N/A) z_p - (Z/A) z_n, alpha_D = 2 e^2 m_-1 (= (8 pi/9) e^2 m_-1 of the
        B(E1) strength). With the moving window (no K-symmetry) m_-1 = Tr(F drho(u0)) / 2, the solver's
        -d<F>/d(2 eps) under h -> h + eps F.
        """
        op, M = self._static_system(constraints)
        b = self._field_drive(field)
        U, rel = _gmres(op, self._rhs(b, 2 + len(constraints), 1.0), M, tol=self.tol, restart=restart, maxiter=maxiter)
        if self.win is None:
            return self._ipK(U[0], b).real / 2, rel
        return float(self.grid.wxyz * jnp.sum(jnp.asarray(field) * jnp.real(self._dens_of(U)[0].rho))) / 2, rel

    def sensitivities(self, observables: dict, names: Optional[Sequence[str]] = None, restart=40, maxiter=40,
                      verbose: bool = True, constraints=()) -> dict:
        """
        Adjoint derivatives: d<O>/d(theta) for every parameter of every observable O = int o . rho, with one
        omega = 0 solve per observable. observables: name -> field o (2, nx, ny, nz) (per isospin).

        With K = eta L Hermitian, the adjoint of the number-conserving static problem is the same problem
        driven by o itself: its solution nu is the static response to the field o (for o = r^2 the monopole
        polarizability), and d<O>/dtheta = -<eta nu, dH/dtheta> for each theta at the cost of one drive.
        Returns {name: {theta: derivative}} plus 'polarizability' <O|nu> = d<O>/d(strength of o).

        The moving window breaks the K-symmetry (the solver's fixed point is not a stationary point), so a
        paired state with window='moving' gets the same numbers by forward solves: one per parameter and one
        per observable (for 'polarizability').
        """
        force, grid = self.force, self.grid
        theta = params_of(force, names)
        op, M = self._static_system(constraints)
        nc = 2 + len(constraints)
        drives = self._param_drives(theta)
        out = {}
        if self.win is not None:
            w = grid.wxyz
            rho = {}
            for k in theta:
                U, rel = _gmres(op, self._rhs(drives[k], nc), M, tol=self.tol, restart=restart, maxiter=maxiter)
                rho[k] = (jnp.real(self._dens_of(U)[0].rho), rel)
            for name, o in observables.items():
                o = jnp.asarray(o, float)
                U, rel = _gmres(op, self._rhs(self._ext_field(o), nc, 1.0), M, tol=self.tol, restart=restart,
                                maxiter=maxiter)
                res = {k: float(w * jnp.sum(o * rho[k][0])) for k in theta}
                res['polarizability'] = -float(w * jnp.sum(o * jnp.real(self._dens_of(U)[0].rho)))
                res['residual'] = max([rel] + [rho[k][1] for k in theta])
                out[name] = res
                if verbose:
                    print(f"  forward {name}: residual {res['residual']:.1e}  " + "  ".join(f"{k} {res[k]:+.5e}" for k in theta),
                          flush=True)
            return out
        for name, o in observables.items():
            g = self.drive_field(jnp.asarray(o, float))
            (nu, _), rel = _gmres(op, (g, jnp.zeros(nc, complex)), M, tol=self.tol, restart=restart, maxiter=maxiter)
            res = {k: -self._ipK(nu, drives[k]).real for k in theta}        # <eta nu, b> = <nu, eta b>
            res['polarizability'] = -self._ipK(nu, g).real
            res['residual'] = rel
            out[name] = res
            if verbose:
                print(f"  adjoint {name}: residual {rel:.1e}  " + "  ".join(f"{k} {res[k]:+.5e}" for k in theta), flush=True)
        return out
