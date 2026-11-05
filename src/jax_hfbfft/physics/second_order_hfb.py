"""
Second-order (2n+1) sensitivities of static polarizabilities for paired (two-basis HFB) states.

The HF version (second_order.HFHessian) writes the energy in Thouless coordinates of the occupied orbitals.
Here the state is a quasiparticle vacuum: its occupied generalized orbitals are the negative-energy
columns A_b = (v_b; w_b) of the Bogoliubov problem in the HF basis {phi_k} (physics/response.py), plus all
lower-component functions (0; f) outside the basis. Near the ground state every vacuum is the graph of an
antisymmetric Thouless map Z from the occupied to the empty space,

    Z = sum_ab beta_ab |A~_a><A_b| + sum_b |(x_b; 0)><A_b| - sum_b |A~_b><(0; x_b*)|,

beta antisymmetric (two-quasiparticle amplitudes inside the basis), x_b orthogonal to the basis (the
continuum amplitudes of the hole components v_b, the third term their partner required by the
Bogoliubov symmetry), A~_a the empty partner of A_a. The generalized density is the projector on the graph,

    R = P0 + Z + Z^+ + (Z Z^+ - Z^+ Z) - (Z Z^+ Z + Z^+ Z Z^+) + O(Z^4),

exact through third derivatives at Z = 0 (the same role as the S^-1/2 series of the HF code). Every
operator is a finite matrix over the frame {A_b, A~_a, (x_b; 0), (0; x_b*)}, whose Gram matrix is
diag(1, 1, <x|x>, <x|x>*), so rho and the pairing tensor follow from small matrix products and one
density evaluation of the basis plus the x_b. The pairing window F = sum_k f_k |phi_k><phi_k| enters the
pair density chi = -(F kappa F^T)(r up, r down) as a fixed operator: E(Z) is the frozen-window HFB energy,
whose Hessian is the QRPA matrix K = eta L of response.Response(window='frozen').

Particle numbers are not conserved by the vacua, so N and Z are held by multipliers (KKT Lagrangian with
the Fermi energies as ground-state multipliers), together with any one-body constraints.

The two-basis solver moves its window with the HF energies (response.py, window='moving'), so its fixed
point is the stationary point of E(.; F(theta)) with F(theta) itself theta-dependent. The QRPA
polarizability of the solver's state, P(theta) = 2 m_-1 = g^T K(theta)^-1 g, then has

    dP/dtheta = c . ds/dtheta - u . (d_theta K) u - d_t [u . K(F + t dF/dtheta) u],

u = K^-1 g, c = grad_y [2 u.grad <F>(y) - u.K(y) u] (the third derivative along u, u), ds/dtheta and
dF/dtheta the moving-window ground-state derivative (Response.derivatives' solve). With the window frozen
(a variational solver) ds/dtheta = -K^-1 d_theta grad Lag and the 2n+1 adjoint mu = K^-1 c gives every
coupling from one solve (HFBHessian.polarizability_gradient(moving=None)); with the moving window each
coupling takes one forward solve of the moving-window statics.
"""

import dataclasses
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from jax_hfbfft.physics.densities import compute_densities
from jax_hfbfft.physics.response import total_energy, params_of, _ext_w, _gmres
from jax_hfbfft.physics import second_order as so
from jax_hfbfft.physics.second_order import (HFHessian, _gmres_real, _third, _dP, _dd, _dot, _lag, _cvals, _S)

# orbitals per density chunk (memory of the nested derivatives scales with it, not with the number of orbitals)
DENSITY_CHUNK = 64


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class _Q:
    """One isospin of the vacuum: basis, Bogoliubov columns, kept quasiparticles, frozen window and its motion."""
    phi: jax.Array      # (n, 2, nx, ny, nz) HF basis
    psi: jax.Array      # (n, ...) f_k phi_k
    W: jax.Array        # (2n, 2n) columns: n negative-energy (occupied), then their partners
    keep: jax.Array     # (m,) negative columns with continuum amplitudes x_b
    pd: jax.Array       # (n, ...) window motion dF phi_l (zero: frozen)
    yt: jax.Array       # (nw, ...) out-of-basis bra amplitudes y~_k of dF
    idx: jax.Array      # (nw,) their states
    fw: jax.Array       # (nw,) their f_k


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class _HFB:
    qs: tuple
    csolver: object
    t: jax.Array        # () window-motion parameter: F = F0 + t dF

    def _frame(self, q, z, grid):
        """rho~ and kappa~ over the functions [phi, x] and those functions, for isospin q."""
        d, w = self.qs[q], grid.wxyz
        (br, bi), (xr, xi) = z
        n, m = d.phi.shape[0], d.keep.shape[0]
        B = br + 1j * bi
        beta = 0.5 * (B - B.T)
        x = xr + 1j * xi
        x = x - jnp.einsum('km,ksxyz->msxyz', w * jnp.einsum('ksxyz,msxyz->km', jnp.conj(d.phi), x), d.phi)
        G = w * jnp.einsum('asxyz,bsxyz->ab', jnp.conj(x), x)
        N = 2 * n + 2 * m
        ar = jnp.arange(m)
        Zf = (jnp.zeros((N, N), complex).at[n:2 * n, :n].set(beta)
              .at[2 * n + ar, d.keep].set(1.0).at[n + d.keep, 2 * n + m + ar].set(-1.0))
        Gf = (jnp.zeros((N, N), complex).at[jnp.arange(2 * n), jnp.arange(2 * n)].set(1.0)
              .at[2 * n:2 * n + m, 2 * n:2 * n + m].set(G).at[2 * n + m:, 2 * n + m:].set(jnp.conj(G)))
        Zd = jnp.conj(Zf.T)
        ZZd, ZdZ = Zf @ Gf @ Zd, Zd @ Gf @ Zf
        P0 = jnp.zeros((N, N), complex).at[jnp.arange(n), jnp.arange(n)].set(1.0)
        R = P0 + Zf + Zd + ZZd - ZdZ - (ZZd @ Gf @ Zf + ZdZ @ Gf @ Zd)
        U = jnp.zeros((n + m, N), complex).at[:n, :2 * n].set(d.W[:n]).at[n + ar, 2 * n + ar].set(1.0)
        L = jnp.zeros((n + m, N), complex).at[:n, :2 * n].set(jnp.conj(d.W[n:])).at[n + ar, 2 * n + m + ar].set(1.0)
        return U @ R @ jnp.conj(U.T), U @ R @ L.T, jnp.concatenate([d.phi, x]), x

    def densities(self, z, grid, tensor, with_pair=False):
        """Densities of the vacuum z = per-isospin ((Re B, Im B), (Re x, Im x)); chi = Re of the windowed pair density."""
        w = grid.wxyz
        orbs, partners, iso, chi, zeta = [], [], [], [], []
        for q in (0, 1):
            d = self.qs[q]
            rho, kap, b, x = self._frame(q, z[q], grid)
            orbs.append(b)
            partners.append(jnp.einsum('pq,qsxyz->psxyz', jnp.conj(rho), b))       # |b_p><G_p| = rho
            iso.append(jnp.full(b.shape[0], q))
            # F b: phi_l -> f_l phi_l + t dF phi_l, x_b -> t sum_k f_k phi_k <y~_k|x_b>
            ov = w * jnp.einsum('ksxyz,bsxyz->kb', jnp.conj(d.yt), x)
            Fx = self.t * jnp.einsum('kb,ksxyz->bsxyz', d.fw[:, None] * ov, d.phi[d.idx])
            Fb = jnp.concatenate([d.psi + self.t * d.pd, Fx])
            k = -jnp.einsum('pxyz,pxyz->xyz', Fb[:, 0], jnp.einsum('pq,qxyz->pxyz', kap, Fb[:, 1]))
            chi.append(jnp.real(k))
            zeta.append(jnp.imag(k))
        a, c, iso = jnp.concatenate(orbs), jnp.concatenate(partners), jnp.concatenate(iso)
        # rho = sum_p |a_p><c_p| is Hermitian, so it is 1/4 sum_p (|a_p + c_p><a_p + c_p| - |a_p - c_p><a_p - c_p|):
        # one density evaluation of 2(n + m) orbitals with weights +-1/4
        q = 0.25 * jnp.ones(a.shape[0])
        two = jnp.ones(2 * a.shape[0])
        dens = compute_densities(jnp.concatenate([a + c, a - c]), jnp.concatenate([q, -q]), 0 * two, two, two,
                                 jnp.concatenate([iso, iso]), grid, chunk=DENSITY_CHUNK, tensor=tensor)
        dens = dataclasses.replace(dens, chi=jnp.stack(chi))
        return (dens, jnp.stack(zeta)) if with_pair else dens

    def energy(self, f, grid, A, uc, z):
        """E(rho, chi) + [E(rho, zeta) - E(rho, 0)]: |kappa|^2 = chi^2 + zeta^2 in a functional quadratic in chi."""
        dens, zeta = self.densities(z, grid, f.tensor, with_pair=True)
        E = lambda ch: total_energy(dataclasses.replace(dens, chi=ch), f, grid, self.csolver, A, uc)
        return E(dens.chi) + E(zeta) - E(jnp.zeros_like(zeta))


@partial(jax.jit, static_argnames=_S)
def _window_term(d, C, C0, force, grid, y, u, A, uc):
    """d_t [u . K(y; F0 + t dF) u] at t = 0."""
    def s(t):
        dt = dataclasses.replace(d, t=t)
        return _dd(lambda v: _lag(dt, C, C0, force, grid, A, uc, v, {}), y, u)
    return jax.jvp(s, (jnp.asarray(0.0),), (jnp.asarray(1.0),))[1]      # all forward mode


class HFBHessian(HFHessian):
    """
    Thouless-coordinate (frozen-window) energy of a converged paired two-basis state and the 2n+1
    sensitivities of its QRPA polarizabilities (module docstring). Built from a response.Response of the
    state (its basis, Bogoliubov columns and kept quasiparticles); pass a Response with window='moving' to
    polarizability_gradient for the derivatives of the solver's own (moving-window) states.

    Variables y = (z, lam): z per isospin ((Re B, Im B), (Re x, Im x)), lam the multipliers of N, Z and the
    one-body constraints, at the ground state the Fermi energies and 0.
    """

    def __init__(self, response, constraints=(), tol: float = 1e-9):
        R = response
        state, force, grid = R.state, R.force, R.grid
        qs, zs = [], []
        for b, keep in zip(R.data.blocks, R.keeps):
            n, m = b.e.shape[0], len(keep)
            zero_w = jnp.zeros((0,) + b.phi.shape[1:], complex)
            qs.append(_Q(phi=b.phi, psi=b.fphi, W=b.W, keep=jnp.asarray(keep), pd=jnp.zeros_like(b.phi),
                         yt=zero_w, idx=jnp.zeros(0, int), fw=jnp.zeros(0)))
            zb, zx = jnp.zeros((n, n)), jnp.zeros((m,) + b.phi.shape[1:])
            zs.append(((zb, zb), (zx, zx)))
        self.response = R
        self.d = _HFB(qs=tuple(qs), csolver=state.coulomb_solver, t=jnp.asarray(0.0))
        self.force, self.grid, self.uc, self.tol = force, grid, R.use_coulomb, tol
        self.A = int(state.nneut + state.nprot)
        one = jnp.ones_like(grid.k2)
        fields = [jnp.stack([one * (p == q) for p in (0, 1)]) for q in (0, 1)] + [jnp.asarray(c, float) for c in constraints]
        self.C = jnp.stack(fields)
        lam0 = jnp.concatenate([jnp.stack([b.lam for b in R.data.blocks]).astype(float), jnp.zeros(len(constraints))])
        self.zero = (tuple(zs), lam0)
        self.C0 = _cvals(self.d, self.C, grid, force.tensor, self.zero[0])
        self.h2m = jnp.asarray(force.h2m)
        self._args = (self.d, self.C, self.C0, self.force, self.grid)
        self._kw = dict(A=self.A, uc=self.uc)

    def _project(self, v):
        """Antisymmetrize B and keep x orthogonal to the basis (the redundant directions)."""
        z, lam = v
        w = self.grid.wxyz
        out = []
        for d, ((br, bi), (xr, xi)) in zip(self.d.qs, z):
            x = xr + 1j * xi
            x = x - jnp.einsum('km,ksxyz->msxyz', w * jnp.einsum('ksxyz,msxyz->km', jnp.conj(d.phi), x), d.phi)
            out.append((((br - br.T) / 2, (bi - bi.T) / 2), (jnp.real(x), jnp.imag(x))))
        return tuple(out), lam

    def _precond(self, sigma=10.0):
        k2 = self.grid.k2
        diag = []
        for q, b in enumerate(self.response.data.blocks):
            n = b.e.shape[0]
            Ep = b.E[n:]
            diag.append((4 * (Ep[:, None] + Ep[None, :]), 4 * (self.h2m[q] * k2 + sigma)))

        def M(v):
            z, lam = v
            out = []
            for (db, kin), ((br, bi), (xr, xi)) in zip(diag, z):
                inv = lambda a: jnp.real(jnp.fft.ifftn(jnp.fft.fftn(a, axes=(-3, -2, -1)) / kin, axes=(-3, -2, -1)))
                out.append(((br / db, bi / db), (inv(xr), inv(xi))))
            return self._project((tuple(out), lam))
        return M

    def ground_check(self):
        """(max |rho - state rho|, max |chi - state chi|, |grad Lag|) at the ground state."""
        dens = self.d.densities(self.zero[0], self.grid, self.force.tensor)
        s = self.response.state.densities
        (z, lam) = self.grad_L(self.zero, {})
        gn = float(jnp.sqrt(sum(jnp.sum(a ** 2) for a in jax.tree_util.tree_leaves(z))))
        return float(jnp.max(jnp.abs(dens.rho - s.rho))), float(jnp.max(jnp.abs(dens.chi - s.chi))), gn

    def gradient_norm(self):
        return self.ground_check()[2]

    # -- the moving-window ground-state derivative in these coordinates ------------------------------
    def _moving_direction(self, U, moving):
        """(ds in y, window motion per isospin) of a moving-window static solution U = (Z, Yw, dlam)."""
        Z, Yw, dl = U
        zs, wins = [], []
        for (X, Y, x, wv), (y, yc, mm), wn, b in zip(Z, Yw, moving.win, moving.data.blocks):
            zs.append(((jnp.real(X), jnp.imag(X)), (jnp.real(x), jnp.imag(x))))
            fw = wn.fw[:, None, None, None, None]
            pd = jnp.einsum('kl,ksxyz->lsxyz', mm, b.phi).at[wn.idx].add(fw * y)
            wins.append((pd, jnp.conj(yc), wn.idx, wn.fw))
        lam = jnp.concatenate([jnp.real(jnp.asarray(dl[:2])), jnp.real(jnp.asarray(dl[2:]))]) if len(dl) else jnp.zeros(0)
        return (tuple(zs), lam), wins

    def polarizability_gradient(self, field, names=None, restart=40, maxiter=40, verbose=True, moving=None):
        """
        P = 2 m_-1 (frozen-window QRPA) and dP/dtheta. moving=None: the variational 2n+1 adjoint (exact for a solver
        whose window is frozen). moving = Response(state, window='moving') with the same constraints: the derivative of
        the QRPA polarizability of the solver's own states, one moving-window forward solve per coupling.
        """
        if moving is None:
            return super().polarizability_gradient(field, names, restart, maxiter, verbose)
        field = jnp.asarray(field, float)
        theta = params_of(self.force, names)
        P, u, rel_u = self.polarizability(field, restart, maxiter)
        c = _third(*self._args, self.zero, u, field, **self._kw)
        dP_th = _dP(*self._args, self.zero, u, jax.tree_util.tree_map(jnp.zeros_like, u), theta, **self._kw)
        nc = self.C.shape[0]
        if nc > 2:
            raise NotImplementedError("moving-window gradient with extra constraints: pass them to the Response too")
        op, M = moving._static_system()
        drives = moving._param_drives(theta)
        out, info = {}, dict(rel_u=rel_u, rel_s={})
        for k in theta:
            U, rel = _gmres(op, moving._rhs(drives[k], 2), M, tol=moving.tol, restart=restart, maxiter=maxiter)
            ds, wins = self._moving_direction(U, moving)
            # the multipliers of the Lagrangian are lam = Fermi energies: d lam / d theta = dlam of the solve
            cs = float(_dot(c, ds))
            dw = dataclasses.replace(self.d, qs=tuple(dataclasses.replace(qd, pd=pd, yt=yt, idx=idx, fw=fw)
                                                    for qd, (pd, yt, idx, fw) in zip(self.d.qs, wins)))
            wt = float(_window_term(dw, self.C, self.C0, self.force, self.grid, self.zero, u, **self._kw))
            out[k] = cs - float(dP_th[k]) - wt
            info['rel_s'][k] = rel
            if verbose:
                print(f"  dP/d{k}: state {cs:+.6e}  explicit {-float(dP_th[k]):+.6e}  window {-wt:+.6e}  "
                      f"total {out[k]:+.6e} (residual {rel:.1e})", flush=True)
        if verbose:
            print(f"  P = 2 m_-1 = {P:.10f} (response residual {rel_u:.1e})", flush=True)
        return P, out, info

    def constrained_energy(self, field, names=None, restart=40, maxiter=40, verbose=True, m1_weight=None):
        """HFHessian.constrained_energy (frozen window); m1_weight is required (the autodiff m_1 route is HF only)."""
        if m1_weight is None:
            raise ValueError("paired constrained_energy needs m1_weight")
        return super().constrained_energy(field, names, restart, maxiter, verbose, m1_weight)
