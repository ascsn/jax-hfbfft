"""
Second-order (2n+1) sensitivities of static polarizabilities for Hartree-Fock states.

For a converged HF state, the energy is written as a function of Thouless coordinates Z of the
occupied orbitals: psi_k(Z) = sum_j (phi + QZ)_j (S^-1/2)_jk with Q the projector off the occupied
space and S = 1 + (QZ)^+ QZ (the series 1 - A/2 + 3A^2/8 for S^-1/2 is exact through third
derivatives at Z = 0, where an eigen-decomposition would be degenerate). Particle numbers are
conserved by construction. With Phi(Z; theta, eps) = E(Z; theta) - eps <F>(Z):

    P = 2 m_-1 = g^T H^-1 g,         g = grad_Z <F>,  H = grad_Z^2 E          (dielectric theorem)
    dP/dtheta = -d/dtheta [ mu . grad_Z E(0; theta) + u . H(0; theta) u ]      (2n+1 theorem)

with u = H^-1 g (the static response) and mu = H^-1 c, c = grad_Z [2 u . grad <F>(Z) - u . H(Z) u]
at Z = 0 (the second-order adjoint). One extra linear solve gives dP/dtheta for every coupling;
the third derivatives come from nested automatic differentiation. alpha_D = e^2 P for the
isovector dipole D_z = (N/A) z_p - (Z/A) z_n.
"""

import dataclasses
from functools import partial

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from jax_hfbfft.physics.densities import compute_densities
from jax_hfbfft.physics.response import total_energy, with_params, params_of

E2 = 1.4399784


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class _HF:
    phi: jax.Array       # (n, 2, nx, ny, nz) occupied orbitals
    iso: jax.Array       # (n,) isospin
    same: jax.Array      # (n, n) 1 where the isospins agree
    csolver: object


def _orbitals(d, z, w):
    Z = z[0] + 1j * z[1]
    ov = w * jnp.einsum('jsxyz,ksxyz->jk', jnp.conj(d.phi), Z) * d.same
    QZ = Z - jnp.einsum('jk,jsxyz->ksxyz', ov, d.phi)
    A = w * jnp.einsum('jsxyz,ksxyz->jk', jnp.conj(QZ), QZ) * d.same
    Sm = jnp.eye(A.shape[0]) - 0.5 * A + 0.375 * A @ A
    return jnp.einsum('jk,jsxyz->ksxyz', Sm, d.phi + QZ)


def _dens(d, z, grid, tensor):
    if not isinstance(d, _HF):          # other state models (second_order_hfb) carry their own densities
        return d.densities(z, grid, tensor)
    psi = _orbitals(d, z, grid.wxyz)
    one = jnp.ones(psi.shape[0])
    return compute_densities(psi, one, jnp.zeros_like(one), one, one, d.iso, grid, tensor=tensor)



def _gmres_real(A, b, M, tol=1e-9, restart=40, maxiter=40, host=None):
    """
    Restarted right-preconditioned GMRES in real arithmetic for the real symmetric (KKT) Hessian, on pytrees.
    Half the memory of the complex solver in physics/response.py, so the Krylov basis of 208Pb (~19 GB) stays on an
    A100/H200; host=True keeps it in host memory instead (default: when it would exceed 40% of the device memory).
    Returns (x, relative residual).
    """
    bf, unravel = ravel_pytree(b)
    Af = lambda v: ravel_pytree(A(unravel(jnp.asarray(v))))[0]
    Mf = lambda v: ravel_pytree(M(unravel(jnp.asarray(v))))[0]
    if host is None:
        try:
            limit = jax.devices()[0].memory_stats()['bytes_limit']
        except Exception:
            limit = 16e9
        host = bf.size * bf.dtype.itemsize * 2 * (restart + 1) > 0.4 * limit
    keep = (lambda v: np.asarray(v)) if host else (lambda v: v)
    x = jnp.zeros_like(bf)
    bn = float(jnp.linalg.norm(bf))
    for _ in range(maxiter):
        r = bf - Af(x)
        beta = float(jnp.linalg.norm(r))
        if beta / bn < tol:
            break
        V, Zs = [keep(r / beta)], []
        Hm = np.zeros((restart + 1, restart))
        for j in range(restart):
            z = Mf(V[j])
            Zs.append(keep(z))
            u = keep(Af(z))
            for i in range(j + 1):
                Hm[i, j] = float(np.dot(V[i], u)) if host else float(jnp.dot(V[i], u))
                u = u - Hm[i, j] * V[i]
            Hm[j + 1, j] = float(np.linalg.norm(u)) if host else float(jnp.linalg.norm(u))
            V.append(u / Hm[j + 1, j])
            e1 = np.zeros(j + 2); e1[0] = beta
            y = np.linalg.lstsq(Hm[:j + 2, :j + 1], e1, rcond=None)[0]
            if np.linalg.norm(Hm[:j + 2, :j + 1] @ y - e1) / bn < tol:
                break
        x = x + jnp.asarray(sum(float(c) * zc for c, zc in zip(y, Zs)))
    return unravel(x), float(jnp.linalg.norm(bf - Af(x))) / bn


def density(d, y, grid, tensor=False):
    """Densities of the state y = (z, lam) (for observables obs(d, y, theta))."""
    return _dens(d, y[0], grid, tensor)


def _dot(a, b):
    return sum(jnp.vdot(x, y).real for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)))


# The kernels take the orbitals, constraint fields, force and grid as arguments, so nothing large is baked into the
# compiled code as a constant (closing over them made XLA constant-fold 90 x 1e5 products and duplicate the orbitals
# per executable).
def _energy(d, force, grid, A, uc, z, theta):
    f = with_params(force, theta) if theta else force
    if not isinstance(d, _HF):
        return d.energy(f, grid, A, uc, z)
    return total_energy(_dens(d, z, grid, f.tensor), f, grid, d.csolver, A, uc)


def _cvals(d, C, grid, tensor, z):
    return grid.wxyz * jnp.einsum('cqxyz,qxyz->c', C, _dens(d, z, grid, tensor).rho)


def _lag(d, C, C0, force, grid, A, uc, y, theta):
    z, lam = y
    return _energy(d, force, grid, A, uc, z, theta) - jnp.dot(lam, _cvals(d, C, grid, force.tensor, z) - C0)


def _expect(d, grid, tensor, y, field):
    return grid.wxyz * jnp.sum(field * _dens(d, y[0], grid, tensor).rho)


def _ews(d, force, grid, A, uc, y, field, theta):
    """m_1 = 1/2 d^2 E / d eps^2 for the orbitals rotated by exp(i eps F) (Thouless theorem)."""
    f = with_params(force, theta) if theta else force
    psi = _orbitals(d, y[0], grid.wxyz)
    fq = field[d.iso][:, None]
    one = jnp.ones(psi.shape[0])

    def E(eps):
        dens = compute_densities(jnp.exp(1j * eps * fq) * psi, one, jnp.zeros_like(one), one, one, d.iso, grid, tensor=f.tensor)
        return total_energy(dens, f, grid, d.csolver, A, uc)
    return 0.5 * jax.grad(jax.grad(E))(0.0)


_S = ('A', 'uc')


@partial(jax.jit, static_argnames=_S)
def _grad_lag(d, C, C0, force, grid, y, theta, A, uc):
    return jax.grad(lambda t: _lag(d, C, C0, force, grid, A, uc, t, theta))(y)


@partial(jax.jit, static_argnames=_S)
def _hvp(d, C, C0, force, grid, y, v, theta, A, uc):
    return jax.jvp(jax.grad(lambda t: _lag(d, C, C0, force, grid, A, uc, t, theta)), (y,), (v,))[1]


@partial(jax.jit, static_argnames=('tensor',))
def _grad_expect(d, grid, y, field, tensor):
    return jax.grad(lambda t: _expect(d, grid, tensor, t, field))(y)


def _dd(fun, t, u):
    """u . Hess fun(t) u = d^2/ds^2 fun(t + s u) at s = 0, forward over forward: reverse-mode derivatives of it
    (third derivatives) then cost about what a Hessian-vector product does, not a reverse pass over one."""
    return jax.jvp(lambda tt: jax.jvp(fun, (tt,), (u,))[1], (t,), (u,))[1]


@partial(jax.jit, static_argnames=_S)
def _third(d, C, C0, force, grid, y, u, field, A, uc):
    """c = grad_y [2 u . grad <F>(y) - u . K(y) u]: the third derivative of the Lagrangian along u, u."""
    def q(t):
        Fu = jax.jvp(lambda s: _expect(d, grid, force.tensor, s, field), (t,), (u,))[1]
        return 2 * Fu - _dd(lambda s: _lag(d, C, C0, force, grid, A, uc, s, {}), t, u)
    return jax.grad(q)(y)


@partial(jax.jit, static_argnames=_S)
def _dP(d, C, C0, force, grid, y, u, mu, theta, A, uc):
    """grad_theta [mu . grad Lag(y; theta) + u . K(y; theta) u]."""
    def s(th):
        L = lambda t: _lag(d, C, C0, force, grid, A, uc, t, th)
        return jax.jvp(L, (y,), (mu,))[1] + _dd(L, y, u)
    return jax.grad(s)(theta)


@partial(jax.jit, static_argnames=('obs',))
def _obs_grad_y(obs, d, y, theta):
    return jax.grad(lambda t: obs(d, t, theta))(y)


@partial(jax.jit, static_argnames=('obs',) + _S)
def _obs_theta(obs, d, C, C0, force, grid, y, nu, theta, A, uc):
    s = lambda th: obs(d, y, th) - _dot(nu, jax.grad(lambda t: _lag(d, C, C0, force, grid, A, uc, t, th))(y))
    return jax.value_and_grad(s)(theta)


class HFHessian:
    """
    Thouless-coordinate energy of a converged HF state, its Hessian and the 2n+1 sensitivities.

    One-body fields in `constraints` (e.g. the c.m. coordinate, which removes the spurious translational
    mode) are held at their ground-state values with multipliers: the variables are y = (Z, lam) and the
    function is the Lagrangian Lag(y) = E(Z) - sum_c lam_c (<C_c>(Z) - <C_c>(0)), whose Hessian K (the KKT
    matrix) replaces H. The 2n+1 formulas hold for any stationary point, so they apply unchanged with y, K.
    """

    def __init__(self, state, force, grid, constraints=(), use_coulomb: bool = True, tol: float = 1e-9):
        occ = np.nonzero(np.asarray(state.wocc) > 0.5)[0]
        iso = jnp.asarray(np.asarray(state.isospin)[occ])
        self.d = _HF(phi=state.psi[occ], iso=iso, same=(iso[:, None] == iso[None, :]).astype(float),
                     csolver=state.coulomb_solver)
        self.force, self.grid, self.uc, self.tol = force, grid, use_coulomb, tol
        self.A = int(state.nneut + state.nprot)
        shape = (2,) + tuple(grid.k2.shape)
        self.C = (jnp.stack([jnp.asarray(c, float) for c in constraints]) if len(constraints)
                  else jnp.zeros((0,) + shape))
        z = jnp.zeros(self.d.phi.shape)
        self.zero = ((z, z), jnp.zeros(self.C.shape[0]))
        self.C0 = _cvals(self.d, self.C, grid, force.tensor, (z, z))
        self.h2m = jnp.asarray(force.h2m)
        self._args = (self.d, self.C, self.C0, self.force, self.grid)
        self._kw = dict(A=self.A, uc=self.uc)

    def grad_L(self, y, theta):
        return _grad_lag(*self._args, y, theta, **self._kw)

    def hvp(self, y, v, theta):
        return _hvp(*self._args, y, v, theta, **self._kw)

    def grad_F(self, y, field):
        return _grad_expect(self.d, self.grid, y, field, tensor=self.force.tensor)

    def gradient_norm(self):
        (gr, gi), _ = self.grad_L(self.zero, {})
        return float(jnp.sqrt(jnp.sum(gr ** 2) + jnp.sum(gi ** 2)))

    def _project(self, v):
        """Remove components of Z along the occupied orbitals (redundant directions)."""
        (zr, zi), lam = v
        w = self.grid.wxyz
        Z = zr + 1j * zi
        ov = w * jnp.einsum('jsxyz,ksxyz->jk', jnp.conj(self.d.phi), Z) * self.d.same
        Z = Z - jnp.einsum('jk,jsxyz->ksxyz', ov, self.d.phi)
        return ((jnp.real(Z), jnp.imag(Z)), lam)

    def _precond(self, sigma=10.0):
        kin = self.h2m[self.d.iso][:, None, None, None, None] * self.grid.k2[None, None] + sigma
        inv = lambda x: jnp.real(jnp.fft.ifftn(jnp.fft.fftn(x, axes=(-3, -2, -1)) / (4 * kin), axes=(-3, -2, -1)))
        return lambda v: self._project(((inv(v[0][0]), inv(v[0][1])), v[1]))

    def solve(self, rhs, restart=40, maxiter=40, host=None):
        """y with K y = rhs (KKT Hessian of the Lagrangian at the ground state), by real GMRES."""
        A = lambda v: self._project(self.hvp(self.zero, self._project(v), {}))
        return _gmres_real(A, self._project(rhs), self._precond(), tol=self.tol, restart=restart, maxiter=maxiter, host=host)

    # -- the polarizability and its gradient ------------------------------------------------------
    _dot = staticmethod(_dot)

    def polarizability(self, field, restart=40, maxiter=40):
        """P = 2 m_-1 = g^T K^-1 g, the response u = K^-1 g and the residual."""
        g = self.grad_F(self.zero, jnp.asarray(field, float))
        u, rel = self.solve(g, restart, maxiter)
        return float(_dot(g, u)), u, rel

    def polarizability_gradient(self, field, names=None, restart=40, maxiter=40, verbose=True):
        """
        dP/dtheta for every coupling by the second-order adjoint (2n+1 theorem): P is the stationary value of
        2 g(y0).u - u.K(y0)u over u, so dP/dtheta = -d/dtheta [mu.grad Lag(y0; theta) + u.K(y0; theta)u] with
        mu = K^-1 c, c = grad_y [2 u.grad <F>(y) - u.K(y)u] at y0 (the third derivative along u, u).
        """
        field = jnp.asarray(field, float)
        theta = params_of(self.force, names)
        P, u, rel_u = self.polarizability(field, restart, maxiter)
        c = _third(*self._args, self.zero, u, field, **self._kw)
        mu, rel_mu = self.solve(c, restart, maxiter)
        dP = _dP(*self._args, self.zero, u, mu, theta, **self._kw)
        out = {k: -float(v) for k, v in dP.items()}
        if verbose:
            print(f"  P = 2 m_-1 = {P:.10f} (response residual {rel_u:.1e}, adjoint residual {rel_mu:.1e})", flush=True)
        return P, out, dict(rel_u=rel_u, rel_mu=rel_mu)

    # -- first-order adjoint for any observable, and the energy-weighted sum rule -----------------
    def observable_gradient(self, obs, names=None, restart=40, maxiter=40):
        """
        O = obs(d, y, theta) at the ground state and dO/dtheta for every coupling: dO/dtheta = d_theta O - nu .
        d_theta grad Lag with K nu = grad_y O (one solve). obs gets the orbital data d explicitly (use density(d, y,
        grid)) so that nothing large is closed over.
        """
        theta = params_of(self.force, names)
        nu, rel = self.solve(_obs_grad_y(obs, self.d, self.zero, theta), restart, maxiter)
        val, gr = _obs_theta(obs, *self._args, self.zero, nu, theta, **self._kw)
        return float(val), {k: float(v) for k, v in gr.items()}, rel

    def constrained_energy(self, field, names=None, restart=40, maxiter=40, verbose=True, m1_weight=None):
        """
        E_-1 = sqrt(m_1 / m_-1) for the one-body field F and its gradient in every coupling:
        m_-1 = P/2 by the second-order adjoint, m_1 by the first-order adjoint of the energy-weighted sum.

        m1_weight (2, nx, ny, nz), if given, is the sum-rule density of a Galilean-invariant functional,
        m_1 = sum_q int (hbar^2/2m_q) |grad f_q|^2 rho_q: then m_1 is a linear observable. Use it for fields that grow
        towards the box edge (r^2): the autodiff route differentiates exp(i eps F) psi spectrally, and F^2 psi is not
        periodic on the box (1e-3 error in m_1 for r^2 in 16O on a 25^3 box).
        """
        field = jnp.asarray(field, float)
        P, dP, info = self.polarizability_gradient(field, names, restart, maxiter, verbose=False)
        force, grid, A, uc = self.force, self.grid, self.A, self.uc
        if m1_weight is None:
            m1_obs = lambda d, y, th: _ews(d, force, grid, A, uc, y, field, th)
        else:
            wgt = jnp.asarray(m1_weight, float)
            m1_obs = lambda d, y, th: grid.wxyz * jnp.sum(wgt * _dens(d, y[0], grid, force.tensor).rho)
        m1, dm1, rel1 = self.observable_gradient(m1_obs, names, restart, maxiter)
        mm1, dmm1 = P / 2, {k: v / 2 for k, v in dP.items()}
        Ec = float(np.sqrt(m1 / mm1))
        dEc = {k: 0.5 * Ec * (dm1[k] / m1 - dmm1[k] / mm1) for k in dP}
        if verbose:
            print(f"  m1 {m1:.8e}  m-1 {mm1:.8e}  E_-1 {Ec:.6f} MeV (residuals {info['rel_u']:.1e} {info['rel_mu']:.1e} {rel1:.1e})", flush=True)
        return dict(Ec=Ec, m1=m1, m_1=mm1, dEc=dEc, dm1=dm1, dm_1=dmm1, residuals=dict(u=info['rel_u'], mu=info['rel_mu'], nu=rel1))
