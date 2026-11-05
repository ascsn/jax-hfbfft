"""Linear response of two-basis states (physics/response.py): structure, derivatives, adjoint, Lanczos."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_hfbfft import Force
from jax_hfbfft.core.grid import Grid
from jax_hfbfft.physics import response as rs
from jax_hfbfft.physics.solver import run_hfb, apply_cm_correction, two_basis_config


@pytest.fixture(scope="module")
def o16():
    g = Grid.create_cubic(n=15, d=1.0)
    f = apply_cm_correction(Force.from_name("SLy4").with_pairing(ipair=0), 16)
    cfg = two_basis_config(etol=1e-7, max_iterations=2000)
    st = run_hfb(g, f, nucleus_z=8, nucleus_n=8, npsi_n=10, config=cfg)
    return g, f, cfg, st, rs.Response(st, f, g)


def _r2(g):
    X, Y, Z = g.get_meshgrid()
    return X ** 2 + Y ** 2 + Z ** 2


def test_ground_state_representation(o16):
    """The Bogoliubov eigenvectors of the HF basis reproduce the state's density and particle numbers."""
    g, f, cfg, st, R = o16
    rho, _ = R.ground_densities()
    np.testing.assert_allclose(rho, st.densities.rho, atol=1e-6)
    assert [r['N'] for r in R.report] == pytest.approx([8.0, 8.0], abs=1e-8)


def test_K_hermitian_on_physical_vectors(o16):
    """eta L is Hermitian (and positive) in the half-weighted inner product on antisymmetric 2qp blocks."""
    g, f, cfg, st, R = o16
    w, key = g.wxyz, jax.random.PRNGKey(1)

    def rnd(k):
        out = []
        for b in R.data.blocks:
            n = b.e.shape[0]
            k, *ks = jax.random.split(k, 5)
            c = lambda kk, s: jax.random.normal(kk, s) + 1j * jax.random.normal(jax.random.fold_in(kk, 1), s)
            X, Y = c(ks[0], (n, n)), c(ks[1], (n, n))
            out.append(((X - X.T) / 2, (Y - Y.T) / 2, rs._Q(b, c(ks[2], b.v.shape), w), rs._Qc(b, c(ks[3], b.v.shape), w)))
        return tuple(out), k

    Z1, key = rnd(key)
    Z2, key = rnd(key)
    L = lambda Z: R.apply_L(R.data, R.mf0, f, Z)
    a, b = R._ipK(Z1, L(Z2)), R._ipK(Z2, L(Z1))
    assert abs(a - np.conj(b)) < 1e-10 * abs(a)
    assert R._ipK(Z1, L(Z1)).real > 0


def test_derivatives_match_finite_differences(o16):
    """dE/dt0 (response = Hellmann-Feynman) and d r_p/dt0 against re-converged neighbours."""
    g, f, cfg, st, R = o16
    d = R.derivatives(['t0'], verbose=False)['t0']
    assert d['dE'] == pytest.approx(rs.energy_gradient(st, f, g, ['t0'])['t0'], rel=1e-8)
    assert abs(d['dE_implicit']) < 1e-8
    th = rs.params_of(f, ['t0'])
    h = 1e-4 * abs(float(th['t0']))
    warm = dataclasses.replace(cfg, bcs_start=0, diag_start=0, iteranneal=0)
    r2, vals = _r2(g), []
    for s in (1, -1):
        ss = run_hfb(g, rs.with_params(f, {'t0': th['t0'] + s * h}), nucleus_z=8, nucleus_n=8, npsi_n=10,
                     config=warm, initial_state=st)
        vals.append((float(ss.energies.ehfint),
                     float(jnp.sqrt(jnp.sum(r2 * ss.densities.rho[1]) / jnp.sum(ss.densities.rho[1])))))
    assert d['dE'] == pytest.approx((vals[0][0] - vals[1][0]) / (2 * h), rel=1e-5)
    assert d['d_rp'] == pytest.approx((vals[0][1] - vals[1][1]) / (2 * h), rel=1e-4)


def test_adjoint_equals_forward(o16):
    g, f, cfg, st, R = o16
    r2, w, rho = _r2(g), g.wxyz, st.densities.rho
    Np = float(w * jnp.sum(rho[1]))
    rp = float(jnp.sqrt(w * jnp.sum(r2 * rho[1]) / Np))
    adj = R.sensitivities({'r_p': jnp.stack([0 * r2, r2 / (2 * rp * Np)])}, ['t0', 't3'], verbose=False)['r_p']
    fwd = R.derivatives(['t0', 't3'], verbose=False)
    for k in ('t0', 't3'):
        assert adj[k] == pytest.approx(fwd[k]['d_rp'], rel=1e-5)


def test_lanczos_matches_direct_solve(o16):
    """The monopole strength of a parity-projected Lanczos chain equals the GMRES solution of (omega - L) dR = [F, R0]
    (to the chain's convergence at the peak), and its energy-weighted sum is exact."""
    g, f, cfg, st, R = o16
    field = jnp.stack([_r2(g)] * 2).astype(float)
    E = np.array([15.0, 22.0])
    _, S, _ = R.strength(field, E, gamma=2.0, verbose=False)
    chain = R.lanczos(field, steps=800, verbose=False, parity=1)
    assert len(chain['alpha']) == 800
    np.testing.assert_allclose(rs.Response.spectrum(chain, E, 2.0), S, rtol=5e-3)


def test_static_polarizability_is_the_chain_m_minus1(o16):
    """m_-1 from the static solve equals <u0, K u0> / 2 of the chain started from it."""
    g, f, cfg, st, R = o16
    field = jnp.stack([_r2(g)] * 2).astype(float)
    m, _ = R.polarizability(field)
    assert m == pytest.approx(R.lanczos(field, steps=2, verbose=False)['m_minus1'], rel=1e-6)
