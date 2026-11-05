"""Second-order sensitivities from the Thouless-coordinate Hessian (physics/second_order.py)."""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from jax_hfbfft import Force
from jax_hfbfft.core.grid import Grid
from jax_hfbfft.physics import response as rs
from jax_hfbfft.physics.second_order import HFHessian, density
from jax_hfbfft.physics.solver import run_hfb, apply_cm_correction, two_basis_config


@pytest.fixture(scope="module")
def o16():
    g = Grid.create_cubic(n=15, d=1.0)
    f = apply_cm_correction(Force.from_name("SLy4").with_pairing(ipair=0), 16)
    cfg = two_basis_config(etol=1e-7, max_iterations=2000)
    st = run_hfb(g, f, nucleus_z=8, nucleus_n=8, npsi_n=10, config=cfg)
    X, Y, Z = g.get_meshgrid()
    fields = dict(D=jnp.stack([-0.5 * Z, 0.5 * Z]), zcm=jnp.stack([Z, Z]), r2=jnp.stack([X**2 + Y**2 + Z**2] * 2))
    return g, f, cfg, st, fields, HFHessian(st, f, g, constraints=(fields['zcm'],))


def test_ground_state_is_stationary(o16):
    """The gradient of E in Thouless coordinates vanishes at the converged HF state."""
    *_, H = o16
    assert H.gradient_norm() < 1e-4


def test_hessian_polarizability_equals_response_operator(o16):
    """g^T K^-1 g from the Hessian of E equals 2 m_-1 from the response operator L: L is the Hessian of E."""
    g, f, cfg, st, fl, H = o16
    for name in ('D', 'r2'):
        P, _, rel = H.polarizability(fl[name])
        m, _ = rs.Response(st, f, g).polarizability(fl[name], constraints=(fl['zcm'],))
        assert rel < 1e-8
        assert P / 2 == pytest.approx(m, rel=1e-7)


def test_second_order_adjoint_matches_finite_differences(o16):
    """dP/dtheta by the 2n+1 formula equals central differences of P from re-converged ground states."""
    g, f, cfg, st, fl, H = o16
    _, dP, _ = H.polarizability_gradient(fl['D'], ['t0', 'x0'], verbose=False)
    warm = dataclasses.replace(cfg, bcs_start=0, diag_start=0, iteranneal=0)
    th = rs.params_of(f, ['t0', 'x0'])
    for k in ('t0', 'x0'):
        h = 1e-4 * abs(float(th[k]))
        P = []
        for s in (1, -1):
            fk = rs.with_params(f, {**th, k: th[k] + s * h})
            sk = run_hfb(g, fk, nucleus_z=8, nucleus_n=8, npsi_n=10, config=warm, initial_state=st)
            P.append(2 * rs.Response(sk, fk, g).polarizability(fl['D'], constraints=(fl['zcm'],))[0])
        assert dP[k] == pytest.approx((P[0] - P[1]) / (2 * h), rel=1e-4)


def test_observable_gradient_equals_response_adjoint(o16):
    """The first-order adjoint in Thouless coordinates reproduces Response.sensitivities for r_p."""
    g, f, cfg, st, fl, H = o16
    r2 = fl['r2'][0]
    rp_of = lambda d, y, th: jnp.sqrt(g.wxyz * jnp.sum(r2 * density(d, y, g).rho[1]) / 8.0)
    rp, d, _ = H.observable_gradient(rp_of, ['t0', 't3'])
    ref = rs.Response(st, f, g).sensitivities({'r_p': jnp.stack([0 * r2, r2 / (2 * rp * 8.0)])}, ['t0', 't3'],
                                              verbose=False)['r_p']
    for k in ('t0', 't3'):
        assert d[k] == pytest.approx(ref[k], rel=1e-6)


def test_constrained_monopole_energy(o16):
    """E_-1 = sqrt(m1/m-1) with the analytic sum-rule weight: m1 equals 4 sum_q h2m_q int r^2 rho_q."""
    g, f, cfg, st, fl, H = o16
    h2m = np.asarray(f.h2m)
    W = jnp.stack([4 * h2m[q] * fl['r2'][0] for q in (0, 1)])
    out = H.constrained_energy(fl['r2'], ['t0'], verbose=False, m1_weight=W)
    m1 = float(sum(g.wxyz * jnp.sum(W[q] * st.densities.rho[q]) for q in (0, 1)))
    assert out['m1'] == pytest.approx(m1, rel=1e-7)     # the state's stored densities are from the solver's last update
    assert 15.0 < out['Ec'] < 35.0
