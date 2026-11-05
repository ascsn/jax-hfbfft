"""
Tests for the response tools (physics/dynamics.py): the energy-weighted sum rule
and the response operators of the kicks.
"""

import dataclasses

import jax.numpy as jnp
import pytest

from jax_hfbfft.core.force import Force
from jax_hfbfft.core.grid import Grid
from jax_hfbfft.physics import dynamics as D
from jax_hfbfft.physics.densities import Densities

N, Z = 12, 8


@pytest.fixture(scope="module")
def setup():
    g = Grid.create(nx=24, ny=24, nz=24, dx=0.8, dy=0.8, dz=0.8)
    X, Y, Zc = g.get_meshgrid()
    shape = lambda a: jnp.exp(-(X**2 + Y**2 + Zc**2) / a)
    unit = lambda f: f / (jnp.sum(f) * g.wxyz)
    rho = jnp.stack([N * unit(shape(6.0)), Z * unit(shape(5.0))])
    return g, dataclasses.replace(Densities.zeros(24, 24, 24), rho=rho), Force.from_name("SLy4")


def test_monopole_ewsr_is_kinetic(setup):
    g, d, f = setup
    m1 = D.ewsr(d, f, g, *D.response_operator(g, 'monopole', N, Z))
    X, Y, Zc = g.get_meshgrid()
    r2 = X**2 + Y**2 + Zc**2
    want = 4 * sum(float(f.h2m[q]) * float(jnp.sum(d.rho[q] * r2) * g.wxyz) for q in (0, 1))
    assert m1 == pytest.approx(want, rel=1e-6)


def test_dipole_ewsr_is_trk_without_ctau(setup):
    g, d, f = setup
    bare = dataclasses.replace(f, Ctau0=0.0, Ctau1=0.0)
    m1 = D.ewsr(d, bare, g, *D.response_operator(g, 'isovector_dipole', N, Z))
    assert m1 == pytest.approx(float(f.h2m[0]) * N * Z / (N + Z), rel=1e-6)
    enhanced = D.ewsr(d, f, g, *D.response_operator(g, 'isovector_dipole', N, Z))
    assert enhanced > m1          # SLy4: TRK enhancement kappa > 0


def test_strength_function_recovers_the_modes():
    """Two modes after a kick exp(-i eta Q): delta<Q> = -2 eta sum_n B_n sin(E_n t / hbar).
    The peaks sit at E_n with total strength sum B_n; the -eta run cancels an eta^2 offset."""
    import numpy as np
    t = np.arange(0.0, 4000.0, 0.5)
    eta, modes = 1e-3, ((12.0, 3.0), (25.0, 1.0))
    signal = lambda e: 5.0 - 2 * e * sum(B * np.sin(En * t / D.HBARC) for En, B in modes) + 2000.0 * e**2 * (t > 0)
    E, S = D.strength_function(t, signal(eta), eta, e_max=60.0, n_e=3000, q_minus=signal(-eta))
    m0, m1, _ = D.strength_moments(E, S)
    assert m0 == pytest.approx(4.0, rel=0.01)
    assert m1 == pytest.approx(12.0 * 3.0 + 25.0 * 1.0, rel=0.01)
    assert E[np.argmax(S)] == pytest.approx(12.0, abs=0.05)
    _, S_offset = D.strength_function(t, signal(eta), eta, e_max=60.0, n_e=3000)
    assert np.max(np.abs(S_offset - S)) > 0.1 * np.max(S)         # the offset matters without -eta


def test_kick_strength_scales_r2_by_A():
    import numpy as np
    t = np.linspace(0.0, 2000.0, 4001)
    h = dict(time=t, A=np.full_like(t, 16.0), r2=7.0 - 2e-3 * np.sin(20.0 * t / D.HBARC) / 16.0)
    E, S = D.kick_strength(h, 'monopole', 1e-3, gamma=1.0, e_max=40.0, n_e=2000)
    assert D.strength_moments(E, S)[0] == pytest.approx(1.0, rel=0.03)
