"""Tests for the form factors (physics/form_factors.py) and the multipole deformations of compute_radii."""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from jax_hfbfft.core.grid import Grid
from jax_hfbfft.physics.densities import Densities
from jax_hfbfft.physics.energies import compute_radii
from jax_hfbfft.physics.form_factors import form_factors

N, Z = 28, 20


@pytest.fixture(scope="module")
def gaussians():
    g = Grid.create_cubic(n=32, d=0.6)
    X, Y, Zc = g.get_meshgrid()
    unit = lambda a: (lambda f: f / (jnp.sum(f) * g.wxyz))(jnp.exp(-(X**2 + Y**2 + Zc**2) / a**2))
    rho = jnp.stack([N * unit(2.6), Z * unit(2.4)])
    return g, dataclasses.replace(Densities.zeros(32, 32, 32), rho=rho)


def test_point_form_factor_of_a_gaussian(gaussians):
    g, d = gaussians
    q = np.linspace(0.0, 2.0, 9)
    out = form_factors(d, g, q)
    assert np.allclose(out['F_p'], np.exp(-q**2 * 2.4**2 / 4), atol=1e-8)
    assert np.allclose(out['F_n'], np.exp(-q**2 * 2.6**2 / 4), atol=1e-8)
    assert out['skin'] == pytest.approx(np.sqrt(1.5) * (2.6 - 2.4), rel=1e-4)


def test_slopes_give_the_radii(gaussians):
    """F(q) = 1 - q^2 R^2 / 6 + ...: the charge slope is compute_radii's charge radius."""
    g, d = gaussians
    q = 0.004
    out = form_factors(d, g, [0.0, q])
    assert out['F_ch'][0] == pytest.approx(1.0, abs=1e-12) and out['F_W'][0] == pytest.approx(1.0, abs=1e-12)
    assert out['R_ch'] == pytest.approx(compute_radii(d, g).charge, rel=1e-10)
    for F, R in ((out['F_ch'][1], out['R_ch']), (out['F_W'][1], out['R_W'])):
        assert np.sqrt(6 * (1 - F) / q**2) == pytest.approx(R, rel=1e-4)
    assert out['weak_skin'] > 0              # neutrons spread wider than protons


def test_octupole_deformation_of_a_shifted_pear():
    """beta3 is measured about the centre of mass: shifting the density does not change it,
    and a sharp pear r < R0 (1 + beta3 Y30) gives beta3 to first order."""
    g = Grid.create_cubic(n=40, d=0.4)
    X, Y, Zc = g.get_meshgrid()
    A = 100.0
    R0 = 1.2 * A ** (1 / 3)
    b3 = 0.05

    def pear(dz):
        r = jnp.sqrt(X**2 + Y**2 + (Zc - dz)**2) + 1e-12
        c = (Zc - dz) / r
        Y30 = np.sqrt(7 / (4 * np.pi)) * (5 * c**3 - 3 * c) / 2
        f = 1 / (1 + jnp.exp((r - R0 * (1 + b3 * Y30)) / 0.15))
        rho = A * f / (jnp.sum(f) * g.wxyz)
        return dataclasses.replace(Densities.zeros(40, 40, 40), rho=jnp.stack([0.5 * rho, 0.5 * rho]))

    r0, r1 = compute_radii(pear(0.0), g), compute_radii(pear(0.8), g)
    assert r1.beta3 == pytest.approx(r0.beta3, rel=1e-3)
    assert r0.beta3 == pytest.approx(b3, rel=0.1) and abs(r0.beta4) < 0.01
