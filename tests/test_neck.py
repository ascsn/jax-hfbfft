"""Tests for the neck constraint QN (physics/constraints.py)."""

import dataclasses

import jax.numpy as jnp
import pytest

from jax_hfbfft import Constraint, Force, Grid
from jax_hfbfft.core.constraint import NECK, multipole_name, parse_multipole
from jax_hfbfft.physics.constraints import build_constraint_state, compute_constraint_expectations
from jax_hfbfft.physics.densities import Densities


def test_name_round_trip():
    assert parse_multipole('QN') == NECK and multipole_name(*NECK) == 'QN'


def test_neck_expectation():
    g = Grid.create_cubic(n=16, d=1.0)
    X, Y, Z = g.get_meshgrid()
    rho = jnp.exp(-(X**2 + Y**2 + (Z - 1.0)**2) / 4.0)
    d = dataclasses.replace(Densities.zeros(16, 16, 16), rho=jnp.stack([rho, 0.5 * rho]))
    c = Constraint.from_multipoles({'QN': 1.0}, neck_z=0.5, neck_width=1.2)
    got = compute_constraint_expectations(build_constraint_state(c, g, 16), d, g)
    want = 1.5 * jnp.sum(rho * jnp.exp(-((Z - 0.5) / 1.2) ** 2)) * g.wxyz
    assert float(got[0]) == pytest.approx(float(want), rel=1e-10)


def test_scan_drives_the_neck():
    """run_constrained_scan reaches QN targets below the 16O ground-state value (~5.45)."""
    from jax_hfbfft.physics import run_constrained_scan
    from jax_hfbfft.physics.solver import apply_cm_correction
    force = apply_cm_correction(Force.from_name("SLy4", ipair=0), 16)
    res, _ = run_constrained_scan(Grid.create_cubic(n=16, d=1.0), force, nucleus_z=8, nucleus_n=8,
                                  npsi_n=8, multipole='QN', targets=[5.0], inner_iters=150, verbose=False)
    assert res[0]['achieved'] == pytest.approx(5.0, abs=0.1)
