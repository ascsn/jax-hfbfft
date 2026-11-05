"""Tests for run_constrained_scan with several driven multipoles (a deformation surface)."""

import pytest

from jax_hfbfft import Force, Grid
from jax_hfbfft.physics import run_constrained_scan
from jax_hfbfft.physics.solver import apply_cm_correction


def test_two_dimensional_scan_hits_both_moments():
    force = apply_cm_correction(Force.from_name("SLy4", ipair=0), 16)
    res, _ = run_constrained_scan(Grid.create_cubic(n=16, d=1.0), force, nucleus_z=8, nucleus_n=8,
                                  npsi_n=8, multipole=('Q20', 'Q30'), targets=[(15.0, 10.0), (15.0, 25.0)],
                                  inner_iters=150, verbose=False)
    for r in res:
        assert len(r['achieved']) == 2 and len(r['lam']) == 2
        for got, want in zip(r['achieved'], r['target']):
            assert got == pytest.approx(want, abs=0.05 * want)


def test_target_shape_is_checked():
    force = apply_cm_correction(Force.from_name("SLy4", ipair=0), 16)
    with pytest.raises(ValueError):
        run_constrained_scan(Grid.create_cubic(n=16, d=1.0), force, nucleus_z=8, nucleus_n=8, npsi_n=8,
                             multipole=('Q20', 'Q30'), targets=[15.0], verbose=False)
