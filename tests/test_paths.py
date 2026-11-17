"""Tests for the paths on a potential-energy surface (physics/paths.py)."""

import numpy as np
import pytest

from jax_hfbfft.physics.paths import least_action_path, minimum_energy_path

x = np.linspace(-1.5, 1.5, 61)
y = np.linspace(-1.0, 2.5, 71)
X, Y = np.meshgrid(x, y, indexing='ij')
start, end = (10, 20), (50, 20)          # (x, y) = (-1, 0) and (1, 0)


def test_the_barrier_is_the_lowest_pass():
    """Double well whose pass is lowered near y = 1.5: the path detours through it."""
    E = (X**2 - 1) ** 2 * (1 - 0.5 * np.exp(-(Y - 1.5) ** 2 / 0.1)) + 0.1 * Y**2
    out = minimum_energy_path(E, start, end)
    assert out['barrier'] == pytest.approx(E[30].min(), abs=1e-12)
    assert out['saddle'] == (30, int(np.argmin(E[30])))
    assert out['path'][0] == start and out['path'][-1] == end


def test_unconverged_points_are_walls():
    E = (X**2 - 1) ** 2 + Y**2
    E[30, :] = np.nan
    E[30, 60] = 3.0                      # the only gap, at y = 2
    out = minimum_energy_path(E, start, end)
    assert (30, 60) in out['path'] and out['barrier'] == pytest.approx(E[29, 59])   # entered diagonally
    E[30, 60] = np.nan
    with pytest.raises(ValueError):
        minimum_energy_path(E, start, end)


def test_action_of_a_straight_path():
    """S = Int sqrt(2 M (E - E0)) dx along y = 0, sqrt(2 M) 4/3 for E = (x^2 - 1)^2."""
    E = (X**2 - 1) ** 2 + 10 * Y**2
    out = least_action_path(E, (x, y), start, end)
    assert all(p[1] == 20 for p in out['path'])
    assert out['action'] == pytest.approx(np.sqrt(2) * 4 / 3, rel=1e-3)
    M = np.zeros(E.shape + (2, 2))
    M[..., 0, 0], M[..., 1, 1] = 4.0, 1.0
    heavy = least_action_path(E, (x, y), start, end, mass=M)
    assert heavy['action'] == pytest.approx(2 * out['action'], rel=1e-12)
    assert heavy['penetrability'] == pytest.approx(1 / (1 + np.exp(2 * heavy['action'])))
