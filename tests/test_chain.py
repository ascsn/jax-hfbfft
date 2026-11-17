"""Tests for core/chain.py: chains of nuclei and separation energies."""

import math

import pytest

from jax_hfbfft import Force, Grid
from jax_hfbfft.core.chain import chain_basis, run_chain, separation_energies

OXYGEN = [(8, 8), (8, 10), (8, 12)]


@pytest.fixture(scope="module")
def setup():
    return Force.from_name("SLy4").with_pairing(ipair=0), Grid.create_cubic(n=16, d=1.0)


def test_basis_covers_the_chain(setup):
    force, _ = setup
    assert chain_basis(OXYGEN, force) == (12, 8)


def test_separation_energies(setup):
    force, grid = setup
    pts = run_chain(OXYGEN, force, grid, max_iterations=150)
    rows = separation_energies(pts)
    assert math.isnan(rows[0]['S2n']) and all(math.isnan(r['S2p']) for r in rows)
    for r, lo, hi in zip(rows[1:], pts[:-1], pts[1:]):
        assert r['S2n'] == pytest.approx(lo.E - hi.E)
        assert r['S2n'] > 0


def test_warm_start_converges_faster(setup):
    force, grid = setup
    warm = run_chain(OXYGEN[:2], force, grid, max_iterations=40)[1]
    cold = run_chain(OXYGEN[1:2], force, grid, npsi=(12, 8), max_iterations=40)[0]
    assert warm.E < cold.E          # after the same 40 iterations the warm start is further down
