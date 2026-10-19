"""Tests for finite temperature (thermal BCS, physics/pairing.py)."""

import jax.numpy as jnp
import numpy as np
import pytest

from jax_hfbfft import HFBFFT, Force, Grid, Nucleus
from jax_hfbfft.physics.pairing import solve_pairing_isospin

SP = jnp.repeat(jnp.arange(10, dtype=jnp.float64) - 4.5, 2)    # 10 Kramers pairs


def _solve(gap, T, n=10):
    ones = jnp.ones_like(SP)
    return solve_pairing_isospin(0, float(n), SP, gap * ones, ones, ones, smallp=0.0, temperature=T)


def test_no_gap_gives_fermi_dirac():
    T = 1.3
    ef, v2, wguv, _, _, st = _solve(0.0, T)
    f = 1.0 / (1.0 + np.exp((np.asarray(SP) - float(ef)) / T))
    assert np.allclose(np.asarray(v2), f, atol=1e-8)
    assert float(jnp.sum(v2)) == pytest.approx(10.0, abs=1e-8)
    S = -np.sum(f * np.log(f) + (1 - f) * np.log(1 - f))
    assert float(st['entropy']) == pytest.approx(S, rel=1e-8)
    assert np.allclose(np.asarray(wguv), 0.0)


def test_thermal_pair_amplitude():
    gap, T = 1.0, 0.7
    ef, v2, wguv, _, _, st = _solve(gap, T)
    E = np.sqrt((np.asarray(SP) - float(ef)) ** 2 + gap**2)
    assert np.allclose(np.asarray(wguv), gap / (2 * E) * np.tanh(E / (2 * T)), atol=1e-10)
    assert float(st['entropy']) > 0.0
    _, _, _, _, _, st0 = _solve(gap, 0.0)
    assert float(st0['entropy']) == 0.0


@pytest.fixture(scope="module")
def o20():
    def solve(T):
        force = Force.from_name("SLy4").with_pairing(ipair=6, v0neut=350.0, v0prot=350.0, rho0pr=0.32,
                                                     tbcs=False, pair_cutoff=(0.0, 0.0), state_cutoff=(0.0, 0.0))
        calc = HFBFFT(nucleus=Nucleus(protons=8, neutrons=12), force=force,
                      grid=Grid.create_cubic(n=12, d=1.2), npsi=(20, 12))
        calc.diag_start = calc.bcs_start = 30
        calc.temperature = T
        return calc.run(max_iterations=200, convergence_threshold=1e-20, print_interval=10**9, sinfo_interval=10**9)
    return solve(0.0), solve(3.0)


def test_pairing_collapses_when_hot(o20):
    cold, hot = o20
    assert cold.pairing_gap_n > 0.5
    assert hot.pairing_gap_n < 0.1 * cold.pairing_gap_n
    assert hot.entropy > 1.0 and cold.entropy == 0.0
    assert hot.total_energy > cold.total_energy                  # internal energy rises
    assert hot.free_energy < hot.total_energy                     # F = E - T S
