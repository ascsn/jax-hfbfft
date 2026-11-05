"""
Tests for the microscopic centre-of-mass correction (zpe = 3,
physics/cm_correction.py).

For a Slater determinant of filled harmonic-oscillator shells the centre of
mass sits in its oscillator ground state, so <P^2>/2mA = (3/4) hbar omega
exactly; with the exchange term included the result is independent of how
many shells are filled.  hbar omega = hbar^2 / (m b^2) = 2 h2m / b^2.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from jax_hfbfft.core.grid import Grid
from jax_hfbfft.physics.cm_correction import microscopic_cm_energy

H2M = 20.73553
B = 1.8   # oscillator length (fm)


def _ho_states(grid, shells):
    """Normalised HO orbitals (0s, then 0p x/y/z), each with spin up and down."""
    x = grid.x[:, None, None]
    y = grid.y[None, :, None]
    z = grid.z[None, None, :]
    g = jnp.exp(-(x**2 + y**2 + z**2) / (2 * B**2))
    spatial = [g] if shells == 1 else [g, x * g, y * g, z * g]
    psi = []
    for f in spatial:
        f = f / jnp.sqrt(jnp.sum(f**2) * grid.wxyz)
        for spin in (0, 1):
            s = jnp.zeros((2,) + f.shape, dtype=jnp.complex128).at[spin].set(f)
            psi.append(s)
    return jnp.stack(psi)


@pytest.fixture(scope="module")
def grid():
    return Grid.create(nx=24, ny=24, nz=24, dx=0.8, dy=0.8, dz=0.8)


@pytest.mark.parametrize("shells", [1, 2])
def test_filled_ho_shells_give_three_quarters_hbar_omega(grid, shells):
    one = _ho_states(grid, shells)
    psi = jnp.concatenate([one, one])                         # neutrons, protons
    n = one.shape[0]
    isospin = jnp.array([0] * n + [1] * n)
    occ = jnp.ones(2 * n)
    ecm = float(microscopic_cm_energy(psi, occ, occ, isospin, grid, jnp.array([H2M, H2M])))
    assert ecm == pytest.approx(0.75 * 2 * H2M / B**2, rel=1e-6)


def test_exchange_term_matters(grid):
    """Without the exchange sum the p shell would overestimate E_cm."""
    one = _ho_states(grid, 2)
    psi = jnp.concatenate([one, one])
    n = one.shape[0]
    isospin_same = jnp.array([0] * n + [1] * n)
    isospin_all_distinct = jnp.arange(2 * n) % 2 + 2 * (jnp.arange(2 * n) // 2)  # no shared isospin
    occ = jnp.ones(2 * n)
    h2m = jnp.array([H2M, H2M])
    with_x = float(microscopic_cm_energy(psi, occ, occ, isospin_same, grid, h2m))
    without_x = float(microscopic_cm_energy(psi, occ, occ, isospin_all_distinct, grid, h2m))
    assert without_x > with_x * 1.2


def test_estimate_used_during_iterations():
    """zpe = 3 subtracts the 17.3/A^0.2 estimate until run_hfb's final step."""
    from jax_hfbfft.core.force import Force
    from jax_hfbfft.physics.densities import Densities
    from jax_hfbfft.physics.energies import compute_integrated_energy
    import dataclasses
    g = Grid.create(nx=8, ny=8, nz=8, dx=1.0, dy=1.0, dz=1.0)
    f = dataclasses.replace(Force.from_name("SLy4"), zpe=3)
    e = compute_integrated_energy(Densities.zeros(8, 8, 8), f, g, use_coulomb=False, mass_number=40)
    assert float(e.e_zpe) == pytest.approx(17.3 / 40**0.2)
