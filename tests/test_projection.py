"""Tests for the particle-number projection (physics/projection.py)."""

from itertools import combinations
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from jax_hfbfft import HFBFFT, Force, Grid, Nucleus
from jax_hfbfft.physics.densities import compute_densities
from jax_hfbfft.physics.projection import project_particle_number


def test_kinetic_energy_of_the_projected_state():
    """With only the kinetic energy, E_N = sum_k t_k <n_k>_N, the occupations of the projected
    state counted over pair configurations: 5 neutron pairs at 6 particles, 1 full proton pair."""
    g = Grid.create_cubic(n=14, d=0.8)
    X, Y, Z = g.get_meshgrid()
    gauss = jnp.exp(-(X**2 + Y**2 + Z**2) / 6.0)
    orbs = [f * gauss for f in (1.0, X, Y, Z, X * Y)] + [gauss]
    orbs = [o / jnp.sqrt(jnp.sum(o**2) * g.wxyz) for o in orbs]
    psi = jnp.stack([jnp.stack(s) for o in orbs for s in ((o, 0 * o), (0 * o, o))]).astype(complex)
    v2p = np.array([0.95, 0.8, 0.6, 0.4, 0.25])          # 3 pairs on average
    v2 = jnp.asarray(np.repeat(np.append(v2p, 1.0), 2))
    ones = jnp.ones(12)
    iso = jnp.array([0] * 10 + [1] * 2)
    dens = compute_densities(psi, v2, jnp.sqrt(v2 * (1 - v2)), ones, ones, iso, g)
    state = SimpleNamespace(psi=psi, wocc=v2, wguv=jnp.sqrt(v2 * (1 - v2)), pairwg=ones, wstates=ones,
                            isospin=iso, densities=dens, pairing=SimpleNamespace())
    t = [2 * 20.73553 * float(jnp.sum(compute_densities(psi, jnp.eye(12)[2 * k], 0 * ones, ones, ones, iso, g)
                                      .tau.sum(axis=0)) * g.wxyz) for k in range(6)]   # per pair
    W = {S: np.prod([v2p[j] if j in S else 1 - v2p[j] for j in range(5)]) for S in combinations(range(5), 3)}
    occ = [sum(w for S, w in W.items() if j in S) / sum(W.values()) for j in range(5)]
    out = project_particle_number(state, g, Force(name="kinetic", ex=0, zpe=2), use_coulomb=False)
    assert out['energy'] == pytest.approx(np.dot(t[:5], occ) + t[5], rel=1e-10)
    assert out['unprojected'] == pytest.approx(np.dot(t[:5], v2p) + t[5], rel=1e-10)
    assert out['distribution']['n'][6] == pytest.approx(sum(W.values()), rel=1e-10)
    assert out['distribution']['p'][2] == pytest.approx(1.0, abs=1e-12)


@pytest.fixture(scope="module")
def o20():
    force = Force.from_name("SLy4").with_pairing(ipair=6, v0neut=350.0, v0prot=350.0, rho0pr=0.32,
                                                 tbcs=False, pair_cutoff=(0.0, 0.0), state_cutoff=(0.0, 0.0))
    calc = HFBFFT(nucleus=Nucleus(protons=8, neutrons=12), force=force,
                  grid=Grid.create_cubic(n=12, d=1.2), npsi=(20, 12))
    calc.diag_start = calc.bcs_start = 30
    calc.run(max_iterations=600, convergence_threshold=1e-20, print_interval=10**9, sinfo_interval=10**9)
    return calc


def test_paired_nucleus(o20):
    """20O: phi = 0 is the static functional, the distribution has mean N and the BCS variance
    2 sum u^2 v^2, the projection lowers the energy and is converged in the number of points."""
    st = o20.solver_state
    out = {L: o20.project_particle_number(L) for L in (9, 13)}
    assert out[13]['total_energy'] == pytest.approx(o20.results.total_energy + out[13]['correction'], abs=1e-12)
    assert out[9]['unprojected'] == pytest.approx(float(st.energies.ehfint), abs=2e-4)
    d = out[13]['distribution']['n']
    v2, iso = np.asarray(st.wocc), np.asarray(st.isospin)
    mean = sum(n * p for n, p in d.items())
    assert sum(d.values()) == pytest.approx(1.0, abs=1e-12) and mean == pytest.approx(12.0, abs=1e-8)
    assert sum((n - 12) ** 2 * p for n, p in d.items()) == pytest.approx(2 * np.sum((v2 * (1 - v2))[iso == 0]), rel=1e-8)
    assert -1.0 < out[13]['correction'] < -0.2
    assert out[9]['correction'] == pytest.approx(out[13]['correction'], abs=1e-3)
    assert abs(out[13]['imag']) < 1e-10


def test_rejects_odd_particle_numbers():
    st = SimpleNamespace(isospin=jnp.array([0, 0]), wocc=jnp.array([1.0, 0.0]), wstates=jnp.ones(2),
                         pairing=SimpleNamespace())
    with pytest.raises(NotImplementedError):
        project_particle_number(st, None, None)
