"""Tests for the equal-filling blocking (physics/blocking.py)."""

import jax.numpy as jnp
import numpy as np
import pytest

from jax_hfbfft.core.grid import Grid
from jax_hfbfft.physics.blocking import efa_hf_occupations, time_reverse, track_doublet
from jax_hfbfft.physics.pairing import solve_pairing_isospin


@pytest.fixture(scope="module")
def pshell():
    """A degenerate p shell (x, y, z times a Gaussian, both spins) plus an s state below it."""
    g = Grid.create_cubic(n=14, d=0.8)
    X, Y, Z = g.get_meshgrid()
    gauss = jnp.exp(-(X**2 + Y**2 + Z**2) / 4.0)
    unit = lambda f: f / jnp.sqrt(jnp.sum(jnp.abs(f) ** 2) * g.wxyz)
    orbs = [unit(f * gauss) for f in (1.0, X, Y, Z)]
    psi = jnp.stack([jnp.stack(s) for o in orbs for s in ((o, 0 * o), (0 * o, o))]).astype(complex)
    e = jnp.array([-20.0, -20.0] + [-5.0] * 6)
    return g, psi, e


def test_time_reversal(pshell):
    g, psi, _ = pshell
    phi = psi[3] + 0.3j * psi[4]
    ip = lambda a, b: complex(g.wxyz * jnp.sum(jnp.conj(a) * b))
    assert ip(phi, time_reverse(phi)) == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(np.asarray(time_reverse(time_reverse(phi))), -np.asarray(phi))


def test_track_doublet_aligns_the_degenerate_shell(pshell):
    """r = (p_x + i p_y)/sqrt2 spin up (Omega = 3/2) is spread over the shell's orbitals; after the
    step two orbitals span {r, Tr} exactly and are the blocked pair, the s orbitals are untouched."""
    g, psi, e = pshell
    r = (psi[2] + 1j * psi[4]) / np.sqrt(2.0)
    iso = jnp.zeros(8, dtype=int)
    psi2, e2, _, _, blocked = track_doublet(psi, e, 0 * e, 0 * psi, iso, r, 0, g.wxyz)
    b = np.nonzero(np.asarray(blocked))[0]
    P = lambda k: sum(abs(complex(g.wxyz * jnp.sum(jnp.conj(psi2[k]) * v))) ** 2 for v in (r, time_reverse(r)))
    assert sorted(b) == sorted(np.argsort([-P(k) for k in range(8)])[:2]) and all(b >= 2)
    assert sum(P(k) for k in b) == pytest.approx(2.0, abs=1e-10)
    assert np.allclose(np.asarray(psi2[:2]), np.asarray(psi[:2])) and np.allclose(np.asarray(e2), np.asarray(e))
    S = g.wxyz * jnp.einsum('ksxyz,lsxyz->kl', jnp.conj(psi2), psi2)
    assert np.allclose(np.asarray(S), np.eye(8), atol=1e-10)


def test_hf_occupations():
    e = jnp.array([-30.0, -30.0, -10.0, -10.0, -8.0, -8.0, -2.0, -2.0, -40.0])
    iso = jnp.array([0] * 8 + [1])
    blocked = jnp.zeros(9).at[4].set(1.0).at[5].set(1.0)
    w = efa_hf_occupations(jnp.ones(9), e, iso, blocked, 0, 5)
    assert np.allclose(np.asarray(w), [1, 1, 1, 1, 0.5, 0.5, 0, 0, 1])


def test_blocked_pairing_conserves_particle_number():
    sp = jnp.repeat(jnp.arange(10, dtype=jnp.float64) - 4.5, 2)
    ones = jnp.ones(20)
    blocked = jnp.zeros(20).at[10].set(1.0).at[11].set(1.0)
    _, v2, uv, _, _, st = solve_pairing_isospin(0, 11.0, sp, 1.0 * ones, ones, ones, smallp=0.0, blocked=blocked)
    assert float(jnp.sum(v2)) == pytest.approx(11.0, abs=1e-8)
    assert np.allclose(np.asarray(v2[10:12]), 0.5) and np.allclose(np.asarray(uv[10:12]), 0.0)
    assert float(st['avdelt']) > 0.5
