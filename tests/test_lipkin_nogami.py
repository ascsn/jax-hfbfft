"""
Tests for the Lipkin-Nogami correction (physics/pairing.py).

The properties checked are the ones the 2026-09 bugs broke: lambda_2 must be
self-consistent (lambda_2 = F(lambda_2) on its own LN occupations, including
open shells where F rises and the root lies above F(0)), particle number must
be conserved, and collapsed pairing must give lambda_2 = 0 exactly.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from jax_hfbfft.physics.pairing import compute_lambda2, solve_pairing_isospin


def _kramers(levels):
    """Each level twice (both Kramers partners are explicit states)."""
    return jnp.repeat(jnp.asarray(levels, dtype=jnp.float64), 2)


def _solve(sp, n, gap, use_ln=True):
    ones = jnp.ones_like(sp)
    return solve_pairing_isospin(0, float(n), sp, gap * ones, ones, ones,
                                 smallp=1e-10, use_ln=use_ln, ln_passes=12)


SPECTRA = {
    # 12 equally spaced Kramers pairs, half filled
    "uniform": (_kramers(np.arange(12) - 5.5), 12, 1.0),
    # closed core, two particles in a degenerate 8-fold shell (like 42Ca neutrons)
    "open_shell": (jnp.concatenate([_kramers([-12, -9, -8, -7, -6]),
                                    _kramers([0.0] * 4), _kramers([5, 6, 7])]), 12, 1.0),
}


@pytest.mark.parametrize("case", sorted(SPECTRA))
def test_lambda2_self_consistent(case):
    sp, n, gap = SPECTRA[case]
    _, v2, _, _, wst, st = _solve(sp, n, gap)
    lam2 = float(st["lambda2"])
    assert lam2 > 0.0
    again = float(compute_lambda2(v2, wst, st["avdelt"], st["epair"]))
    assert again == pytest.approx(lam2, rel=1e-4)


@pytest.mark.parametrize("case", sorted(SPECTRA))
def test_particle_number_and_energy(case):
    sp, n, gap = SPECTRA[case]
    _, v2, _, _, wst, st = _solve(sp, n, gap)
    assert float(jnp.sum(v2 * wst)) == pytest.approx(n, abs=1e-6)
    s_u2v2 = 0.5 * float(jnp.sum(v2 * (1 - v2) * wst))
    assert float(st["e_ln"]) == pytest.approx(-4 * float(st["lambda2"]) * s_u2v2, rel=1e-10)
    assert float(st["e_ln"]) < 0.0


def test_collapsed_pairing_gives_zero():
    # closed shell with a 10 MeV gap and a vanishing pairing gap
    sp = jnp.concatenate([_kramers([-15, -12, -11, -10]), _kramers([0, 1, 2, 3])])
    _, _, _, _, _, st = _solve(sp, 8, 1e-5)
    assert float(st["lambda2"]) == 0.0
    assert float(st["e_ln"]) == 0.0


def test_ln_off_is_plain_bcs():
    sp, n, gap = SPECTRA["uniform"]
    ef0, v2_0, _, _, _, st0 = _solve(sp, n, gap, use_ln=False)
    assert float(st0["lambda2"]) == 0.0 and float(st0["e_ln"]) == 0.0
    edif = sp - ef0
    bcs = 0.5 * (1 - edif / jnp.sqrt(edif**2 + gap**2))
    assert np.allclose(np.asarray(v2_0), np.asarray(bcs), atol=1e-8)
