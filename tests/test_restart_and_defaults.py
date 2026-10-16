"""
Guards for checkpoint/restart, the constraint damping radius, the charge
radius and the static -> dynamics warnings.
"""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from jax_hfbfft.core.constraint import Constraint
from jax_hfbfft.core.grid import Grid
from jax_hfbfft.core.hfbfft import HFBFFT
from jax_hfbfft.core.nucleus import Nucleus
from jax_hfbfft.physics.constraints import default_damprad, resolved_damprad
from jax_hfbfft.physics.densities import Densities
from jax_hfbfft.physics.energies import compute_radii
from jax_hfbfft.physics.solver import statics_only_terms

KW = dict(force_name="SLy4", nx=16, ny=16, nz=16, dx=1.0, dy=1.0, dz=1.0, npsi=(10, 10))


def test_checkpoint_round_trip(tmp_path):
    path = str(tmp_path / "o16.npz")
    a = HFBFFT(Nucleus(protons=8, neutrons=8), **KW)
    a.run(max_iterations=5, print_interval=100, checkpoint_file=path)
    b = HFBFFT(Nucleus(protons=8, neutrons=8), **KW)
    b.initialize_wavefunctions(method="restart", filename=path)
    assert np.allclose(np.asarray(b.state.psi), np.asarray(a.state.psi))
    assert np.allclose(np.asarray(b.state.wocc), np.asarray(a.state.wocc))
    with pytest.raises(ValueError):
        HFBFFT(Nucleus(protons=8, neutrons=8), **dict(KW, npsi=(12, 12))).load_checkpoint(path)
    with pytest.raises(ValueError):
        HFBFFT(Nucleus(protons=8, neutrons=10), **KW).load_checkpoint(path)


def test_damprad_default_scales_with_nucleus():
    g = Grid.create(nx=48, ny=48, nz=48, dx=1.0, dy=1.0, dz=1.0)
    assert default_damprad(16, g) == pytest.approx(2.2 * 1.2 * 16 ** (1 / 3))
    assert default_damprad(238, g, 0.3) > default_damprad(238, g, 0.0) > 15.0
    small = Grid.create(nx=16, ny=16, nz=16, dx=1.0, dy=1.0, dz=1.0)
    assert default_damprad(238, small, verbose=False) == pytest.approx(5.0)     # 3 fm inside the box
    assert resolved_damprad(Constraint.prolate(40, beta2=0.2, damprad=6.0), g, 40) == 6.0
    assert resolved_damprad(Constraint.prolate(40, beta2=0.2), g, 40) > 9.0


def test_charge_radius_formula():
    g = Grid.create(nx=16, ny=16, nz=16, dx=1.0, dy=1.0, dz=1.0)
    x, y, z = g.x[:, None, None], g.y[None, :, None], g.z[None, None, :]
    shape = jnp.exp(-(x**2 + y**2 + z**2) / 4.0)
    unit = shape / (jnp.sum(shape) * g.wxyz)
    d = dataclasses.replace(Densities.zeros(16, 16, 16), rho=jnp.stack([28 * unit, 20 * unit]))
    r = compute_radii(d, g)
    want = np.sqrt(float(r.rms_p) ** 2 + 0.8409**2 - 0.1155 * 28 / 20 + 0.0332)
    assert float(r.charge) == pytest.approx(want, rel=1e-6)


def test_statics_only_terms():
    assert statics_only_terms({"lambda2": [0.0, 0.0], "zpe": 0}) == []
    assert len(statics_only_terms({"lambda2": [0.3, 0.6], "zpe": 0})) == 1
    assert len(statics_only_terms({"lambda2": [0.3, 0.6], "zpe": 3})) == 2
