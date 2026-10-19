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


@pytest.mark.parametrize("axis", [0, 2])
def test_oblate_beta_gamma(axis):
    """An axial oblate shape has gamma = 60 deg and the beta of its symmetry axis, along any axis."""
    g = Grid.create(nx=24, ny=24, nz=24, dx=0.8, dy=0.8, dz=0.8)
    R = list(g.get_meshgrid())
    widths = [9.0, 9.0, 9.0]
    widths[axis] = 4.0                                    # flattened along `axis`
    rho = jnp.exp(-sum(r**2 / w for r, w in zip(R, widths)))
    d = dataclasses.replace(Densities.zeros(24, 24, 24), rho=jnp.stack([rho, rho]))
    r = compute_radii(d, g)
    A = float(jnp.sum(2 * rho) * g.wxyz)
    r2 = float(jnp.sum(2 * rho * sum(x**2 for x in R)) * g.wxyz) / A
    q_axis = float(jnp.sum(2 * rho * (3 * R[axis]**2 - sum(x**2 for x in R))) * g.wxyz)
    want = abs(np.sqrt(5 / (16 * np.pi)) * q_axis * 4 * np.pi / (5 * r2 * A))
    assert float(r.gamma) == pytest.approx(60.0, abs=0.1)
    assert float(r.beta2) == pytest.approx(want, rel=1e-6)


def test_converges_with_only_occupied_orbitals():
    """16O HF with 8 + 8 orbitals: equal occupations make the Lagrange asymmetry vanish
    identically, so the residual |(1 - P) h psi| must measure convergence (the run used to
    stop at iteration 1 on the initial guess)."""
    from jax_hfbfft.core.force import Force
    E = {}
    for npsi in ((8, 8), (12, 12)):
        calc = HFBFFT(nucleus=Nucleus(protons=8, neutrons=8), force=Force.from_name("SLy4").with_pairing(ipair=0),
                      grid=Grid.create_cubic(n=12, d=1.2), npsi=npsi)
        r = calc.run(max_iterations=1500, convergence_threshold=1e-5, print_interval=10**9, sinfo_interval=10**9)
        assert r.converged and r.iterations > 20
        E[npsi] = r.total_energy
    assert E[(8, 8)] == pytest.approx(E[(12, 12)], abs=0.01)
