"""Two-basis HFB solver (physics/gcg.py, SolverConfig.orbital_solver = 'two_basis') and axial averaging."""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from jax_hfbfft import Force
from jax_hfbfft.core.grid import Grid
from jax_hfbfft.physics.densities import Densities
from jax_hfbfft.physics.solver import SolverConfig, run_hfb, apply_cm_correction, two_basis_config, _c4_average


def test_axial_average():
    """Idempotent, Q22 = 0, particle number kept, and the J_mu,nu rotation consistent with scalars (its trace)."""
    g = Grid.create_cubic(n=9, d=1.0)
    r = np.random.default_rng(0)
    f = lambda *s: jnp.asarray(r.random((2,) + s + (9, 9, 9)))
    d = Densities(rho=f(), tau=f(), chi=f(), current=f(3), sdens=f(3), sodens=f(3), jsym=f(6), tspin=f(3))
    a = _c4_average(d)
    for x, y in zip(dataclasses.astuple(a), dataclasses.astuple(_c4_average(a))):
        np.testing.assert_allclose(x, y, atol=1e-12)
    X, Y, _ = g.get_meshgrid()
    assert abs(float(jnp.sum(a.rho * (X ** 2 - Y ** 2)))) < 1e-9
    np.testing.assert_allclose(jnp.sum(a.rho), jnp.sum(d.rho), rtol=1e-12)
    trace = lambda t: t[:, 0] + t[:, 1] + t[:, 2]
    np.testing.assert_allclose(trace(a.jsym), _c4_average(dataclasses.replace(d, rho=trace(d.jsym))).rho, atol=1e-12)


def _oxygen(N, two_basis, npsi_n, **force_kw):
    g = Grid.create_cubic(n=15, d=1.0)
    force = apply_cm_correction(Force.from_name("SLy4").with_pairing(**force_kw), 8 + N)
    cfg = two_basis_config(max_iterations=400) if two_basis else SolverConfig(max_iterations=1500, convergence_criterion=1e-7)
    return g, run_hfb(g, force, nucleus_z=8, nucleus_n=N, npsi_n=npsi_n, config=cfg)


def test_two_basis_hf_matches_gradient():
    """Without pairing the two-basis solver is a GCG eigensolver: the same HF state as the gradient solver."""
    E = [float(_oxygen(8, tb, 14, ipair=0)[1].energies.ehfint) for tb in (False, True)]
    assert E[1] == pytest.approx(E[0], abs=2e-3)


def test_two_basis_hfb_converges_consistently():
    """20O with pairing: converges by the energy rule, conserves N, and E_pair = int v_pair chi (its windowed value)."""
    g, st = _oxygen(12, True, 40, ipair=6, v0neut=500.0, v0prot=500.0, rho0pr=0.32, state_cutoff=(8.0, 8.0))
    assert bool(st.converged)
    n = np.asarray(st.wocc)[np.asarray(st.isospin) == 0].sum()
    assert n == pytest.approx(12.0, abs=1e-6)
    ep = float(st.pairing.epair[0])
    assert ep > 0.1
    assert ep == pytest.approx(float(g.wxyz * jnp.sum(st.meanfield.v_pair[0] * st.densities.chi[0])), rel=1e-3)


def test_two_basis_needs_odd_grid():
    g = Grid.create_cubic(n=16, d=1.0)
    force = apply_cm_correction(Force.from_name("SLy4").with_pairing(ipair=0), 16)
    with pytest.raises(ValueError, match="odd grid"):
        run_hfb(g, force, nucleus_z=8, nucleus_n=8, npsi_n=14, config=two_basis_config(max_iterations=5))


def test_two_basis_constrained_scan_and_derivatives():
    """Q20-constrained 16O with the two-basis solver: the response Newton step hits the target, and the
    constrained response (constraint_potential='auto') holds every constrained moment fixed, so the
    implicit part of dE/dt0 vanishes (dE = Hellmann-Feynman at fixed Q)."""
    from jax_hfbfft.physics.solver import run_constrained_scan
    from jax_hfbfft.physics.constraints import compute_constraint_expectations
    from jax_hfbfft.physics.response import Response
    g = Grid.create_cubic(n=15, d=1.0)
    force = apply_cm_correction(Force.from_name("SLy4").with_pairing(ipair=0), 16)
    res, st = run_constrained_scan(g, force, 8, 8, 10, [15.0], "Q20", config=two_basis_config(etol=1e-8),
                                   n_outer=8, tol=1e-6, inner_iters=300, verbose=False)
    assert res[0]['achieved'] == pytest.approx(15.0, rel=1e-5)
    cs = st.constraint_state
    fields = [jnp.transpose(cs.constr_field[i], (3, 0, 1, 2)) for i in range(cs.constr_field.shape[0])]
    d = Response(st, force, g, constraint_potential='auto').derivatives(['t0'], constraints=fields, verbose=False)['t0']
    dC = [float(g.wxyz * jnp.sum(fc * d['densities'].rho)) for fc in fields]
    assert max(abs(x) for x in dC) < 1e-8
    assert abs(d['dE_implicit']) < 1e-6 * abs(d['dE'])
