"""Paired two-basis states: moving-window static response (physics/response.py) and the paired second order
(physics/second_order_hfb.py), against finite differences of the re-converged solver."""

import dataclasses

import jax.numpy as jnp
import pytest

from jax_hfbfft import Force
from jax_hfbfft.core.grid import Grid
from jax_hfbfft.physics import response as rs
from jax_hfbfft.physics.second_order_hfb import HFBHessian
from jax_hfbfft.physics.solver import run_hfb, apply_cm_correction, two_basis_config

Z, N, NPSI = 8, 12, 36


@pytest.fixture(scope="module")
def o20():
    g = Grid.create_cubic(n=17, d=1.0)
    f = apply_cm_correction(Force.from_name("SLy4").with_pairing(ipair=6, v0neut=325.0, v0prot=325.0, rho0pr=0.32,
                                                                 pair_cutoff=(0.0, 0.0), state_cutoff=(8.0, 8.0)), Z + N)
    f = dataclasses.replace(f, uv_floor=0.0)          # floored u v makes the statics non-variational
    cfg = two_basis_config(etol=1e-8, max_iterations=6000)
    st = run_hfb(g, f, nucleus_z=Z, nucleus_n=N, npsi_n=NPSI, config=cfg)
    assert float(st.pairing.avdeltv2[0]) > 0.5
    return g, f, cfg, st


def _r2(g):
    X, Y, Zc = g.get_meshgrid()
    return X ** 2 + Y ** 2 + Zc ** 2


def _fd(g, f, cfg, st, name, obs):
    th = rs.params_of(f, [name])
    h = 1e-4 * abs(float(th[name]))
    warm = dataclasses.replace(cfg, bcs_start=0, diag_start=0, iteranneal=0)
    v = []
    for s in (1, -1):
        fs = rs.with_params(f, {name: th[name] + s * h})
        ss = run_hfb(g, fs, nucleus_z=Z, nucleus_n=N, npsi_n=NPSI, config=warm, initial_state=st)
        v.append(obs(ss, fs))
    return [(a - b) / (2 * h) for a, b in zip(*v)]


def test_moving_window_derivatives_match_solver(o20):
    """The moving window reproduces the solver's derivatives; the frozen window misses them at the 1% level."""
    g, f, cfg, st = o20
    r2 = _r2(g)
    rn = lambda s, fs: (float(s.energies.ehfint),
                        float(jnp.sqrt(jnp.sum(r2 * s.densities.rho[0]) / jnp.sum(s.densities.rho[0]))))
    fd = _fd(g, f, cfg, st, 't0', rn)
    mv = rs.Response(st, f, g, window='moving').derivatives(['t0'], verbose=False)['t0']
    fr = rs.Response(st, f, g, window='frozen').derivatives(['t0'], verbose=False)['t0']
    assert mv['dE'] == pytest.approx(fd[0], rel=1e-5)
    assert mv['d_rn'] == pytest.approx(fd[1], rel=1e-3)
    assert abs(fr['d_rn'] / fd[1] - 1) > 1e-3
    # the solver is not variational in its window: dE is not Hellmann-Feynman
    assert abs(mv['dE'] / rs.energy_gradient(st, f, g, ['t0'])['t0'] - 1) > 1e-4


def test_paired_hessian_is_the_qrpa(o20):
    """The Thouless-coordinate vacuum reproduces the state, and its Hessian polarizability is the frozen QRPA m_-1."""
    g, f, cfg, st = o20
    Rf = rs.Response(st, f, g, window='frozen')
    H = HFBHessian(Rf)
    drho, dchi, grad = H.ground_check()
    assert drho < 1e-8 and dchi < 1e-5
    F1 = _r2(g)
    field = jnp.stack([F1, F1])
    P, _, _ = H.polarizability(field)
    assert P == pytest.approx(2 * Rf.polarizability(field)[0], rel=1e-6)


def test_paired_polarizability_gradient_matches_solver(o20):
    """dP/dt0 of the QRPA monopole polarizability of the solver's states (moving-window chain) vs re-solved states."""
    g, f, cfg, st = o20
    F1 = _r2(g)
    field = jnp.stack([F1, F1])
    H = HFBHessian(rs.Response(st, f, g, window='frozen'))
    _, dP, _ = H.polarizability_gradient(field, ['t0'], verbose=False, moving=rs.Response(st, f, g, window='moving'))
    fd = _fd(g, f, cfg, st, 't0', lambda s, fs: (2 * rs.Response(s, fs, g, window='frozen').polarizability(field)[0],))
    assert dP['t0'] == pytest.approx(fd[0], rel=1e-3)
