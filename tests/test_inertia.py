"""Tests for the cranking inertia (physics/inertia.py)."""

import dataclasses
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from jax_hfbfft.core.grid import Grid
from jax_hfbfft.physics.inertia import cranking_inertia


def test_hartree_fock_limit():
    """Occupied 1s (e = -10) and empty 1p_z (e = -2): M^(k) = 2 |<p|z|s>|^2 / (e_p - e_s)^k."""
    g = Grid.create_cubic(n=15, d=0.8)
    X, Y, Z = g.get_meshgrid()
    gauss = jnp.exp(-(X**2 + Y**2 + Z**2) / 4.0)
    norm = lambda f: f / jnp.sqrt(jnp.sum(f**2) * g.wxyz)
    s, p = norm(gauss), norm(Z * gauss)
    psi = jnp.stack([jnp.stack([s, 0 * s]), jnp.stack([p, 0 * p])]).astype(complex)
    state = SimpleNamespace(psi=psi, wocc=jnp.array([1.0, 0.0]), sp_energy=jnp.array([-10.0, -2.0]),
                            deltaf=jnp.zeros(2), isospin=jnp.array([0, 0]),
                            pairing=SimpleNamespace(eferm=jnp.zeros(2)))
    out = cranking_inertia(state, g, [Z], complement=False)
    q2 = float(jnp.sum(s * Z * p) * g.wxyz) ** 2
    for k, M in ((1, out['M1']), (2, out['M2']), (3, out['M3'])):
        assert float(M[0, 0]) == pytest.approx(2 * q2 / 8.0**k, rel=1e-10)
    assert float(out['mass'][0, 0]) == pytest.approx((2 * q2 / 512) / (2 * q2 / 8) ** 2, rel=1e-10)


def test_complement_makes_the_moments_basis_independent():
    """16O: with the Sternheimer complement, 8 orbitals (just the occupied ones) and 16 give
    the same energy moments; the explicit sum alone misses most of M^(1)."""
    from jax_hfbfft import Force
    from jax_hfbfft.physics.solver import SolverConfig, run_hfb, apply_cm_correction
    g = Grid.create_cubic(n=16, d=1.0)
    force = apply_cm_correction(Force.from_name("SLy4").with_pairing(ipair=0), 16)
    X, Y, Z = g.get_meshgrid()
    q20 = [2 * Z**2 - X**2 - Y**2]
    cfg = SolverConfig(max_iterations=600, diag_start=30, bcs_start=30, tvaryx_0=True,
                       precond_floor=1e-8, verbose=False)
    full = [cranking_inertia(run_hfb(g, force, nucleus_z=8, nucleus_n=8, npsi_n=n, config=cfg), g, q20)
            for n in (8, 16)]
    for k in ('M1', 'M2', 'M3'):
        assert float(full[0][k][0, 0]) == pytest.approx(float(full[1][k][0, 0]), rel=1e-3)
    explicit = cranking_inertia(run_hfb(g, force, nucleus_z=8, nucleus_n=8, npsi_n=16, config=cfg), g, q20,
                                complement=False)
    assert float(explicit['M1'][0, 0]) < 0.5 * float(full[1]['M1'][0, 0])


def test_moment_of_inertia_of_a_prolate_nucleus():
    """20Ne (prolate about z): J_x = J_y, J_z = 0, the same J with 10 or 20 orbitals, and after
    mixing the occupied and the empty orbitals among themselves (the state is unchanged; a
    weakly paired solve leaves them mixed, which made J 2.5x too small)."""
    from jax_hfbfft import Constraint, Force
    from jax_hfbfft.physics.inertia import moment_of_inertia
    from jax_hfbfft.physics.solver import SolverConfig, run_hfb, apply_cm_correction
    g = Grid.create_cubic(n=16, d=1.0)
    f = apply_cm_correction(Force.from_name("SLy4").with_pairing(ipair=0), 20)
    cfg = lambda n: SolverConfig(max_iterations=n, diag_start=30, bcs_start=30, tvaryx_0=True,
                                 precond_floor=1e-8, verbose=False)

    def solve(npsi):
        st = run_hfb(g, f, nucleus_z=10, nucleus_n=10, npsi_n=npsi, config=cfg(300),
                     constraint=Constraint.prolate(20, beta2=0.3, principal_axes=True, damprad=5.0))
        return run_hfb(g, f, nucleus_z=10, nucleus_n=10, npsi_n=npsi, config=cfg(800), initial_state=st)

    st = solve(10)
    J = {a: moment_of_inertia(st, g, a)['J'] for a in 'xyz'}
    assert J['x'] == pytest.approx(J['y'], rel=1e-3) and abs(J['z']) < 1e-3 * J['x']
    st20 = solve(20)
    assert moment_of_inertia(st20, g, 'x')['J'] == pytest.approx(J['x'], rel=1e-3)
    rng, psi = np.random.default_rng(0), st20.psi
    iso, occ = np.asarray(st20.isospin), np.asarray(st20.wocc) > 0.5
    for idx in [np.nonzero((iso == q) & (occ == o))[0] for q in (0, 1) for o in (True, False)]:
        U, _ = np.linalg.qr(rng.normal(size=(len(idx),) * 2) + 1j * rng.normal(size=(len(idx),) * 2))
        psi = psi.at[idx].set(jnp.einsum('kl,ksxyz->lsxyz', U, psi[idx]))
    mixed = dataclasses.replace(st20, psi=psi)
    assert moment_of_inertia(mixed, g, 'x')['J'] == pytest.approx(moment_of_inertia(st20, g, 'x')['J'], rel=1e-6)
