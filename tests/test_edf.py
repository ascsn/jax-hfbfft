"""EDF coupling-constant parametrization and the autodiff functional layer (physics/edf.py)."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_hfbfft import Force
from jax_hfbfft.core.grid import Grid
from jax_hfbfft.physics import response as rs
from jax_hfbfft.physics.coulomb import solve_poisson
from jax_hfbfft.physics.densities import compute_densities
from jax_hfbfft.physics.energies import compute_integrated_energy
from jax_hfbfft.physics.meanfield import compute_skyrme_meanfield, compute_laplacian
from jax_hfbfft.physics.solver import run_hfb, apply_cm_correction, two_basis_config

FIELDS = ('upot', 'bmass', 'divaq', 'v_pair', 'aq', 'spot', 'wlspot', 'dbmass')


@pytest.fixture(scope="module")
def o16():
    g = Grid.create_cubic(n=15, d=1.0)
    f = apply_cm_correction(Force.from_name("SLy4").with_pairing(ipair=0), 16)
    st = run_hfb(g, f, nucleus_z=8, nucleus_n=8, npsi_n=10, config=two_basis_config(etol=1e-7, max_iterations=2000))
    return g, f, st


def _odd_densities(st, g, tensor=False, chi_scale=0.0):
    """Densities of the state's orbitals boosted and spin-twisted, so that j, s (and T) are non-zero."""
    X, Y, Z = g.get_meshgrid()
    ph = jnp.exp(1j * (0.3 * X - 0.2 * Z))
    c, s_ = jnp.cos(0.1 * Y), jnp.sin(0.1 * Y)
    psi = st.psi * ph[None, None]
    psi = jnp.stack([c * psi[:, 0] - s_ * psi[:, 1], s_ * psi[:, 0] + c * psi[:, 1]], axis=1)
    wguv = jnp.full_like(st.wocc, chi_scale)          # a fake pair density for the pairing channel
    return compute_densities(psi, st.wocc, wguv, jnp.ones_like(st.wocc), st.wstates, st.isospin, g, tensor=tensor)


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def test_coupling_roundtrip():
    """t/x -> C -> t/x and the b coefficients the hand-coded fields use are reproduced."""
    f = Force.from_name("SLy4")
    fc = f.to_couplings()
    for k in ('b0', 'b0p', 'b1', 'b1p', 'b2', 'b2p', 'b3', 'b3p', 'b4', 'b4p'):
        assert float(getattr(fc, k)) == pytest.approx(float(getattr(f, k)), rel=1e-12, abs=1e-12)
    for k in ('t0', 't1', 't2', 't3', 't4', 'x0', 'x1', 'x2', 'x3'):
        assert float(getattr(fc, k)) == pytest.approx(float(getattr(f, k)), rel=1e-10, abs=1e-12)
    assert float(fc.Crho0) == pytest.approx(3 / 8 * f.t0)
    f2 = Force.from_couplings("SLy4C", f.couplings(), power=f.power)
    assert float(f2.b3p) == pytest.approx(float(f.b3p), rel=1e-12)


def test_coupling_hellmann_feynman(o16):
    """dE/dC at fixed densities is the integral of the density product the coupling multiplies."""
    g, f, st = o16
    fc = f.to_couplings()
    d = rs.energy_gradient(st, fc, g, names=['Crho0', 'Crho1', 'Ctau0', 'Cdrho1'])
    rho0, rho1 = st.densities.rho[0] + st.densities.rho[1], st.densities.rho[1] - st.densities.rho[0]
    tau0 = st.densities.tau[0] + st.densities.tau[1]
    w = g.wxyz
    assert d['Crho0'] == pytest.approx(float(w * jnp.sum(rho0 ** 2)), rel=1e-10)
    assert d['Crho1'] == pytest.approx(float(w * jnp.sum(rho1 ** 2)), rel=1e-8)
    assert d['Ctau0'] == pytest.approx(float(w * jnp.sum(rho0 * tau0)), rel=1e-10)
    assert d['Cdrho1'] == pytest.approx(float(w * jnp.sum(rho1 * compute_laplacian(rho1, g))), rel=1e-8)


def test_coupling_chain_rule(o16):
    """dE/dt0 = sum_C dE/dC dC/dt0 (dCrho0/dt0 = 3/8, dCrho1/dt0 = -(1/2 + x0)/4)."""
    g, f, st = o16
    dt = rs.energy_gradient(st, f, g, names=['t0'])['t0']
    dc = rs.energy_gradient(st, f.to_couplings(), g, names=['Crho0', 'Crho1'])
    assert dt == pytest.approx(3 / 8 * dc['Crho0'] - (0.5 + f.x0) / 4 * dc['Crho1'], rel=1e-10)


@pytest.mark.parametrize("case", ["ground", "time_odd", "paired"])
def test_autodiff_fields_equal_handcoded(o16, case):
    """Every field of the autodiff Skyrme functional equals the hand-coded field on the same densities."""
    g, f, st = o16
    if case == "ground":
        d, ff = st.densities, f
    elif case == "time_odd":
        d, ff = _odd_densities(st, g), f.with_time_odd_from_skyrme()
    else:
        d, ff = _odd_densities(st, g, chi_scale=0.2), f.with_pairing_type('mixed', v0=300.0)
    vc = solve_poisson(d.rho[1], st.coulomb_solver, g)
    hand = compute_skyrme_meanfield(d, ff, g, coulomb_potential=vc)
    ad = compute_skyrme_meanfield(d, ff.with_functional('skyrme_ad'), g, coulomb_potential=vc)
    for k in FIELDS:
        if float(np.max(np.abs(np.asarray(getattr(hand, k))))) > 0:
            assert _rel(getattr(ad, k), getattr(hand, k)) < 1e-9, k
    assert float(ad.ecorrp) == pytest.approx(float(hand.ecorrp), rel=1e-9, abs=1e-12)


def test_autodiff_tensor_fields(o16):
    """J^2 forces: the symmetric spin-current and spin-kinetic fields agree too."""
    g, _, st = o16
    f = apply_cm_correction(Force.from_name("SLy5").with_pairing(ipair=0), 16)
    d = _odd_densities(st, g, tensor=True)
    hand = compute_skyrme_meanfield(d, f, g)
    ad = compute_skyrme_meanfield(d, f.with_functional('skyrme_ad'), g)
    for k in FIELDS + ('wtens', 'fspin'):
        if float(np.max(np.abs(np.asarray(getattr(hand, k))))) > 0:
            assert _rel(getattr(ad, k), getattr(hand, k)) < 1e-9, k


def test_autodiff_energy_and_koopman(o16):
    """Integrated energy and the Koopman rearrangement R = e3corr + ecorrp + ecorc - E_pair agree."""
    g, f, st = o16
    ff = f.with_pairing_type('mixed', v0=300.0)
    d = _odd_densities(st, g, chi_scale=0.2)
    vc = solve_poisson(d.rho[1], st.coulomb_solver, g)
    mf = compute_skyrme_meanfield(d, ff, g, coulomb_potential=vc)
    epair = g.wxyz * jnp.sum(mf.v_pair * d.chi, axis=(1, 2, 3))
    z = jnp.zeros_like(st.wocc)
    kw = dict(coulomb_potential=vc, mass_number=16, wocc=z, wstates=z, sp_kinetic=z, sp_energy=z)
    eh = compute_integrated_energy(d, ff, g, pairing_energy=epair, ecorrp=mf.ecorrp, **kw)
    ea = compute_integrated_energy(d, ff.with_functional('skyrme_ad'), g, **kw)
    assert float(ea.ehfint) == pytest.approx(float(eh.ehfint), rel=1e-12)
    assert float(ea.ehf) == pytest.approx(float(eh.ehf), rel=1e-9, abs=1e-9)
    np.testing.assert_allclose(np.asarray(ea.epair), np.asarray(eh.epair), rtol=1e-10)


def test_autodiff_solve_equals_handcoded(o16):
    """
    The hand-coded ground state is already a fixed point of the autodiff functional: a solve started from it stays
    put. An independent cold solve agrees to its convergence tolerance (two solves to etol 1e-7 differ by ~5e-5 in
    the density, the convergence floor, so that comparison is looser).
    """
    g, f, st = o16
    cfg = two_basis_config(etol=1e-7, max_iterations=2000)
    warm = dataclasses.replace(cfg, bcs_start=0, diag_start=0, iteranneal=0)
    fa = f.with_functional('skyrme_ad')
    st_w = run_hfb(g, fa, nucleus_z=8, nucleus_n=8, npsi_n=10, config=warm, initial_state=st)
    assert float(st_w.energies.ehfint) == pytest.approx(float(st.energies.ehfint), abs=1e-7)
    np.testing.assert_allclose(np.asarray(st_w.densities.rho), np.asarray(st.densities.rho), atol=5e-7)
    st_c = run_hfb(g, fa, nucleus_z=8, nucleus_n=8, npsi_n=10, config=cfg)
    assert float(st_c.energies.ehfint) == pytest.approx(float(st.energies.ehfint), abs=1e-5)
    np.testing.assert_allclose(np.asarray(st_c.densities.rho), np.asarray(st.densities.rho), atol=5e-5)


def test_fayans_loads_and_differentiates(o16):
    """Fayans forces load as an autodiff functional and its parameters are differentiable (Hellmann-Feynman)."""
    g, _, st = o16
    fy = Force.from_name("FyDr").with_pairing(ipair=0)
    assert fy.functional == 'fayans'
    d = rs.energy_gradient(st, fy, g, names=['fy.avp', 'fy.asp'])
    assert np.isfinite(d['fy.avp']) and d['fy.avp'] != 0.0
    # avp multiplies a volume term linear in it: dE/d avp = (E(avp + h) - E(avp - h)) / 2h exactly
    th = rs.params_of(fy, ['fy.avp'])
    E = lambda a: float(rs.total_energy(st.densities, rs.with_params(fy, {'fy.avp': a}), g, st.coulomb_solver, 16))
    h = 1e-3
    assert d['fy.avp'] == pytest.approx((E(th['fy.avp'] + h) - E(th['fy.avp'] - h)) / (2 * h), rel=1e-6)


def test_user_defined_functional(o16):
    """
    A new functional in a few lines (Skyrme plus a second density-dependent term with its own exponent): its fields
    are the functional derivatives of its energy (directional finite differences) and its new parameters are
    differentiable (Hellmann-Feynman).
    """
    from jax_hfbfft.physics import edf

    @edf.register('test_skyrme_dd2')
    def skyrme_dd2(d, f, grid, use_coulomb):
        e, pair = edf.skyrme(d, f, grid, use_coulomb)
        rho0, rho1 = d.rho[0] + d.rho[1], d.rho[1] - d.rho[0]
        rb = edf._pos(rho0, edf.FLOOR) ** f.extra['beta']
        return e + f.extra['C0B'] * rb * rho0 ** 2 + f.extra['C1B'] * rb * rho1 ** 2, pair

    g, f, st = o16
    fu = dataclasses.replace(f, functional='test_skyrme_dd2', extra={'C0B': -100.0, 'C1B': 50.0, 'beta': 1.0 / 3.0})
    d = st.densities
    mf = compute_skyrme_meanfield(d, fu, g, use_coulomb=False)
    X, Y, Z = g.get_meshgrid()
    E = lambda dd: float(edf._energy(dd, fu, g, False))
    for name, field, pert in (('rho', mf.upot, 0.02 * d.rho * jnp.cos(0.3 * X)[None]),
                              ('tau', mf.bmass, 0.02 * d.tau * jnp.sin(0.2 * Y + 0.1)[None])):
        eps = 1e-4
        fd = (E(dataclasses.replace(d, **{name: getattr(d, name) + eps * pert}))
              - E(dataclasses.replace(d, **{name: getattr(d, name) - eps * pert}))) / (2 * eps)
        assert float(g.wxyz * jnp.sum(field * pert)) == pytest.approx(fd, rel=1e-7), name
    dE = rs.energy_gradient(st, fu, g, names=['extra.C0B', 'extra.beta'])
    rho0 = d.rho[0] + d.rho[1]
    assert dE['extra.C0B'] == pytest.approx(float(g.wxyz * jnp.sum(jnp.maximum(rho0, 1e-25) ** (1 / 3) * rho0 ** 2)),
                                            rel=1e-10)
    assert np.isfinite(dE['extra.beta'])


def test_regularized_coupling():
    """Bulgac-Yu: identity without a cutoff; otherwise 0 < v_eff < v at surface densities, weaker for a larger
    pairing space (E_c), and 1/v_eff - 1/v = K independent of v (a renormalisation of the coupling only).
    Where E_c is not large against the local Fermi energy (k_F/k_c > 0.83) K < 0 and v_eff > v."""
    from jax_hfbfft.physics.edf import regularized_coupling
    f = Force.from_name("SLy4")
    rho = jnp.stack([jnp.linspace(1e-4, 0.05, 50), jnp.linspace(1e-4, 0.04, 50)])
    v = jnp.full_like(rho, 300.0)
    assert jnp.array_equal(regularized_coupling(v, rho, f), v)
    veff = [regularized_coupling(v, rho, dataclasses.replace(f, pair_reg_ecut=e)) for e in (20.0, 60.0)]
    assert bool(jnp.all((veff[0] > 0) & (veff[0] < v) & (veff[1] < veff[0])))
    f20 = dataclasses.replace(f, pair_reg_ecut=20.0)
    np.testing.assert_allclose(1 / regularized_coupling(2 * v, rho, f20) - 1 / (2 * v), 1 / veff[0] - 1 / v, rtol=1e-10)
