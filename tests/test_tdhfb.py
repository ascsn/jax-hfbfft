"""
Tests for TDHFB (physics/tdhfb.py, core/tdhfb.py) on a small paired 20O, and
for the odd-grid kinetic operator they rely on.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_hfbfft import HFBFFT, Force, Grid, Nucleus, TDHFB
from jax_hfbfft.core.tdhfb import save_static
from jax_hfbfft.physics import tdhfb as T
from jax_hfbfft.physics.meanfield import Meanfield, apply_hamiltonian


@pytest.fixture(scope="module")
def static():
    """20O on 9^3, DDDI pairing, every state weight 1 (state_cutoff = 0)."""
    force = Force.from_name("SLy4").with_pairing(
        ipair=6, v0neut=300.0, v0prot=300.0, rho0pr=0.32, tbcs=False,
        pair_cutoff=(0.0, 0.0), state_cutoff=(0.0, 0.0))
    calc = HFBFFT(nucleus=Nucleus(protons=8, neutrons=12), force=force,
                  grid=Grid.create_cubic(n=9, d=1.4), npsi=(20, 12))
    calc.diag_start = calc.bcs_start = 30
    calc.tvaryx_0 = True
    calc.initialize_wavefunctions(method="harmonic_oscillator", radinx=2.7, radiny=2.7, radinz=2.7)
    calc.run(max_iterations=80, convergence_threshold=1e-20, print_interval=10**9, sinfo_interval=10**9)
    return calc


@pytest.fixture(scope="module")
def kicked(static):
    td = TDHFB.from_static(static)
    td.kick_quadrupole(0.01)
    return td


class TestOddGrid:

    def test_derivative_matrices(self):
        g = Grid.create_cubic(n=11, d=1.0)
        x = np.asarray(g.x)
        k = 2 * np.pi * 2 / 11.0
        assert np.allclose(np.asarray(g.der1x) @ np.sin(k * x), k * np.cos(k * x), atol=1e-10)
        assert np.allclose(np.asarray(g.der1x @ g.der1x), np.asarray(g.der2x), atol=1e-12)

    @pytest.mark.parametrize("n", [9, 10])
    def test_kinetic_is_gradient_on_odd_grids(self, n):
        """<phi|h_kin psi> = Int B grad(phi)* . grad(psi) on an odd grid; not at grid scale on an even one."""
        g = Grid.create_cubic(n=n, d=1.0)
        key = jax.random.split(jax.random.PRNGKey(0), 3)
        shp = (2, n, n, n)
        phi = jax.random.normal(key[0], shp) + 1j * jax.random.normal(key[1], shp)
        psi = jnp.roll(phi, 1, axis=1) * (1 + 0.5j)
        B = 1.0 + 0.3 * jax.random.uniform(key[2], (n, n, n))
        dB = jnp.stack([jnp.einsum('ij,jkl->ikl', g.der1x, B), jnp.einsum('ij,kjl->kil', g.der1y, B),
                        jnp.einsum('ij,klj->kli', g.der1z, B)])
        mf = dataclasses.replace(Meanfield.zeros(n, n, n), bmass=jnp.stack([B, B]), dbmass=jnp.stack([dB, dB]))
        lhs = jnp.vdot(phi, apply_hamiltonian(psi, mf, 0, g))
        grads = lambda f: [jnp.einsum('ij,sjkl->sikl', g.der1x, f), jnp.einsum('ij,skjl->skil', g.der1y, f),
                           jnp.einsum('ij,sklj->skli', g.der1z, f)]
        rhs = sum(jnp.vdot(a, B[None] * b) for a, b in zip(grads(phi), grads(psi)))
        rel = abs(lhs - rhs) / abs(rhs)
        if n % 2:
            assert rel < 1e-12
        else:
            assert rel > 1e-6        # the expanded form is not the gradient at grid-scale momenta


class TestInitialState:

    def test_kramers_pairs_and_pair_density(self, static):
        st, g = static.solver_state, static.grid
        G = T.from_static(st.psi, st.wocc, st.wguv, st.isospin, g, wstates=st.wstates, pairwg=st.pairwg)
        E = T.ebata_state(st.psi, st.wocc, st.wguv, st.isospin, g, wstates=st.wstates, pairwg=st.pairwg)
        pt = np.asarray(E.partners)
        assert np.all(pt[pt] == np.arange(len(pt))) and np.all(pt != np.arange(len(pt)))
        xi_m = T.pair_density(G.psi, G.kap, G.isospin)
        xi_c = T.pair_density_canonical(E.psi, E.kappa, E.isospin)
        assert float(jnp.linalg.norm(xi_m - xi_c)) < 1e-9 * float(jnp.linalg.norm(xi_c))
        assert float(jnp.linalg.norm(xi_c[0])) > 0, "neutrons should be paired"

    def test_soft_weights_refused(self, static):
        st = static.solver_state
        w = np.asarray(st.wstates).copy()
        w[-1] = 0.5
        with pytest.raises(ValueError):
            T.from_static(st.psi, st.wocc, st.wguv, st.isospin, static.grid, wstates=w)

    def test_purified(self, kicked):
        assert T.purity(kicked.state) < 1e-12


class TestGlobalScheme:

    def test_conservation(self, static):
        td = TDHFB.from_static(static)
        td.kick_quadrupole(0.01)
        h = td.run(4)
        assert abs(h[-1]['N'] - h[0]['N']) < 1e-9
        assert abs(h[-1]['Z'] - h[0]['Z']) < 1e-9
        assert h[-1]['purity'] < 1e-11
        assert abs(h[-1]['E'] - h[0]['E']) < 1e-3
        assert h[-1]['Q20'] != h[0]['Q20']

    def test_exact_predictor_close(self, kicked):
        a = T.make_step(kicked.state, kicked.grid, kicked.force, kicked.cs, n_comp=4, pred_n_comp=1)(kicked.state, 0.05)
        b = T.make_step(kicked.state, kicked.grid, kicked.force, kicked.cs, n_comp=4, pred_n_comp=None)(kicked.state, 0.05)
        ra = T.fields(a, kicked.grid, kicked.force, kicked.cs)[0].rho
        rb = T.fields(b, kicked.grid, kicked.force, kicked.cs)[0].rho
        assert float(jnp.linalg.norm(ra - rb) / jnp.linalg.norm(rb)) < 1e-8

    @pytest.mark.parametrize("midpoint, order", [(False, 1), (True, 2)])
    def test_energy_error_order(self, kicked, midpoint, order):
        """strang + Euler complement rotations: energy error O(dt); with midpoint rotations O(dt^2)."""
        S0, g, f, cs = kicked.state, kicked.grid, kicked.force, kicked.cs
        E0 = T.energy(S0, g, f, cs)[0]
        err = []
        for dt in (0.2, 0.1):
            step = T.make_step(S0, g, f, cs, n_comp=1, strang=True, midpoint=midpoint)
            S = S0
            for _ in range(int(round(2.0 / dt))):
                S = step(S, dt)
            err.append(abs(T.energy(S, g, f, cs)[0] - E0))
        p = np.log2(err[0] / err[1])          # measured 0.96 (Euler) and 2.31 (midpoint)
        assert (abs(p - 1) < 0.3) if order == 1 else (p > 1.8)

    def test_local_continuity_at_floor(self, kicked):
        c = kicked.continuity()
        assert c['n'] < 3 * c['floor_n']
        assert c['p'] < 3 * c['floor_p']

    def test_checkpoint_roundtrip(self, kicked, tmp_path):
        kicked.save(str(tmp_path / "ck.npz"))
        td = TDHFB(kicked.state, kicked.grid, kicked.force).load(str(tmp_path / "ck.npz"))
        assert np.allclose(np.asarray(td.state.kap), np.asarray(kicked.state.kap))
        assert td.time == kicked.time


class TestEbata:

    def test_runs_and_conserves_number(self, static):
        td = TDHFB.from_static(static, scheme='ebata')
        td.kick_quadrupole(0.01)
        h = td.run(4)
        assert td.scheme == 'ebata'
        assert abs(h[-1]['N'] - h[0]['N']) < 1e-8


class TestCollision:

    def test_symmetric_composite(self, static, tmp_path):
        path = str(tmp_path / "O20.npz")
        save_static(path, static.solver_state, static.grid, v0=300.0, ipair=6, rho0pr=0.32)
        td = TDHFB.collision(path, nz=25, sep=14.0, e_asym=20.0, phase=np.pi)
        rec = td.observables()
        st = static.solver_state
        assert td.state.nstates == 2 * st.psi.shape[0]
        assert abs(rec['N'] - 2 * float(np.sum(np.asarray(st.wocc)[np.asarray(st.isospin) == 0]))) < 1e-8
        assert T.purity(td.state) < 1e-10
        assert td.info['e_kin'] + td.info['e_coul'] == pytest.approx(20.0)


class TestRelax:

    def test_relax_lowers_energy_and_residual(self, static):
        td = TDHFB.from_static(static)
        E0 = T.energy(td.state, td.grid, td.force, td.cs)[0]
        occ0, r0 = T.stationarity(td.state, td.grid, td.force, td.cs)
        S, _ = T.relax(td.state, td.grid, td.force, td.cs, n_iter=30)
        E1 = T.energy(S, td.grid, td.force, td.cs)[0]
        occ1, r1 = T.stationarity(S, td.grid, td.force, td.cs)
        assert E1 < E0
        assert np.max(r1[np.minimum(occ1, 1 - occ1) > 1e-3]) < np.max(r0[np.minimum(occ0, 1 - occ0) > 1e-3])
        assert T.purity(S) < 1e-10
        assert sum(T.particle_numbers(S)) == pytest.approx(sum(T.particle_numbers(td.state)), abs=1e-6)


class TestPairCutoff:

    def test_far_cutoff_is_bare(self, kicked):
        f = kicked.force.with_pairing(pair_kcut=50.0)
        a = T.energy(kicked.state, kicked.grid, kicked.force, kicked.cs)
        b = T.energy(kicked.state, kicked.grid, f, kicked.cs)
        assert abs(a[0] - b[0]) < 1e-10 and np.allclose(a[1], b[1], atol=1e-10)

    def test_cutoff_conserves(self, static):
        td = TDHFB.from_static(static, force=static.solver_force.with_pairing(pair_kcut=1.5, pair_kwidth=0.1))
        Ebare = T.energy(td.state, td.grid, static.solver_force, td.cs)[1][0]
        td.kick_quadrupole(0.01)
        h = td.run(4)
        assert h[0]['Epair_n'] < Ebare                 # the filter removes pairing from high-k components
        assert abs(h[-1]['N'] - h[0]['N']) < 1e-9
        assert abs(h[-1]['E'] - h[0]['E']) < 1e-3
        S, _ = T.relax(td.state, td.grid, td.force, td.cs, n_iter=20)
        assert T.energy(S, td.grid, td.force, td.cs)[0] < h[-1]['E']
