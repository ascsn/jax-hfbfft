"""
Tests for the J^2 (tensor) terms: the symmetric spin-current tensor
(Densities.jsym), its Hamiltonian term, and the energy decomposition.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_hfbfft.core.grid import Grid
from jax_hfbfft.physics.densities import compute_densities
from jax_hfbfft.physics.meanfield import _tensor_term

N = 9                                   # odd grid: der1 and the density derivatives agree


@pytest.fixture(scope="module")
def setup():
    g = Grid.create_cubic(n=N, d=1.0)
    X, Y, Z = g.get_meshgrid()
    env = jnp.exp(-(X**2 + Y**2 + Z**2) / 6.0)
    k = jax.random.split(jax.random.PRNGKey(3), 2)
    psi = (jax.random.normal(k[0], (1, 2, N, N, N)) + 1j * jax.random.normal(k[1], (1, 2, N, N, N))) * env
    return g, psi


def _dens(psi, g):
    one = jnp.ones(1)
    return compute_densities(psi, one, 0 * one, one, one, jnp.zeros(1, dtype=int), g, tensor=True)


def _j_full(psi, g):
    """J_mu,nu = Im psi^+ sigma_nu d_mu psi directly, (3, 3, ...)."""
    sig = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])]
    p = psi[0]
    d = [jnp.einsum('ij,sjkl->sikl', g.der1x, p), jnp.einsum('ij,skjl->skil', g.der1y, p),
         jnp.einsum('ij,sklj->skli', g.der1z, p)]
    return jnp.stack([jnp.stack([jnp.imag(jnp.einsum('s...,st,t...->...', jnp.conj(p), s, dm)) for s in sig])
                      for dm in d])


def test_symmetric_part_and_decomposition(setup):
    g, psi = setup
    d, J = _dens(psi, g), _j_full(psi, g)
    S = 0.5 * (J + jnp.swapaxes(J, 0, 1))
    js = d.jsym[0]
    for c, (m, n) in enumerate(((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))):
        assert np.allclose(js[c], S[m, n], atol=1e-10)
    full = jnp.sum(J**2)
    split = 0.5 * jnp.sum(d.sodens[0]**2) + jnp.sum(js[:3]**2) + 2 * jnp.sum(js[3:]**2)
    assert float(split) == pytest.approx(float(full), rel=1e-10)


def test_hamiltonian_term_is_the_gradient(setup):
    """h_T psi = dE/dpsi^+ for E = Int sum_mu,nu W_mu,nu J_mu,nu with a symmetric W."""
    g, psi = setup
    W = jax.random.normal(jax.random.PRNGKey(5), (6, N, N, N))

    def energy(re, im):
        js = _dens((re + 1j * im)[None], g).jsym[0]
        return g.wxyz * (jnp.sum(W[:3] * js[:3]) + 2 * jnp.sum(W[3:] * js[3:]))

    gr, gi = jax.grad(energy, argnums=(0, 1))(jnp.real(psi[0]), jnp.imag(psi[0]))
    dE_dpsi_conj = 0.5 * (gr + 1j * gi) / g.wxyz
    assert np.allclose(np.asarray(_tensor_term(psi[0], W, g)), np.asarray(dE_dpsi_conj), atol=1e-9)


def test_spin_kinetic_term_is_the_gradient(setup):
    """h psi = dE/dpsi^+ for E = Int F . T, T_nu = sum_mu (d_mu psi)^+ sigma_nu d_mu psi."""
    from jax_hfbfft.physics.meanfield import _spin_kinetic_term
    g, psi = setup
    F = jax.random.normal(jax.random.PRNGKey(7), (3, N, N, N))

    def energy(re, im):
        return g.wxyz * jnp.sum(F * _dens((re + 1j * im)[None], g).tspin[0])

    gr, gi = jax.grad(energy, argnums=(0, 1))(jnp.real(psi[0]), jnp.imag(psi[0]))
    assert np.allclose(np.asarray(_spin_kinetic_term(psi[0], F, g)),
                       np.asarray(0.5 * (gr + 1j * gi) / g.wxyz), atol=1e-9)


def _smooth_orbitals(g, n, seed):
    """n spin-polarised, band-limited orbitals: Gaussians times random complex linear polynomials."""
    X, Y, Z = g.get_meshgrid()
    basis = jnp.stack([jnp.ones_like(X), X, Y, Z, X * Y, Y * Z]) * jnp.exp(-(X**2 + Y**2 + Z**2) / 6.0)
    c = jax.random.normal(jax.random.PRNGKey(seed), (n, 2, basis.shape[0], 2))
    return jnp.einsum('nsb,b...->ns...', c[..., 0] + 1j * c[..., 1], basis)


@pytest.mark.parametrize("name", ["SLy4", "UNEDF2"])
def test_galilean_invariance(name):
    """A boost exp(ikz) changes only the kinetic energy; for UNEDF2 this needs the J^2 terms'
    time-odd partner s.T, and fails without it."""
    import dataclasses
    from jax_hfbfft.core.force import Force
    from jax_hfbfft.physics.energies import compute_integrated_energy
    g = Grid.create_cubic(n=27, d=0.75)          # band-limited and inside the box after the boost
    f = Force.from_name(name)
    psi = _smooth_orbitals(g, 2, 11)                                  # a neutron and a proton
    iso, one = jnp.array([0, 1]), jnp.ones(2)
    k = 2 * np.pi / (27 * g.dz)                                        # periodic on the grid
    boosted = psi * jnp.exp(1j * k * g.get_meshgrid()[2])

    def energy(p, drop_st=False):
        d = compute_densities(p, one, 0 * one, one, one, iso, g, tensor=True)
        if drop_st:
            d = dataclasses.replace(d, tspin=None)
        return float(compute_integrated_energy(d, f, g, use_coulomb=False).ehfint), d

    e0, d0 = energy(psi)
    kin = sum(float(f.h2m[q]) * float(jnp.sum(2 * k * d0.current[q, 2] + k**2 * d0.rho[q]) * g.wxyz)
              for q in (0, 1))
    assert energy(boosted)[0] - e0 == pytest.approx(kin, rel=1e-6)
    if name == "UNEDF2":
        broken = energy(boosted, True)[0] - energy(psi, True)[0]
        assert abs(broken - kin) > 1e-3 * abs(kin)
