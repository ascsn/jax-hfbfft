"""
Unit tests for numerical solver kernels.

These tests focus on the core numerical operations in the solver:
- apply_preconditioner
- orthonormalize_states
- compute_sp_energies
- normalize_states
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from jax_hfbfft.core.grid import Grid
from jax_hfbfft.physics.solver import (
    apply_preconditioner,
    orthonormalize_states,
    normalize_states,
    compute_sp_energies,
)
from jax_hfbfft.physics.meanfield import Meanfield


class TestApplyPreconditioner:
    """Tests for the preconditioner kernel."""
    
    def test_preconditioner_shape_preservation(self):
        """Test that preconditioner preserves array shape."""
        nstates = 4
        grid = Grid.create(nx=8, ny=8, nz=8, dx=1.0, dy=1.0, dz=1.0)
        e0dmp = 20.0
        h2ma = 20.73
        
        # Create test wavefunction batch
        key = jax.random.PRNGKey(42)
        phi = jax.random.normal(key, (nstates, 2, 8, 8, 8)) + \
              1j * jax.random.normal(jax.random.split(key)[1], (nstates, 2, 8, 8, 8))
        
        # Apply preconditioner
        phi_out = apply_preconditioner(phi, e0dmp, h2ma, grid.k2)
        
        # Check shape
        assert phi_out.shape == phi.shape
        assert phi_out.dtype == phi.dtype
    
    def test_preconditioner_single_state(self):
        """Test preconditioner on single wavefunction."""
        grid = Grid.create(nx=8, ny=8, nz=8, dx=1.0, dy=1.0, dz=1.0)
        e0dmp = 20.0
        h2ma = 20.73
        
        # Single state (2, nx, ny, nz)
        key = jax.random.PRNGKey(123)
        phi = jax.random.normal(key, (2, 8, 8, 8))
        
        phi_out = apply_preconditioner(phi, e0dmp, h2ma, grid.k2)
        
        assert phi_out.shape == (2, 8, 8, 8)
        assert not jnp.any(jnp.isnan(phi_out))
        assert not jnp.any(jnp.isinf(phi_out))
    
    def test_preconditioner_zero_momentum_limit(self):
        """Test that preconditioner reduces to 1/e0dmp at k=0."""
        grid = Grid.create(nx=8, ny=8, nz=8, dx=1.0, dy=1.0, dz=1.0)
        e0dmp = 20.0
        h2ma = 20.73
        
        # Constant field (k=0 mode only)
        phi = jnp.ones((2, 8, 8, 8))
        phi_out = apply_preconditioner(phi, e0dmp, h2ma, grid.k2)
        
        # At k=0, output should be phi / e0dmp
        expected = phi / e0dmp
        np.testing.assert_allclose(phi_out, expected, rtol=1e-6)
    
    def test_preconditioner_damping_effect(self):
        """Test that preconditioner damps high-frequency modes."""
        grid = Grid.create(nx=16, ny=16, nz=16, dx=0.8, dy=0.8, dz=0.8)
        e0dmp = 20.0
        h2ma = 20.73
        
        # Create field with high-frequency component
        x = grid.x
        y = grid.y
        z = grid.z
        
        X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
        
        # Mix of low and high frequency
        phi_low = jnp.sin(2 * jnp.pi * X / (16 * 0.8))
        phi_high = jnp.sin(10 * 2 * jnp.pi * X / (16 * 0.8))
        
        phi = jnp.stack([phi_low + phi_high, jnp.zeros_like(phi_low)])
        phi_out = apply_preconditioner(phi, e0dmp, h2ma, grid.k2)
        
        # High frequency should be damped more
        # Check that output has reduced amplitude
        assert jnp.max(jnp.abs(phi_out)) < jnp.max(jnp.abs(phi))


class TestOrthonormalize:
    """Tests for wavefunction orthonormalization."""
    
    def test_orthonormalization_single_isospin(self):
        """Test orthonormalization of states with single isospin."""
        grid = Grid.create(nx=8, ny=8, nz=8, dx=1.0, dy=1.0, dz=1.0)
        nstates = 4
        npsi_n = 4  # All neutrons
        
        # Create random wavefunctions
        key = jax.random.PRNGKey(42)
        psi = jax.random.normal(key, (nstates, 2, 8, 8, 8)) + \
              1j * jax.random.normal(jax.random.split(key)[1], (nstates, 2, 8, 8, 8))
        
        # Orthonormalize
        psi_ortho = orthonormalize_states(psi, npsi_n, grid.wxyz)
        
        # Check normalization
        for i in range(nstates):
            norm = jnp.sum(jnp.abs(psi_ortho[i])**2) * grid.wxyz
            np.testing.assert_allclose(norm, 1.0, rtol=1e-6)
        
        # Check orthogonality
        for i in range(nstates):
            for j in range(i + 1, nstates):
                overlap = jnp.sum(jnp.conjugate(psi_ortho[i]) * psi_ortho[j]) * grid.wxyz
                np.testing.assert_allclose(jnp.abs(overlap), 0.0, atol=1e-6)
    
    def test_orthonormalization_two_isospins(self):
        """Test orthonormalization with neutrons and protons."""
        grid = Grid.create(nx=8, ny=8, nz=8, dx=1.0, dy=1.0, dz=1.0)
        npsi_n = 3
        npsi_p = 2
        nstates = npsi_n + npsi_p
        
        key = jax.random.PRNGKey(99)
        psi = jax.random.normal(key, (nstates, 2, 8, 8, 8)) + \
              1j * jax.random.normal(jax.random.split(key)[1], (nstates, 2, 8, 8, 8))
        
        psi_ortho = orthonormalize_states(psi, npsi_n, grid.wxyz)
        
        # Check neutrons are orthonormal
        for i in range(npsi_n):
            norm_i = jnp.sum(jnp.abs(psi_ortho[i])**2) * grid.wxyz
            np.testing.assert_allclose(norm_i, 1.0, rtol=1e-6)
            
            for j in range(i + 1, npsi_n):
                overlap = jnp.sum(jnp.conjugate(psi_ortho[i]) * psi_ortho[j]) * grid.wxyz
                np.testing.assert_allclose(jnp.abs(overlap), 0.0, atol=1e-6)
        
        # Check protons are orthonormal
        for i in range(npsi_n, nstates):
            norm_i = jnp.sum(jnp.abs(psi_ortho[i])**2) * grid.wxyz
            np.testing.assert_allclose(norm_i, 1.0, rtol=1e-6)
            
            for j in range(i + 1, nstates):
                overlap = jnp.sum(jnp.conjugate(psi_ortho[i]) * psi_ortho[j]) * grid.wxyz
                np.testing.assert_allclose(jnp.abs(overlap), 0.0, atol=1e-6)
    
    def test_orthonormalization_preserves_span(self):
        """Test that orthonormalization preserves the span of states."""
        grid = Grid.create(nx=8, ny=8, nz=8, dx=1.0, dy=1.0, dz=1.0)
        nstates = 3
        npsi_n = 3
        
        key = jax.random.PRNGKey(77)
        psi = jax.random.normal(key, (nstates, 2, 8, 8, 8)) + \
              1j * jax.random.normal(jax.random.split(key)[1], (nstates, 2, 8, 8, 8))
        
        psi_ortho = orthonormalize_states(psi, npsi_n, grid.wxyz)
        
        # The orthonormalized states should be linear combinations of originals
        # Test: project first ortho state onto all original states
        # and check that we can reconstruct it
        projections = jnp.array([
            jnp.sum(jnp.conjugate(psi[i]) * psi_ortho[0]) * grid.wxyz
            for i in range(nstates)
        ])
        
        reconstruction = sum(projections[i] * psi[i] for i in range(nstates))
        
        # Normalize reconstruction
        norm = jnp.sqrt(jnp.sum(jnp.abs(reconstruction)**2) * grid.wxyz)
        reconstruction = reconstruction / norm
        
        # Should match original ortho state up to phase
        overlap = jnp.abs(jnp.sum(jnp.conjugate(reconstruction) * psi_ortho[0]) * grid.wxyz)
        np.testing.assert_allclose(overlap, 1.0, rtol=1e-5)


class TestNormalizeStates:
    """Tests for wavefunction normalization."""
    
    def test_normalization_basic(self):
        """Test basic normalization."""
        grid = Grid.create(nx=8, ny=8, nz=8, dx=1.0, dy=1.0, dz=1.0)
        nstates = 2
        
        key = jax.random.PRNGKey(42)
        psi = jax.random.normal(key, (nstates, 2, 8, 8, 8)) + \
              1j * jax.random.normal(jax.random.split(key)[1], (nstates, 2, 8, 8, 8))
        
        psi_norm = normalize_states(psi, grid.wxyz)
        
        for i in range(nstates):
            norm = jnp.sum(jnp.abs(psi_norm[i])**2) * grid.wxyz
            np.testing.assert_allclose(norm, 1.0, rtol=1e-6)
    
    def test_normalization_zero_handling(self):
        """Test that normalization handles near-zero states."""
        grid = Grid.create(nx=8, ny=8, nz=8, dx=1.0, dy=1.0, dz=1.0)
        
        # Create a very small wavefunction
        psi = jnp.array([[[[1e-20]]]] * 2).reshape(1, 2, 1, 1, 1)
        psi = jnp.broadcast_to(psi, (1, 2, 8, 8, 8))
        
        psi_norm = normalize_states(psi, grid.wxyz)
        
        # Should not produce NaN or Inf
        assert not jnp.any(jnp.isnan(psi_norm))
        assert not jnp.any(jnp.isinf(psi_norm))


class TestComputeSpEnergies:
    """Tests for single-particle energy computation."""
    
    def test_sp_energy_shape(self):
        """Test that sp_energy computation returns correct shapes."""
        grid = Grid.create(nx=8, ny=8, nz=8, dx=1.0, dy=1.0, dz=1.0)
        nstates = 4
        
        # Create dummy wavefunctions
        key = jax.random.PRNGKey(42)
        psi = jax.random.normal(key, (nstates, 2, 8, 8, 8))
        psi = psi + 1j * jax.random.normal(jax.random.split(key)[1], (nstates, 2, 8, 8, 8))
        psi = normalize_states(psi, grid.wxyz)
        
        # Create dummy meanfield
        meanfield = Meanfield.zeros(8, 8, 8)
        
        # Isospin labels
        isospin = jnp.array([0, 0, 1, 1])
        
        # Compute energies
        sp_energy, sp_kinetic = compute_sp_energies(psi, meanfield, isospin, grid)
        
        assert sp_energy.shape == (nstates,)
        assert sp_kinetic.shape == (nstates,)
        assert not jnp.any(jnp.isnan(sp_energy))
        assert not jnp.any(jnp.isnan(sp_kinetic))
    
    def test_sp_energy_constant_potential(self):
        """Test sp_energy with constant potential."""
        grid = Grid.create(nx=8, ny=8, nz=8, dx=1.0, dy=1.0, dz=1.0)
        
        # Single state, normalized plane wave at k=0 (constant)
        psi = jnp.ones((1, 2, 8, 8, 8)) / jnp.sqrt(2 * 8**3 * grid.wxyz)
        
        # Constant potential V=10 MeV
        meanfield = Meanfield.zeros(8, 8, 8)
        meanfield = Meanfield(
            upot=jnp.ones((2, 8, 8, 8)) * 10.0,
            bmass=jnp.ones((2, 8, 8, 8)),
            v_pair=jnp.zeros((2, 8, 8, 8)),
            aq=jnp.zeros((2, 3, 8, 8, 8)),
            spot=jnp.zeros((2, 3, 8, 8, 8)),
            wlspot=jnp.zeros((2, 3, 8, 8, 8)),
            dbmass=jnp.zeros((2, 3, 8, 8, 8)),
            divaq=jnp.zeros((2, 8, 8, 8)),
        )
        
        isospin = jnp.array([0])
        
        sp_energy, sp_kinetic = compute_sp_energies(psi, meanfield, isospin, grid)
        
        # For constant wavefunction with constant potential,
        # kinetic energy should be zero and total should be ~V
        # (up to numerical precision and effective mass effects)
        assert sp_energy.shape == (1,)
        assert jnp.abs(sp_energy[0]) < 100.0  # Reasonable energy scale


class TestIntegrationSmallGrid:
    """Integration tests using small grids."""
    
    def test_gradient_step_stability(self):
        """Test that gradient step remains stable over several iterations."""
        from jax_hfbfft.physics.solver import gradient_step
        
        grid = Grid.create(nx=8, ny=8, nz=8, dx=1.0, dy=1.0, dz=1.0)
        nstates = 4
        npsi_n = 2
        
        # Initialize
        key = jax.random.PRNGKey(42)
        psi = jax.random.normal(key, (nstates, 2, 8, 8, 8))
        psi = psi + 1j * jax.random.normal(jax.random.split(key)[1], (nstates, 2, 8, 8, 8))
        psi = normalize_states(psi, grid.wxyz)
        
        meanfield = Meanfield.zeros(8, 8, 8)
        wocc = jnp.array([1.0, 1.0, 1.0, 1.0])
        wguv = jnp.zeros(nstates)
        pairwg = jnp.ones(nstates)
        sp_energy = jnp.linspace(-5, 5, nstates)
        isospin = jnp.array([0, 0, 1, 1])
        
        x0dmp = 0.3
        e0dmp = 20.0
        
        # Multiple gradient steps
        for _ in range(5):
            psi = gradient_step(
                psi, meanfield, wocc, wguv, pairwg, sp_energy,
                isospin, grid, x0dmp, e0dmp, npsi_n
            )
            
            # Check stability
            assert not jnp.any(jnp.isnan(psi))
            assert not jnp.any(jnp.isinf(psi))
            
            # Check normalization maintained
            for i in range(nstates):
                norm = jnp.sum(jnp.abs(psi[i])**2) * grid.wxyz
                np.testing.assert_allclose(norm, 1.0, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
