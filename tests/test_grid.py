"""
Tests for the Grid class.
"""

import pytest
import jax.numpy as jnp
from jax_hfbfft.core.grid import Grid


class TestGrid:
    """Tests for Grid class."""
    
    def test_basic_creation(self):
        """Test basic grid creation."""
        grid = Grid.create(nx=32, ny=32, nz=32, dx=0.8, dy=0.8, dz=0.8)
        
        assert grid.nx == 32
        assert grid.ny == 32
        assert grid.nz == 32
        assert grid.dx == 0.8
        assert grid.dy == 0.8
        assert grid.dz == 0.8
    
    def test_cubic_creation(self):
        """Test cubic grid creation."""
        grid = Grid.create_cubic(n=48, d=1.0)
        
        assert grid.nx == 48
        assert grid.ny == 48
        assert grid.nz == 48
        assert grid.dx == 1.0
        assert grid.dy == 1.0
        assert grid.dz == 1.0
    
    def test_shape_property(self):
        """Test shape property."""
        grid = Grid.create(nx=24, ny=32, nz=48)
        
        assert grid.shape == (24, 32, 48)
    
    def test_volume_property(self):
        """Test volume property."""
        grid = Grid.create(nx=10, ny=10, nz=10, dx=1.0, dy=1.0, dz=1.0)
        
        assert grid.volume == 1000.0
    
    def test_integration_weight(self):
        """Test integration weight calculation."""
        grid = Grid.create(dx=0.8, dy=0.8, dz=0.8)
        
        expected_wxyz = 0.8 * 0.8 * 0.8
        assert abs(grid.wxyz - expected_wxyz) < 1e-10
    
    def test_coordinate_arrays(self):
        """Test coordinate arrays are centered at origin."""
        grid = Grid.create_cubic(n=32, d=1.0)
        
        # Check x array is centered
        assert jnp.abs(jnp.mean(grid.x)) < 1e-10
        
        # Check array length
        assert len(grid.x) == 32
        assert len(grid.y) == 32
        assert len(grid.z) == 32
    
    def test_derivative_matrices(self):
        """Test derivative matrices are created."""
        grid = Grid.create_cubic(n=16, d=0.8)
        
        # Check shapes
        assert grid.der1x.shape == (16, 16)
        assert grid.der2x.shape == (16, 16)
        assert grid.der1y.shape == (16, 16)
        assert grid.der2y.shape == (16, 16)
        assert grid.der1z.shape == (16, 16)
        assert grid.der2z.shape == (16, 16)
    
    def test_meshgrid(self):
        """Test meshgrid generation."""
        grid = Grid.create_cubic(n=8, d=1.0)
        
        xx, yy, zz = grid.get_meshgrid()
        
        assert xx.shape == (8, 8, 8)
        assert yy.shape == (8, 8, 8)
        assert zz.shape == (8, 8, 8)
    
    def test_periodic_boundary(self):
        """Test periodic boundary condition flag."""
        grid_periodic = Grid.create(periodic=True)
        grid_nonperiodic = Grid.create(periodic=False)
        
        assert grid_periodic.periodic == True
        assert grid_nonperiodic.periodic == False
    
    def test_bloch_angles(self):
        """Test Bloch angle initialization."""
        grid = Grid.create(bangx=0.5, bangy=0.25, bangz=0.0)
        
        # Angles are multiplied by pi in create()
        import numpy as np
        assert abs(grid.bangx - 0.5 * np.pi) < 1e-10
        assert abs(grid.bangy - 0.25 * np.pi) < 1e-10
        assert abs(grid.bangz) < 1e-10


class TestDerivativeOperators:
    """Tests for derivative operator accuracy."""
    
    def test_first_derivative_sine(self):
        """Test first derivative on a sine function."""
        grid = Grid.create_cubic(n=64, d=0.5)
        
        # f(x) = sin(kx), f'(x) = k*cos(kx)
        k = 2 * jnp.pi / (grid.nx * grid.dx)  # One wavelength in box
        
        f = jnp.sin(k * grid.x)
        f_prime_exact = k * jnp.cos(k * grid.x)
        f_prime_computed = jnp.dot(grid.der1x, f)
        
        # Check accuracy (tolerance depends on float32 vs float64)
        error = jnp.max(jnp.abs(f_prime_computed - f_prime_exact))
        assert error < 1e-4  # Relaxed for float32 when JAX_ENABLE_X64 is not set
    
    def test_second_derivative_sine(self):
        """Test second derivative on a sine function."""
        grid = Grid.create_cubic(n=64, d=0.5)
        
        # f(x) = sin(kx), f''(x) = -k^2*sin(kx)
        k = 2 * jnp.pi / (grid.nx * grid.dx)
        
        f = jnp.sin(k * grid.x)
        f_pp_exact = -k**2 * jnp.sin(k * grid.x)
        f_pp_computed = jnp.dot(grid.der2x, f)
        
        error = jnp.max(jnp.abs(f_pp_computed - f_pp_exact))
        assert error < 1e-4  # Relaxed for float32 when JAX_ENABLE_X64 is not set
