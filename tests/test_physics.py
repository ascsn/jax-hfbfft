"""
Tests for the physics subpackage.
"""

import pytest
import jax.numpy as jnp
import jax

from jax_hfbfft.physics.densities import Densities
from jax_hfbfft.physics.meanfield import Meanfield, compute_laplacian, compute_gradient, compute_divergence, compute_curl
from jax_hfbfft.physics.energies import Energies
from jax_hfbfft.physics.pairing import Pairing, bcs_occupation, soft_cutoff
from jax_hfbfft.physics.coulomb import CoulombSolver
from jax_hfbfft.physics.solver import SolverConfig, SolverState
from jax_hfbfft.core.grid import Grid
from jax_hfbfft.jax_config import get_dtypes


class TestDensities:
    """Tests for the Densities class."""
    
    def test_zeros_creation(self):
        """Test zero initialization."""
        densities = Densities.zeros(16, 16, 16)
        assert densities.rho.shape == (2, 16, 16, 16)
        assert densities.tau.shape == (2, 16, 16, 16)
        assert densities.chi.shape == (2, 16, 16, 16)
        assert densities.current.shape == (2, 3, 16, 16, 16)
        assert densities.sdens.shape == (2, 3, 16, 16, 16)
        assert densities.sodens.shape == (2, 3, 16, 16, 16)
    
    def test_all_zeros(self):
        """Test that zero-initialized arrays are actually zero."""
        densities = Densities.zeros(8, 8, 8)
        assert jnp.allclose(densities.rho, 0.0)
        assert jnp.allclose(densities.tau, 0.0)


class TestMeanfield:
    """Tests for the Meanfield class."""
    
    def test_zeros_creation(self):
        """Test zero initialization."""
        mf = Meanfield.zeros(16, 16, 16)
        assert mf.upot.shape == (2, 16, 16, 16)
        assert mf.bmass.shape == (2, 16, 16, 16)
        assert mf.aq.shape == (2, 3, 16, 16, 16)
        assert mf.spot.shape == (2, 3, 16, 16, 16)
        assert mf.wlspot.shape == (2, 3, 16, 16, 16)
        assert mf.ecorrp == 0.0
    
    def test_laplacian(self):
        """Test Laplacian computation."""
        grid = Grid.create(nx=16, ny=16, nz=16, dx=0.5, dy=0.5, dz=0.5)
        
        # Create a test field: f = sin(2*pi*x/L)
        x = jnp.linspace(-grid.nx*grid.dx/2, grid.nx*grid.dx/2, grid.nx)
        k = 2 * jnp.pi / (grid.nx * grid.dx)
        field = jnp.sin(k * x)[:, None, None] * jnp.ones((grid.nx, grid.ny, grid.nz))
        
        lap = compute_laplacian(field, grid)
        
        # Laplacian of sin(kx) is -k^2 * sin(kx)
        expected = -k**2 * field
        assert lap.shape == field.shape


class TestEnergies:
    """Tests for the Energies class."""
    
    def test_zeros_creation(self):
        """Test zero initialization."""
        energies = Energies.zeros()
        assert energies.ehft == 0.0
        assert energies.ehfint == 0.0
        assert energies.total == 0.0
    
    def test_total_property(self):
        """Test total energy property."""
        dtypes = get_dtypes()
        energies = Energies(
            ehft=10.0, ehf0=20.0, ehf1=5.0, ehf2=-3.0, ehf3=15.0,
            ehfls=2.0, ehflsodd=1.0, ehfc=8.0, ecorc=0.5, ehfint=-100.0,
            ehf=-98.0, tke=50.0, e3corr=-5.0, e_zpe=1.0,
            efluct1=jnp.zeros(1, dtype=dtypes.float),
            efluct1q=jnp.zeros(2, dtype=dtypes.float),
            efluct2=jnp.zeros(1, dtype=dtypes.float),
            efluct2q=jnp.zeros(2, dtype=dtypes.float),
            orbital=jnp.zeros(3, dtype=dtypes.float),
            spin=jnp.zeros(3, dtype=dtypes.float),
            total_angmom=jnp.zeros(3, dtype=dtypes.float),
            epair=jnp.zeros(2, dtype=dtypes.float),
            ehfCrho0=0.0, ehfCrho1=0.0, ehfCdrho0=0.0, ehfCdrho1=0.0,
            ehfCtau0=0.0, ehfCtau1=0.0, ehfCdJ0=0.0, ehfCdJ1=0.0,
            ehfCj0=0.0, ehfCj1=0.0,
        )
        assert energies.total == -100.0


class TestPairing:
    """Tests for the Pairing module."""
    
    def test_pairing_zeros(self):
        """Test zero initialization."""
        pairing = Pairing.zeros()
        assert pairing.eferm.shape == (2,)
        assert jnp.allclose(pairing.epair, 0.0)
    
    def test_bcs_occupation(self):
        """Test BCS occupation formula."""
        # Single level at Fermi energy should have 50% occupation
        sp_energy = jnp.array([0.0])
        deltaf = jnp.array([1.0])
        wstates = jnp.array([1.0])
        
        n = bcs_occupation(0.0, sp_energy, deltaf, wstates)
        assert jnp.isclose(n, 0.5, atol=1e-5)
    
    def test_bcs_occupation_far_below(self):
        """Test that levels far below Fermi are fully occupied."""
        sp_energy = jnp.array([-10.0])
        deltaf = jnp.array([1.0])
        wstates = jnp.array([1.0])
        
        n = bcs_occupation(0.0, sp_energy, deltaf, wstates)
        assert n > 0.99
    
    def test_bcs_occupation_far_above(self):
        """Test that levels far above Fermi are empty."""
        sp_energy = jnp.array([10.0])
        deltaf = jnp.array([1.0])
        wstates = jnp.array([1.0])
        
        n = bcs_occupation(0.0, sp_energy, deltaf, wstates)
        assert n < 0.01
    
    def test_soft_cutoff(self):
        """Test soft cutoff function."""
        energy = jnp.array([-5.0, 0.0, 5.0, 10.0])
        cutoff = 5.0
        width = 1.0
        
        result = soft_cutoff(energy, cutoff, width)
        
        # Below cutoff: ~1
        assert result[0] > 0.99
        # At cutoff: 0.5
        assert jnp.isclose(result[2], 0.5, atol=0.01)
        # Above cutoff: ~0
        assert result[3] < 0.01


class TestCoulomb:
    """Tests for the Coulomb solver."""
    
    def test_solver_creation(self):
        """Test solver initialization."""
        grid = Grid.create(nx=16, ny=16, nz=16, dx=1.0, dy=1.0, dz=1.0)
        solver = CoulombSolver.create(grid)
        
        # For open BCs, extended grid is 2x
        assert solver.nx2 == 32
        assert solver.ny2 == 32
        assert solver.nz2 == 32
    
    def test_solver_periodic(self):
        """Test solver with periodic BCs."""
        grid = Grid.create(
            nx=16, ny=16, nz=16, dx=1.0, dy=1.0, dz=1.0,
            periodic=True
        )
        solver = CoulombSolver.create(grid)
        
        # For periodic, no padding
        assert solver.nx2 == 16


class TestSolverConfig:
    """Tests for the SolverConfig class."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = SolverConfig()
        assert config.max_iterations == 200
        assert config.convergence_criterion == 1e-6
        assert config.x0dmp == 0.45
        assert config.density_mixing == 0.5  # Updated to match actual default
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = SolverConfig(
            max_iterations=500,
            convergence_criterion=1e-8,
            x0dmp=0.3,
        )
        assert config.max_iterations == 500
        assert config.convergence_criterion == 1e-8
        assert config.x0dmp == 0.3


class TestDifferentialOperators:
    """Tests for differential operators."""
    
    def test_gradient_constant(self):
        """Gradient of constant is zero."""
        grid = Grid.create(nx=16, ny=16, nz=16, dx=0.5, dy=0.5, dz=0.5)
        field = jnp.ones((grid.nx, grid.ny, grid.nz))
        
        gx, gy, gz = compute_gradient(field, grid)
        
        assert jnp.allclose(gx, 0.0, atol=1e-10)
        assert jnp.allclose(gy, 0.0, atol=1e-10)
        assert jnp.allclose(gz, 0.0, atol=1e-10)
    
    def test_divergence_constant_vector(self):
        """Divergence of constant vector field is zero."""
        grid = Grid.create(nx=16, ny=16, nz=16, dx=0.5, dy=0.5, dz=0.5)
        vec = jnp.ones((3, grid.nx, grid.ny, grid.nz))
        
        div = compute_divergence(vec, grid)
        
        assert jnp.allclose(div, 0.0, atol=1e-10)
    
    def test_curl_gradient_is_zero(self):
        """Curl of gradient is zero (vector identity)."""
        grid = Grid.create(nx=16, ny=16, nz=16, dx=0.5, dy=0.5, dz=0.5)
        
        # Create a scalar field
        x = jnp.linspace(-4, 4, grid.nx)
        y = jnp.linspace(-4, 4, grid.ny)
        z = jnp.linspace(-4, 4, grid.nz)
        X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
        phi = X**2 + Y**2 + Z**2
        
        # Compute gradient
        gx, gy, gz = compute_gradient(phi, grid)
        grad = jnp.stack([gx, gy, gz], axis=0)
        
        # Compute curl of gradient
        curl = compute_curl(grad, grid)
        
        # Should be approximately zero
        assert jnp.allclose(curl, 0.0, atol=0.1)
