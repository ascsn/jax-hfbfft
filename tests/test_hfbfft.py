"""
Tests for the HFBFFT class.
"""

import pytest
import jax.numpy as jnp
from jax_hfbfft import HFBFFT, Nucleus, Force


class TestHFBFFTCreation:
    """Tests for HFBFFT instance creation."""
    
    def test_basic_creation(self):
        """Test basic HFBFFT creation."""
        nucleus = Nucleus(protons=20, neutrons=20)
        
        calc = HFBFFT(nucleus=nucleus)
        
        assert calc.nucleus.protons == 20
        assert calc.nucleus.neutrons == 20
        assert calc.force.name == "SLy4"  # Default force
    
    def test_creation_with_force(self):
        """Test creation with explicit force."""
        nucleus = Nucleus(protons=50, neutrons=82)
        force = Force.from_name("SkM*")
        
        calc = HFBFFT(nucleus=nucleus, force=force)
        
        assert calc.force.name == "SkM*"
    
    def test_creation_with_force_name(self):
        """Test creation with force name string."""
        nucleus = Nucleus(protons=50, neutrons=82)
        
        calc = HFBFFT(nucleus=nucleus, force_name="SLy4")
        
        assert calc.force.name == "SLy4"
    
    def test_grid_parameters(self):
        """Test grid parameter specification."""
        nucleus = Nucleus(protons=20, neutrons=20)
        
        calc = HFBFFT(
            nucleus=nucleus,
            nx=48, ny=48, nz=48,
            dx=1.0, dy=1.0, dz=1.0
        )
        
        assert calc.grid.nx == 48
        assert calc.grid.ny == 48
        assert calc.grid.nz == 48
        assert calc.grid.dx == 1.0
    
    def test_basis_size_auto(self):
        """Test automatic basis size calculation."""
        nucleus = Nucleus(protons=50, neutrons=82)
        
        calc = HFBFFT(nucleus=nucleus)
        
        # Basis should be at least particle numbers
        # When ipair=0 (no pairing, which is the default), basis equals particle numbers
        # When pairing is enabled, basis is larger
        assert calc._npsi[0] >= nucleus.neutrons
        assert calc._npsi[1] >= nucleus.protons
    
    def test_basis_size_explicit(self):
        """Test explicit basis size specification."""
        nucleus = Nucleus(protons=20, neutrons=20)
        
        calc = HFBFFT(nucleus=nucleus, npsi=(40, 40))
        
        assert calc._npsi == (40, 40)
        assert calc._nstmax == 80


class TestHFBFFTState:
    """Tests for HFBFFT state initialization."""
    
    def test_state_arrays_initialized(self):
        """Test that state arrays are initialized."""
        nucleus = Nucleus(protons=8, neutrons=8)
        
        calc = HFBFFT(nucleus=nucleus, nx=16, ny=16, nz=16)
        
        assert calc.state.psi is not None
        assert calc.state.rho is not None
        assert calc.state.upot is not None
    
    def test_state_array_shapes(self):
        """Test state array shapes are correct."""
        nucleus = Nucleus(protons=8, neutrons=8)
        
        calc = HFBFFT(nucleus=nucleus, nx=16, ny=16, nz=16, npsi=(16, 16))
        
        # Wavefunction shape: (nstmax, 2, nx, ny, nz)
        assert calc.state.psi.shape == (32, 2, 16, 16, 16)
        
        # Density shape: (2, nx, ny, nz)
        assert calc.state.rho.shape == (2, 16, 16, 16)
    
    def test_isospin_assignment(self):
        """Test isospin labels are assigned correctly."""
        nucleus = Nucleus(protons=8, neutrons=8)
        
        calc = HFBFFT(nucleus=nucleus, npsi=(16, 16))
        
        # First npsi[0] states should be neutrons (isospin=0)
        assert jnp.all(calc.state.isospin[:16] == 0)
        
        # Remaining states should be protons (isospin=1)
        assert jnp.all(calc.state.isospin[16:] == 1)
    
    def test_initial_occupations(self):
        """Test initial occupation numbers."""
        nucleus = Nucleus(protons=8, neutrons=8)
        
        calc = HFBFFT(nucleus=nucleus, npsi=(16, 16))
        
        # First N neutrons should be occupied
        assert jnp.sum(calc.state.wocc[:8]) == 8.0
        assert jnp.sum(calc.state.wocc[8:16]) == 0.0
        
        # First Z protons should be occupied
        assert jnp.sum(calc.state.wocc[16:24]) == 8.0
        assert jnp.sum(calc.state.wocc[24:]) == 0.0


class TestHFBFFTInitialization:
    """Tests for wavefunction initialization."""
    
    def test_harmonic_oscillator_init(self):
        """Test harmonic oscillator initialization."""
        nucleus = Nucleus(protons=8, neutrons=8)
        calc = HFBFFT(nucleus=nucleus, nx=16, ny=16, nz=16)
        
        calc.initialize_wavefunctions(method="harmonic_oscillator")
        
        # Check that wavefunctions are non-zero
        assert jnp.max(jnp.abs(calc.state.psi)) > 0
    
    def test_random_init(self):
        """Test random initialization."""
        nucleus = Nucleus(protons=8, neutrons=8)
        calc = HFBFFT(nucleus=nucleus, nx=16, ny=16, nz=16)
        
        calc.initialize_wavefunctions(method="random", seed=42)
        
        # Check that wavefunctions are non-zero
        assert jnp.max(jnp.abs(calc.state.psi)) > 0


class TestHFBFFTIndependence:
    """Tests for independence of multiple HFBFFT instances."""
    
    def test_independent_instances(self):
        """Test that different instances are independent."""
        n1 = Nucleus(protons=20, neutrons=20)
        n2 = Nucleus(protons=50, neutrons=82)
        
        calc1 = HFBFFT(nucleus=n1, nx=16, ny=16, nz=16)
        calc2 = HFBFFT(nucleus=n2, nx=16, ny=16, nz=16)
        
        # Modify calc1
        calc1.state.rho = calc1.state.rho.at[0, 0, 0, 0].set(999.0)
        
        # Check calc2 is unaffected
        assert calc2.state.rho[0, 0, 0, 0] != 999.0
    
    def test_copy_independence(self):
        """Test that copied instances are independent."""
        nucleus = Nucleus(protons=20, neutrons=20)
        
        calc1 = HFBFFT(nucleus=nucleus, nx=16, ny=16, nz=16)
        calc2 = calc1.copy()
        
        # Modify calc1
        calc1.state.rho = calc1.state.rho.at[0, 0, 0, 0].set(999.0)
        
        # Check calc2 is unaffected
        assert calc2.state.rho[0, 0, 0, 0] != 999.0
