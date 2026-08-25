"""
JAX-HFBFFT: A JAX-based Hartree-Fock-Bogoliubov solver with Fast Fourier Transform.

This package provides object-oriented tools for nuclear structure calculations
using the Hartree-Fock-Bogoliubov (HFB) method with Skyrme-type effective interactions.

Main Classes:
    HFBFFT: Central calculation object encapsulating all state and methods
    Force: Nuclear force/interaction definitions (Skyrme parameterizations)
    Constraint: Constraint configuration for constrained HFB calculations
    Nucleus: Nuclear system specification (proton/neutron numbers)

Example:
    >>> from jax_hfbfft import HFBFFT, Force, Nucleus
    >>> 
    >>> # Create a calculation for Sn-132
    >>> nucleus = Nucleus(protons=50, neutrons=82)
    >>> force = Force.from_name("SLy4")
    >>> 
    >>> # Create and run the HFB calculation
    >>> calc = HFBFFT(nucleus=nucleus, force=force)
    >>> calc.run(max_iterations=1000)
    >>> 
    >>> # Access results
    >>> print(f"Binding energy: {calc.total_energy:.3f} MeV")
"""

__version__ = "0.1.0"
__author__ = "ASCSN"

# JAX configuration - MUST be imported first to set 64-bit mode
from jax_hfbfft.jax_config import (
    set_precision,
    get_dtypes,
    check_precision,
    Precision,
    DTypes,
)

# Core classes - main user-facing API
from jax_hfbfft.core.hfbfft import HFBFFT
from jax_hfbfft.core.nucleus import Nucleus
from jax_hfbfft.core.force import Force
from jax_hfbfft.core.constraint import Constraint
from jax_hfbfft.core.grid import Grid

# Configuration utilities
from jax_hfbfft.config import Config

# Convenient access to force presets
from jax_hfbfft.forces import AVAILABLE_FORCES

__all__ = [
    # Main classes
    "HFBFFT",
    "Nucleus", 
    "Force",
    "Constraint",
    "Grid",
    # Configuration
    "Config",
    # Force presets
    "AVAILABLE_FORCES",
    # Version info
    "__version__",
]
