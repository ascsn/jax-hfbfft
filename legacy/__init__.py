"""
Legacy implementation of JAX-HFBFFT.

This package contains the original implementation before the OOP refactoring.
It is kept for backward compatibility and reference.

Usage:
    from legacy import static, levels, densities, meanfield, ...
    
Or via the main package:
    from jax_hfbfft import HFBFFT
    calc = HFBFFT(...)
    calc.run(use_legacy=True)
"""

# Import legacy modules for convenience
from legacy.params import init_params
from legacy.grids import init_grids
from legacy.forces import init_forces
from legacy.levels import init_levels
from legacy.densities import init_densities
from legacy.meanfield import init_meanfield
from legacy.energies import init_energies
from legacy.coulomb import init_coulomb
from legacy.pairs import Pairs
from legacy.static import init_static, statichf

__all__ = [
    'init_params',
    'init_grids',
    'init_forces',
    'init_levels',
    'init_densities',
    'init_meanfield',
    'init_energies',
    'init_coulomb',
    'Pairs',
    'init_static',
    'statichf',
]
