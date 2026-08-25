"""
Legacy compatibility layer for JAX-HFBFFT.

This module provides adapters to use existing module-based code with the new
object-oriented API. It allows gradual migration while preserving the ability
to use the original implementation.

Legacy code is now located in the 'legacy/' folder at the repository root.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add legacy directory to path to import legacy modules
_repo_root = Path(__file__).parent.parent.parent.parent
_legacy_path = _repo_root / 'legacy'
if str(_legacy_path) not in sys.path and _legacy_path.exists():
    sys.path.insert(0, str(_legacy_path))
# Also add repo root for backward compatibility
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


def create_legacy_objects(
    config: Dict[str, Any],
    nucleus_z: int,
    nucleus_n: int,
) -> Dict[str, Any]:
    """
    Create legacy-style objects from the original codebase.
    
    This function bridges the gap between the new OOP API and the
    original module-based implementation.
    
    Args:
        config: Configuration dictionary (from YAML or programmatic)
        nucleus_z: Number of protons
        nucleus_n: Number of neutrons
        
    Returns:
        Dictionary containing all initialized legacy objects:
        - params: Params object
        - grids: Grids object
        - forces: Forces object
        - levels: Levels object
        - densities: Densities object
        - meanfield: Meanfield object
        - energies: Energies object
        - coulomb: Coulomb object
        - static: Static object
        - pairs: Pairs object
    """
    # Import legacy modules
    from params import init_params
    from grids import init_grids
    from forces import init_forces
    from levels import init_levels
    from densities import init_densities
    from meanfield import init_meanfield
    from energies import init_energies
    from coulomb import init_coulomb
    from static import init_static
    from pairs import Pairs
    
    # Extract configuration sections
    params_config = config.get('params', {})
    grids_config = config.get('grids', config.get('grid', {}))
    force_config = config.get('force', {})
    levels_config = config.get('levels', {})
    static_config = config.get('static', {})
    
    # Set nucleus in levels config
    levels_config['nneut'] = nucleus_n
    levels_config['nprot'] = nucleus_z
    
    # Initialize objects
    params = init_params(imode=params_config.get('imode', 1), **params_config)
    grids = init_grids(params, **grids_config)
    forces = init_forces(params, **force_config)
    levels = init_levels(grids, **levels_config)
    densities = init_densities(grids)
    meanfield = init_meanfield(grids)
    energies = init_energies()
    coulomb = init_coulomb(grids)
    forces, static = init_static(forces, levels, **static_config)
    pairs = Pairs()
    
    return {
        'params': params,
        'grids': grids,
        'forces': forces,
        'levels': levels,
        'densities': densities,
        'meanfield': meanfield,
        'energies': energies,
        'coulomb': coulomb,
        'static': static,
        'pairs': pairs,
    }


def run_legacy_static(
    objects: Dict[str, Any],
    max_iterations: int = 1000,
    print_interval: int = 10,
) -> Dict[str, Any]:
    """
    Run the legacy static HFB calculation.
    
    Args:
        objects: Dictionary of legacy objects from create_legacy_objects
        max_iterations: Maximum number of iterations
        print_interval: Print interval for progress
        
    Returns:
        Updated objects dictionary with converged state
    """
    from static import statichf
    
    # Extract objects
    coulomb = objects['coulomb']
    densities = objects['densities']
    energies = objects['energies']
    forces = objects['forces']
    grids = objects['grids']
    levels = objects['levels']
    meanfield = objects['meanfield']
    params = objects['params']
    static = objects['static']
    pairs = objects['pairs']
    
    # Update static parameters
    import dataclasses
    static = dataclasses.replace(static, maxiter=max_iterations)
    params = dataclasses.replace(params, mprint=print_interval)
    
    # Run static calculation
    result = statichf(
        coulomb, densities, energies, forces, 
        grids, levels, meanfield, params, static, pairs
    )
    
    # Unpack results
    (coulomb, densities, energies, forces, grids, 
     levels, meanfield, params, static, pairs) = result
    
    # Update and return
    objects.update({
        'coulomb': coulomb,
        'densities': densities,
        'energies': energies,
        'forces': forces,
        'grids': grids,
        'levels': levels,
        'meanfield': meanfield,
        'params': params,
        'static': static,
        'pairs': pairs,
    })
    
    return objects


class LegacyHFBFFT:
    """
    Wrapper class that provides an OOP interface to the legacy code.
    
    This class allows using the original implementation through the new API.
    
    Examples:
        >>> from jax_hfbfft.legacy import LegacyHFBFFT
        >>> 
        >>> calc = LegacyHFBFFT.from_config("_config.yml")
        >>> calc.run(max_iterations=1000)
        >>> print(f"E = {calc.total_energy:.3f} MeV")
    """
    
    def __init__(
        self,
        nucleus_z: int,
        nucleus_n: int,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a legacy HFBFFT wrapper.
        
        Args:
            nucleus_z: Number of protons
            nucleus_n: Number of neutrons
            config: Configuration dictionary
        """
        self.nucleus_z = nucleus_z
        self.nucleus_n = nucleus_n
        self.config = config or {}
        
        self._objects = None
        self._initialized = False
    
    @classmethod
    def from_config(cls, config_file: str) -> "LegacyHFBFFT":
        """
        Create a LegacyHFBFFT from a YAML configuration file.
        
        Args:
            config_file: Path to YAML configuration file
            
        Returns:
            Initialized LegacyHFBFFT instance
        """
        from reader import read_yaml
        
        config = read_yaml(config_file)
        
        levels_config = config.get('levels', {})
        nucleus_n = levels_config.get('nneut', 20)
        nucleus_z = levels_config.get('nprot', 20)
        
        return cls(nucleus_z=nucleus_z, nucleus_n=nucleus_n, config=config)
    
    def initialize(self):
        """Initialize all calculation objects."""
        self._objects = create_legacy_objects(
            self.config, 
            self.nucleus_z, 
            self.nucleus_n
        )
        self._initialized = True
    
    def run(
        self,
        max_iterations: int = 1000,
        print_interval: int = 10,
    ):
        """
        Run the HFB calculation.
        
        Args:
            max_iterations: Maximum number of iterations
            print_interval: Print interval for progress
        """
        if not self._initialized:
            self.initialize()
        
        self._objects = run_legacy_static(
            self._objects,
            max_iterations=max_iterations,
            print_interval=print_interval,
        )
    
    @property
    def total_energy(self) -> float:
        """Get the total binding energy."""
        if self._objects is None:
            return 0.0
        return float(self._objects['energies'].ehf)
    
    @property
    def kinetic_energy(self) -> float:
        """Get the kinetic energy."""
        if self._objects is None:
            return 0.0
        return float(self._objects['energies'].ehft)
    
    @property
    def fermi_energy_n(self) -> float:
        """Get the neutron Fermi energy."""
        if self._objects is None:
            return 0.0
        return float(self._objects['pairs'].eferm[0])
    
    @property
    def fermi_energy_p(self) -> float:
        """Get the proton Fermi energy."""
        if self._objects is None:
            return 0.0
        return float(self._objects['pairs'].eferm[1])
    
    def get_object(self, name: str):
        """
        Get a specific internal object.
        
        Args:
            name: Object name (params, grids, forces, levels, etc.)
            
        Returns:
            The requested object
        """
        if self._objects is None:
            raise RuntimeError("Call initialize() or run() first")
        return self._objects.get(name)
