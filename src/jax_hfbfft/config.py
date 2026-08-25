"""
Configuration management for JAX-HFBFFT.

This module provides utilities for loading and managing configuration
from YAML files and programmatic settings.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path
import yaml


@dataclass
class Config:
    """
    Configuration container for HFBFFT calculations.
    
    This class provides a convenient way to specify all parameters
    for an HFB calculation, either programmatically or from a YAML file.
    
    Examples:
        >>> # Load from YAML file
        >>> config = Config.from_yaml("calculation.yml")
        
        >>> # Create programmatically
        >>> config = Config(
        ...     nucleus={"protons": 50, "neutrons": 82},
        ...     force={"name": "SLy4"},
        ...     grid={"nx": 32, "ny": 32, "nz": 32}
        ... )
        
        >>> # Create HFBFFT from config
        >>> from jax_hfbfft import HFBFFT
        >>> calc = HFBFFT.from_config(config)
    """
    
    # Nucleus configuration
    nucleus: Dict[str, Any] = field(default_factory=dict)
    
    # Force configuration
    force: Dict[str, Any] = field(default_factory=lambda: {"name": "SLy4"})
    
    # Grid configuration
    grid: Dict[str, Any] = field(default_factory=lambda: {
        "nx": 32, "ny": 32, "nz": 32,
        "dx": 0.8, "dy": 0.8, "dz": 0.8
    })
    
    # Constraint configuration
    constraint: Dict[str, Any] = field(default_factory=dict)
    
    # Static iteration parameters
    static: Dict[str, Any] = field(default_factory=lambda: {
        "max_iterations": 1000,
        "convergence_threshold": 1e-6,
        "x0dmp": 0.45,
        "e0dmp": 100.0
    })
    
    # Levels/basis configuration
    levels: Dict[str, Any] = field(default_factory=dict)
    
    # General parameters
    params: Dict[str, Any] = field(default_factory=lambda: {
        "include_coulomb": True,
        "use_fft": True,
        "mprint": 10
    })
    
    @classmethod
    def from_yaml(cls, filepath: str) -> "Config":
        """
        Load configuration from a YAML file.
        
        Args:
            filepath: Path to the YAML configuration file
            
        Returns:
            Config object with loaded parameters
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}
        
        return cls(
            nucleus=data.get('nucleus', data.get('levels', {})),
            force=data.get('force', {}),
            grid=data.get('grid', data.get('grids', {})),
            constraint=data.get('constraint', data.get('constraints', {})),
            static=data.get('static', {}),
            levels=data.get('levels', {}),
            params=data.get('params', {}),
        )
    
    def to_yaml(self, filepath: str):
        """
        Save configuration to a YAML file.
        
        Args:
            filepath: Path to save the YAML configuration
        """
        data = {
            'nucleus': self.nucleus,
            'force': self.force,
            'grid': self.grid,
            'constraint': self.constraint,
            'static': self.static,
            'levels': self.levels,
            'params': self.params,
        }
        
        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    
    def update(self, **kwargs) -> "Config":
        """
        Create a new Config with updated parameters.
        
        Args:
            **kwargs: Section dictionaries to update (e.g., force={...})
            
        Returns:
            New Config with updated values
        """
        import dataclasses
        
        updates = {}
        for key, value in kwargs.items():
            if hasattr(self, key):
                current = getattr(self, key)
                if isinstance(current, dict) and isinstance(value, dict):
                    updates[key] = {**current, **value}
                else:
                    updates[key] = value
        
        return dataclasses.replace(self, **updates)
    
    def __repr__(self) -> str:
        return (
            f"Config(nucleus={self.nucleus}, "
            f"force={self.force.get('name', 'custom')}, "
            f"grid={self.grid.get('nx', '?')}x{self.grid.get('ny', '?')}x{self.grid.get('nz', '?')})"
        )
