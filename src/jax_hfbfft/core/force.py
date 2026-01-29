"""
Force class for nuclear interaction definitions.

This module provides the Force class which encapsulates Skyrme-type
effective nuclear interactions for HFB calculations.
"""

import jax
import jax.numpy as jnp
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import yaml


@jax.tree_util.register_dataclass
@dataclass
class Force:
    """
    Skyrme-type effective nuclear interaction.
    
    This class encapsulates the parameters of a Skyrme interaction including
    the standard Skyrme parameters (t0, t1, t2, t3, x0, x1, x2, x3) and 
    derived coefficients (b0, b1, b2, b3, etc.).
    
    Attributes:
        name (str): Name of the force parameterization
        t0, t1, t2, t3, t4 (float): Skyrme interaction strengths
        x0, x1, x2, x3 (float): Skyrme exchange parameters
        power (float): Density-dependence power (typically 1/6 to 1)
        
    Examples:
        >>> # Load a predefined force
        >>> force = Force.from_name("SLy4")
        
        >>> # Create custom force with parameters
        >>> force = Force(name="custom", t0=-2488.91, t1=486.82, ...)
        
        >>> # Modify pairing parameters
        >>> force = force.with_pairing(v0prot=362.0, v0neut=362.0)
    """
    
    # Force identification
    name: str = field(metadata=dict(static=True))
    
    # Exchange type
    ex: int = field(metadata=dict(static=True))
    
    # Zero-point energy correction flag
    zpe: int = 0
    
    # Effective mass (h-bar^2 / 2m for each isospin)
    h2m: Optional[jax.Array] = None
    
    # Standard Skyrme parameters
    t0: float = 0.0
    t1: float = 0.0
    t2: float = 0.0
    t3: float = 0.0
    t4: float = 0.0
    x0: float = 0.0
    x1: float = 0.0
    x2: float = 0.0
    x3: float = 0.0
    
    # Spin-orbit parameter
    b4p: float = 0.0
    
    # Density-dependence power
    power: float = 1.0
    
    # Pairing parameters
    ipair: int = field(default=0, metadata=dict(static=True))  # Pairing type
    v0prot: float = 0.0  # Proton pairing strength
    v0neut: float = 0.0  # Neutron pairing strength
    rho0pr: float = 0.16  # Pairing reference density
    pair_reg: bool = False  # Regularization flag
    delta_fit: Optional[jax.Array] = None  # Fitted gap parameters
    pair_cutoff: Optional[jax.Array] = None  # Energy cutoff for pairing
    state_cutoff: Optional[jax.Array] = None  # State cutoff
    softcut_range: float = 0.1  # Soft cutoff width parameter
    tbcs: bool = False  # Use BCS approximation
    
    # Physical constants
    h2ma: float = 20.73553  # h-bar^2 / (2 * nucleon mass)
    nucleon_mass: float = 938.9  # Nucleon mass in MeV
    
    # Derived Skyrme coefficients (calculated from t, x parameters)
    b0: float = 0.0
    b0p: float = 0.0
    b1: float = 0.0
    b1p: float = 0.0
    b2: float = 0.0
    b2p: float = 0.0
    b3: float = 0.0
    b3p: float = 0.0
    b4: float = 0.0
    
    # Slater determinant parameter
    slate: float = 0.0
    
    # EDF coupling constants (derived)
    Crho0: float = 0.0
    Crho1: float = 0.0
    Crho0D: float = 0.0
    Crho1D: float = 0.0
    Cdrho0: float = 0.0
    Cdrho1: float = 0.0
    Ctau0: float = 0.0
    Ctau1: float = 0.0
    CdJ0: float = 0.0
    CdJ1: float = 0.0
    
    def __post_init__(self):
        """Calculate derived coefficients after initialization."""
        # Initialize arrays if not set
        if self.delta_fit is None:
            object.__setattr__(self, 'delta_fit', jnp.array([-1.0, -1.0]))
        if self.pair_cutoff is None:
            object.__setattr__(self, 'pair_cutoff', jnp.array([0.0, 0.0]))
        if self.state_cutoff is None:
            object.__setattr__(self, 'state_cutoff', jnp.array([0.0, 0.0]))
        if self.h2m is None:
            object.__setattr__(self, 'h2m', jnp.array([self.h2ma, self.h2ma]))
    
    @classmethod
    def from_name(
        cls, 
        name: str,
        forces_file: Optional[str] = None,
        **kwargs
    ) -> "Force":
        """
        Create a Force from a predefined parameterization.
        
        Args:
            name: Name of the force (e.g., "SLy4", "SkM*", "UNEDF1")
            forces_file: Path to YAML file with force definitions
            **kwargs: Override parameters (e.g., pairing strengths)
            
        Returns:
            Initialized Force object
            
        Examples:
            >>> force = Force.from_name("SLy4")
            >>> force = Force.from_name("SLy4", v0prot=362.0, v0neut=362.0)
        """
        # Try to find forces file
        if forces_file is None:
            # Look in package data or current directory
            possible_paths = [
                Path(__file__).parent.parent / "data" / "_forces.yml",
                Path("_forces.yml"),
            ]
            for path in possible_paths:
                if path.exists():
                    forces_file = str(path)
                    break
        
        if forces_file is None:
            raise FileNotFoundError(
                "Could not find forces definition file. "
                "Please provide forces_file argument."
            )
        
        # Load force parameters from YAML
        with open(forces_file, 'r') as f:
            forces_data = yaml.safe_load(f)
        
        if name not in forces_data:
            available = ", ".join(forces_data.keys())
            raise ValueError(
                f"Force '{name}' not found. Available forces: {available}"
            )
        
        force_params = forces_data[name]
        
        # Build initialization parameters
        init_kwargs = {
            'name': name,
            'ex': force_params.get('ex', 1),
            'zpe': force_params.get('zpe', 0),
            'h2m': jnp.array(force_params.get('h2m', [20.73553, 20.73553])),
            't0': force_params.get('t0', 0.0),
            't1': force_params.get('t1', 0.0),
            't2': force_params.get('t2', 0.0),
            't3': force_params.get('t3', 0.0),
            't4': force_params.get('t4', 0.0),
            'x0': force_params.get('x0', 0.0),
            'x1': force_params.get('x1', 0.0),
            'x2': force_params.get('x2', 0.0),
            'x3': force_params.get('x3', 0.0),
            'b4p': force_params.get('b4p', 0.0),
            'power': force_params.get('power', 1.0),
        }
        
        # Apply user overrides
        init_kwargs.update(kwargs)
        
        # Create force and calculate derived coefficients
        force = cls(**init_kwargs)
        return force._calculate_derived_coefficients()
    
    def _calculate_derived_coefficients(self) -> "Force":
        """Calculate derived Skyrme coefficients from primary parameters."""
        import dataclasses
        
        # Calculate b coefficients from t and x parameters
        b0 = self.t0 * (1 + 0.5 * self.x0)
        b0p = self.t0 * (0.5 + self.x0)
        
        b1 = (self.t1 + 0.5 * self.x1 * self.t1 + 
              self.t2 + 0.5 * self.x2 * self.t2) / 4
        b1p = (self.t1 * (0.5 + self.x1) - 
               self.t2 * (0.5 + self.x2)) / 4
        
        b2 = (3 * self.t1 * (1 + 0.5 * self.x1) - 
              self.t2 * (1 + 0.5 * self.x2)) / 8
        b2p = (3 * self.t1 * (0.5 + self.x1) + 
               self.t2 * (0.5 + self.x2)) / 8
        
        b3 = self.t3 * (1 + 0.5 * self.x3) / 4
        b3p = self.t3 * (0.5 + self.x3) / 4
        
        b4 = self.t4 / 2
        
        # Calculate EDF coupling constants
        Crho0 = 3 * self.t0 / 8
        Crho1 = -self.t0 * (0.5 + self.x0) / 4
        Crho0D = self.t3 / 16
        Crho1D = -self.t3 * (0.5 + self.x3) / 24
        
        Cdrho0 = (9 * self.t1 - 5 * self.t2 * (1 + 0.5 * self.x2)) / 64
        Cdrho1 = -(3 * self.t1 * (0.5 + self.x1) + 
                   self.t2 * (0.5 + self.x2)) / 32
        
        Ctau0 = (3 * self.t1 + self.t2 * (5 + 4 * self.x2)) / 16
        Ctau1 = (self.t2 * (0.5 + self.x2) - self.t1 * (0.5 + self.x1)) / 8
        
        CdJ0 = -self.b4p * 1.5
        CdJ1 = -self.b4p * 0.5
        
        # Calculate Slater parameter
        slate = (3.0 / jnp.pi) ** (1.0/3.0) * 0.75
        
        return dataclasses.replace(
            self,
            b0=b0, b0p=b0p,
            b1=b1, b1p=b1p,
            b2=b2, b2p=b2p,
            b3=b3, b3p=b3p,
            b4=b4,
            slate=slate,
            Crho0=Crho0, Crho1=Crho1,
            Crho0D=Crho0D, Crho1D=Crho1D,
            Cdrho0=Cdrho0, Cdrho1=Cdrho1,
            Ctau0=Ctau0, Ctau1=Ctau1,
            CdJ0=CdJ0, CdJ1=CdJ1,
        )
    
    def with_pairing(
        self,
        ipair: Optional[int] = None,
        v0prot: Optional[float] = None,
        v0neut: Optional[float] = None,
        rho0pr: Optional[float] = None,
        tbcs: Optional[bool] = None,
        pair_cutoff: Optional[Tuple[float, float]] = None,
        state_cutoff: Optional[Tuple[float, float]] = None,
    ) -> "Force":
        """
        Create a new Force with modified pairing parameters.
        
        Args:
            ipair: Pairing type (0=none, 5=VDI, 6=DDDI)
            v0prot: Proton pairing strength (MeV·fm³)
            v0neut: Neutron pairing strength (MeV·fm³)
            rho0pr: Reference density for DDDI pairing
            tbcs: Use BCS approximation
            pair_cutoff: Energy cutoff (neutron, proton) in MeV
            state_cutoff: State cutoff (neutron, proton) in MeV
            
        Returns:
            New Force with updated pairing parameters
        """
        import dataclasses
        
        updates = {}
        if ipair is not None:
            updates['ipair'] = ipair
        if v0prot is not None:
            updates['v0prot'] = v0prot
        if v0neut is not None:
            updates['v0neut'] = v0neut
        if rho0pr is not None:
            updates['rho0pr'] = rho0pr
        if tbcs is not None:
            updates['tbcs'] = tbcs
        if pair_cutoff is not None:
            updates['pair_cutoff'] = jnp.array(pair_cutoff)
        if state_cutoff is not None:
            updates['state_cutoff'] = jnp.array(state_cutoff)
        
        return dataclasses.replace(self, **updates)
    
    def __str__(self) -> str:
        return f"Force({self.name})"
    
    def __repr__(self) -> str:
        return (
            f"Force(name={self.name!r}, t0={self.t0:.2f}, t1={self.t1:.2f}, "
            f"t2={self.t2:.2f}, t3={self.t3:.2f}, power={self.power:.3f})"
        )


# Dictionary of commonly used forces with their parameters
BUILTIN_FORCES = {
    "SLy4": {
        "ex": 1, "zpe": 0,
        "h2m": [20.73553, 20.73553],
        "t0": -2488.91, "t1": 486.82, "t2": -546.39, "t3": 13777.0, "t4": 123.0,
        "x0": 0.834, "x1": -0.344, "x2": -1.0, "x3": 1.354,
        "b4p": 61.5, "power": 1.0/6.0,
    },
    "SkM*": {
        "ex": 1, "zpe": 0,
        "h2m": [20.73553, 20.73553],
        "t0": -2645.0, "t1": 410.0, "t2": -135.0, "t3": 15595.0, "t4": 130.0,
        "x0": 0.09, "x1": 0.0, "x2": 0.0, "x3": 0.0,
        "b4p": 65.0, "power": 1.0/6.0,
    },
}
