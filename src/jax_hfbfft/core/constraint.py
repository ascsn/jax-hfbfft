"""
Constraint class for constrained HFB calculations.

This module provides the Constraint class which encapsulates constraint
configurations for constrained mean-field calculations.
"""

import jax
import jax.numpy as jnp
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import dataclasses


@jax.tree_util.register_dataclass
@dataclass
class Constraint:
    """
    Constraint configuration for constrained HFB calculations.
    
    This class defines constraints that can be applied during HFB iterations
    to control nuclear deformation and other collective degrees of freedom.
    
    Supported constraint types:
        - Quadrupole moments (Q20, Q22)
        - Principal axes alignment
        - Custom constraint fields
    
    Attributes:
        alpha20_wanted (float): Target Q20 deformation parameter
        alpha22_wanted (float): Target Q22 deformation parameter
        tq_prin_axes (bool): Enable principal axes constraints
        
    Examples:
        >>> # Create a spherical constraint
        >>> constraint = Constraint.spherical()
        
        >>> # Create a prolate deformation constraint
        >>> constraint = Constraint(alpha20_wanted=0.3)
        
        >>> # Create constraint with triaxiality
        >>> constraint = Constraint(alpha20_wanted=0.3, alpha22_wanted=0.1)
    """
    
    # Control flags
    tconstraint: bool = False
    tq_prin_axes: bool = False  # Principal axes constraints (6 additional constraints)
    
    # Desired constraint values (-1e99 means unconstrained)
    alpha20_wanted: float = -1e99  # Q20 quadrupole moment target
    alpha22_wanted: float = -1e99  # Q22 quadrupole moment target
    
    # Constraint algorithm parameters
    c0constr: float = 0.8          # Parameter for Q-corrective step
    d0constr: float = 1e-4         # Small parameter to avoid division by zero
    qepsconstr: float = 0.3        # Parameter for Lagrange multiplier update
    dampgamma: float = 1.0         # Damping parameter for masking function
    damprad: float = 6.0           # Damping radius for masking function
    
    # Physical constants
    r0rms: float = 0.93            # RMS radius parameter (corresponds to R0=1.2 fm)
    
    # Constraint arrays (allocated dynamically)
    numconstraint: int = 0
    constr_field: Optional[jax.Array] = None      # [iconstr, nx, ny, nz, isospin]
    lambda_crank: Optional[jax.Array] = None      # Lagrange multipliers
    dlambda_crank: Optional[jax.Array] = None     # Corrections to multipliers
    goal_crank: Optional[jax.Array] = None        # Target expectation values
    actual_crank: Optional[jax.Array] = None      # Actual expectation values
    old_crank: Optional[jax.Array] = None         # Previous expectation values
    actual_crank2: Optional[jax.Array] = None     # Variances
    qcorr: Optional[jax.Array] = None             # Q-correction factors
    
    @classmethod
    def spherical(cls) -> "Constraint":
        """
        Create a constraint configuration for spherical calculations.
        
        No deformation constraints are applied; nucleus evolves freely.
        
        Returns:
            Constraint object for spherical calculation
        """
        return cls(tconstraint=False)
    
    @classmethod
    def prolate(cls, beta2: float = 0.3, **kwargs) -> "Constraint":
        """
        Create a prolate deformation constraint.
        
        Args:
            beta2: Axial deformation parameter
            **kwargs: Additional constraint parameters
            
        Returns:
            Constraint object for prolate deformation
        """
        return cls(alpha20_wanted=beta2, tconstraint=True, **kwargs)
    
    @classmethod
    def oblate(cls, beta2: float = -0.3, **kwargs) -> "Constraint":
        """
        Create an oblate deformation constraint.
        
        Args:
            beta2: Axial deformation parameter (negative for oblate)
            **kwargs: Additional constraint parameters
            
        Returns:
            Constraint object for oblate deformation
        """
        return cls(alpha20_wanted=beta2, tconstraint=True, **kwargs)
    
    @classmethod
    def triaxial(
        cls, 
        alpha20: float = 0.3, 
        alpha22: float = 0.1, 
        **kwargs
    ) -> "Constraint":
        """
        Create a triaxial deformation constraint.
        
        Args:
            alpha20: Q20 deformation parameter
            alpha22: Q22 deformation parameter (non-axial)
            **kwargs: Additional constraint parameters
            
        Returns:
            Constraint object for triaxial deformation
        """
        return cls(
            alpha20_wanted=alpha20, 
            alpha22_wanted=alpha22, 
            tconstraint=True, 
            **kwargs
        )
    
    def with_principal_axes(self, enable: bool = True) -> "Constraint":
        """
        Enable or disable principal axes constraints.
        
        Args:
            enable: Whether to enable principal axes alignment
            
        Returns:
            New Constraint with updated setting
        """
        return dataclasses.replace(self, tq_prin_axes=enable)
    
    def initialize_arrays(self, grid_shape: tuple, mass_number: float) -> "Constraint":
        """
        Initialize constraint arrays for a given grid.
        
        Args:
            grid_shape: Tuple of (nx, ny, nz) grid dimensions
            mass_number: Mass number of the nucleus
            
        Returns:
            New Constraint with initialized arrays
        """
        # Count number of constraints
        numconstraint = 0
        if self.alpha20_wanted > -1e90:
            numconstraint += 1
        if self.alpha22_wanted > -1e90:
            numconstraint += 1
        if self.tq_prin_axes:
            numconstraint += 6  # xy, xz, yz, x, y, z
        
        if numconstraint == 0:
            return dataclasses.replace(self, numconstraint=0, tconstraint=False)
        
        nx, ny, nz = grid_shape
        
        return dataclasses.replace(
            self,
            numconstraint=numconstraint,
            tconstraint=True,
            constr_field=jnp.zeros((numconstraint, nx, ny, nz, 2)),
            lambda_crank=jnp.full(numconstraint, -0.2),
            dlambda_crank=jnp.zeros(numconstraint),
            goal_crank=jnp.zeros(numconstraint),
            actual_crank=jnp.zeros(numconstraint),
            old_crank=jnp.zeros(numconstraint),
            actual_crank2=jnp.zeros(numconstraint),
            qcorr=jnp.zeros(numconstraint),
        )
    
    def __str__(self) -> str:
        if not self.tconstraint:
            return "Constraint(spherical)"
        
        parts = []
        if self.alpha20_wanted > -1e90:
            parts.append(f"α20={self.alpha20_wanted:.3f}")
        if self.alpha22_wanted > -1e90:
            parts.append(f"α22={self.alpha22_wanted:.3f}")
        if self.tq_prin_axes:
            parts.append("principal_axes")
        
        return f"Constraint({', '.join(parts)})"
    
    def __repr__(self) -> str:
        return (
            f"Constraint(alpha20_wanted={self.alpha20_wanted}, "
            f"alpha22_wanted={self.alpha22_wanted}, "
            f"tq_prin_axes={self.tq_prin_axes})"
        )
