"""
Constraint class for constrained HFB calculations.

This module provides the Constraint class which encapsulates constraint
configurations for constrained mean-field calculations, including arbitrary
multipole moment constraints.
"""

import jax
import jax.numpy as jnp
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union, Tuple
import dataclasses
import numpy as np


# Mapping from common multipole names to (lambda, mu) tuples
MULTIPOLE_NAMES = {
    # Monopole (lambda=0)
    'Q00': (0, 0),
    
    # Dipole (lambda=1)
    'Q10': (1, 0),
    'Q11': (1, 1),
    
    # Quadrupole (lambda=2)
    'Q20': (2, 0),
    'Q21': (2, 1),
    'Q22': (2, 2),
    
    # Octupole (lambda=3)
    'Q30': (3, 0),
    'Q31': (3, 1),
    'Q32': (3, 2),
    'Q33': (3, 3),
    
    # Hexadecapole (lambda=4)
    'Q40': (4, 0),
    'Q41': (4, 1),
    'Q42': (4, 2),
    'Q43': (4, 3),
    'Q44': (4, 4),
    
    # Triakontadipole (lambda=5)
    'Q50': (5, 0),
    'Q51': (5, 1),
    'Q52': (5, 2),
    'Q53': (5, 3),
    'Q54': (5, 4),
    'Q55': (5, 5),
    
    # Hexacontatetrapole (lambda=6)
    'Q60': (6, 0),
    'Q61': (6, 1),
    'Q62': (6, 2),
    'Q63': (6, 3),
    'Q64': (6, 4),
    'Q65': (6, 5),
    'Q66': (6, 6),
}


def parse_multipole(spec: Union[str, Tuple[int, int]]) -> Tuple[int, int]:
    """
    Parse a multipole specification into (lambda, mu) tuple.
    
    Args:
        spec: Either a string name like 'Q20', 'Q30', etc., or a tuple (lambda, mu)
        
    Returns:
        Tuple of (lambda, mu) multipole quantum numbers
        
    Examples:
        >>> parse_multipole('Q20')
        (2, 0)
        >>> parse_multipole('Q30')
        (3, 0)
        >>> parse_multipole((4, 2))
        (4, 2)
    
    Raises:
        ValueError: If multipole specification is invalid
    """
    if isinstance(spec, tuple):
        if len(spec) != 2:
            raise ValueError(f"Multipole tuple must have 2 elements, got {len(spec)}")
        lam, mu = spec
        if not isinstance(lam, int) or not isinstance(mu, int):
            raise ValueError(f"Multipole quantum numbers must be integers, got ({lam}, {mu})")
        if lam < 0:
            raise ValueError(f"Lambda must be non-negative, got {lam}")
        if abs(mu) > lam:
            raise ValueError(f"Mu must satisfy |mu| <= lambda, got lambda={lam}, mu={mu}")
        return (lam, mu)
    
    elif isinstance(spec, str):
        spec_upper = spec.upper()
        if spec_upper not in MULTIPOLE_NAMES:
            raise ValueError(
                f"Unknown multipole name '{spec}'. "
                f"Use standard names like 'Q20', 'Q30', etc., or specify as tuple (lambda, mu). "
                f"Available names: {', '.join(sorted(MULTIPOLE_NAMES.keys()))}"
            )
        return MULTIPOLE_NAMES[spec_upper]
    
    else:
        raise ValueError(
            f"Multipole specification must be string or tuple, got {type(spec).__name__}"
        )


def multipole_name(lam: int, mu: int) -> str:
    """
    Get the standard name for a multipole moment.
    
    Args:
        lam: Lambda quantum number
        mu: Mu quantum number
        
    Returns:
        Standard name like 'Q20', 'Q30', etc.
    """
    name = f"Q{lam}{abs(mu)}"
    if name in MULTIPOLE_NAMES and MULTIPOLE_NAMES[name] == (lam, mu):
        return name
    return f"Q({lam},{mu})"


def beta_gamma_to_multipoles(
    beta2: Optional[float] = None,
    gamma: Optional[float] = None,
    beta3: Optional[float] = None,
    beta4: Optional[float] = None,
    mass_number: float = None,
    r0: float = 1.2
) -> Dict[Tuple[int, int], float]:
    """
    Convert beta-gamma deformation parameters to multipole moments.
    
    The Hill-Wheeler parameterization relates deformation parameters to
    multipole moments via:
    
    Q_λμ = β_λ × A^((2λ+1)/3) × R₀^λ × Y_λμ_factor
    
    where R₀ = r₀ × A^(1/3) is the nuclear radius parameter.
    
    For quadrupole deformation:
    - Q₂₀ = (3/√(5π)) × |β₂| × A^(5/3) × R₀² × cos(γ)
    - Q₂₂ = (1/√(5π)) × |β₂| × A^(5/3) × R₀² × sin(γ)
    
    Sign convention for β₂:
    - β₂ > 0: Prolate (elongated, football shape)
    - β₂ < 0: Oblate (flattened, pancake shape)
    - Negative β₂ is equivalent to β₂ > 0 with γ = 60°
    
    For octupole:
    - Q₃₀ = (1/√(7π)) × β₃ × A^(7/3) × R₀³
    
    For hexadecapole:
    - Q₄₀ = (3/√(9π)) × β₄ × A^(3) × R₀⁴
    
    Args:
        beta2: Quadrupole deformation parameter
               Positive = prolate, negative = oblate
        gamma: Triaxiality angle in degrees (0° = axially symmetric)
               Only needed for triaxial shapes
        beta3: Octupole deformation parameter (pear shape)
        beta4: Hexadecapole deformation parameter
        mass_number: Nuclear mass number A (required if any beta is provided)
        r0: Radius parameter in fm (default: 1.2 fm)
        
    Returns:
        Dictionary mapping (lambda, mu) to multipole moment values in fm^λ
        
    Examples:
        >>> # Prolate quadrupole deformation
        >>> beta_gamma_to_multipoles(beta2=0.3, mass_number=16)
        {(2, 0): 5.123}
        
        >>> # Oblate (using negative beta)
        >>> beta_gamma_to_multipoles(beta2=-0.25, mass_number=40)
        {(2, 0): -8.456}
        
        >>> # Triaxial
        >>> beta_gamma_to_multipoles(beta2=0.25, gamma=30, mass_number=40)
        {(2, 0): 8.456, (2, 2): 4.882}
        
        >>> # Octupole deformation (pear shape)
        >>> beta_gamma_to_multipoles(beta3=0.1, mass_number=224)
        {(3, 0): 15.234}
    
    Raises:
        ValueError: If mass_number is not provided when beta parameters are given
    """
    multipoles = {}
    
    # Check if we need mass_number
    needs_A = beta2 is not None or beta3 is not None or beta4 is not None
    if needs_A and mass_number is None:
        raise ValueError("mass_number is required when specifying beta deformation parameters")
    
    if mass_number is not None:
        A = float(mass_number)
        R0 = r0 * A**(1.0/3.0)  # Nuclear radius parameter
    
    # Quadrupole deformation (lambda=2)
    if beta2 is not None:
        # Handle sign convention: negative beta2 = oblate
        # Convert negative beta2 to positive with gamma shift
        if beta2 < 0 and (gamma is None or gamma == 0):
            # Negative beta means oblate (gamma = 60°)
            beta2_mag = abs(beta2)
            gamma = 60.0
        else:
            beta2_mag = abs(beta2)
            if gamma is None:
                gamma = 0.0
        
        gamma_rad = np.deg2rad(gamma)
        
        # Normalization factors from spherical harmonics
        # Y_20: sqrt(5/(4π)) → Q_20 factor is 3/sqrt(5π) for Hill-Wheeler
        # Y_22: sqrt(15/(8π)) → Q_22 factor is 1/sqrt(5π) for Hill-Wheeler
        factor_base = beta2_mag * A**(5.0/3.0) * R0**2
        
        Q20 = factor_base * (3.0 / np.sqrt(5.0 * np.pi)) * np.cos(gamma_rad)
        Q22 = factor_base * (1.0 / np.sqrt(5.0 * np.pi)) * np.sin(gamma_rad)
        
        # Apply sign to Q20 (negative beta2 makes Q20 negative for oblate)
        if beta2 < 0:
            Q20 = -Q20
        
        multipoles[(2, 0)] = float(Q20)
        if abs(Q22) > 1e-10:  # Only include Q22 if non-negligible
            multipoles[(2, 2)] = float(Q22)
    
    # Octupole deformation (lambda=3)
    if beta3 is not None:
        factor_base = beta3 * A**(7.0/3.0) * R0**3
        Q30 = factor_base * (1.0 / np.sqrt(7.0 * np.pi))
        multipoles[(3, 0)] = float(Q30)
    
    # Hexadecapole deformation (lambda=4)
    if beta4 is not None:
        factor_base = beta4 * A**(3.0) * R0**4
        Q40 = factor_base * (3.0 / np.sqrt(9.0 * np.pi))
        multipoles[(4, 0)] = float(Q40)
    
    return multipoles


def multipoles_to_beta_gamma(
    multipoles: Dict[Tuple[int, int], float],
    mass_number: float,
    r0: float = 1.2
) -> Dict[str, float]:
    """
    Convert multipole moments to beta-gamma deformation parameters.
    
    Inverts the Hill-Wheeler parameterization to extract deformation
    parameters from multipole moments.
    
    Args:
        multipoles: Dictionary mapping (lambda, mu) to multipole values in fm^λ
        mass_number: Nuclear mass number A
        r0: Radius parameter in fm (default: 1.2 fm)
        
    Returns:
        Dictionary with keys 'beta2', 'gamma', 'beta3', 'beta4' as available
        
    Examples:
        >>> moments = {(2, 0): 10.0, (2, 2): 2.0}
        >>> multipoles_to_beta_gamma(moments, mass_number=40, r0=1.2)
        {'beta2': 0.289, 'gamma': 23.4}
    """
    result = {}
    
    A = float(mass_number)
    R0 = r0 * A**(1.0/3.0)
    
    # Extract quadrupole parameters
    Q20 = multipoles.get((2, 0), 0.0)
    Q22 = multipoles.get((2, 2), 0.0)
    
    if Q20 != 0.0 or Q22 != 0.0:
        factor_base = A**(5.0/3.0) * R0**2
        
        # Extract beta2 from magnitude
        q20_norm = Q20 / (factor_base * 3.0 / np.sqrt(5.0 * np.pi))
        q22_norm = Q22 / (factor_base * 1.0 / np.sqrt(5.0 * np.pi))
        
        beta2 = np.sqrt(q20_norm**2 + q22_norm**2)
        result['beta2'] = float(beta2)
        
        # Extract gamma from ratio
        if abs(q20_norm) > 1e-10:
            gamma_rad = np.arctan2(q22_norm, q20_norm)
            result['gamma'] = float(np.rad2deg(gamma_rad))
        elif abs(q22_norm) > 1e-10:
            result['gamma'] = 90.0  # Pure Q22 deformation
        else:
            result['gamma'] = 0.0
    
    # Extract octupole parameter
    Q30 = multipoles.get((3, 0), 0.0)
    if Q30 != 0.0:
        factor_base = A**(7.0/3.0) * R0**3
        beta3 = Q30 / (factor_base / np.sqrt(7.0 * np.pi))
        result['beta3'] = float(beta3)
    
    # Extract hexadecapole parameter
    Q40 = multipoles.get((4, 0), 0.0)
    if Q40 != 0.0:
        factor_base = A**(3.0) * R0**4
        beta4 = Q40 / (factor_base * 3.0 / np.sqrt(9.0 * np.pi))
        result['beta4'] = float(beta4)
    
    return result


@jax.tree_util.register_dataclass
@dataclass
class Constraint:
    """
    Constraint configuration for constrained HFB calculations.
    
    This class defines constraints that can be applied during HFB iterations
    to control nuclear deformation and other collective degrees of freedom.
    
    Supports arbitrary multipole moment constraints specified either by name
    (e.g., 'Q20', 'Q30', 'Q22') or by quantum numbers (e.g., (2, 0), (3, 0)).
    
    Examples:
        >>> # Spherical (no constraints)
        >>> constraint = Constraint.spherical()
        
        >>> # Single Q20 constraint
        >>> constraint = Constraint.from_multipoles({'Q20': 5.0})
        
        >>> # Multiple multipole constraints
        >>> constraint = Constraint.from_multipoles({
        ...     'Q20': 5.0,   # Quadrupole deformation
        ...     'Q30': 2.0,   # Octupole deformation
        ...     'Q22': 1.0,   # Triaxial deformation
        ... })
        
        >>> # Using tuple notation for arbitrary multipoles
        >>> constraint = Constraint.from_multipoles({
        ...     (2, 0): 5.0,   # Same as 'Q20'
        ...     (4, 0): 1.0,   # Hexadecapole
        ...     (6, 2): 0.5,   # Higher multipole
        ... })
        
        >>> # Legacy interface (deprecated, use from_multipoles instead)
        >>> constraint = Constraint(alpha20_wanted=0.3)
    
    Attributes:
        multipole_constraints: Dictionary mapping multipoles to target values.
            Keys can be string names ('Q20', 'Q30') or tuples ((2, 0), (3, 0)).
        tq_prin_axes: Enable principal axes alignment constraints
    """
    
    # Control flags
    tconstraint: bool = False
    tq_prin_axes: bool = False  # Principal axes constraints (6 additional constraints)
    
    # Modern multipole constraint system
    multipole_constraints: Dict[Union[str, Tuple[int, int]], float] = field(default_factory=dict)
    
    # Legacy interface (deprecated - use multipole_constraints instead)
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
    
    # Metadata about constraints
    constraint_multipoles: List[Tuple[int, int]] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate and normalize multipole constraints after initialization."""
        # Convert any string keys to tuples and validate
        if self.multipole_constraints:
            normalized = {}
            for key, value in self.multipole_constraints.items():
                lam, mu = parse_multipole(key)
                normalized[(lam, mu)] = float(value)
            
            # Use replace to update the field (needed for frozen dataclasses)
            object.__setattr__(self, 'multipole_constraints', normalized)
        
        # Handle legacy alpha20/alpha22 interface
        if self.alpha20_wanted > -1e90:
            if (2, 0) not in self.multipole_constraints:
                self.multipole_constraints[(2, 0)] = self.alpha20_wanted
        if self.alpha22_wanted > -1e90:
            if (2, 2) not in self.multipole_constraints:
                self.multipole_constraints[(2, 2)] = self.alpha22_wanted
    
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
    def from_multipoles(
        cls, 
        multipoles: Dict[Union[str, Tuple[int, int]], float],
        principal_axes: bool = False,
        **kwargs
    ) -> "Constraint":
        """
        Create constraint from arbitrary multipole moments.
        
        This is the recommended way to create multipole constraints.
        
        Args:
            multipoles: Dictionary mapping multipole specs to target values.
                Keys can be:
                - String names: 'Q20', 'Q30', 'Q22', etc.
                - Tuples: (lambda, mu) quantum numbers like (2, 0), (3, 0), etc.
                Values are the target expectation values in fm^lambda.
            principal_axes: Whether to enable principal axes constraints
            **kwargs: Additional constraint parameters
            
        Returns:
            Constraint object with specified multipole constraints
            
        Examples:
            >>> # Prolate Q20 constraint
            >>> c = Constraint.from_multipoles({'Q20': 5.0})
            
            >>> # Multiple constraints
            >>> c = Constraint.from_multipoles({
            ...     'Q20': 5.0,
            ...     'Q30': 2.0,
            ...     'Q22': 1.0,
            ... })
            
            >>> # Using tuple notation
            >>> c = Constraint.from_multipoles({
            ...     (2, 0): 5.0,
            ...     (4, 0): 1.0,
            ... })
        """
        return cls(
            multipole_constraints=multipoles,
            tq_prin_axes=principal_axes,
            tconstraint=True,
            **kwargs
        )
    
    @classmethod
    def from_beta_gamma(
        cls,
        mass_number: float,
        beta2: Optional[float] = None,
        gamma: Optional[float] = None,
        beta3: Optional[float] = None,
        beta4: Optional[float] = None,
        r0: float = 1.2,
        principal_axes: bool = False,
        **kwargs
    ) -> "Constraint":
        """
        Create constraint from beta-gamma deformation parameters.
        
        This method converts Hill-Wheeler (β, γ) parameterization to multipole
        moments and creates the appropriate constraints. Commonly used in
        publications and experimental data.
        
        The Hill-Wheeler parameterization:
        - β₂: Quadrupole deformation parameter
          * β₂ > 0: Prolate (football shape, elongated)
          * β₂ < 0: Oblate (pancake shape, flattened)
          * Typical magnitude: 0.0-0.6
        - γ: Triaxiality angle in degrees (for non-axially symmetric shapes)
          * γ = 0°: Axially symmetric
          * 0° < γ < 60°: Triaxial (asymmetric)
        - β₃: Octupole deformation (pear shape, typical: 0.0-0.2)
        - β₄: Hexadecapole deformation (typical: 0.0-0.1)
        
        Sign convention: Use β₂ > 0 for prolate or β₂ < 0 for oblate.
        This is often more intuitive than using γ = 60°.
        
        Args:
            mass_number: Nuclear mass number A (protons + neutrons)
            beta2: Quadrupole deformation parameter
                   Positive = prolate, negative = oblate
            gamma: Triaxiality angle in degrees (default: 0 = axially symmetric)
            beta3: Octupole deformation parameter
            beta4: Hexadecapole deformation parameter
            r0: Radius parameter in fm (default: 1.2 fm)
            principal_axes: Whether to enable principal axes constraints
            **kwargs: Additional constraint parameters
            
        Returns:
            Constraint object with multipole constraints corresponding to
            the specified deformation parameters
            
        Examples:
            >>> # Prolate deformation (typical for rare-earth nuclei)
            >>> c = Constraint.from_beta_gamma(
            ...     mass_number=238,
            ...     beta2=0.25
            ... )
            
            >>> # Oblate deformation (using negative beta)
            >>> c = Constraint.from_beta_gamma(
            ...     mass_number=40,
            ...     beta2=-0.30
            ... )
            ...     beta2=0.3,
            ...     gamma=60
            ... )
            
            >>> # Triaxial deformation
            >>> c = Constraint.from_beta_gamma(
            ...     mass_number=190,
            ...     beta2=0.2,
            ...     gamma=30
            ... )
            
            >>> # Octupole deformation (pear shape, e.g., for Ra-224)
            >>> c = Constraint.from_beta_gamma(
            ...     mass_number=224,
            ...     beta2=0.1,
            ...     gamma=0,
            ...     beta3=0.08
            ... )
            
            >>> # Multiple deformations
            >>> c = Constraint.from_beta_gamma(
            ...     mass_number=152,
            ...     beta2=0.3,
            ...     gamma=0,
            ...     beta4=0.05
            ... )
        
        Notes:
            - The conversion uses R₀ = r₀ × A^(1/3) for the nuclear radius
            - Standard r₀ = 1.2 fm, but can be adjusted (e.g., 1.1-1.3 fm)
            - For axially symmetric shapes, set gamma=0 (prolate) or gamma=60 (oblate)
            - For reflection-asymmetric shapes, include beta3 (octupole)
        """
        # Convert beta-gamma to multipoles
        multipoles = beta_gamma_to_multipoles(
            beta2=beta2,
            gamma=gamma,
            beta3=beta3,
            beta4=beta4,
            mass_number=mass_number,
            r0=r0
        )
        
        # Create constraint from multipoles
        return cls.from_multipoles(
            multipoles=multipoles,
            principal_axes=principal_axes,
            **kwargs
        )
    
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
    
    def add_multipole(
        self, 
        multipole: Union[str, Tuple[int, int]], 
        value: float
    ) -> "Constraint":
        """
        Add a multipole constraint to existing constraints.
        
        Args:
            multipole: Multipole specification (name or tuple)
            value: Target value
            
        Returns:
            New Constraint with added multipole
            
        Example:
            >>> c = Constraint.spherical()
            >>> c = c.add_multipole('Q20', 5.0)
            >>> c = c.add_multipole((3, 0), 2.0)
        """
        lam, mu = parse_multipole(multipole)
        new_multipoles = dict(self.multipole_constraints)
        new_multipoles[(lam, mu)] = value
        return dataclasses.replace(
            self, 
            multipole_constraints=new_multipoles,
            tconstraint=True
        )
    
    def remove_multipole(
        self, 
        multipole: Union[str, Tuple[int, int]]
    ) -> "Constraint":
        """
        Remove a multipole constraint.
        
        Args:
            multipole: Multipole specification to remove
            
        Returns:
            New Constraint without the specified multipole
        """
        lam, mu = parse_multipole(multipole)
        new_multipoles = dict(self.multipole_constraints)
        if (lam, mu) in new_multipoles:
            del new_multipoles[(lam, mu)]
        
        tconstraint = len(new_multipoles) > 0 or self.tq_prin_axes
        return dataclasses.replace(
            self, 
            multipole_constraints=new_multipoles,
            tconstraint=tconstraint
        )
    
    def get_multipole_list(self) -> List[Tuple[Tuple[int, int], float]]:
        """
        Get list of all multipole constraints as (quantum numbers, value) pairs.
        
        Returns:
            List of ((lambda, mu), value) tuples sorted by lambda then mu
        """
        return sorted(self.multipole_constraints.items(), key=lambda x: (x[0][0], x[0][1]))
    
    def to_beta_gamma(self, mass_number: float, r0: float = 1.2) -> Dict[str, float]:
        """
        Convert current multipole constraints to beta-gamma parameters.
        
        Useful for reporting deformation in standard parameterization.
        
        Args:
            mass_number: Nuclear mass number A
            r0: Radius parameter in fm (default: 1.2 fm)
            
        Returns:
            Dictionary with 'beta2', 'gamma', 'beta3', 'beta4' as available
            
        Example:
            >>> c = Constraint.from_multipoles({'Q20': 10.0, 'Q22': 2.0})
            >>> params = c.to_beta_gamma(mass_number=40)
            >>> print(f"β₂ = {params['beta2']:.3f}, γ = {params['gamma']:.1f}°")
        """
        return multipoles_to_beta_gamma(
            self.multipole_constraints,
            mass_number=mass_number,
            r0=r0
        )
    
    def initialize_arrays(self, grid_shape: tuple, mass_number: float) -> "Constraint":
        """
        Initialize constraint arrays for a given grid.
        
        Args:
            grid_shape: Tuple of (nx, ny, nz) grid dimensions
            mass_number: Mass number of the nucleus
            
        Returns:
            New Constraint with initialized arrays
        """
        # Count number of constraints from multipoles
        numconstraint = len(self.multipole_constraints)
        
        # Add principal axes constraints if enabled
        if self.tq_prin_axes:
            numconstraint += 6  # xy, xz, yz, x, y, z
        
        if numconstraint == 0:
            return dataclasses.replace(self, numconstraint=0, tconstraint=False)
        
        nx, ny, nz = grid_shape
        
        # Build list of constraint multipoles in order
        constraint_multipoles = [mp for mp, _ in self.get_multipole_list()]
        
        # Add principal axes if needed
        if self.tq_prin_axes:
            # These are represented as special "multipoles" internally
            constraint_multipoles.extend([
                (-1, 0),  # xy cross term
                (-1, 1),  # xz cross term
                (-1, 2),  # yz cross term
                (-2, 0),  # x displacement
                (-2, 1),  # y displacement
                (-2, 2),  # z displacement
            ])
        
        # Build goal array from multipole constraints
        goal_crank = jnp.zeros(numconstraint)
        for i, (mp, val) in enumerate(self.get_multipole_list()):
            goal_crank = goal_crank.at[i].set(val)
        
        return dataclasses.replace(
            self,
            numconstraint=numconstraint,
            tconstraint=True,
            constraint_multipoles=constraint_multipoles,
            constr_field=jnp.zeros((numconstraint, nx, ny, nz, 2)),
            lambda_crank=jnp.full(numconstraint, -0.2),
            dlambda_crank=jnp.zeros(numconstraint),
            goal_crank=goal_crank,
            actual_crank=jnp.zeros(numconstraint),
            old_crank=jnp.zeros(numconstraint),
            actual_crank2=jnp.zeros(numconstraint),
            qcorr=jnp.zeros(numconstraint),
        )
    
    def __str__(self) -> str:
        if not self.tconstraint:
            return "Constraint(spherical)"
        
        parts = []
        
        # Show multipole constraints
        for (lam, mu), value in self.get_multipole_list():
            name = multipole_name(lam, mu)
            parts.append(f"{name}={value:.3f}")
        
        # Legacy compatibility
        if self.alpha20_wanted > -1e90 and (2, 0) not in self.multipole_constraints:
            parts.append(f"α20={self.alpha20_wanted:.3f}")
        if self.alpha22_wanted > -1e90 and (2, 2) not in self.multipole_constraints:
            parts.append(f"α22={self.alpha22_wanted:.3f}")
        
        if self.tq_prin_axes:
            parts.append("principal_axes")
        
        return f"Constraint({', '.join(parts)})"
        
        return f"Constraint({', '.join(parts)})"
    
    def __repr__(self) -> str:
        return (
            f"Constraint(alpha20_wanted={self.alpha20_wanted}, "
            f"alpha22_wanted={self.alpha22_wanted}, "
            f"tq_prin_axes={self.tq_prin_axes})"
        )
