"""
Nucleus class for specifying nuclear systems.

This module provides the Nucleus class which encapsulates the specification
of a nuclear system including proton and neutron numbers.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Nucleus:
    """
    Specification of a nuclear system.
    
    This immutable class defines the basic properties of a nucleus including
    proton number (Z), neutron number (N), and optionally a name/symbol.
    
    Attributes:
        protons (int): Number of protons (Z)
        neutrons (int): Number of neutrons (N)
        name (str, optional): Optional name or symbol for the nucleus
        
    Properties:
        mass_number (int): Total nucleon number A = Z + N
        charge_number (int): Same as protons (Z)
        
    Examples:
        >>> # Create Sn-132 by specifying protons and neutrons
        >>> nucleus = Nucleus(protons=50, neutrons=82)
        >>> print(f"A={nucleus.mass_number}, Z={nucleus.charge_number}")
        A=132, Z=50
        
        >>> # Create with a name
        >>> pb208 = Nucleus(protons=82, neutrons=126, name="Pb-208")
        
        >>> # Use class method for common isotopes
        >>> ca40 = Nucleus.from_symbol("Ca", mass_number=40)
    """
    
    protons: int
    neutrons: int
    name: Optional[str] = None
    
    def __post_init__(self):
        """Validate nucleus parameters."""
        if self.protons < 0:
            raise ValueError(f"Number of protons must be non-negative, got {self.protons}")
        if self.neutrons < 0:
            raise ValueError(f"Number of neutrons must be non-negative, got {self.neutrons}")
    
    @property
    def mass_number(self) -> int:
        """Total nucleon number A = Z + N."""
        return self.protons + self.neutrons
    
    @property
    def charge_number(self) -> int:
        """Charge number (same as proton number)."""
        return self.protons
    
    @property
    def Z(self) -> int:
        """Proton number (alias for charge_number)."""
        return self.protons
    
    @property
    def N(self) -> int:
        """Neutron number."""
        return self.neutrons
    
    @property
    def A(self) -> int:
        """Mass number (alias for mass_number)."""
        return self.mass_number
    
    @classmethod
    def from_symbol(cls, symbol: str, mass_number: int) -> "Nucleus":
        """
        Create a Nucleus from element symbol and mass number.
        
        Args:
            symbol: Element symbol (e.g., "Ca", "Sn", "Pb")
            mass_number: Total number of nucleons
            
        Returns:
            Nucleus instance
            
        Examples:
            >>> ca40 = Nucleus.from_symbol("Ca", 40)
            >>> sn132 = Nucleus.from_symbol("Sn", 132)
        """
        # Mapping of element symbols to proton numbers
        element_z = {
            "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
            "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
            "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22,
            "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29,
            "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
            "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43,
            "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
            "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57,
            "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64,
            "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71,
            "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78,
            "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85,
            "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90, "Pa": 91, "U": 92,
            "Np": 93, "Pu": 94, "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98, "Es": 99,
            "Fm": 100, "Md": 101, "No": 102, "Lr": 103, "Rf": 104, "Db": 105,
            "Sg": 106, "Bh": 107, "Hs": 108, "Mt": 109, "Ds": 110, "Rg": 111,
            "Cn": 112, "Nh": 113, "Fl": 114, "Mc": 115, "Lv": 116, "Ts": 117,
            "Og": 118
        }
        
        if symbol not in element_z:
            raise ValueError(f"Unknown element symbol: {symbol}")
        
        z = element_z[symbol]
        n = mass_number - z
        
        if n < 0:
            raise ValueError(
                f"Invalid mass number {mass_number} for {symbol} (Z={z}): "
                f"would give negative neutron number"
            )
        
        return cls(protons=z, neutrons=n, name=f"{symbol}-{mass_number}")
    
    def __str__(self) -> str:
        if self.name:
            return self.name
        return f"Z={self.protons}, N={self.neutrons}"
    
    def __repr__(self) -> str:
        return f"Nucleus(protons={self.protons}, neutrons={self.neutrons}, name={self.name!r})"
