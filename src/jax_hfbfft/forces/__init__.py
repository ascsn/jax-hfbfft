"""
Force definitions and presets for JAX-HFBFFT.

This module provides predefined Skyrme force parameterizations
and utilities for working with nuclear interactions.
"""

from jax_hfbfft.core.force import Force, BUILTIN_FORCES

# List of all available force names
AVAILABLE_FORCES = list(BUILTIN_FORCES.keys())


def list_forces() -> list:
    """
    Get a list of all available predefined forces.
    
    Returns:
        List of force names that can be used with Force.from_name()
    """
    return AVAILABLE_FORCES.copy()


def get_force_info(name: str) -> dict:
    """
    Get information about a predefined force.
    
    Args:
        name: Name of the force
        
    Returns:
        Dictionary with force parameters
    """
    if name not in BUILTIN_FORCES:
        available = ", ".join(AVAILABLE_FORCES)
        raise ValueError(f"Unknown force '{name}'. Available: {available}")
    
    return BUILTIN_FORCES[name].copy()


__all__ = [
    "Force",
    "AVAILABLE_FORCES",
    "list_forces",
    "get_force_info",
]
