"""JAX configuration for jax-hfbfft.

This module sets up JAX with 64-bit precision by default and provides
utilities for dtype management.
"""

import os

# Set 64-bit precision BEFORE importing JAX
# This must happen before any JAX imports
os.environ.setdefault('JAX_ENABLE_X64', 'True')

import jax
import jax.numpy as jnp
from typing import Union, Type
from dataclasses import dataclass
from enum import Enum


class Precision(Enum):
    """Precision modes for JAX calculations."""
    SINGLE = 32
    DOUBLE = 64


@dataclass
class DTypes:
    """Container for consistent dtype usage throughout the package."""
    float: Type
    complex: Type
    int: Type
    
    @classmethod
    def from_precision(cls, precision: Union[int, Precision] = 64) -> 'DTypes':
        """Create DTypes for given precision.
        
        Args:
            precision: 32 for single precision, 64 for double precision
            
        Returns:
            DTypes instance with appropriate types
        """
        if isinstance(precision, Precision):
            precision = precision.value
            
        if precision == 64:
            return cls(
                float=jnp.float64,
                complex=jnp.complex128,
                int=jnp.int64
            )
        elif precision == 32:
            return cls(
                float=jnp.float32,
                complex=jnp.complex64,
                int=jnp.int32
            )
        else:
            raise ValueError(f"Unsupported precision: {precision}. Use 32 or 64.")


# Default dtypes (64-bit precision)
_dtypes = DTypes.from_precision(64)


def get_dtypes() -> DTypes:
    """Get current dtype configuration."""
    return _dtypes


def set_precision(precision: Union[int, Precision] = 64) -> DTypes:
    """Set the precision for all calculations.
    
    Args:
        precision: 32 for single precision, 64 for double precision
        
    Returns:
        Updated DTypes instance
        
    Note:
        This should be called before any calculations are performed.
        JAX_ENABLE_X64 must be True for 64-bit precision to work.
    """
    global _dtypes
    
    if isinstance(precision, Precision):
        precision = precision.value
        
    if precision == 64:
        if not jax.config.x64_enabled:
            # Try to enable it
            jax.config.update("jax_enable_x64", True)
            if not jax.config.x64_enabled:
                raise RuntimeError(
                    "64-bit precision requested but JAX x64 mode could not be enabled. "
                    "Set JAX_ENABLE_X64=True environment variable before importing jax."
                )
    
    _dtypes = DTypes.from_precision(precision)
    return _dtypes


def check_precision() -> None:
    """Check and report current precision settings."""
    x64_enabled = jax.config.x64_enabled
    print(f"JAX x64 mode: {'enabled' if x64_enabled else 'disabled'}")
    print(f"Float dtype: {_dtypes.float}")
    print(f"Complex dtype: {_dtypes.complex}")
    print(f"Int dtype: {_dtypes.int}")
    
    if not x64_enabled:
        print("\nWARNING: 64-bit mode is disabled. For best precision, set:")
        print("  export JAX_ENABLE_X64=True")
        print("  or call: jax.config.update('jax_enable_x64', True)")


# Convenience functions for creating arrays with correct dtypes
def zeros(shape, dtype=None):
    """Create zeros array with default float dtype."""
    if dtype is None:
        dtype = _dtypes.float
    return jnp.zeros(shape, dtype=dtype)


def zeros_complex(shape):
    """Create zeros array with complex dtype."""
    return jnp.zeros(shape, dtype=_dtypes.complex)


def ones(shape, dtype=None):
    """Create ones array with default float dtype."""
    if dtype is None:
        dtype = _dtypes.float
    return jnp.ones(shape, dtype=dtype)


def array(data, dtype=None):
    """Create array with default float dtype."""
    if dtype is None:
        dtype = _dtypes.float
    return jnp.array(data, dtype=dtype)


def arange(*args, dtype=None, **kwargs):
    """Create arange with default int dtype."""
    if dtype is None:
        dtype = _dtypes.int
    return jnp.arange(*args, dtype=dtype, **kwargs)


# Verify 64-bit is enabled on import
if not jax.config.x64_enabled:
    try:
        jax.config.update("jax_enable_x64", True)
    except:
        pass  # Will warn when check_precision is called
