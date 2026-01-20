"""
Mathematical utility functions for JAX-HFBFFT.

This module provides common mathematical operations used in HFB calculations.
"""

import jax
import jax.numpy as jnp


@jax.jit
def rpsnorm(ps: jax.Array, wxyz: float) -> jax.Array:
    """
    Calculate the squared norm of a wavefunction.
    
    Args:
        ps: Wavefunction array
        wxyz: Integration weight (volume element)
        
    Returns:
        Squared norm of the wavefunction
    """
    return wxyz * jnp.sum(jnp.real(jnp.conjugate(ps) * ps))


@jax.jit
def overlap(pl: jax.Array, pr: jax.Array, wxyz: float) -> jax.Array:
    """
    Calculate the overlap between two wavefunctions.
    
    Args:
        pl: Left wavefunction (will be conjugated)
        pr: Right wavefunction
        wxyz: Integration weight (volume element)
        
    Returns:
        Complex overlap integral <pl|pr>
    """
    return wxyz * jnp.sum(jnp.conjugate(pl) * pr)


@jax.jit
def normalize(psi: jax.Array, wxyz: float) -> jax.Array:
    """
    Normalize a wavefunction.
    
    Args:
        psi: Wavefunction to normalize
        wxyz: Integration weight
        
    Returns:
        Normalized wavefunction
    """
    norm = jnp.sqrt(rpsnorm(psi, wxyz))
    return psi / norm


def gram_schmidt(psi_array: jax.Array, wxyz: float) -> jax.Array:
    """
    Orthonormalize a set of wavefunctions using Gram-Schmidt.
    
    Args:
        psi_array: Array of wavefunctions with shape (nstates, ...)
        wxyz: Integration weight
        
    Returns:
        Orthonormalized wavefunctions
    """
    nstates = psi_array.shape[0]
    result = psi_array.copy()
    
    for i in range(nstates):
        # Normalize the current vector
        result = result.at[i].set(normalize(result[i], wxyz))
        
        # Orthogonalize remaining vectors
        for j in range(i + 1, nstates):
            proj = overlap(result[i], result[j], wxyz)
            result = result.at[j].add(-proj * result[i])
    
    return result
