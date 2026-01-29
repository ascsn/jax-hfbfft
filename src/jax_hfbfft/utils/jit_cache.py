"""
JAX JIT Compilation Cache Configuration.

This module provides utilities to enable persistent JIT caching for JAX,
which can significantly reduce startup time for repeated calculations.

Usage:
    # At the start of your script (before any JAX operations):
    from jax_hfbfft.utils.jit_cache import enable_jit_cache
    enable_jit_cache()
    
    # Or set environment variables BEFORE importing JAX:
    # export JAX_ENABLE_COMPILATION_CACHE=true
    # export JAX_COMPILATION_CACHE_DIR=/path/to/cache
    # export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
"""

import os
from pathlib import Path
from typing import Optional


def enable_jit_cache(
    cache_dir: Optional[str] = None,
    min_entry_size: int = 0,
    min_compile_time: float = 0.0,
) -> str:
    """
    Enable persistent JIT compilation caching for JAX.
    
    This stores compiled XLA computations on disk so they can be reused
    across Python sessions. This is particularly useful for:
    
    1. **Uncertainty Quantification**: Varying force parameters (t0, t1, v0, etc.)
       doesn't change the computation graph, so cached compilations are reused.
       
    2. **Constrained Calculations**: Adding constraint potentials uses the same
       compiled kernels as unconstrained calculations.
       
    3. **Same Grid/Basis Calculations**: Any calculations with the same grid size
       and number of states will reuse cached compilations.
    
    Things that trigger recompilation (cache miss):
    - Different grid dimensions (nx, ny, nz)
    - Different number of states (npsi_n, npsi_p)
    - Different pairing type (ipair)
    - Different static configuration options
    
    Args:
        cache_dir: Directory for cache storage. Defaults to ~/.jax_cache
        min_entry_size: Minimum compiled object size to cache (bytes). 
                       Default 0 caches everything.
        min_compile_time: Minimum compile time to cache (seconds).
                         Default 0 caches everything.
    
    Returns:
        The path to the cache directory being used.
        
    Example:
        >>> from jax_hfbfft.utils.jit_cache import enable_jit_cache
        >>> cache_path = enable_jit_cache()
        >>> print(f"JIT cache at: {cache_path}")
        
        # Now run your calculation - first run will be slow
        # Subsequent runs with same grid/basis will be fast
    """
    import jax
    
    if cache_dir is None:
        cache_dir = str(Path.home() / ".jax_cache" / "jax_hfbfft")
    
    # Create cache directory if it doesn't exist
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    # Use JAX's config API (works even after JAX is imported)
    jax.config.update("jax_enable_compilation_cache", True)
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", min_compile_time)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", min_entry_size)
    
    return cache_dir


def is_cache_enabled() -> bool:
    """Check if JAX compilation caching is currently enabled."""
    import jax
    return jax.config.jax_enable_compilation_cache


def get_cache_config() -> dict:
    """Get current JAX cache configuration settings."""
    import jax
    return {
        "enabled": jax.config.jax_enable_compilation_cache,
        "cache_dir": jax.config.jax_compilation_cache_dir,
        "min_compile_time_secs": jax.config.jax_persistent_cache_min_compile_time_secs,
        "min_entry_size_bytes": jax.config.jax_persistent_cache_min_entry_size_bytes,
    }


def clear_jit_cache(cache_dir: Optional[str] = None) -> int:
    """
    Clear the JIT compilation cache.
    
    Args:
        cache_dir: Directory to clear. Defaults to ~/.jax_cache/jax_hfbfft
        
    Returns:
        Number of files deleted.
    """
    import shutil
    
    if cache_dir is None:
        cache_dir = str(Path.home() / ".jax_cache" / "jax_hfbfft")
    
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return 0
    
    count = sum(1 for _ in cache_path.rglob("*") if _.is_file())
    shutil.rmtree(cache_path)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    return count


def get_cache_info(cache_dir: Optional[str] = None) -> dict:
    """
    Get information about the JIT cache.
    
    Returns:
        Dictionary with cache statistics.
    """
    if cache_dir is None:
        cache_dir = str(Path.home() / ".jax_cache" / "jax_hfbfft")
    
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return {"exists": False, "path": cache_dir}
    
    files = list(cache_path.rglob("*"))
    file_count = sum(1 for f in files if f.is_file())
    total_size = sum(f.stat().st_size for f in files if f.is_file())
    
    return {
        "exists": True,
        "path": cache_dir,
        "file_count": file_count,
        "total_size_mb": total_size / (1024 * 1024),
    }


# What triggers recompilation vs what is traced
RECOMPILATION_TRIGGERS = """
=== JAX Compilation Behavior for jax-hfbfft ===

Things that REUSE cached compilations (can vary freely):
- Force parameters: t0, t1, t2, t3, x0, x1, x2, x3, power
- Pairing strengths: v0prot, v0neut, rho0pr
- Effective mass: h2m values
- Grid spacing: dx, dy, dz
- Solver config: x0dmp, e0dmp, density_mixing
- Constraint potentials (array values)
- Coulomb potential values
- All wavefunction and density arrays

Things that TRIGGER recompilation (changing these = new compile):
- Grid dimensions: nx, ny, nz (static)
- Number of states: npsi_n, npsi_p (static)
- Pairing type: ipair (static)
- Coulomb on/off: use_coulomb (static)
- Energy computation: compute_energy (static)
- Particle numbers: nucleus_n, nucleus_z (static for some functions)
- Exchange type: ex (static)

Typical scenarios:
1. UQ with varying t0, t1, etc. → ALL CACHED (same compile reused)
2. Constrained vs unconstrained → ALL CACHED (same compile reused)
3. Different nuclei, same grid → PARTIAL (some recompile due to N, Z)
4. Different grid sizes → FULL RECOMPILE needed
"""


def print_compilation_info():
    """Print information about what triggers recompilation."""
    print(RECOMPILATION_TRIGGERS)
