"""
Grid class for spatial discretization.

This module provides the Grid class which encapsulates the 3D spatial grid
used for wavefunctions and densities in HFB calculations.
"""

import jax
import jax.numpy as jnp
from dataclasses import dataclass, field
from typing import Tuple, Optional

from jax_hfbfft.jax_config import get_dtypes


@jax.tree_util.register_dataclass
@dataclass
class Grid:
    """
    3D spatial grid for HFB calculations.
    
    This class encapsulates the spatial discretization including grid dimensions,
    spacing, derivative operators, and mesh arrays. It is designed to be used
    as a JAX pytree for compatibility with JIT compilation.
    
    Attributes:
        nx, ny, nz (int): Number of grid points in each dimension
        dx, dy, dz (float): Grid spacing in each dimension (fm)
        periodic (bool): Whether to use periodic boundary conditions
        
    Properties:
        wxyz (float): Integration weight (volume element)
        shape (tuple): Grid shape (nx, ny, nz)
        
    Examples:
        >>> # Create a 32^3 grid with 0.8 fm spacing
        >>> grid = Grid.create(nx=32, ny=32, nz=32, dx=0.8, dy=0.8, dz=0.8)
        
        >>> # Create a cubic grid
        >>> grid = Grid.create_cubic(n=48, d=0.8)
    """
    
    # Grid dimensions (static - cannot change during JIT)
    nx: int = field(metadata=dict(static=True))
    ny: int = field(metadata=dict(static=True))
    nz: int = field(metadata=dict(static=True))
    
    # Grid spacing
    dx: float
    dy: float
    dz: float
    
    # Boundary conditions
    periodic: bool = field(metadata=dict(static=True))
    
    # Bloch angle boundary conditions (for twisted boundary conditions)
    bangx: float = 0.0
    bangy: float = 0.0
    bangz: float = 0.0
    
    # TABC (twisted average boundary conditions) indices
    tabc_x: int = 0
    tabc_y: int = 0
    tabc_z: int = 0
    
    # Integration weight
    wxyz: float = 0.0
    
    # Coordinate arrays
    x: Optional[jax.Array] = None
    y: Optional[jax.Array] = None
    z: Optional[jax.Array] = None
    
    # Derivative operators (Fourier-based)
    der1x: Optional[jax.Array] = None
    der2x: Optional[jax.Array] = None
    cdmpx: Optional[jax.Array] = None
    der1y: Optional[jax.Array] = None
    der2y: Optional[jax.Array] = None
    cdmpy: Optional[jax.Array] = None
    der1z: Optional[jax.Array] = None
    der2z: Optional[jax.Array] = None
    cdmpz: Optional[jax.Array] = None
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        """Return the grid shape as a tuple."""
        return (self.nx, self.ny, self.nz)
    
    @property
    def volume(self) -> float:
        """Return the total grid volume in fm^3."""
        return self.nx * self.ny * self.nz * self.dx * self.dy * self.dz
    
    @classmethod
    def create(
        cls,
        nx: int = 32,
        ny: int = 32, 
        nz: int = 32,
        dx: float = 0.8,
        dy: float = 0.8,
        dz: float = 0.8,
        periodic: bool = False,
        bangx: float = 0.0,
        bangy: float = 0.0,
        bangz: float = 0.0,
    ) -> "Grid":
        """
        Create a new Grid with the specified parameters.
        
        Args:
            nx, ny, nz: Number of grid points in each dimension
            dx, dy, dz: Grid spacing in each dimension (fm)
            periodic: Whether to use periodic boundary conditions
            bangx, bangy, bangz: Bloch angles for twisted boundary conditions
            
        Returns:
            Initialized Grid object
        """
        # Calculate integration weight
        wxyz = dx * dy * dz
        
        # Create coordinate arrays (centered at origin)
        x = jnp.linspace(-(nx-1)/2 * dx, (nx-1)/2 * dx, nx)
        y = jnp.linspace(-(ny-1)/2 * dy, (ny-1)/2 * dy, ny)
        z = jnp.linspace(-(nz-1)/2 * dz, (nz-1)/2 * dz, nz)
        
        # Initialize derivative operators
        der1x = _sder(nx, dx)
        der2x = _sder2(nx, dx)
        der1y = _sder(ny, dy)
        der2y = _sder2(ny, dy)
        der1z = _sder(nz, dz)
        der2z = _sder2(nz, dz)
        
        # Damping matrices (identity for now, can be modified for boundaries)
        cdmpx = jnp.eye(nx)
        cdmpy = jnp.eye(ny)
        cdmpz = jnp.eye(nz)
        
        return cls(
            nx=nx, ny=ny, nz=nz,
            dx=dx, dy=dy, dz=dz,
            periodic=periodic,
            bangx=bangx * jnp.pi,
            bangy=bangy * jnp.pi,
            bangz=bangz * jnp.pi,
            tabc_x=0, tabc_y=0, tabc_z=0,
            wxyz=wxyz,
            x=x, y=y, z=z,
            der1x=der1x, der2x=der2x, cdmpx=cdmpx,
            der1y=der1y, der2y=der2y, cdmpy=cdmpy,
            der1z=der1z, der2z=der2z, cdmpz=cdmpz,
        )
    
    @classmethod
    def create_cubic(cls, n: int = 32, d: float = 0.8, **kwargs) -> "Grid":
        """
        Create a cubic grid with equal dimensions and spacing.
        
        Args:
            n: Number of grid points in each dimension
            d: Grid spacing in each dimension (fm)
            **kwargs: Additional arguments passed to create()
            
        Returns:
            Initialized cubic Grid object
        """
        return cls.create(nx=n, ny=n, nz=n, dx=d, dy=d, dz=d, **kwargs)
    
    def get_meshgrid(self) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        Return 3D meshgrid arrays for x, y, z coordinates.
        
        Returns:
            Tuple of (xx, yy, zz) meshgrid arrays with shape (nx, ny, nz)
        """
        return jnp.meshgrid(self.x, self.y, self.z, indexing='ij')


def _sder(nmax: int, d: float) -> jnp.ndarray:
    """
    Compute first derivative matrix using Fourier spectral method.
    
    Args:
        nmax: Number of grid points
        d: Grid spacing
        
    Returns:
        First derivative matrix
    """
    icn = (nmax + 1) // 2
    afac = jnp.pi / icn

    i = jnp.arange(1, nmax + 1, dtype=jnp.int64)
    j = jnp.arange(1, icn, dtype=jnp.int64)[:, jnp.newaxis, jnp.newaxis]
    grid = i[jnp.newaxis, :] - i[:, jnp.newaxis]

    der = -afac * (
        (jnp.sum(-j * jnp.sin(j * afac * grid), axis=0)) - 
        (0.5 * icn * jnp.sin(icn * afac * grid))
    ) / (icn * d)

    return der


def _sder2(nmax: int, d: float) -> jnp.ndarray:
    """
    Compute second derivative matrix using Fourier spectral method.
    
    Args:
        nmax: Number of grid points
        d: Grid spacing
        
    Returns:
        Second derivative matrix
    """
    icn = (nmax + 1) // 2
    afac = jnp.pi / icn

    i = jnp.arange(1, nmax + 1, dtype=jnp.int64)
    j = jnp.arange(1, icn, dtype=jnp.int64)[:, jnp.newaxis, jnp.newaxis]
    grid = i[jnp.newaxis, :] - i[:, jnp.newaxis]

    der = -(afac * afac) * (
        (jnp.sum((j ** 2) * jnp.cos(j * afac * grid), axis=0)) + 
        (0.5 * (icn ** 2) * jnp.cos(icn * afac * grid))
    ) / (icn * d * d)

    return der


# =============================================================================
# FFT-based derivative operators for wavefunctions
# These are more efficient than matrix multiplication for 3D fields
# =============================================================================

@jax.jit
def deriv_x(psi: jax.Array, dx: float) -> jax.Array:
    """
    Compute first derivative in x-direction using FFT.
    
    Args:
        psi: Wavefunction array with shape (..., nx, ny, nz) or (2, nx, ny, nz) for spinor
        dx: Grid spacing in x
        
    Returns:
        Derivative array with same shape as input
    """
    axis = -3  # x is third-to-last axis
    n = psi.shape[axis]
    kfac = (2.0 * jnp.pi) / (dx * n)
    half_n = n // 2

    # FFT along x
    psi_k = jnp.fft.fft(psi, axis=axis, norm="backward")
    
    # Build wave numbers
    k = jnp.zeros(n)
    k = k.at[:half_n].set(jnp.arange(half_n))
    k = k.at[half_n+1:].set(jnp.arange(-(half_n-1), 0))
    
    # Reshape k for broadcasting
    shape = [1] * psi.ndim
    shape[axis] = n
    k = k.reshape(shape)
    
    # Multiply by ik
    psi_k = psi_k * (1j * k * kfac)
    psi_k = psi_k.at[..., half_n, :, :].set(0.0)  # Zero Nyquist frequency
    
    return jnp.fft.ifft(psi_k, axis=axis, norm="forward")


@jax.jit
def deriv_y(psi: jax.Array, dy: float) -> jax.Array:
    """Compute first derivative in y-direction using FFT."""
    axis = -2  # y is second-to-last axis
    n = psi.shape[axis]
    kfac = (2.0 * jnp.pi) / (dy * n)
    half_n = n // 2

    psi_k = jnp.fft.fft(psi, axis=axis, norm="backward")
    
    k = jnp.zeros(n)
    k = k.at[:half_n].set(jnp.arange(half_n))
    k = k.at[half_n+1:].set(jnp.arange(-(half_n-1), 0))
    
    shape = [1] * psi.ndim
    shape[axis] = n
    k = k.reshape(shape)
    
    psi_k = psi_k * (1j * k * kfac)
    psi_k = psi_k.at[..., :, half_n, :].set(0.0)
    
    return jnp.fft.ifft(psi_k, axis=axis, norm="forward")


@jax.jit  
def deriv_z(psi: jax.Array, dz: float) -> jax.Array:
    """Compute first derivative in z-direction using FFT."""
    axis = -1  # z is last axis
    n = psi.shape[axis]
    kfac = (2.0 * jnp.pi) / (dz * n)
    half_n = n // 2

    psi_k = jnp.fft.fft(psi, axis=axis, norm="backward")
    
    k = jnp.zeros(n)
    k = k.at[:half_n].set(jnp.arange(half_n))
    k = k.at[half_n+1:].set(jnp.arange(-(half_n-1), 0))
    
    shape = [1] * psi.ndim
    shape[axis] = n
    k = k.reshape(shape)
    
    psi_k = psi_k * (1j * k * kfac)
    psi_k = psi_k.at[..., half_n].set(0.0)
    
    return jnp.fft.ifft(psi_k, axis=axis, norm="forward")


@jax.jit
def laplacian(psi: jax.Array, dx: float, dy: float, dz: float) -> jax.Array:
    """
    Compute Laplacian using FFT.
    
    Args:
        psi: Wavefunction array with shape (..., nx, ny, nz)
        dx, dy, dz: Grid spacings
        
    Returns:
        Laplacian with same shape as input
    """
    nx, ny, nz = psi.shape[-3:]
    
    # Wave numbers
    kx = jnp.fft.fftfreq(nx, dx / (2.0 * jnp.pi))
    ky = jnp.fft.fftfreq(ny, dy / (2.0 * jnp.pi))
    kz = jnp.fft.fftfreq(nz, dz / (2.0 * jnp.pi))
    
    # Create k^2 grid
    kx2 = kx**2
    ky2 = ky**2
    kz2 = kz**2
    
    # Reshape for broadcasting
    kx2 = kx2.reshape((-1, 1, 1))
    ky2 = ky2.reshape((1, -1, 1))
    kz2 = kz2.reshape((1, 1, -1))
    
    k2 = kx2 + ky2 + kz2
    
    # Transform, multiply, transform back
    psi_k = jnp.fft.fftn(psi, axes=(-3, -2, -1))
    psi_k = psi_k * (-k2)
    
    return jnp.fft.ifftn(psi_k, axes=(-3, -2, -1))


@jax.jit
def gradient(psi: jax.Array, dx: float, dy: float, dz: float) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Compute gradient (∂/∂x, ∂/∂y, ∂/∂z) using FFT.
    
    Args:
        psi: Wavefunction array with shape (..., nx, ny, nz)
        dx, dy, dz: Grid spacings
        
    Returns:
        Tuple of (dpsi_dx, dpsi_dy, dpsi_dz)
    """
    return deriv_x(psi, dx), deriv_y(psi, dy), deriv_z(psi, dz)

