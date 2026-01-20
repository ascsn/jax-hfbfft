"""
Coulomb interaction for HFB.

This module computes the Coulomb potential from the proton density
using FFT-based Poisson solver. Supports:
- Periodic boundary conditions (momentum space)
- Open boundary conditions (zero-padded convolution)
"""

import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Tuple

from jax_hfbfft.jax_config import get_dtypes
from jax_hfbfft.core.grid import Grid


@jax.tree_util.register_dataclass
@dataclass
class CoulombSolver:
    """
    Coulomb solver state.
    
    For periodic BCs: q contains 1/k^2 in momentum space
    For open BCs: q contains FFT of the Green's function
    """
    nx2: int  # Extended grid size (x)
    ny2: int  # Extended grid size (y)
    nz2: int  # Extended grid size (z)
    q: jax.Array  # Coulomb kernel
    wcoul: jax.Array  # Last computed Coulomb potential
    periodic: bool  # Boundary condition flag
    
    @classmethod
    def create(cls, grid: Grid, e2: float = 1.4399784) -> "CoulombSolver":
        """
        Create and initialize Coulomb solver.
        
        Args:
            grid: Spatial grid
            e2: Coulomb constant e^2 in MeV*fm
            
        Returns:
            Initialized CoulombSolver
        """
        dtypes = get_dtypes()
        
        if grid.periodic:
            nx2, ny2, nz2 = grid.nx, grid.ny, grid.nz
        else:
            nx2, ny2, nz2 = 2 * grid.nx, 2 * grid.ny, 2 * grid.nz
        
        # Initialize momentum/position space indices
        q = _init_coulomb_kernel(
            nx2, ny2, nz2,
            grid.dx, grid.dy, grid.dz,
            grid.periodic
        )
        
        wcoul = jnp.zeros((grid.nx, grid.ny, grid.nz), dtype=dtypes.float)
        
        return cls(
            nx2=nx2,
            ny2=ny2, 
            nz2=nz2,
            q=q,
            wcoul=wcoul,
            periodic=grid.periodic,
        )


def _init_iq(n: int, d: float, periodic: bool) -> jax.Array:
    """
    Initialize momentum-space or position-space indices.
    
    For periodic: returns k^2 = (2*pi*n / L)^2
    For open: returns r^2 = (d * n)^2
    """
    idx = jnp.arange(n)
    idx = jnp.where(idx <= n // 2, idx, idx - n)
    
    if periodic:
        return (2.0 * jnp.pi * idx / (n * d)) ** 2
    else:
        return (d * idx) ** 2


def _init_coulomb_kernel(
    nx2: int, ny2: int, nz2: int,
    dx: float, dy: float, dz: float,
    periodic: bool,
) -> jax.Array:
    """
    Initialize the Coulomb kernel.
    
    For periodic BCs: 4*pi/k^2 in momentum space
    For open BCs: FFT of 1/r Green's function
    """
    dtypes = get_dtypes()
    
    iqx = _init_iq(nx2, dx, periodic)
    iqy = _init_iq(ny2, dy, periodic)
    iqz = _init_iq(nz2, dz, periodic)
    
    # Create 3D grid of k^2 or r^2
    i = iqx[:, None, None]
    j = iqy[None, :, None]
    k = iqz[None, None, :]
    q = (i + j + k).astype(dtypes.complex)
    
    # Avoid division by zero at origin
    q = q.at[0, 0, 0].set(1.0)
    
    if periodic:
        # Periodic: use 1/k^2 directly
        q = 1.0 / jnp.real(q)
        q = q.at[0, 0, 0].set(0.0)  # No k=0 contribution
    else:
        # Open: use 1/r and FFT
        q = 1.0 / jnp.sqrt(jnp.real(q))
        # Origin value from analytic integration
        q = q.at[0, 0, 0].set(2.84 / (dx * dy * dz) ** (1.0 / 3.0))
        q = jnp.fft.fftn(q)
    
    return q


def solve_poisson(
    rho_proton: jax.Array,
    solver: CoulombSolver,
    grid: Grid,
    e2: float = 1.4399784,
) -> jax.Array:
    """
    Solve Poisson equation for Coulomb potential.
    
    Computes: V_C(r) = e^2 * integral(rho_p(r') / |r - r'|)
    
    Args:
        rho_proton: Proton density (nx, ny, nz)
        solver: CoulombSolver object
        grid: Spatial grid
        e2: Coulomb constant
        
    Returns:
        Coulomb potential (nx, ny, nz)
    """
    dtypes = get_dtypes()
    
    # Zero-pad for open BCs
    rho2 = jnp.zeros(
        (solver.nx2, solver.ny2, solver.nz2),
        dtype=dtypes.complex
    )
    rho2 = rho2.at[:grid.nx, :grid.ny, :grid.nz].set(rho_proton)
    
    # FFT
    rho2 = jnp.fft.fftn(rho2)
    
    # Multiply by kernel
    if solver.periodic:
        rho2 = rho2 * (4.0 * jnp.pi * e2 * jnp.real(solver.q))
    else:
        rho2 = rho2 * (e2 * grid.wxyz * solver.q)
    
    # Inverse FFT
    wcoul = jnp.fft.ifftn(rho2)
    
    # Extract result
    return jnp.real(wcoul[:grid.nx, :grid.ny, :grid.nz])


def compute_coulomb_energy(
    rho_proton: jax.Array,
    wcoul: jax.Array,
    force,
    wxyz: float,
) -> Tuple[float, float]:
    """
    Compute Coulomb energy contributions.
    
    Args:
        rho_proton: Proton density
        wcoul: Coulomb potential
        force: Force parameters (for Slater exchange)
        wxyz: Integration weight
        
    Returns:
        (direct_energy, exchange_energy)
    """
    # Direct Coulomb
    e_direct = 0.5 * wxyz * jnp.sum(rho_proton * wcoul)
    
    # Slater exchange
    e_exchange = 0.0
    if force.ex != 0:
        slater_coeff = -3.0 / 4.0 * force.slate
        e_exchange = wxyz * jnp.sum(slater_coeff * rho_proton ** (4.0 / 3.0))
    
    return float(e_direct), float(e_exchange)
