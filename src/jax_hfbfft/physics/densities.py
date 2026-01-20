"""
Density calculations for HFB.

This module computes nuclear densities from single-particle wavefunctions:
- rho: particle density
- tau: kinetic density  
- current: current density vector
- sdens: spin density vector
- sodens: spin-orbit density vector
- chi: pairing density (anomalous density)
"""

import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Tuple

from jax_hfbfft.jax_config import get_dtypes
from jax_hfbfft.core.grid import deriv_x, deriv_y, deriv_z


@jax.tree_util.register_dataclass
@dataclass
class Densities:
    """
    Container for nuclear densities.
    
    All densities have shape (2, nx, ny, nz) where the first index is isospin:
    - 0: neutrons
    - 1: protons
    
    Vector densities (current, sdens, sodens) have shape (2, 3, nx, ny, nz)
    where the second index is the spatial component (x, y, z).
    """
    rho: jax.Array      # Particle density
    tau: jax.Array      # Kinetic density
    chi: jax.Array      # Pairing density (anomalous)
    current: jax.Array  # Current density (vector)
    sdens: jax.Array    # Spin density (vector)
    sodens: jax.Array   # Spin-orbit density (vector)
    
    @classmethod
    def zeros(cls, nx: int, ny: int, nz: int) -> "Densities":
        """Create zero-initialized densities."""
        dtypes = get_dtypes()
        shape4d = (2, nx, ny, nz)
        shape5d = (2, 3, nx, ny, nz)
        
        return cls(
            rho=jnp.zeros(shape4d, dtype=dtypes.float),
            tau=jnp.zeros(shape4d, dtype=dtypes.float),
            chi=jnp.zeros(shape4d, dtype=dtypes.float),
            current=jnp.zeros(shape5d, dtype=dtypes.float),
            sdens=jnp.zeros(shape5d, dtype=dtypes.float),
            sodens=jnp.zeros(shape5d, dtype=dtypes.float),
        )


@jax.jit
def _add_single_state_density(
    carry: Tuple[jax.Array, ...],
    state_data: Tuple[jax.Array, float, float, int]
) -> Tuple[Tuple[jax.Array, ...], None]:
    """
    Add contribution from one single-particle state to densities.
    
    This is designed to be used with jax.lax.scan for efficiency.
    
    Args:
        carry: (rho, tau, chi, current, sdens, sodens, dx, dy, dz)
        state_data: (psi, weight, weightuv, iq) - wavefunction and occupation
        
    Returns:
        Updated densities (same tuple structure as carry)
    """
    rho, tau, chi, current, sdens, sodens, dx, dy, dz = carry
    psi, weight, weightuv, iq = state_data
    
    # psi has shape (2, nx, ny, nz) - spinor components
    psi0 = psi[0]  # Spin-up component
    psi1 = psi[1]  # Spin-down component
    
    # Conjugates
    psi0_conj = jnp.conjugate(psi0)
    psi1_conj = jnp.conjugate(psi1)
    
    # Particle density: rho = sum_sigma |psi_sigma|^2
    rho = rho.at[iq].add(
        weight * jnp.real(psi0_conj * psi0 + psi1_conj * psi1)
    )
    
    # Pairing density (chi): same form but with weightuv
    chi = chi.at[iq].add(
        0.5 * weightuv * jnp.real(psi0_conj * psi0 + psi1_conj * psi1)
    )
    
    # Spin density (sdens = <psi|sigma|psi>)
    # s_x = 2 Re(psi0* psi1)
    sdens = sdens.at[iq, 0].add(2.0 * weight * jnp.real(psi0_conj * psi1))
    # s_y = 2 Im(psi0* psi1)  
    sdens = sdens.at[iq, 1].add(2.0 * weight * jnp.imag(psi0_conj * psi1))
    # s_z = |psi0|^2 - |psi1|^2
    sdens = sdens.at[iq, 2].add(
        weight * jnp.real(psi0_conj * psi0 - psi1_conj * psi1)
    )
    
    # Derivatives for kinetic and spin-orbit densities
    # d/dx
    dpsi_dx = deriv_x(psi, dx)
    dpsi0_dx = dpsi_dx[0]
    dpsi1_dx = dpsi_dx[1]
    
    # Kinetic density contribution from x: |d psi / dx|^2
    tau = tau.at[iq].add(
        weight * jnp.real(
            jnp.conjugate(dpsi0_dx) * dpsi0_dx + 
            jnp.conjugate(dpsi1_dx) * dpsi1_dx
        )
    )
    
    # Current density x-component: Im(psi* d psi/dx)
    current = current.at[iq, 0].add(
        weight * jnp.imag(psi0_conj * dpsi0_dx + psi1_conj * dpsi1_dx)
    )
    
    # Spin-orbit density from x-derivative
    # J_y contribution from d/dx
    sodens = sodens.at[iq, 1].add(
        -weight * jnp.imag(psi0_conj * dpsi0_dx - psi1_conj * dpsi1_dx)
    )
    # J_z contribution from d/dx  
    sodens = sodens.at[iq, 2].add(
        -weight * jnp.real(psi0 * jnp.conjugate(dpsi1_dx) - psi1 * jnp.conjugate(dpsi0_dx))
    )
    
    # d/dy
    dpsi_dy = deriv_y(psi, dy)
    dpsi0_dy = dpsi_dy[0]
    dpsi1_dy = dpsi_dy[1]
    
    tau = tau.at[iq].add(
        weight * jnp.real(
            jnp.conjugate(dpsi0_dy) * dpsi0_dy + 
            jnp.conjugate(dpsi1_dy) * dpsi1_dy
        )
    )
    
    current = current.at[iq, 1].add(
        weight * jnp.imag(psi0_conj * dpsi0_dy + psi1_conj * dpsi1_dy)
    )
    
    # Spin-orbit from y-derivative
    sodens = sodens.at[iq, 0].add(
        weight * jnp.imag(psi0_conj * dpsi0_dy - psi1_conj * dpsi1_dy)
    )
    sodens = sodens.at[iq, 2].add(
        -weight * jnp.imag(psi1_conj * dpsi0_dy + psi0_conj * dpsi1_dy)
    )
    
    # d/dz
    dpsi_dz = deriv_z(psi, dz)
    dpsi0_dz = dpsi_dz[0]
    dpsi1_dz = dpsi_dz[1]
    
    tau = tau.at[iq].add(
        weight * jnp.real(
            jnp.conjugate(dpsi0_dz) * dpsi0_dz + 
            jnp.conjugate(dpsi1_dz) * dpsi1_dz
        )
    )
    
    current = current.at[iq, 2].add(
        weight * jnp.imag(psi0_conj * dpsi0_dz + psi1_conj * dpsi1_dz)
    )
    
    # Spin-orbit from z-derivative
    sodens = sodens.at[iq, 0].add(
        -weight * jnp.real(psi0 * jnp.conjugate(dpsi1_dz) + psi1 * jnp.conjugate(dpsi0_dz))
    )
    sodens = sodens.at[iq, 1].add(
        weight * jnp.imag(psi0 * jnp.conjugate(dpsi1_dz) + psi1 * jnp.conjugate(dpsi0_dz))
    )
    
    return (rho, tau, chi, current, sdens, sodens, dx, dy, dz), None


def compute_densities(
    psi: jax.Array,
    wocc: jax.Array,
    wguv: jax.Array, 
    pairwg: jax.Array,
    isospin: jax.Array,
    grid,
) -> Densities:
    """
    Compute all nuclear densities from wavefunctions.
    
    Args:
        psi: Wavefunctions array with shape (nstates, 2, nx, ny, nz)
        wocc: Occupation weights (BCS v^2) with shape (nstates,)
        wguv: Pairing weights (u*v) with shape (nstates,)
        pairwg: Pairing cutoff weights with shape (nstates,)
        isospin: Isospin indices (0=neutron, 1=proton) with shape (nstates,)
        grid: Grid object with dx, dy, dz
        
    Returns:
        Densities object containing all computed densities
    """
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    nstates = psi.shape[0]
    
    # Initialize densities
    densities = Densities.zeros(nx, ny, nz)
    
    # Compute weights
    weights = wocc
    weightsuv = wguv * pairwg
    
    # Use a simple loop for now (can be optimized with vmap/scan later)
    rho = densities.rho
    tau = densities.tau
    chi = densities.chi
    current = densities.current
    sdens = densities.sdens
    sodens = densities.sodens
    
    # Process each state
    carry = (rho, tau, chi, current, sdens, sodens, grid.dx, grid.dy, grid.dz)
    
    for nst in range(nstates):
        psi_n = psi[nst]
        weight = weights[nst]
        weightuv = weightsuv[nst]
        iq = int(isospin[nst])
        
        carry, _ = _add_single_state_density(
            carry, 
            (psi_n, weight, weightuv, iq)
        )
    
    rho, tau, chi, current, sdens, sodens, _, _, _ = carry
    
    return Densities(
        rho=rho,
        tau=tau,
        chi=chi,
        current=current,
        sdens=sdens,
        sodens=sodens,
    )


@jax.jit
def compute_densities_vmapped(
    psi: jax.Array,
    wocc: jax.Array,
    wguv: jax.Array,
    pairwg: jax.Array,
    isospin: jax.Array,
    dx: float,
    dy: float, 
    dz: float,
    nx: int,
    ny: int,
    nz: int,
) -> Tuple[jax.Array, ...]:
    """
    Vectorized density computation for better GPU performance.
    
    This version uses vmap for parallelization over states.
    """
    nstates = psi.shape[0]
    dtypes = get_dtypes()
    
    # Initialize output arrays
    rho = jnp.zeros((2, nx, ny, nz), dtype=dtypes.float)
    tau = jnp.zeros((2, nx, ny, nz), dtype=dtypes.float)
    chi = jnp.zeros((2, nx, ny, nz), dtype=dtypes.float)
    current = jnp.zeros((2, 3, nx, ny, nz), dtype=dtypes.float)
    sdens = jnp.zeros((2, 3, nx, ny, nz), dtype=dtypes.float)
    sodens = jnp.zeros((2, 3, nx, ny, nz), dtype=dtypes.float)
    
    # Compute contributions per state (vmapped)
    def single_state_contrib(psi_n, weight, weightuv, iq):
        psi0 = psi_n[0]
        psi1 = psi_n[1]
        psi0_c = jnp.conjugate(psi0)
        psi1_c = jnp.conjugate(psi1)
        
        # Particle density
        rho_contrib = weight * jnp.real(psi0_c * psi0 + psi1_c * psi1)
        
        # Derivatives
        dpsi_dx = deriv_x(psi_n, dx)
        dpsi_dy = deriv_y(psi_n, dy)
        dpsi_dz = deriv_z(psi_n, dz)
        
        # Kinetic density
        tau_contrib = weight * jnp.real(
            jnp.conjugate(dpsi_dx[0]) * dpsi_dx[0] +
            jnp.conjugate(dpsi_dx[1]) * dpsi_dx[1] +
            jnp.conjugate(dpsi_dy[0]) * dpsi_dy[0] +
            jnp.conjugate(dpsi_dy[1]) * dpsi_dy[1] +
            jnp.conjugate(dpsi_dz[0]) * dpsi_dz[0] +
            jnp.conjugate(dpsi_dz[1]) * dpsi_dz[1]
        )
        
        return rho_contrib, tau_contrib, iq
    
    # Apply vmap
    rho_contribs, tau_contribs, iqs = jax.vmap(
        single_state_contrib
    )(psi, wocc, wguv * pairwg, isospin)
    
    # Scatter-add to isospin channels
    # This requires segment_sum or similar
    for iq in range(2):
        mask = (isospin == iq)
        rho = rho.at[iq].add(jnp.sum(rho_contribs * mask[:, None, None, None], axis=0))
        tau = tau.at[iq].add(jnp.sum(tau_contribs * mask[:, None, None, None], axis=0))
    
    return rho, tau, chi, current, sdens, sodens
