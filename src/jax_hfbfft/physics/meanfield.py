"""
Mean-field potentials for Skyrme HFB.

This module computes the Skyrme mean-field potentials from nuclear densities:
- upot: single-particle potential
- bmass: effective mass (B field)
- aq: current coupling (A vector)
- spot: spin potential (S vector)
- wlspot: spin-orbit potential (W vector)
- v_pair: pairing potential

The potentials are derived from the Skyrme energy density functional.
"""

import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Optional, Tuple

from jax_hfbfft.jax_config import get_dtypes
from jax_hfbfft.core.grid import Grid, deriv_x, deriv_y, deriv_z
from jax_hfbfft.physics.densities import Densities


@jax.tree_util.register_dataclass
@dataclass
class Meanfield:
    """
    Container for mean-field potentials.
    
    All potentials have shape (2, nx, ny, nz) where the first index is isospin:
    - 0: neutrons  
    - 1: protons
    
    Vector potentials (aq, spot, wlspot, dbmass) have shape (2, 3, nx, ny, nz)
    where the second index is the spatial component (x, y, z).
    """
    upot: jax.Array      # Single-particle potential
    bmass: jax.Array     # Effective mass field
    divaq: jax.Array     # Divergence of A vector
    v_pair: jax.Array    # Pairing potential
    aq: jax.Array        # Current coupling vector
    spot: jax.Array      # Spin potential vector
    wlspot: jax.Array    # Spin-orbit potential vector
    dbmass: jax.Array    # Gradient of effective mass
    ecorrp: float        # Pairing correlation energy correction
    
    @classmethod
    def zeros(cls, nx: int, ny: int, nz: int) -> "Meanfield":
        """Create zero-initialized meanfield."""
        dtypes = get_dtypes()
        shape4d = (2, nx, ny, nz)
        shape5d = (2, 3, nx, ny, nz)
        
        return cls(
            upot=jnp.zeros(shape4d, dtype=dtypes.float),
            bmass=jnp.zeros(shape4d, dtype=dtypes.float),
            divaq=jnp.zeros(shape4d, dtype=dtypes.float),
            v_pair=jnp.zeros(shape4d, dtype=dtypes.float),
            aq=jnp.zeros(shape5d, dtype=dtypes.float),
            spot=jnp.zeros(shape5d, dtype=dtypes.float),
            wlspot=jnp.zeros(shape5d, dtype=dtypes.float),
            dbmass=jnp.zeros(shape5d, dtype=dtypes.float),
            ecorrp=0.0,
        )


def compute_laplacian(field: jax.Array, grid: Grid) -> jax.Array:
    """Compute Laplacian of a scalar field using matrix derivatives."""
    lap_x = jnp.einsum('ij,jkl->ikl', grid.der2x, field)
    lap_y = jnp.einsum('jl,ilk->ijk', grid.der2y, field)
    lap_z = jnp.einsum('kl,ijl->ijk', grid.der2z, field)
    return lap_x + lap_y + lap_z


def compute_gradient(field: jax.Array, grid: Grid) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Compute gradient of a scalar field."""
    grad_x = jnp.einsum('ij,jkl->ikl', grid.der1x, field)
    grad_y = jnp.einsum('jl,ilk->ijk', grid.der1y, field)
    grad_z = jnp.einsum('kl,ijl->ijk', grid.der1z, field)
    return grad_x, grad_y, grad_z


def compute_divergence(vec: jax.Array, grid: Grid) -> jax.Array:
    """Compute divergence of a vector field. vec has shape (3, nx, ny, nz)."""
    div_x = jnp.einsum('ij,jkl->ikl', grid.der1x, vec[0])
    div_y = jnp.einsum('jl,ilk->ijk', grid.der1y, vec[1])
    div_z = jnp.einsum('kl,ijl->ijk', grid.der1z, vec[2])
    return div_x + div_y + div_z


def compute_curl(vec: jax.Array, grid: Grid) -> jax.Array:
    """Compute curl of a vector field. vec has shape (3, nx, ny, nz)."""
    # x-component: d/dy(v_z) - d/dz(v_y)
    curl_x = (jnp.einsum('jl,ilk->ijk', grid.der1y, vec[2]) -
              jnp.einsum('kl,ijl->ijk', grid.der1z, vec[1]))
    # y-component: d/dz(v_x) - d/dx(v_z)
    curl_y = (jnp.einsum('kl,ijl->ijk', grid.der1z, vec[0]) -
              jnp.einsum('ij,jkl->ikl', grid.der1x, vec[2]))
    # z-component: d/dx(v_y) - d/dy(v_x)
    curl_z = (jnp.einsum('ij,jkl->ikl', grid.der1x, vec[1]) -
              jnp.einsum('jl,ilk->ijk', grid.der1y, vec[0]))
    return jnp.stack([curl_x, curl_y, curl_z], axis=0)


def compute_skyrme_meanfield(
    densities: Densities,
    force,  # Force object
    grid: Grid,
    coulomb_potential: Optional[jax.Array] = None,
    constraint_potential: Optional[jax.Array] = None,
    use_coulomb: bool = True,
) -> Meanfield:
    """
    Compute Skyrme mean-field potentials from densities.
    
    This implements the full Skyrme energy density functional including:
    - Central terms (t0, t1, t2, t3)
    - Spin-orbit coupling (W0)
    - Effective mass
    - Current coupling (time-odd for dynamics)
    - Pairing (VDI or DDDI)
    
    Args:
        densities: Nuclear densities
        force: Force parameters (Skyrme interaction)
        grid: Spatial grid
        coulomb_potential: Pre-computed Coulomb potential (optional)
        constraint_potential: External constraint potential (optional)
        use_coulomb: Whether to include Coulomb interaction
        
    Returns:
        Meanfield object with all potentials
    """
    dtypes = get_dtypes()
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    epsilon = 1.0e-25
    
    # Initialize
    upot = jnp.zeros((2, nx, ny, nz), dtype=dtypes.float)
    workden = jnp.zeros((2, nx, ny, nz), dtype=dtypes.float)
    workvec = jnp.zeros((2, 3, nx, ny, nz), dtype=dtypes.float)
    
    # Total density
    rho_tot = densities.rho[0] + densities.rho[1]
    rho_tot_pow = rho_tot ** force.power
    
    # =========================================================================
    # Step 1: Three-body density-dependent term
    # =========================================================================
    for iq in range(2):
        ic = 1 - iq
        three_body = (
            (force.b3 * (force.power + 2.0) / 3.0 - 2.0 * force.b3p / 3.0) * densities.rho[iq] +
            force.b3 * (force.power + 2.0) / 3.0 * densities.rho[ic] -
            (force.b3p * force.power / 3.0) * (densities.rho[0]**2 + densities.rho[1]**2) /
            (rho_tot + epsilon)
        )
        upot = upot.at[iq].set(rho_tot_pow * three_body)
    
    # =========================================================================
    # Step 2: Divergence of spin-orbit current contribution
    # =========================================================================
    for iq in range(2):
        workden = workden.at[iq].set(compute_divergence(densities.sodens[iq], grid))
    
    for iq in range(2):
        ic = 1 - iq
        upot = upot.at[iq].add(
            -(force.b4 + force.b4p) * workden[iq] - force.b4 * workden[ic]
        )
    
    # =========================================================================
    # Step 3: Coulomb potential (protons only)
    # =========================================================================
    if use_coulomb and coulomb_potential is not None:
        upot = upot.at[1].add(coulomb_potential)
        
        # Slater exchange correction
        if force.ex != 0:
            slater = -force.slate * jnp.power(densities.rho[1] + epsilon, 1.0/3.0)
            upot = upot.at[1].add(slater)
    
    # =========================================================================
    # Step 4: Standard Skyrme terms (central + kinetic)
    # =========================================================================
    # Laplacian of density
    for iq in range(2):
        workden = workden.at[iq].set(compute_laplacian(densities.rho[iq], grid))
    
    for iq in range(2):
        ic = 1 - iq
        standard = (
            (force.b0 - force.b0p) * densities.rho[iq] + force.b0 * densities.rho[ic] +
            (force.b1 - force.b1p) * densities.tau[iq] + force.b1 * densities.tau[ic] -
            (force.b2 - force.b2p) * workden[iq] - force.b2 * workden[ic]
        )
        upot = upot.at[iq].add(standard)
    
    # Add constraint potential if present
    if constraint_potential is not None:
        for iq in range(2):
            upot = upot.at[iq].add(constraint_potential[iq])
    
    # =========================================================================
    # Step 5: Effective mass
    # =========================================================================
    bmass = jnp.zeros((2, nx, ny, nz), dtype=dtypes.float)
    for iq in range(2):
        ic = 1 - iq
        bmass_val = (force.h2m[iq] + 
                     (force.b1 - force.b1p) * densities.rho[iq] + 
                     force.b1 * densities.rho[ic])
        bmass = bmass.at[iq].set(bmass_val)
    
    # =========================================================================
    # Step 6: Spin-orbit potential (gradient of density)
    # =========================================================================
    wlspot = jnp.zeros((2, 3, nx, ny, nz), dtype=dtypes.float)
    for iq in range(2):
        grad_x, grad_y, grad_z = compute_gradient(densities.rho[iq], grid)
        workvec = workvec.at[iq, 0].set(grad_x)
        workvec = workvec.at[iq, 1].set(grad_y)
        workvec = workvec.at[iq, 2].set(grad_z)
    
    for iq in range(2):
        ic = 1 - iq
        wlspot_val = (force.b4 + force.b4p) * workvec[iq] + force.b4 * workvec[ic]
        wlspot = wlspot.at[iq].set(wlspot_val)
    
    # =========================================================================
    # Step 7: Curl of spin density
    # =========================================================================
    for iq in range(2):
        workvec = workvec.at[iq].set(compute_curl(densities.sdens[iq], grid))
    
    # =========================================================================
    # Step 8: Current coupling (A vector)
    # =========================================================================
    aq = jnp.zeros((2, 3, nx, ny, nz), dtype=dtypes.float)
    for iq in range(2):
        ic = 1 - iq
        aq_val = (
            -2.0 * (force.b1 - force.b1p) * densities.current[iq] -
            2.0 * force.b1 * densities.current[ic] -
            (force.b4 + force.b4p) * workvec[iq] - force.b4 * workvec[ic]
        )
        aq = aq.at[iq].set(aq_val)
    
    # =========================================================================
    # Step 9: Spin potential from curl of current
    # =========================================================================
    spot = jnp.zeros((2, 3, nx, ny, nz), dtype=dtypes.float)
    for iq in range(2):
        spot = spot.at[iq].set(compute_curl(densities.current[iq], grid))
    
    # =========================================================================
    # Step 10: Combine isospin for spin potential
    # =========================================================================
    spot_temp = jnp.copy(spot)
    for iq in range(2):
        ic = 1 - iq
        spot_combined = -(force.b4 + force.b4p) * spot_temp[iq] - force.b4 * spot_temp[ic]
        spot = spot.at[iq].set(spot_combined)
    
    # =========================================================================
    # Step 11: Divergence of A vector
    # =========================================================================
    divaq = jnp.zeros((2, nx, ny, nz), dtype=dtypes.float)
    for iq in range(2):
        divaq = divaq.at[iq].set(compute_divergence(aq[iq], grid))
    
    # =========================================================================
    # Step 12: Gradient of effective mass
    # =========================================================================
    dbmass = jnp.zeros((2, 3, nx, ny, nz), dtype=dtypes.float)
    for iq in range(2):
        grad_x, grad_y, grad_z = compute_gradient(bmass[iq], grid)
        dbmass = dbmass.at[iq, 0].set(grad_x)
        dbmass = dbmass.at[iq, 1].set(grad_y)
        dbmass = dbmass.at[iq, 2].set(grad_z)
    
    # =========================================================================
    # Step 13: Pairing potential
    # =========================================================================
    v_pair = jnp.zeros((2, nx, ny, nz), dtype=dtypes.float)
    ecorrp = 0.0
    
    if force.ipair == 6:  # DDDI pairing
        # Rearrangement potential
        rearrange = (
            (force.v0neut / force.rho0pr) * densities.chi[0]**2 +
            (force.v0prot / force.rho0pr) * densities.chi[1]**2
        )
        upot = upot.at[0].add(rearrange)
        upot = upot.at[1].add(rearrange)
        
        ecorrp = -jnp.sum(rho_tot * rearrange) * grid.wxyz / 2.0
        
        # DDDI pairing potential
        density_factor = 1.0 - rho_tot / force.rho0pr
        v_pair = v_pair.at[0].set(force.v0neut * densities.chi[0] * density_factor)
        v_pair = v_pair.at[1].set(force.v0prot * densities.chi[1] * density_factor)
        
    elif force.ipair == 5:  # VDI pairing
        v_pair = v_pair.at[0].set(force.v0neut * densities.chi[0])
        v_pair = v_pair.at[1].set(force.v0prot * densities.chi[1])
    
    return Meanfield(
        upot=upot,
        bmass=bmass,
        divaq=divaq,
        v_pair=v_pair,
        aq=aq,
        spot=spot,
        wlspot=wlspot,
        dbmass=dbmass,
        ecorrp=float(ecorrp),
    )


@jax.jit
def apply_hamiltonian(
    psi: jax.Array,
    meanfield: Meanfield,
    iq: int,
    dx: float,
    dy: float,
    dz: float,
) -> jax.Array:
    """
    Apply the single-particle HFB Hamiltonian to a wavefunction.
    
    This computes h|psi> where h is the Skyrme single-particle Hamiltonian
    including kinetic energy with effective mass and spin-orbit coupling.
    
    Args:
        psi: Spinor wavefunction with shape (2, nx, ny, nz)
        meanfield: Mean-field potentials
        iq: Isospin index (0=neutron, 1=proton)
        dx, dy, dz: Grid spacings
        
    Returns:
        h|psi> with same shape as input
    """
    sigis = jnp.array([0.5, -0.5])
    
    # Step 1: Local potential (upot)
    pout = psi * meanfield.upot[iq]
    
    # Step 2: Spin-current coupling
    pout = pout.at[0].add(
        (meanfield.spot[iq, 0] - 1j * meanfield.spot[iq, 1]) * psi[1] +
        meanfield.spot[iq, 2] * psi[0]
    )
    pout = pout.at[1].add(
        (meanfield.spot[iq, 0] + 1j * meanfield.spot[iq, 1]) * psi[0] -
        meanfield.spot[iq, 2] * psi[1]
    )
    
    # Step 3: x-derivatives (kinetic + spin-orbit)
    pout = _add_derivative_terms_x(pout, psi, meanfield, iq, dx, sigis)
    
    # Step 4: y-derivatives
    pout = _add_derivative_terms_y(pout, psi, meanfield, iq, dy, sigis)
    
    # Step 5: z-derivatives
    pout = _add_derivative_terms_z(pout, psi, meanfield, iq, dz, sigis)
    
    return pout


def _add_derivative_terms_x(pout, psi, mf, iq, dx, sigis):
    """Add x-direction derivative terms."""
    # First derivatives with effective mass
    dpsi_dx = deriv_x(psi, dx)
    d2psi_dx2 = deriv_x(deriv_x(psi, dx), dx)  # Simple second derivative
    
    # Effective mass contribution: -B * d²/dx² - dB/dx * d/dx
    pout = pout.at[0].add(
        -mf.bmass[iq] * d2psi_dx2[0] - mf.dbmass[iq, 0] * dpsi_dx[0]
    )
    pout = pout.at[1].add(
        -mf.bmass[iq] * d2psi_dx2[1] - mf.dbmass[iq, 0] * dpsi_dx[1]
    )
    
    # Spin-orbit coupling terms from x-derivative
    pout = pout.at[0].add(
        -(1j * (0.5 * mf.aq[iq, 0] - sigis[0] * mf.wlspot[iq, 1])) * dpsi_dx[0] -
        sigis[0] * mf.wlspot[iq, 2] * dpsi_dx[1]
    )
    pout = pout.at[1].add(
        -(1j * (0.5 * mf.aq[iq, 0] - sigis[1] * mf.wlspot[iq, 1])) * dpsi_dx[1] -
        sigis[1] * mf.wlspot[iq, 2] * dpsi_dx[0]
    )
    
    # Additional spin-orbit from derivative of psi weighted by potential
    pswk = jnp.zeros_like(psi)
    pswk = pswk.at[0].set(
        (-1j * 0.5) * (mf.aq[iq, 0] - mf.wlspot[iq, 1]) * psi[0] -
        0.5 * mf.wlspot[iq, 2] * psi[1]
    )
    pswk = pswk.at[1].set(
        (-1j * 0.5) * (mf.aq[iq, 0] + mf.wlspot[iq, 1]) * psi[1] +
        0.5 * mf.wlspot[iq, 2] * psi[0]
    )
    pout = pout + deriv_x(pswk, dx)
    
    return pout


def _add_derivative_terms_y(pout, psi, mf, iq, dy, sigis):
    """Add y-direction derivative terms."""
    dpsi_dy = deriv_y(psi, dy)
    d2psi_dy2 = deriv_y(deriv_y(psi, dy), dy)
    
    # Effective mass contribution
    pout = pout.at[0].add(
        -mf.bmass[iq] * d2psi_dy2[0] - mf.dbmass[iq, 1] * dpsi_dy[0]
    )
    pout = pout.at[1].add(
        -mf.bmass[iq] * d2psi_dy2[1] - mf.dbmass[iq, 1] * dpsi_dy[1]
    )
    
    # Spin-orbit coupling
    pout = pout.at[0].add(
        -(1j * (0.5 * mf.aq[iq, 1] + sigis[0] * mf.wlspot[iq, 0])) * dpsi_dy[0] +
        (1j * 0.5 * mf.wlspot[iq, 2]) * dpsi_dy[1]
    )
    pout = pout.at[1].add(
        -(1j * (0.5 * mf.aq[iq, 1] + sigis[1] * mf.wlspot[iq, 0])) * dpsi_dy[1] +
        (1j * 0.5 * mf.wlspot[iq, 2]) * dpsi_dy[0]
    )
    
    pswk = jnp.zeros_like(psi)
    pswk = pswk.at[0].set(
        (-1j * 0.5) * (mf.aq[iq, 1] + mf.wlspot[iq, 0]) * psi[0] +
        (1j * 0.5) * mf.wlspot[iq, 2] * psi[1]
    )
    pswk = pswk.at[1].set(
        (-1j * 0.5) * (mf.aq[iq, 1] - mf.wlspot[iq, 0]) * psi[1] +
        (1j * 0.5) * mf.wlspot[iq, 2] * psi[0]
    )
    pout = pout + deriv_y(pswk, dy)
    
    return pout


def _add_derivative_terms_z(pout, psi, mf, iq, dz, sigis):
    """Add z-direction derivative terms."""
    dpsi_dz = deriv_z(psi, dz)
    d2psi_dz2 = deriv_z(deriv_z(psi, dz), dz)
    
    # Effective mass contribution
    pout = pout.at[0].add(
        -mf.bmass[iq] * d2psi_dz2[0] - mf.dbmass[iq, 2] * dpsi_dz[0]
    )
    pout = pout.at[1].add(
        -mf.bmass[iq] * d2psi_dz2[1] - mf.dbmass[iq, 2] * dpsi_dz[1]
    )
    
    # Spin-orbit coupling
    pout = pout.at[0].add(
        -(1j * 0.5 * mf.aq[iq, 2]) * dpsi_dz[0] +
        (sigis[0] * mf.wlspot[iq, 0] - 1j * 0.5 * mf.wlspot[iq, 1]) * dpsi_dz[1]
    )
    pout = pout.at[1].add(
        -(1j * 0.5 * mf.aq[iq, 2]) * dpsi_dz[1] +
        (sigis[1] * mf.wlspot[iq, 0] - 1j * 0.5 * mf.wlspot[iq, 1]) * dpsi_dz[0]
    )
    
    pswk = jnp.zeros_like(psi)
    pswk = pswk.at[0].set(
        (-1j * 0.5) * mf.aq[iq, 2] * psi[0] +
        (0.5 * mf.wlspot[iq, 0] - 1j * 0.5 * mf.wlspot[iq, 1]) * psi[1]
    )
    pswk = pswk.at[1].set(
        (-1j * 0.5) * mf.aq[iq, 2] * psi[1] +
        (-0.5 * mf.wlspot[iq, 0] - 1j * 0.5 * mf.wlspot[iq, 1]) * psi[0]
    )
    pout = pout + deriv_z(pswk, dz)
    
    return pout


@jax.jit
def apply_hfb_hamiltonian(
    psi: jax.Array,
    meanfield: Meanfield,
    iq: int,
    weight: float,
    weightuv: float,
    dx: float,
    dy: float,
    dz: float,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Apply full HFB Hamiltonian including pairing.
    
    Returns:
        pout: Full HFB result (h - weightuv * Delta)|psi>
        pout_mf: Mean-field only part h|psi>
        pout_del: Pairing part Delta|psi>
    """
    # Mean-field part
    pout_mf = apply_hamiltonian(psi, meanfield, iq, dx, dy, dz)
    
    # Pairing part (local approximation)
    pout_del = psi * meanfield.v_pair[iq]
    
    # Combined HFB
    pout = weight * pout_mf - weightuv * pout_del
    
    return pout, pout_mf, pout_del
