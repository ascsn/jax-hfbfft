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

import dataclasses
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
    ecorrp: jax.Array    # Pairing correlation energy correction (scalar)
    
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
            ecorrp=jnp.array(0.0, dtype=dtypes.float),
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


# Vectorized versions of Laplacian, gradient, divergence, curl for both isospin channels
@jax.jit
def _compute_laplacian_both(rho: jax.Array, grid: Grid) -> jax.Array:
    """Compute Laplacian for both isospin channels. rho: (2, nx, ny, nz)."""
    # Vectorize over first axis (isospin)
    return jax.vmap(lambda f: compute_laplacian(f, grid))(rho)


@jax.jit
def _compute_gradient_both(rho: jax.Array, grid: Grid) -> jax.Array:
    """Compute gradient for both isospin channels. Returns (2, 3, nx, ny, nz)."""
    def grad_single(f):
        gx, gy, gz = compute_gradient(f, grid)
        return jnp.stack([gx, gy, gz], axis=0)
    return jax.vmap(grad_single)(rho)


@jax.jit
def _compute_divergence_both(vec: jax.Array, grid: Grid) -> jax.Array:
    """Compute divergence for both isospin channels. vec: (2, 3, nx, ny, nz)."""
    return jax.vmap(lambda v: compute_divergence(v, grid))(vec)


@jax.jit
def _compute_curl_both(vec: jax.Array, grid: Grid) -> jax.Array:
    """Compute curl for both isospin channels. vec: (2, 3, nx, ny, nz)."""
    return jax.vmap(lambda v: compute_curl(v, grid))(vec)

#Regularization kernel
def _keff_regularization(
    eferm: jax.Array,   # (2,) Fermi energy per isospin
    ecut: jax.Array,    # (2,) cutoff energy per isospin (pairing-active window)
    upot: jax.Array,    # (2, nx, ny, nz) local potential, BEFORE pairing rearrangement
    bmass: jax.Array,   # (2, nx, ny, nz) local effective-mass field hbar^2/(2M*(r))
) -> jax.Array:
    """
    Bulgac-Yu regularization kernel K_eff(r)
    Returns K_eff with the same (2, nx, ny, nz) shape as upot/bmass.
    Taken from HFBTHO's (hfbtho_solver.f90:4341-4396) 


    
    """
    ef = eferm[:, None, None, None]
    ec = ecut[:, None, None, None]
    bmass_safe = jnp.maximum(bmass, 1.0e-6)
    fac = jnp.sqrt(1.0 / bmass_safe)
    pref = 0.25 / (bmass_safe * jnp.pi ** 2)

    cond1 = (upot + ec - ef) < 0.0
    cond2 = ef > upot
    cond3 = (upot - ec - ef) < 0.0

    kc = fac * jnp.sqrt(jnp.maximum(ef + ec - upot, 1.0e-12))
    kf_occ = fac * jnp.sqrt(jnp.maximum(ef - upot, 1.0e-12))
    log_term_kc = jnp.log((kc + kf_occ) / jnp.maximum(kc - kf_occ, 1.0e-9))
    a = pref * kc * (1.0 - 0.5 * (kf_occ / kc) * log_term_kc)

    lc = fac * jnp.sqrt(jnp.maximum(ef - ec - upot, 1.0e-12))
    log_term_lc = jnp.log((lc + kf_occ) / jnp.maximum(jnp.abs(kf_occ - lc), 1.0e-9))
    b = pref * lc * (1.0 - 0.5 * (kf_occ / lc) * log_term_lc)
    branch1 = a + b

    branch2 = a 
    kf_unocc = fac * jnp.sqrt(jnp.maximum(upot - ef, 1.0e-12))
    branch3 = pref * kc * (1.0 + (kf_unocc / kc) * jnp.arctan(kf_unocc / kc))

    branch4 = jnp.zeros_like(branch1)

    return jnp.where(cond1, branch1, jnp.where(cond2, branch2, jnp.where(cond3, branch3, branch4)))


@jax.jit
def _compute_skyrme_meanfield_core(
    rho: jax.Array,          # (2, nx, ny, nz)
    tau: jax.Array,          # (2, nx, ny, nz)
    chi: jax.Array,          # (2, nx, ny, nz)
    current: jax.Array,      # (2, 3, nx, ny, nz)
    sdens: jax.Array,        # (2, 3, nx, ny, nz)
    sodens: jax.Array,       # (2, 3, nx, ny, nz)
    tdens: jax.Array,        # (2, 3, nx, ny, nz)
    coulomb_potential: jax.Array,  # (nx, ny, nz) or zeros
    constraint_potential: jax.Array,  # (2, nx, ny, nz) or zeros
    # Force parameters (flattened for JIT)
    b0: float, b0p: float, b1: float, b1p: float, b2: float, b2p: float,
    b3: float, b3p: float, b4: float, b4p: float,
    c_s0: float, c_s1: float, c_sDs0: float, c_sDs1: float,
    c_ds0: float, c_ds1: float,
    h2m_n: float, h2m_p: float,
    power: float, slate: float, ex: float,
    v0neut: float, v0prot: float, rho0pr: float, ipair: int,
    # Grid
    grid: Grid,
    # Flags
    use_coulomb: bool,
    # Bulgac-Yu regularized pairing
    eferm: jax.Array,
    pairing_reg_cutoff: jax.Array,
    use_pairing_regularization: jax.Array,
    time_odd_frac: jax.Array,
) -> Meanfield:
    """
    JIT-compiled core of compute_skyrme_meanfield.
    
    All loops over isospin are vectorized for GPU efficiency.
    """
    _RF = 1.0e-25 # rho floor (for gradient instability)
    dtypes = get_dtypes()
    epsilon = 1.0e-25
    
    h2m = jnp.array([h2m_n, h2m_p])
    
    # Total density
    rho_tot = rho[0] + rho[1]
    rho_tot_pow = jnp.maximum(rho_tot,_RF) ** power
    
    # =========================================================================
    # Step 1: Three-body density-dependent term (vectorized)
    # =========================================================================
    # For iq=0: ic=1, for iq=1: ic=0
    rho_sum_sq = rho[0]**2 + rho[1]**2
    
    # Compute both isospin channels at once
    coeff_same = b3 * (power + 2.0) / 3.0 - 2.0 * b3p / 3.0
    coeff_other = b3 * (power + 2.0) / 3.0
    coeff_rho = b3p * power / 3.0
    
    three_body_0 = (coeff_same * rho[0] + coeff_other * rho[1] - 
                    coeff_rho * rho_sum_sq / (rho_tot + epsilon))
    three_body_1 = (coeff_same * rho[1] + coeff_other * rho[0] - 
                    coeff_rho * rho_sum_sq / (rho_tot + epsilon))
    
    upot = jnp.stack([rho_tot_pow * three_body_0, rho_tot_pow * three_body_1], axis=0)
    
    # =========================================================================
    # Step 2: Divergence of spin-orbit current (vectorized)
    # =========================================================================
    div_sodens = _compute_divergence_both(sodens, grid)  # (2, nx, ny, nz)
    
    # upot[iq] += -(b4 + b4p) * div_sodens[iq] - b4 * div_sodens[ic]
    upot = upot.at[0].add(-(b4 + b4p) * div_sodens[0] - b4 * div_sodens[1])
    upot = upot.at[1].add(-(b4 + b4p) * div_sodens[1] - b4 * div_sodens[0])
    
    # =========================================================================
    # Step 3: Coulomb potential (protons only)
    # =========================================================================
    # Use jnp.where for JIT compatibility instead of if statement
    upot = upot.at[1].add(jnp.where(use_coulomb, coulomb_potential, 0.0))
    
    # Slater exchange - apply only if use_coulomb and ex != 0
    slater = -slate * jnp.power(rho[1] + epsilon, 1.0/3.0)
    upot = upot.at[1].add(jnp.where(use_coulomb & (ex != 0), slater, 0.0))
    
    # =========================================================================
    # Step 4: Standard Skyrme terms (vectorized)
    # =========================================================================
    lap_rho = _compute_laplacian_both(rho, grid)  # (2, nx, ny, nz)
    
    # upot[iq] += (b0-b0p)*rho[iq] + b0*rho[ic] + (b1-b1p)*tau[iq] + b1*tau[ic] 
    #             - (b2-b2p)*lap[iq] - b2*lap[ic]
    upot = upot.at[0].add(
        (b0 - b0p) * rho[0] + b0 * rho[1] +
        (b1 - b1p) * tau[0] + b1 * tau[1] -
        (b2 - b2p) * lap_rho[0] - b2 * lap_rho[1]
    )
    upot = upot.at[1].add(
        (b0 - b0p) * rho[1] + b0 * rho[0] +
        (b1 - b1p) * tau[1] + b1 * tau[0] -
        (b2 - b2p) * lap_rho[1] - b2 * lap_rho[0]
    )
    
    # Add constraint potential
    upot = upot + constraint_potential
    
    # =========================================================================
    # Step 5: Effective mass (vectorized)
    # =========================================================================
    # bmass[iq] = h2m[iq] + (b1-b1p)*rho[iq] + b1*rho[ic]
    bmass = jnp.stack([
        h2m[0] + (b1 - b1p) * rho[0] + b1 * rho[1],
        h2m[1] + (b1 - b1p) * rho[1] + b1 * rho[0],
    ], axis=0)
    
    # =========================================================================
    # Step 6: Spin-orbit potential (vectorized)
    # =========================================================================
    grad_rho = _compute_gradient_both(rho, grid)  # (2, 3, nx, ny, nz)
    
    # wlspot[iq] = (b4+b4p) * grad_rho[iq] + b4 * grad_rho[ic]
    wlspot = jnp.stack([
        (b4 + b4p) * grad_rho[0] + b4 * grad_rho[1],
        (b4 + b4p) * grad_rho[1] + b4 * grad_rho[0],
    ], axis=0)
    
    # =========================================================================
    # Step 7: Curl of spin density (vectorized)
    # =========================================================================
    curl_sdens = _compute_curl_both(sdens, grid)  # (2, 3, nx, ny, nz)
    
    # =========================================================================
    # Step 8: Current coupling (vectorized)
    # =========================================================================
    # aq[iq] = -2*(b1-b1p)*current[iq] - 2*b1*current[ic] 
    #          - (b4+b4p)*curl_sdens[iq] - b4*curl_sdens[ic]
    aq = jnp.stack([
        -2.0 * (b1 - b1p) * current[0] - 2.0 * b1 * current[1] -
        (b4 + b4p) * curl_sdens[0] - b4 * curl_sdens[1],
        -2.0 * (b1 - b1p) * current[1] - 2.0 * b1 * current[0] -
        (b4 + b4p) * curl_sdens[1] - b4 * curl_sdens[0],
    ], axis=0)
    
    # =========================================================================
    # Step 9-10: Spin potential from curl of current (vectorized)
    # =========================================================================
    curl_current = _compute_curl_both(current, grid)  # (2, 3, nx, ny, nz)
    
    # spot[iq] = -(b4+b4p)*curl_current[iq] - b4*curl_current[ic]
    spot = jnp.stack([
        -(b4 + b4p) * curl_current[0] - b4 * curl_current[1],
        -(b4 + b4p) * curl_current[1] - b4 * curl_current[0],
    ], axis=0)
    
    # =========================================================================
    # Step 11: Time-odd spin mean-field terms.
    # Should vanish in even-even nuclei
    # =========================================================================
    s_iso = sdens[0] + sdens[1]     # isoscalar spin density  (3, nx, ny, nz)
    s_vec = sdens[0] - sdens[1]     # isovector spin density  (3, nx, ny, nz)

    # Ramping
    c_s0 = c_s0 * time_odd_frac
    c_s1 = c_s1 * time_odd_frac
    c_ds0 = c_ds0 * time_odd_frac
    c_ds1 = c_ds1 * time_odd_frac
    c_sDs0 = c_sDs0 * time_odd_frac
    c_sDs1 = c_sDs1 * time_odd_frac

    # s^2 term
    cs0_eff = c_s0 + c_ds0 * rho_tot_pow
    cs1_eff = c_s1 + c_ds1 * rho_tot_pow
    spot = spot + jnp.stack([
        2.0 * cs0_eff * s_iso + 2.0 * cs1_eff * s_vec,
        2.0 * cs0_eff * s_iso - 2.0 * cs1_eff * s_vec,
    ], axis=0)

    # Rearrangement
    s_iso_sq = jnp.sum(s_iso**2, axis=0)
    s_vec_sq = jnp.sum(s_vec**2, axis=0)
    _RHO_FLOOR = 1.0e-25 # rho floor for gradient
    rho_tot = jnp.maximum(rho_tot, _RHO_FLOOR)
    rearr_odd = (power * rho_tot ** (power - 1.0)
                 * (c_ds0 * s_iso_sq + c_ds1 * s_vec_sq))
    upot = upot + jnp.stack([rearr_odd, rearr_odd], axis=0)

    # s.Lap(s) term
    lap_s_iso = jnp.stack([compute_laplacian(s_iso[k], grid) for k in range(3)], axis=0)
    lap_s_vec = jnp.stack([compute_laplacian(s_vec[k], grid) for k in range(3)], axis=0)
    spot = spot + jnp.stack([
        2.0 * c_sDs0 * lap_s_iso + 2.0 * c_sDs1 * lap_s_vec,
        2.0 * c_sDs0 * lap_s_iso - 2.0 * c_sDs1 * lap_s_vec,
    ], axis=0)

    # s.T coupling is absent on purpose

    # =========================================================================
    # Step 11: Divergence of A vector (vectorized)
    # =========================================================================
    divaq = _compute_divergence_both(aq, grid)  # (2, nx, ny, nz)
    
    # =========================================================================
    # Step 12: Gradient of effective mass (vectorized)
    # =========================================================================
    dbmass = _compute_gradient_both(bmass, grid)  # (2, 3, nx, ny, nz)
    
    # =========================================================================
    # Step 13: Pairing potential (use where for JIT compatibility)
    # =========================================================================
    density_factor = 1.0 - rho_tot / rho0pr
    
    # VDI pairing (ipair == 5)
    v_pair_vdi = jnp.stack([
        v0neut * chi[0],
        v0prot * chi[1],
    ], axis=0)
    
    # DDDI pairing (ipair == 6)
    #Bulgac-Yu regularization applied to density-scaled bare coupling
    g_bare = jnp.stack([v0neut * density_factor, v0prot * density_factor], axis=0) 
    keff = _keff_regularization(eferm, pairing_reg_cutoff, upot, bmass)
    g_reg = g_bare / (1.0 + g_bare * keff)
    g_eff = jnp.where(use_pairing_regularization, g_reg, g_bare)
    v_pair_dddi = chi * g_eff
    
    # Rearrangement for DDDI
    rearrange = (v0neut / rho0pr) * chi[0]**2 + (v0prot / rho0pr) * chi[1]**2
    upot_dddi = upot + jnp.stack([rearrange, rearrange], axis=0)
    ecorrp_dddi = -jnp.sum(rho_tot * rearrange) * grid.wxyz / 2.0
    
    # Select based on ipair using where
    is_dddi = (ipair == 6)
    is_vdi = (ipair == 5)
    
    v_pair = jnp.where(is_dddi, v_pair_dddi, jnp.where(is_vdi, v_pair_vdi, jnp.zeros_like(v_pair_vdi)))
    upot = jnp.where(is_dddi, upot_dddi, upot)
    ecorrp = jnp.where(is_dddi, ecorrp_dddi, 0.0)
    
    return Meanfield(
        upot=upot,
        bmass=bmass,
        divaq=divaq,
        v_pair=v_pair,
        aq=aq,
        spot=spot,
        wlspot=wlspot,
        dbmass=dbmass,
        ecorrp=jnp.asarray(ecorrp),  # Keep as JAX scalar array
    )


def compute_skyrme_meanfield(
    densities: Densities,
    force,  # Force object
    grid: Grid,
    coulomb_potential: Optional[jax.Array] = None,
    constraint_potential: Optional[jax.Array] = None,
    use_coulomb: bool = True,
    eferm: Optional[jax.Array] = None,
    use_pairing_regularization: bool = False,
    time_odd_frac: jax.Array = 1.0,
) -> Meanfield:
    """
    Compute Skyrme mean-field potentials from densities.
    
    This is a wrapper that calls the JIT-compiled core function.
    
    Args:
        densities: Nuclear densities
        force: Force parameters (Skyrme interaction)
        grid: Spatial grid
        coulomb_potential: Pre-computed Coulomb potential (optional)
        constraint_potential: External constraint potential (optional)
        use_coulomb: Whether to include Coulomb interaction
        eferm: Fermi energy per isospin for pairing_regularization (optional).
        use_pairing_regularization: Bulgac-Yu regularize DDDI pairing coupling.
        time_odd_frac: multiplier on time-odd couplings for ramping

        
    Returns:
        Meanfield object with all potentials
    """
    dtypes = get_dtypes()
    
    # Handle optional arguments
    if coulomb_potential is None:
        coulomb_potential = jnp.zeros((grid.nx, grid.ny, grid.nz), dtype=dtypes.float)
    
    if constraint_potential is None:
        constraint_potential = jnp.zeros((2, grid.nx, grid.ny, grid.nz), dtype=dtypes.float)
    
    if eferm is None:
        eferm = jnp.zeros(2, dtype=dtypes.float)
    # Call the JIT-compiled core
    return _compute_skyrme_meanfield_core(
        densities.rho,
        densities.tau,
        densities.chi,
        densities.current,
        densities.sdens,
        densities.sodens,
        densities.tdens,
        coulomb_potential,
        constraint_potential,
        force.b0, force.b0p, force.b1, force.b1p, force.b2, force.b2p,
        force.b3, force.b3p, force.b4, force.b4p,
        force.c_s0, force.c_s1, force.c_sDs0, force.c_sDs1,
        force.c_ds0, force.c_ds1,
        force.h2m[0], force.h2m[1],
        force.power, force.slate, force.ex,
        force.v0neut, force.v0prot, force.rho0pr, force.ipair,
        grid,
        use_coulomb,
        eferm,
        jnp.asarray(
            force.pairing_reg_cutoff if force.pairing_reg_cutoff is not None
            else force.state_cutoff,
            dtype=dtypes.float,
        ),
        jnp.asarray(use_pairing_regularization),
        jnp.asarray(time_odd_frac, dtype=dtypes.float),
    )


@jax.jit
def apply_hamiltonian(
    psi: jax.Array,
    meanfield: Meanfield,
    iq: int,
    grid: Grid,
) -> jax.Array:
    """
    Apply the single-particle HFB Hamiltonian to a wavefunction.
    
    This computes h|psi> where h is the Skyrme single-particle Hamiltonian
    including kinetic energy with effective mass and spin-orbit coupling.
    
    Args:
        psi: Spinor wavefunction with shape (2, nx, ny, nz)
        meanfield: Mean-field potentials
        iq: Isospin index (0=neutron, 1=proton)
        grid: Spatial grid
        
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
    pout = _add_derivative_terms_x(pout, psi, meanfield, iq, grid, sigis)
    
    # Step 4: y-derivatives
    pout = _add_derivative_terms_y(pout, psi, meanfield, iq, grid, sigis)
    
    # Step 5: z-derivatives
    pout = _add_derivative_terms_z(pout, psi, meanfield, iq, grid, sigis)
    
    return pout


def _add_derivative_terms_x(pout, psi, mf, iq, grid, sigis):
    """Add x-direction derivative terms."""
    # Use matrix derivatives for better JIT performance on GPU
    # psi is (2, nx, ny, nz), der1x is (nx, nx)
    dpsi_dx = jnp.einsum('ij,sjkl->sikl', grid.der1x, psi)
    d2psi_dx2 = jnp.einsum('ij,sjkl->sikl', grid.der2x, psi)
    
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
    pswk0 = (-1j * 0.5) * (mf.aq[iq, 0] - mf.wlspot[iq, 1]) * psi[0] - 0.5 * mf.wlspot[iq, 2] * psi[1]
    pswk1 = (-1j * 0.5) * (mf.aq[iq, 0] + mf.wlspot[iq, 1]) * psi[1] + 0.5 * mf.wlspot[iq, 2] * psi[0]
    
    pout = pout.at[0].add(jnp.einsum('ij,jkl->ikl', grid.der1x, pswk0))
    pout = pout.at[1].add(jnp.einsum('ij,jkl->ikl', grid.der1x, pswk1))
    
    return pout


def _add_derivative_terms_y(pout, psi, mf, iq, grid, sigis):
    """Add y-direction derivative terms."""
    dpsi_dy = jnp.einsum('ij,skjl->skil', grid.der1y, psi)
    d2psi_dy2 = jnp.einsum('ij,skjl->skil', grid.der2y, psi)
    
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
    
    pswk0 = (-1j * 0.5) * (mf.aq[iq, 1] + mf.wlspot[iq, 0]) * psi[0] + (1j * 0.5) * mf.wlspot[iq, 2] * psi[1]
    pswk1 = (-1j * 0.5) * (mf.aq[iq, 1] - mf.wlspot[iq, 0]) * psi[1] + (1j * 0.5) * mf.wlspot[iq, 2] * psi[0]
    
    pout = pout.at[0].add(jnp.einsum('ij,kjl->kil', grid.der1y, pswk0))
    pout = pout.at[1].add(jnp.einsum('ij,kjl->kil', grid.der1y, pswk1))
    
    return pout


def _add_derivative_terms_z(pout, psi, mf, iq, grid, sigis):
    """Add z-direction derivative terms."""
    dpsi_dz = jnp.einsum('ij,sklj->skli', grid.der1z, psi)
    d2psi_dz2 = jnp.einsum('ij,sklj->skli', grid.der2z, psi)
    
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
    
    pswk0 = (-1j * 0.5) * mf.aq[iq, 2] * psi[0] + (0.5 * mf.wlspot[iq, 0] - 1j * 0.5 * mf.wlspot[iq, 1]) * psi[1]
    pswk1 = (-1j * 0.5) * mf.aq[iq, 2] * psi[1] + (-0.5 * mf.wlspot[iq, 0] - 1j * 0.5 * mf.wlspot[iq, 1]) * psi[0]
    
    pout = pout.at[0].add(jnp.einsum('ij,klj->kli', grid.der1z, pswk0))
    pout = pout.at[1].add(jnp.einsum('ij,klj->kli', grid.der1z, pswk1))
    
    return pout


@jax.jit
def apply_hfb_hamiltonian(
    psi: jax.Array,
    meanfield: Meanfield,
    iq: int,
    weight: float,
    weightuv: float,
    grid: Grid,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Apply full HFB Hamiltonian including pairing.
    
    Returns:
        pout: Full HFB result (h - weightuv * Delta)|psi>
        pout_mf: Mean-field only part h|psi>
        pout_del: Pairing part Delta|psi>
    """
    # Mean-field part
    pout_mf = apply_hamiltonian(psi, meanfield, iq, grid)
    
    # Pairing part (local approximation)
    pout_del = psi * meanfield.v_pair[iq]
    
    # Combined HFB
    pout = weight * pout_mf - weightuv * pout_del
    
    return pout, pout_mf, pout_del
