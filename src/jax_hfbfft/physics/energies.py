"""
Energy calculations for Skyrme HFB.

This module computes energy contributions from the Skyrme energy density functional:
- Kinetic energy (tau terms)
- Central terms (t0)
- Gradient terms (t1, t2)
- Density-dependent term (t3)
- Spin-orbit energy (W0)
- Coulomb energy (direct + exchange)
- Pairing energy
"""

import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Optional

from jax_hfbfft.jax_config import get_dtypes
from jax_hfbfft.core.grid import Grid
from jax_hfbfft.physics.densities import Densities
from jax_hfbfft.physics.meanfield import compute_laplacian, compute_divergence, compute_curl


@jax.tree_util.register_dataclass
@dataclass
class Energies:
    """
    Container for energy contributions.
    
    All energies are in MeV.
    """
    # Integrated energy contributions (from energy density functional)
    ehft: float       # Kinetic energy
    ehf0: float       # t0 (central) contribution
    ehf1: float       # t1/t2 (momentum-dependent) contribution  
    ehf2: float       # Gradient (Laplacian) contribution
    ehf3: float       # t3 (density-dependent) contribution
    ehfls: float      # Spin-orbit contribution (time-even)
    ehflsodd: float   # Spin-orbit contribution (time-odd)
    ehfc: float       # Coulomb energy
    ecorc: float      # Exchange correlation
    ehfint: float     # Total integrated energy
    
    # Single-particle energies and derived quantities
    ehf: float        # Total energy from s.p. levels
    tke: float        # Total kinetic energy (summed)
    e3corr: float     # Rearrangement energy correction
    e_zpe: float      # Center-of-mass correction
    
    # Convergence measures
    efluct1: jax.Array    # Max |lambda_minus|
    efluct1q: jax.Array   # Max |lambda_minus| per isospin
    efluct2: jax.Array    # RMS of lambda_minus
    efluct2q: jax.Array   # RMS per isospin
    
    # Angular momentum
    orbital: jax.Array     # Orbital angular momentum (Lx, Ly, Lz)
    spin: jax.Array        # Spin angular momentum (Sx, Sy, Sz)
    total_angmom: jax.Array  # Total angular momentum (Jx, Jy, Jz)
    
    # Pairing
    epair: jax.Array  # Pairing energy per isospin
    
    # Coupling constant contributions (for analysis)
    ehfCrho0: float
    ehfCrho1: float
    ehfCdrho0: float
    ehfCdrho1: float
    ehfCtau0: float
    ehfCtau1: float
    ehfCdJ0: float
    ehfCdJ1: float
    ehfCj0: float
    ehfCj1: float
    
    @classmethod
    def zeros(cls) -> "Energies":
        """Create zero-initialized energies."""
        dtypes = get_dtypes()
        return cls(
            ehft=0.0,
            ehf0=0.0,
            ehf1=0.0,
            ehf2=0.0,
            ehf3=0.0,
            ehfls=0.0,
            ehflsodd=0.0,
            ehfc=0.0,
            ecorc=0.0,
            ehfint=0.0,
            ehf=0.0,
            tke=0.0,
            e3corr=0.0,
            e_zpe=0.0,
            efluct1=jnp.zeros(1, dtype=dtypes.float),
            efluct1q=jnp.zeros(2, dtype=dtypes.float),
            efluct2=jnp.zeros(1, dtype=dtypes.float),
            efluct2q=jnp.zeros(2, dtype=dtypes.float),
            orbital=jnp.zeros(3, dtype=dtypes.float),
            spin=jnp.zeros(3, dtype=dtypes.float),
            total_angmom=jnp.zeros(3, dtype=dtypes.float),
            epair=jnp.zeros(2, dtype=dtypes.float),
            ehfCrho0=0.0,
            ehfCrho1=0.0,
            ehfCdrho0=0.0,
            ehfCdrho1=0.0,
            ehfCtau0=0.0,
            ehfCtau1=0.0,
            ehfCdJ0=0.0,
            ehfCdJ1=0.0,
            ehfCj0=0.0,
            ehfCj1=0.0,
        )
    
    @property
    def total(self) -> float:
        """Return total energy."""
        return self.ehfint


@jax.tree_util.register_dataclass
@dataclass
class Radii:
    """Container for nuclear radii and moments."""
    rms_n: float
    rms_p: float
    rms_tot: float
    charge: float
    
    # Quadrupole moments
    q20: float
    q22: float
    beta2: float
    gamma: float


def compute_integrated_energy(
    densities: Densities,
    force,
    grid: Grid,
    coulomb_potential: Optional[jax.Array] = None,
    pairing_energy: Optional[jax.Array] = None,
    mass_number: int = 1,
    use_coulomb: bool = True,
) -> Energies:
    """
    Compute integrated energy from the Skyrme energy density functional.
    
    This integrates the energy density over the spatial grid to get
    the total nuclear binding energy.
    
    Args:
        densities: Nuclear densities
        force: Force parameters
        grid: Spatial grid
        coulomb_potential: Pre-computed Coulomb potential
        pairing_energy: Pairing energy per isospin (2,)
        mass_number: Total mass number (for c.m. correction)
        use_coulomb: Whether to include Coulomb
        
    Returns:
        Energies object with all contributions
    """
    dtypes = get_dtypes()
    wxyz = grid.wxyz
    
    # Total and isoscalar/isovector densities
    rho_n = densities.rho[0]
    rho_p = densities.rho[1]
    rho_tot = rho_n + rho_p
    rho0 = rho_tot  # Isoscalar
    rho1 = rho_p - rho_n  # Isovector
    
    tau_n = densities.tau[0]
    tau_p = densities.tau[1]
    tau0 = tau_n + tau_p
    tau1 = tau_p - tau_n
    
    # Laplacians
    d2rho_n = compute_laplacian(rho_n, grid)
    d2rho_p = compute_laplacian(rho_p, grid)
    d2rho = d2rho_n + d2rho_p
    d2rho0 = d2rho
    d2rho1 = d2rho_p - d2rho_n
    
    # =========================================================================
    # t0 (central) contribution
    # =========================================================================
    ehf0 = wxyz * jnp.sum(
        (force.b0 * rho_tot**2 - force.b0p * (rho_p**2 + rho_n**2)) / 2.0
    )
    
    # C^rho contributions (alternative representation)
    ehfCrho0 = wxyz * jnp.sum(
        (force.Crho0 + force.Crho0D * rho0**force.power) * rho0**2
    )
    ehfCrho1 = wxyz * jnp.sum(
        (force.Crho1 + force.Crho1D * rho0**force.power) * rho1**2
    )
    
    # =========================================================================
    # t3 (density-dependent) contribution
    # =========================================================================
    ehf3 = wxyz * jnp.sum(
        rho_tot**force.power * 
        (force.b3 * rho_tot**2 - force.b3p * (rho_p**2 + rho_n**2)) / 3.0
    )
    e3corr = -force.power * ehf3 / 2.0  # Rearrangement correction
    
    # =========================================================================
    # Gradient (t1/t2 Laplacian) contribution
    # =========================================================================
    ehf2 = wxyz * jnp.sum(
        (-force.b2 * rho_tot * d2rho + 
         force.b2p * (rho_p * d2rho_p + rho_n * d2rho_n)) / 2.0
    )
    
    ehfCdrho0 = wxyz * jnp.sum(force.Cdrho0 * rho0 * d2rho0)
    ehfCdrho1 = wxyz * jnp.sum(force.Cdrho1 * rho1 * d2rho1)
    
    # =========================================================================
    # t1/t2 (momentum-dependent/kinetic) contribution
    # =========================================================================
    ehf1 = wxyz * jnp.sum(
        force.b1 * rho_tot * tau0 -
        force.b1p * (rho_n * tau_n + rho_p * tau_p)
    )
    
    ehfCtau0 = wxyz * jnp.sum(force.Ctau0 * tau0 * rho0)
    ehfCtau1 = wxyz * jnp.sum(force.Ctau1 * tau1 * rho1)
    
    # Current contribution to C^j
    j_n = densities.current[0]  # (3, nx, ny, nz)
    j_p = densities.current[1]
    j_tot = j_n + j_p
    j_diff = j_p - j_n
    
    j_tot_sq = jnp.sum(j_tot**2, axis=0)  # Sum over spatial components
    j_diff_sq = jnp.sum(j_diff**2, axis=0)
    
    ehfCj0 = -wxyz * jnp.sum(force.Ctau0 * j_tot_sq)
    ehfCj1 = -wxyz * jnp.sum(force.Ctau1 * j_diff_sq)
    
    # =========================================================================
    # Spin-orbit (time-even) contribution
    # =========================================================================
    div_J_n = compute_divergence(densities.sodens[0], grid)
    div_J_p = compute_divergence(densities.sodens[1], grid)
    div_J_tot = div_J_n + div_J_p
    
    ehfls = wxyz * jnp.sum(
        -force.b4 * rho_tot * div_J_tot -
        force.b4p * (rho_n * div_J_n + rho_p * div_J_p)
    )
    
    ehfCdJ0 = wxyz * jnp.sum(force.CdJ0 * rho_tot * div_J_tot)
    ehfCdJ1 = wxyz * jnp.sum(
        force.CdJ1 * (rho_p - rho_n) * (div_J_p - div_J_n)
    )
    
    # =========================================================================
    # Spin-orbit (time-odd) contribution: s · curl(j)
    # =========================================================================
    curl_j_n = compute_curl(j_n, grid)
    curl_j_p = compute_curl(j_p, grid)
    curl_j_tot = curl_j_n + curl_j_p
    
    s_n = densities.sdens[0]  # (3, nx, ny, nz)
    s_p = densities.sdens[1]
    s_tot = s_n + s_p
    
    # s · curl(j) dot product summed over spatial components
    s_curl_j_nn = jnp.sum(s_n * curl_j_n, axis=0)
    s_curl_j_pp = jnp.sum(s_p * curl_j_p, axis=0)
    s_curl_j_tot = jnp.sum(s_tot * curl_j_tot, axis=0)
    
    ehflsodd = wxyz * jnp.sum(
        -force.b4 * s_curl_j_tot -
        force.b4p * (s_curl_j_nn + s_curl_j_pp)
    )
    
    ehfls = ehfls + ehflsodd
    
    # =========================================================================
    # Coulomb energy
    # =========================================================================
    ehfc = 0.0
    ecorc = 0.0
    
    if use_coulomb and coulomb_potential is not None:
        # Direct Coulomb
        ehfc = wxyz * jnp.sum(0.5 * rho_p * coulomb_potential)
        
        # Slater exchange
        if force.ex != 0:
            slater_coeff = -3.0 / 4.0 * force.slate
            ehfc = ehfc + wxyz * jnp.sum(slater_coeff * rho_p**(4.0/3.0))
            ecorc = wxyz * jnp.sum(slater_coeff / 3.0 * rho_p**(4.0/3.0))
    
    # =========================================================================
    # Kinetic energy (free nucleon masses)
    # =========================================================================
    ehft = wxyz * jnp.sum(
        force.h2m[0] * tau_n + force.h2m[1] * tau_p
    )
    
    # =========================================================================
    # Center-of-mass correction (simple estimate)
    # =========================================================================
    e_zpe = 0.0
    if force.zpe == 1 and mass_number > 1:
        e_zpe = 17.3 / mass_number**0.2
    
    # =========================================================================
    # Total integrated energy
    # =========================================================================
    epair_total = 0.0
    if pairing_energy is not None:
        epair_total = jnp.sum(pairing_energy)
    
    ehfint = ehft + ehf0 + ehf1 + ehf2 + ehf3 + ehfls + ehfc - epair_total - e_zpe
    
    return Energies(
        ehft=float(ehft),
        ehf0=float(ehf0),
        ehf1=float(ehf1),
        ehf2=float(ehf2),
        ehf3=float(ehf3),
        ehfls=float(ehfls),
        ehflsodd=float(ehflsodd),
        ehfc=float(ehfc),
        ecorc=float(ecorc),
        ehfint=float(ehfint),
        ehf=0.0,  # Computed separately from s.p. levels
        tke=0.0,
        e3corr=float(e3corr),
        e_zpe=float(e_zpe),
        efluct1=jnp.zeros(1, dtype=dtypes.float),
        efluct1q=jnp.zeros(2, dtype=dtypes.float),
        efluct2=jnp.zeros(1, dtype=dtypes.float),
        efluct2q=jnp.zeros(2, dtype=dtypes.float),
        orbital=jnp.zeros(3, dtype=dtypes.float),
        spin=jnp.zeros(3, dtype=dtypes.float),
        total_angmom=jnp.zeros(3, dtype=dtypes.float),
        epair=pairing_energy if pairing_energy is not None else jnp.zeros(2, dtype=dtypes.float),
        ehfCrho0=float(ehfCrho0),
        ehfCrho1=float(ehfCrho1),
        ehfCdrho0=float(ehfCdrho0),
        ehfCdrho1=float(ehfCdrho1),
        ehfCtau0=float(ehfCtau0),
        ehfCtau1=float(ehfCtau1),
        ehfCdJ0=float(ehfCdJ0),
        ehfCdJ1=float(ehfCdJ1),
        ehfCj0=float(ehfCj0),
        ehfCj1=float(ehfCj1),
    )


def compute_sp_energy(
    sp_kinetic: jax.Array,
    sp_potential: jax.Array,
    wocc: jax.Array,
    wstates: jax.Array,
    e3corr: float,
    ecorrp: float,
    ecorc: float,
    pairing_energy: jax.Array,
    e_zpe: float = 0.0,
) -> float:
    """
    Compute total energy from single-particle energies.
    
    This uses the Kohn-Sham relation:
    E = (1/2) sum_i n_i (epsilon_i + t_i) + E_rearrange
    
    Args:
        sp_kinetic: Single-particle kinetic energies
        sp_potential: Single-particle potential energies
        wocc: BCS occupation factors (v^2)
        wstates: State degeneracy weights
        e3corr: Three-body rearrangement correction
        ecorrp: Pairing rearrangement correction
        ecorc: Exchange correlation correction
        pairing_energy: Pairing energy per isospin
        e_zpe: Center-of-mass correction
        
    Returns:
        Total binding energy
    """
    total_pairing = jnp.sum(pairing_energy)
    
    ehf = jnp.sum(
        wocc * wstates * (sp_kinetic + sp_potential)
    ) / 2.0 + e3corr + ecorrp + ecorc - total_pairing - e_zpe
    
    return float(ehf)


def compute_angular_momentum(
    sp_orbital: jax.Array,
    sp_spin: jax.Array,
    wocc: jax.Array,
    wstates: jax.Array,
) -> tuple:
    """
    Compute total angular momentum from single-particle values.
    
    Args:
        sp_orbital: Orbital angular momentum per state (nstates, 3)
        sp_spin: Spin angular momentum per state (nstates, 3)
        wocc: BCS occupation factors
        wstates: State degeneracy weights
        
    Returns:
        (orbital, spin, total) angular momentum vectors
    """
    weights = wocc * wstates
    
    orbital = jnp.zeros(3)
    spin = jnp.zeros(3)
    
    for i in range(3):
        orbital = orbital.at[i].set(jnp.sum(weights * sp_orbital[:, i]))
        spin = spin.at[i].set(jnp.sum(weights * sp_spin[:, i]))
    
    total = orbital + spin
    
    return orbital, spin, total


def compute_radii(densities: Densities, grid: Grid) -> Radii:
    """Compute nuclear radii and deformation parameters."""
    wxyz = grid.wxyz
    x, y, z = grid.x, grid.y, grid.z
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    
    # Create 3D grids for x, y, z
    # Since grid.x/y/z are 1D, we use broadcasting
    X = x[:, jnp.newaxis, jnp.newaxis]
    Y = y[jnp.newaxis, :, jnp.newaxis]
    Z = z[jnp.newaxis, jnp.newaxis, :]
    
    r2 = X**2 + Y**2 + Z**2
    
    rho_n = densities.rho[0]
    rho_p = densities.rho[1]
    
    n_counts = jnp.sum(rho_n) * wxyz
    p_counts = jnp.sum(rho_p) * wxyz
    tot_counts = n_counts + p_counts
    
    rms_n = jnp.sqrt(jnp.sum(rho_n * r2) * wxyz / (n_counts + 1e-10))
    rms_p = jnp.sqrt(jnp.sum(rho_p * r2) * wxyz / (p_counts + 1e-10))
    rms_tot = jnp.sqrt(jnp.sum((rho_n + rho_p) * r2) * wxyz / (tot_counts + 1e-10))
    
    # Simple charge radius estimate (proton radius + nucleon size)
    charge = jnp.sqrt(rms_p**2 + 0.64)  # 0.64 fm^2 is roughly <r^2>_proton
    
    # Quadrupole moments
    q20 = jnp.sum((rho_n + rho_p) * (2*Z**2 - X**2 - Y**2)) * wxyz
    q22 = jnp.sum((rho_n + rho_p) * (X**2 - Y**2)) * wxyz
    
    # Deformation parameters beta/gamma
    # beta = sqrt(5/pi) * (4pi/3AR^2) * Q/2?
    # Simplified version for now
    r_mean_sq = rms_tot**2
    q_all = jnp.sqrt(q20**2 + 3 * q22**2)
    beta = (jnp.sqrt(5 * jnp.pi) / (3 * tot_counts * r_mean_sq + 1e-10)) * q_all
    gamma = jnp.arctan2(jnp.sqrt(3.0) * q22, q20) * 180.0 / jnp.pi
    
    return Radii(
        rms_n=float(rms_n),
        rms_p=float(rms_p),
        rms_tot=float(rms_tot),
        charge=float(charge),
        q20=float(q20),
        q22=float(q22),
        beta2=float(beta),
        gamma=float(gamma)
    )
