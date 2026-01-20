import jax
import jax.numpy as jnp
from dataclasses import dataclass, field


@jax.tree_util.register_dataclass
@dataclass
class Energies:
    ehft: float            # kinetic energy
    ehf0: float            # t0 contribution
    ehf1: float            # b1 contribution (current part)
    ehf2: float            # b2 contribution (Laplacian part)
    ehf3: float            # t3 contribution which models density dependence
    ehfls: float           # spin-orbit contribution (time even)
    ehflsodd: float        # spin-orbit contribution (odd-odd)
    ehfc: float            # Coulomb contribution
    ecorc: float           # Slater & Koopman exchange
    ehfint: float          # integrated total energy
    efluct1: jax.Array     # maximum absolute value of λ⁻_αβ
    efluct1q: jax.Array    # maximum absolute value of λ⁻_αβ for isospin q
    efluct1prev: float     # efluct1 of previous iteration
    efluct2: jax.Array     # rms of λ⁻_αβ
    efluct2q: jax.Array    # rms of λ⁻_αβ for isospin q
    efluct2prev: float     # fluctuation of previous iteration
    tke: float             # kinetic energy summed
    ehf: float             # total energy from s.p. levels
    ehfprev: float         # total energy from s.p. levels of previous iteration
    e3corr: float          # rearrangement energy 3-body term
    e_zpe: float           # c.m. energy correction
    orbital: jax.Array     # components of total orbital angular momentum
    spin: jax.Array        # components of total spin
    total_angmom: jax.Array # components of total angular momentum
    e_extern: float        # energy from external field
    ehfCrho0: float        # C^ρ_0 contribution
    ehfCrho1: float        # C^ρ_1 contribution
    ehfCdrho0: float       # C^∇ρ_0 contribution
    ehfCdrho1: float       # C^∇ρ_1 contribution
    ehfCtau0: float        # C^τ_0 contribution
    ehfCtau1: float        # C^τ_1 contribution
    ehfCdJ0: float         # C^∇J_0 contribution
    ehfCdJ1: float         # C^∇J_1 contribution
    ehfCj0: float          # C^j_0 contribution
    ehfCj1: float          # C^j_1 contribution


def init_energies(**kwargs) -> Energies:
    """Initialize the Energies object with default or provided values."""
    default_kwargs = {
        'ehft': 0.0,
        'ehf0': 0.0,
        'ehf1': 0.0,
        'ehf2': 0.0,
        'ehf3': 0.0,
        'ehfls': 0.0,
        'ehflsodd': 0.0,
        'ehfc': 0.0,
        'ecorc': 0.0,
        'ehfint': 0.0,
        'efluct1': jnp.zeros(1, dtype=jnp.float64),
        'efluct1q': jnp.zeros(2, dtype=jnp.float64),
        'efluct1prev': 0.0,
        'efluct2': jnp.zeros(1, dtype=jnp.float64),
        'efluct2q': jnp.zeros(2, dtype=jnp.float64),
        'efluct2prev': 0.0,
        'tke': 0.0,
        'ehf': 0.0,
        'ehfprev': 0.0,
        'e3corr': 0.0,
        'e_zpe': 0.0,
        'orbital': jnp.zeros(3, dtype=jnp.float64),
        'spin': jnp.zeros(3, dtype=jnp.float64),
        'total_angmom': jnp.zeros(3, dtype=jnp.float64),
        'e_extern': 0.0,
        'ehfCrho0': 0.0,
        'ehfCrho1': 0.0,
        'ehfCdrho0': 0.0,
        'ehfCdrho1': 0.0,
        'ehfCtau0': 0.0,
        'ehfCtau1': 0.0,
        'ehfCdJ0': 0.0,
        'ehfCdJ1': 0.0,
        'ehfCj0': 0.0,
        'ehfCj1': 0.0,
    }

    default_kwargs.update(kwargs)
    
    return Energies(**default_kwargs)

@jax.jit
def integ_energy(coulomb, densities, energies, forces, grids, levels, params, pairs):
    # Step 1: Compute Laplacian of densities
    worka = jnp.zeros((2, grids.nx, grids.ny, grids.nz), dtype=jnp.float64)
    
    for iq in range(2):
        worka = worka.at[iq,...].set(
            jnp.einsum('ij,jkl->ikl', grids.der2x, densities.rho[iq,...]) +
            jnp.einsum('jl,ilk->ijk', grids.der2y, densities.rho[iq,...]) +
            jnp.einsum('kl,ijl->ijk', grids.der2z, densities.rho[iq,...])
        )
    
    # Step 2: Calculate ehf0, ehf2, ehf3, and other density contributions
    rhot = densities.rho[0,...] + densities.rho[1,...]
    rho0 = rhot
    rho1 = -densities.rho[0,...] + densities.rho[1,...]
    rhon = densities.rho[0,...]
    rhop = densities.rho[1,...]
    
    d2rhon = worka[0,...]
    d2rhop = worka[1,...]
    d2rho = d2rhon + d2rhop
    d2rho0 = d2rho
    d2rho1 = -d2rhon + d2rhop
    
    tau0 = densities.tau[0,...] + densities.tau[1,...]
    tau1 = -densities.tau[0,...] + densities.tau[1,...]
    
    ehf0 = grids.wxyz * jnp.sum(
        (forces.b0 * rhot**2 - forces.b0p * (rhop**2 + rhon**2)) / 2.0
    )
    
    ehfCrho0 = grids.wxyz * jnp.sum(
        (forces.Crho0 + forces.Crho0D * rho0**forces.power) * rho0**2
    )
    
    ehfCrho1 = grids.wxyz * jnp.sum(
        (forces.Crho1 + forces.Crho1D * rho0**forces.power) * rho1**2
    )
    
    ehf3 = grids.wxyz * jnp.sum(
        rhot**forces.power * (forces.b3 * rhot**2 - forces.b3p * (rhop**2 + rhon**2)) / 3.0
    )
    
    ehf2 = grids.wxyz * jnp.sum(
        (-forces.b2 * rhot * d2rho + forces.b2p * (rhop * d2rhop + rhon * d2rhon)) / 2.0
    )
    
    ehfCdrho0 = grids.wxyz * jnp.sum(forces.Cdrho0 * rho0 * d2rho0)
    ehfCdrho1 = grids.wxyz * jnp.sum(forces.Cdrho1 * rho1 * d2rho1)
    ehfCtau0 = grids.wxyz * jnp.sum(forces.Ctau0 * tau0 * rho0)
    ehfCtau1 = grids.wxyz * jnp.sum(forces.Ctau1 * tau1 * rho1)
    
    e3corr = -forces.power * ehf3 / 2.0
    
    # Step 3: Calculate ehf1 contribution (current part)
    current_squared = (densities.current[:,:,:,0,:]**2 + 
                       densities.current[:,:,:,1,:]**2 + 
                       densities.current[:,:,:,2,:]**2)
    
    ehf1 = grids.wxyz * jnp.sum(
        forces.b1 * ((densities.rho[0,...] + densities.rho[1,...]) * 
                     (densities.tau[0,...] + densities.tau[1,...])) -
        forces.b1p * (densities.rho[0,...] * densities.tau[0,...] + 
                      densities.rho[1,...] * densities.tau[1,...])
    )
    
    ehfCj0 = -grids.wxyz * jnp.sum(
        forces.Ctau0 * ((densities.current[:,:,:,0,0] + densities.current[:,:,:,0,1])**2 +
                       (densities.current[:,:,:,1,0] + densities.current[:,:,:,1,1])**2 +
                       (densities.current[:,:,:,2,0] + densities.current[:,:,:,2,1])**2)
    )
    
    ehfCj1 = -grids.wxyz * jnp.sum(
        forces.Ctau1 * ((-densities.current[:,:,:,0,0] + densities.current[:,:,:,0,1])**2 +
                        (-densities.current[:,:,:,1,0] + densities.current[:,:,:,1,1])**2 +
                        (-densities.current[:,:,:,2,0] + densities.current[:,:,:,2,1])**2)
    )
    
    # Step 4: Calculate spin-orbit contribution (time even part)
    div_sodens = jnp.zeros((2, grids.nx, grids.ny, grids.nz), dtype=jnp.float64)
    for iq in range(2):
        div_sodens = div_sodens.at[iq,...].set(
            jnp.einsum('ij,jkl->ikl', grids.der1x, densities.sodens[iq,0,...]) +
            jnp.einsum('jl,ilk->ijk', grids.der1y, densities.sodens[iq,1,...]) +
            jnp.einsum('kl,ijl->ijk', grids.der1z, densities.sodens[iq,2,...])
        )
    
    ehfls = grids.wxyz * jnp.sum(
        -forces.b4 * (densities.rho[0,...] + densities.rho[1,...]) *
        (div_sodens[0,...] + div_sodens[1,...]) -
        forces.b4p * (densities.rho[1,...] * div_sodens[1,...] + 
                     densities.rho[0,...] * div_sodens[0,...])
    )
    
    ehfCdJ0 = grids.wxyz * jnp.sum(
        forces.CdJ0 * (densities.rho[0,...] + densities.rho[1,...]) *
        (div_sodens[0,...] + div_sodens[1,...])
    )
    
    ehfCdJ1 = grids.wxyz * jnp.sum(
        forces.CdJ1 * (-densities.rho[0,...] + densities.rho[1,...]) *
        (-div_sodens[0,...] + div_sodens[1,...])
    )
    
    # Step 5: Calculate spin-orbit contribution (odd-odd part)
    curl_current = jnp.zeros((2, 3, grids.nx, grids.ny, grids.nz), dtype=jnp.float64)
    for iq in range(2):
        curl_current = curl_current.at[iq,0,...].set(
            jnp.einsum('jl,ilk->ijk', grids.der1y, densities.current[iq,2,...]) -
            jnp.einsum('kl,ijl->ijk', grids.der1z, densities.current[iq,1,...])
        )
        curl_current = curl_current.at[iq,1,...].set(
            jnp.einsum('kl,ijl->ijk', grids.der1z, densities.current[iq,0,...]) -
            jnp.einsum('ij,jkl->ikl', grids.der1x, densities.current[iq,2,...])
        )
        curl_current = curl_current.at[iq,2,...].set(
            jnp.einsum('ij,jkl->ikl', grids.der1x, densities.current[iq,1,...]) -
            jnp.einsum('jl,ilk->ijk', grids.der1y, densities.current[iq,0,...])
        )
    
    
    ehflsodd = grids.wxyz * jnp.sum(
        -forces.b4 * (
            densities.sdens[0,0,...] * (curl_current[0,0,...] + curl_current[1,0,...]) +
            densities.sdens[0,1,...] * (curl_current[0,1,...] + curl_current[1,1,...]) +
            densities.sdens[0,2,...] * (curl_current[0,2,...] + curl_current[1,2,...]) +
            densities.sdens[1,0,...] * (curl_current[0,0,...] + curl_current[1,0,...]) +
            densities.sdens[1,1,...] * (curl_current[0,1,...] + curl_current[1,1,...]) +
            densities.sdens[1,2,...] * (curl_current[0,2,...] + curl_current[1,2,...])
        ) -
        forces.b4p * (
            densities.sdens[0,0,...] * curl_current[0,0,...] +
            densities.sdens[0,1,...] * curl_current[0,1,...] +
            densities.sdens[0,2,...] * curl_current[0,2,...] +
            densities.sdens[1,0,...] * curl_current[1,0,...] +
            densities.sdens[1,1,...] * curl_current[1,1,...] +
            densities.sdens[1,2,...] * curl_current[1,2,...]
        )
    )
    
    ehfls = ehfls + ehflsodd
    
    # Step 6: Coulomb energy with Slater term
    ehfc = 0.0
    ecorc = 0.0
    if params.tcoul:
        sc = jnp.where(
            forces.ex != 0,
            -3.0 / 4.0 * forces.slate,
            0.0
        )
        
        ehfc = grids.wxyz * jnp.sum(
            0.5 * densities.rho[1,...] * coulomb.wcoul +
            sc * densities.rho[1,...]**(4.0/3.0)
        )
        
        ecorc = grids.wxyz * jnp.sum(
            sc / 3.0 * densities.rho[1,...]**(4.0/3.0)
        )
    
    # Step 7: Kinetic energy contribution
    ehft = grids.wxyz * jnp.sum(
        forces.h2m[0] * densities.tau[0,...] + 
        forces.h2m[1] * densities.tau[1,...]
    )
    
    # Step 8: Optional c.m. correction (simple estimate)
    e_zpe = jnp.where(
        forces.zpe == 1,
        17.3 / levels.mass_number**0.2,
        0.0
    )
    
    # Step 9: Form total energy (for now assuming zero pairing energy)
    total_pairing_energy = jnp.sum(pairs.epair)
    ehfint = ehft + ehf0 + ehf1 + ehf2 + ehf3 + ehfls + ehfc - total_pairing_energy - e_zpe
    
    # Update the energies dataclass
    return Energies(
        ehft=ehft,
        ehf0=ehf0,
        ehf1=ehf1,
        ehf2=ehf2,
        ehf3=ehf3,
        ehfls=ehfls,
        ehflsodd=ehflsodd,
        ehfc=ehfc,
        ecorc=ecorc,
        ehfint=ehfint,
        efluct1=energies.efluct1,
        efluct1q=energies.efluct1q,
        efluct1prev=energies.efluct1prev,
        efluct2=energies.efluct2,
        efluct2q=energies.efluct2q,
        efluct2prev=energies.efluct2prev,
        tke=energies.tke,
        ehf=energies.ehf,
        ehfprev=energies.ehfprev,
        e3corr=e3corr,
        e_zpe=e_zpe,
        orbital=energies.orbital,
        spin=energies.spin,
        total_angmom=energies.total_angmom,
        e_extern=energies.e_extern,
        ehfCrho0=ehfCrho0,
        ehfCrho1=ehfCrho1,
        ehfCdrho0=ehfCdrho0,
        ehfCdrho1=ehfCdrho1,
        ehfCtau0=ehfCtau0,
        ehfCtau1=ehfCtau1,
        ehfCdJ0=ehfCdJ0,
        ehfCdJ1=ehfCdJ1,
        ehfCj0=ehfCj0,
        ehfCj1=ehfCj1
    )


@jax.jit
def sum_energy(energies, levels, meanfield, pairs):

    total_pairing_energy = jnp.sum(pairs.epair)
    ehf = jnp.sum(
        levels.wocc * levels.wstates * (levels.sp_kinetic + levels.sp_energy)
    ) / 2.0 + energies.e3corr + meanfield.ecorrp + energies.ecorc - total_pairing_energy - energies.e_zpe
    
    # Calculate total kinetic energy
    tke = jnp.sum(levels.wocc * levels.wstates * levels.sp_kinetic)
    
    # Calculate angular momentum components
    orbital = jnp.zeros(3, dtype=jnp.float64)
    spin = jnp.zeros(3, dtype=jnp.float64)
    total_angmom = jnp.zeros(3, dtype=jnp.float64)
    
    for i in range(3):
        orbital = orbital.at[i].set(
            jnp.sum(levels.wocc * levels.wstates * levels.sp_orbital[:,i])
        )
        
        spin = spin.at[i].set(
            jnp.sum(levels.wocc * levels.wstates * levels.sp_spin[:,i])
        )
    
    total_angmom = orbital + spin
    
    return Energies(
        ehft=energies.ehft,
        ehf0=energies.ehf0,
        ehf1=energies.ehf1,
        ehf2=energies.ehf2,
        ehf3=energies.ehf3,
        ehfls=energies.ehfls,
        ehflsodd=energies.ehflsodd,
        ehfc=energies.ehfc,
        ecorc=energies.ecorc,
        ehfint=energies.ehfint,
        efluct1=energies.efluct1,
        efluct1q=energies.efluct1q,
        efluct1prev=energies.efluct1prev,
        efluct2=energies.efluct2,
        efluct2q=energies.efluct2q,
        efluct2prev=energies.efluct2prev,
        tke=tke,
        ehf=ehf,
        ehfprev=energies.ehfprev,
        e3corr=energies.e3corr,
        e_zpe=energies.e_zpe,
        orbital=orbital,
        spin=spin,
        total_angmom=total_angmom,
        e_extern=energies.e_extern,
        ehfCrho0=energies.ehfCrho0,
        ehfCrho1=energies.ehfCrho1,
        ehfCdrho0=energies.ehfCdrho0,
        ehfCdrho1=energies.ehfCdrho1,
        ehfCtau0=energies.ehfCtau0,
        ehfCtau1=energies.ehfCtau1,
        ehfCdJ0=energies.ehfCdJ0,
        ehfCdJ1=energies.ehfCdJ1,
        ehfCj0=energies.ehfCj0,
        ehfCj1=energies.ehfCj1
    )