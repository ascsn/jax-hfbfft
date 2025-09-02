import jax
import jax.numpy as jnp
from dataclasses import dataclass, field
from levels import cdervx00, cdervy00, cdervz00

@jax.tree_util.register_dataclass
@dataclass
class Densities:
    rho: jax.Array
    chi: jax.Array
    tau: jax.Array
    current: jax.Array
    sdens: jax.Array
    sodens: jax.Array


def init_densities(grids) -> Densities:
    shape4d = (2, grids.nx, grids.ny, grids.nz)
    shape5d = (2, 3, grids.nx, grids.ny, grids.nz)

    default_kwargs = {
        'rho': jnp.zeros(shape4d, dtype=jnp.float64),
        'chi': jnp.zeros(shape4d, dtype=jnp.float64),
        'tau': jnp.zeros(shape4d, dtype=jnp.float64),
        'current': jnp.zeros(shape5d, dtype=jnp.float64),
        'sdens': jnp.zeros(shape5d, dtype=jnp.float64),
        'sodens': jnp.zeros(shape5d, dtype=jnp.float64),
    }

    return Densities(**default_kwargs)


def add_density_helper(
    iq,
    weight,
    weightuv,
    psin,
    rho,
    chi,
    tau,
    current,
    sdens,
    sodens,
    dx,
    dy,
    dz
):
    rho = rho.at[iq,...].add(
        weight *
        jnp.real(
            psin[0,...] * jnp.conjugate(psin[0,...]) +
            psin[1,...] * jnp.conjugate(psin[1,...])
        )
    )

    chi = chi.at[iq,...].add(
        0.5 * weightuv *
        jnp.real(
            psin[0,...] * jnp.conjugate(psin[0,...]) +
            psin[1,...] * jnp.conjugate(psin[1,...])
        )
    )

    sdens = sdens.at[iq,0,...].add(
        2.0 * weight * jnp.real(jnp.conjugate(psin[0,...]) * psin[1,...])
    )

    sdens = sdens.at[iq,1,...].add(
        2.0 * weight * jnp.imag(jnp.conjugate(psin[0,...]) * psin[1,...])
    )

    sdens = sdens.at[iq,2,...].add(
        weight *
        jnp.real(
            jnp.conjugate(psin[0,...]) * psin[0,...] - jnp.real(jnp.conjugate(psin[1,...]) * psin[1,...])
        )
    )

    ps1 = cdervx00(dx, psin)

    tau = tau.at[iq,...].add(
        weight *
        jnp.real(
            ps1[0,...] * jnp.conjugate(ps1[0,...]) + ps1[1,...] * jnp.conjugate(ps1[1,...])
        )
    )
    current = current.at[iq,0,...].add(
        weight *
        jnp.imag(
            ps1[0,...] * jnp.conjugate(psin[0,...]) + ps1[1,...] * jnp.conjugate(psin[1,...])
        )
    )
    sodens = sodens.at[iq,1,...].add(
        -weight *
        (
            jnp.imag(ps1[0,...] * jnp.conjugate(psin[0,...])) -
            jnp.imag(ps1[1,...] * jnp.conjugate(psin[1,...]))
        )
    )
    sodens = sodens.at[iq,2,...].add(
        -weight *
        (
            jnp.real(psin[0,...] * jnp.conjugate(ps1[1,...])) -
            jnp.real(psin[1,...] * jnp.conjugate(ps1[0,...]))
        )
    )

    ps1 = cdervy00(dy, psin)

    tau = tau.at[iq,...].add(
        weight *
        jnp.real(
            ps1[0,...] * jnp.conjugate(ps1[0,...]) +
            ps1[1,...] * jnp.conjugate(ps1[1,...])
        )
    )
    current = current.at[iq,1,...].add(
        weight *
        jnp.imag(
            ps1[0,...] * jnp.conjugate(psin[0,...]) +
            ps1[1,...] * jnp.conjugate(psin[1,...])
        )
    )
    sodens = sodens.at[iq,0,...].add(
        weight *
        jnp.imag(
            ps1[0,...] * jnp.conjugate(psin[0,...]) -
            ps1[1,...] * jnp.conjugate(psin[1,...])
        )
    )
    sodens = sodens.at[iq,2,...].add(
        -weight *
        jnp.imag(
            ps1[1,...] * jnp.conjugate(psin[0,...]) +
            ps1[0,...] * jnp.conjugate(psin[1,...])
        )
    )

    ps1 = cdervz00(dz, psin)

    tau = tau.at[iq,...].add(
        weight *
        jnp.real(
            ps1[0,...] * jnp.conjugate(ps1[0,...]) +
            ps1[1,...] * jnp.conjugate(ps1[1,...])
        )
    )
    current = current.at[iq,2,...].add(
        weight *
        jnp.imag(
            ps1[0,...] * jnp.conjugate(psin[0,...]) +
            ps1[1,...] * jnp.conjugate(psin[1,...])
        )
    )
    sodens = sodens.at[iq,0,...].add(
        weight *
        jnp.real(
            ps1[1,...] * jnp.conjugate(psin[0,...]) -
            ps1[0,...] * jnp.conjugate(psin[1,...])
        )
    )
    sodens = sodens.at[iq,1,...].add(
        weight *
        jnp.imag(
            ps1[1,...] * jnp.conjugate(psin[0,...]) +
            ps1[0,...] * jnp.conjugate(psin[1,...])
        )
    )

    return rho, chi, sdens, tau, current, sodens

add_density_helper_vmap = jax.vmap(
    add_density_helper,
    in_axes=(0, 0, 0, 0, None, None, None, None, None, None, None, None, None)
)

@jax.jit
def add_density(densities, grids, levels):
    
    # Reset densities to zero first - using direct assignment
    rho = jnp.zeros_like(densities.rho)
    chi = jnp.zeros_like(densities.chi)
    tau = jnp.zeros_like(densities.tau)
    current = jnp.zeros_like(densities.current)
    sdens = jnp.zeros_like(densities.sdens)
    sodens = jnp.zeros_like(densities.sodens)
    
    #Loop over ALL states in the basis, not just a subset
    for nst in range(levels.nstmax):
        iq = levels.isospin[nst]
        weight = levels.wocc[nst] * levels.wstates[nst]
        weightuv = levels.wguv[nst] * levels.pairwg[nst] * levels.wstates[nst]
                    
        
        psin = levels.psi[nst]  # Shape: (2, nx, ny, nz)
        
        # Calculate density contribution |ψ|²
        rho_contrib = weight * jnp.real(
            psin[0,...] * jnp.conjugate(psin[0,...]) +
            psin[1,...] * jnp.conjugate(psin[1,...])
        )
        rho = rho.at[iq,...].add(rho_contrib)
        
        # Calculate pairing density contribution
        chi_contrib = 0.5 * weightuv * jnp.real(
            psin[0,...] * jnp.conjugate(psin[0,...]) +
            psin[1,...] * jnp.conjugate(psin[1,...])
        )
        chi = chi.at[iq,...].add(chi_contrib)
        
        # Calculate spin density components
        sdens_x = 2.0 * weight * jnp.real(jnp.conjugate(psin[0,...]) * psin[1,...])
        sdens_y = 2.0 * weight * jnp.imag(jnp.conjugate(psin[0,...]) * psin[1,...])  
        sdens_z = weight * jnp.real(
            jnp.conjugate(psin[0,...]) * psin[0,...] - 
            jnp.conjugate(psin[1,...]) * psin[1,...]
        )
        
        sdens = sdens.at[iq,0,...].add(sdens_x)
        sdens = sdens.at[iq,1,...].add(sdens_y)
        sdens = sdens.at[iq,2,...].add(sdens_z)
        
        # Calculate kinetic energy density and current from derivatives
        # X derivatives
        ps1 = cdervx00(grids.dx, psin)
        tau_contrib = weight * jnp.real(
            ps1[0,...] * jnp.conjugate(ps1[0,...]) +
            ps1[1,...] * jnp.conjugate(ps1[1,...])
        )
        tau = tau.at[iq,...].add(tau_contrib)
        
        current_x = weight * jnp.imag(
            ps1[0,...] * jnp.conjugate(psin[0,...]) +
            ps1[1,...] * jnp.conjugate(psin[1,...])
        )
        current = current.at[iq,0,...].add(current_x)
        
        sodens_y = -weight * (
            jnp.imag(ps1[0,...] * jnp.conjugate(psin[0,...])) -
            jnp.imag(ps1[1,...] * jnp.conjugate(psin[1,...]))
        )
        sodens_z = -weight * (
            jnp.real(psin[0,...] * jnp.conjugate(ps1[1,...])) -
            jnp.real(psin[1,...] * jnp.conjugate(ps1[0,...]))
        )
        sodens = sodens.at[iq,1,...].add(sodens_y)
        sodens = sodens.at[iq,2,...].add(sodens_z)
        
        # Y derivatives  
        ps1 = cdervy00(grids.dy, psin)
        tau_contrib = weight * jnp.real(
            ps1[0,...] * jnp.conjugate(ps1[0,...]) +
            ps1[1,...] * jnp.conjugate(ps1[1,...])
        )
        tau = tau.at[iq,...].add(tau_contrib)
        
        current_y = weight * jnp.imag(
            ps1[0,...] * jnp.conjugate(psin[0,...]) +
            ps1[1,...] * jnp.conjugate(psin[1,...])
        )
        current = current.at[iq,1,...].add(current_y)
        
        sodens_x = weight * jnp.imag(
            ps1[0,...] * jnp.conjugate(psin[0,...]) -
            ps1[1,...] * jnp.conjugate(psin[1,...])
        )
        sodens_z_y = -weight * jnp.imag(
            ps1[1,...] * jnp.conjugate(psin[0,...]) +
            ps1[0,...] * jnp.conjugate(psin[1,...])
        )
        sodens = sodens.at[iq,0,...].add(sodens_x)
        sodens = sodens.at[iq,2,...].add(sodens_z_y)
        
        # Z derivatives
        ps1 = cdervz00(grids.dz, psin)
        tau_contrib = weight * jnp.real(
            ps1[0,...] * jnp.conjugate(ps1[0,...]) +
            ps1[1,...] * jnp.conjugate(ps1[1,...])
        )
        tau = tau.at[iq,...].add(tau_contrib)
        
        current_z = weight * jnp.imag(
            ps1[0,...] * jnp.conjugate(psin[0,...]) +
            ps1[1,...] * jnp.conjugate(psin[1,...])
        )
        current = current.at[iq,2,...].add(current_z)
        
        sodens_x_z = weight * jnp.real(
            ps1[1,...] * jnp.conjugate(psin[0,...]) -
            ps1[0,...] * jnp.conjugate(psin[1,...])
        )
        sodens_y_z = weight * jnp.imag(
            ps1[1,...] * jnp.conjugate(psin[0,...]) +
            ps1[0,...] * jnp.conjugate(psin[1,...])
        )
        sodens = sodens.at[iq,0,...].add(sodens_x_z)
        sodens = sodens.at[iq,1,...].add(sodens_y_z)
    
    # Return updated densities object by directly updating the fields
    densities.rho = rho
    densities.chi = chi  
    densities.tau = tau
    densities.current = current
    densities.sdens = sdens
    densities.sodens = sodens
    
    return densities