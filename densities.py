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
    # Compute weights and contributions
    weights = levels.wocc * levels.wstates
    weightuvs = levels.wguv * levels.pairwg * levels.wstates
    
    iqs, rho_contribs, chi_contribs, tau_contribs, current_contribs, sdens_contribs, sodens_contribs = compute_contributions_vmap(
        levels.isospin, weights, weightuvs, levels.psi, grids.dx, grids.dy, grids.dz
    )
    
    # Sort by isospin for segment operations
    sorted_indices = jnp.argsort(iqs)
    sorted_iqs = iqs[sorted_indices]
    
    # Find segment boundaries
    num_segments = 2  # neutrons and protons
    segment_ids = sorted_iqs
    
    # Apply sorting to contributions
    sorted_rho = rho_contribs[sorted_indices]
    sorted_chi = chi_contribs[sorted_indices] 
    sorted_tau = tau_contribs[sorted_indices]
    sorted_current = current_contribs[sorted_indices]
    sorted_sdens = sdens_contribs[sorted_indices]
    sorted_sodens = sodens_contribs[sorted_indices]
    
    # Use jax.ops.segment_sum for efficient reduction
    from jax.ops import segment_sum
    
    # Flatten spatial dimensions for segment_sum, then reshape
    spatial_shape = sorted_rho.shape[1:]
    flat_rho = sorted_rho.reshape(sorted_rho.shape[0], -1)
    flat_chi = sorted_chi.reshape(sorted_chi.shape[0], -1)
    flat_tau = sorted_tau.reshape(sorted_tau.shape[0], -1)
    
    # Segment sum and reshape
    rho_sum = segment_sum(flat_rho, segment_ids, num_segments).reshape(num_segments, *spatial_shape)
    chi_sum = segment_sum(flat_chi, segment_ids, num_segments).reshape(num_segments, *spatial_shape)
    tau_sum = segment_sum(flat_tau, segment_ids, num_segments).reshape(num_segments, *spatial_shape)
    
    # For 5D arrays (current, sdens, sodens)
    spatial_5d_shape = sorted_current.shape[1:]
    flat_current = sorted_current.reshape(sorted_current.shape[0], -1)
    flat_sdens = sorted_sdens.reshape(sorted_sdens.shape[0], -1)
    flat_sodens = sorted_sodens.reshape(sorted_sodens.shape[0], -1)
    
    current_sum = segment_sum(flat_current, segment_ids, num_segments).reshape(num_segments, *spatial_5d_shape)
    sdens_sum = segment_sum(flat_sdens, segment_ids, num_segments).reshape(num_segments, *spatial_5d_shape)
    sodens_sum = segment_sum(flat_sodens, segment_ids, num_segments).reshape(num_segments, *spatial_5d_shape)
    
    densities.rho = rho_sum
    densities.chi = chi_sum
    densities.tau = tau_sum
    densities.current = current_sum
    densities.sdens = sdens_sum
    densities.sodens = sodens_sum
    
    return densities


def compute_single_contribution(iq, weight, weightuv, psin, dx, dy, dz):
    # Basic density |ψ|²
    rho_contrib = weight * jnp.real(
        psin[0,...] * jnp.conjugate(psin[0,...]) +
        psin[1,...] * jnp.conjugate(psin[1,...])
    )
    
    # Pairing density
    chi_contrib = 0.5 * weightuv * jnp.real(
        psin[0,...] * jnp.conjugate(psin[0,...]) +
        psin[1,...] * jnp.conjugate(psin[1,...])
    )
    
    # Spin densities (non-derivative terms)
    sdens_contrib = jnp.stack([
        2.0 * weight * jnp.real(jnp.conjugate(psin[0,...]) * psin[1,...]),
        2.0 * weight * jnp.imag(jnp.conjugate(psin[0,...]) * psin[1,...]),
        weight * jnp.real(
            jnp.conjugate(psin[0,...]) * psin[0,...] - 
            jnp.conjugate(psin[1,...]) * psin[1,...]
        )
    ], axis=0)
    
    # X derivatives
    ps1_x = cdervx00(dx, psin)
    tau_x = weight * jnp.real(
        ps1_x[0,...] * jnp.conjugate(ps1_x[0,...]) +
        ps1_x[1,...] * jnp.conjugate(ps1_x[1,...])
    )
    current_x = weight * jnp.imag(
        ps1_x[0,...] * jnp.conjugate(psin[0,...]) +
        ps1_x[1,...] * jnp.conjugate(psin[1,...])
    )
    sodens_y_x = -weight * jnp.imag(
        ps1_x[0,...] * jnp.conjugate(psin[0,...]) -
        ps1_x[1,...] * jnp.conjugate(psin[1,...])
    )
    sodens_z_x = -weight * jnp.real(
        psin[0,...] * jnp.conjugate(ps1_x[1,...]) -
        psin[1,...] * jnp.conjugate(ps1_x[0,...])
    )
    
    # Y derivatives  
    ps1_y = cdervy00(dy, psin)
    tau_y = weight * jnp.real(
        ps1_y[0,...] * jnp.conjugate(ps1_y[0,...]) +
        ps1_y[1,...] * jnp.conjugate(ps1_y[1,...])
    )
    current_y = weight * jnp.imag(
        ps1_y[0,...] * jnp.conjugate(psin[0,...]) +
        ps1_y[1,...] * jnp.conjugate(psin[1,...])
    )
    sodens_x_y = weight * jnp.imag(
        ps1_y[0,...] * jnp.conjugate(psin[0,...]) -
        ps1_y[1,...] * jnp.conjugate(psin[1,...])
    )
    sodens_z_y = -weight * jnp.imag(
        ps1_y[1,...] * jnp.conjugate(psin[0,...]) +
        ps1_y[0,...] * jnp.conjugate(psin[1,...])
    )
    
    # Z derivatives
    ps1_z = cdervz00(dz, psin) 
    tau_z = weight * jnp.real(
        ps1_z[0,...] * jnp.conjugate(ps1_z[0,...]) +
        ps1_z[1,...] * jnp.conjugate(ps1_z[1,...])
    )
    current_z = weight * jnp.imag(
        ps1_z[0,...] * jnp.conjugate(psin[0,...]) +
        ps1_z[1,...] * jnp.conjugate(psin[1,...])
    )
    sodens_x_z = weight * jnp.real(
        ps1_z[1,...] * jnp.conjugate(psin[0,...]) -
        ps1_z[0,...] * jnp.conjugate(psin[1,...])
    )
    sodens_y_z = weight * jnp.imag(
        ps1_z[1,...] * jnp.conjugate(psin[0,...]) +
        ps1_z[0,...] * jnp.conjugate(psin[1,...])
    )
    
    # Total kinetic energy density
    tau_contrib = tau_x + tau_y + tau_z
    
    # Current density components
    current_contrib = jnp.stack([current_x, current_y, current_z], axis=0)
    
    # Spin-orbit density components
    sodens_contrib = jnp.stack([
        sodens_x_y + sodens_x_z,  # sodens_x total
        sodens_y_x + sodens_y_z,  # sodens_y total  
        sodens_z_x + sodens_z_y   # sodens_z total
    ], axis=0)
    
    return iq, rho_contrib, chi_contrib, tau_contrib, current_contrib, sdens_contrib, sodens_contrib


# Vectorize the contribution computation
compute_contributions_vmap = jax.vmap(
    compute_single_contribution,
    in_axes=(0, 0, 0, 0, None, None, None)  # vectorize over first 4 args, broadcast scalars
)