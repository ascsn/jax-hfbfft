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
    weight = levels.wocc * levels.wstates
    weightuv = levels.wguv * levels.pairwg * levels.wstates

    rho, chi, sdens, tau, current, sodens = add_density_helper_vmap(
        levels.isospin,
        weight,
        weightuv,
        levels.psi,
        densities.rho,
        densities.chi,
        densities.tau,
        densities.current,
        densities.sdens,
        densities.sodens,
        grids.dx,
        grids.dy,
        grids.dz
    )

    densities.rho = densities.rho.at[...].set(
        jnp.sum(rho, axis=0)
    )

    densities.chi = densities.chi.at[...].set(
        jnp.sum(chi, axis=0)
    )

    densities.tau = densities.tau.at[...].set(
        jnp.sum(tau, axis=0)
    )

    densities.current = densities.current.at[...].set(
        jnp.sum(current, axis=0)
    )

    densities.sdens = densities.sdens.at[...].set(
        jnp.sum(sdens, axis=0)
    )

    densities.sodens = densities.sodens.at[...].set(
        jnp.sum(sodens, axis=0)
    )

    return densities
