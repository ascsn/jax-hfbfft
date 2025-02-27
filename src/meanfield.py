import jax
import jax.numpy as jnp
from dataclasses import dataclass, field
from coulomb import poisson
from levels import cdervx00, cdervx02, cdervy00, cdervy02, cdervz00, cdervz02

@jax.tree_util.register_dataclass
@dataclass
class Meanfield:
    upot: jax.Array
    bmass: jax.Array
    divaq: jax.Array
    v_pair: jax.Array

    aq: jax.Array
    spot: jax.Array
    wlspot: jax.Array
    dbmass: jax.Array

    ecorrp: float


def init_meanfield(grids):
    shape4d = (2, grids.nx, grids.ny, grids.nz)
    shape5d = (2, 3, grids.nx, grids.ny, grids.nz)

    default_kwargs = {
        'upot': jnp.zeros(shape4d, dtype=jnp.float64),
        'bmass': jnp.zeros(shape4d, dtype=jnp.float64),
        'divaq': jnp.zeros(shape4d, dtype=jnp.float64),
        'v_pair': jnp.zeros(shape4d, dtype=jnp.float64),
        'aq': jnp.zeros(shape5d, dtype=jnp.float64),
        'spot': jnp.zeros(shape5d, dtype=jnp.float64),
        'wlspot': jnp.zeros(shape5d, dtype=jnp.float64),
        'dbmass': jnp.zeros(shape5d, dtype=jnp.float64),
        'ecorrp': 0.0
    }
    return Meanfield(**default_kwargs)


def hpsi00(grids, meanfield, iq, weight, weightuv, pinn):
    sigis = jnp.array([0.5, -0.5])

    # Step 1: non-derivative parts not involving spin
    pout = jnp.multiply(pinn, meanfield.upot[iq,...])

    # Step 2: the spin-current coupling
    pout = pout.at[0,...].add(
        (meanfield.spot[iq,0,...] - 1j * meanfield.spot[iq,1,...]) * \
        pinn[1,...] + meanfield.spot[iq,2,...] * pinn[0,...]
    )

    pout = pout.at[1,...].add(
        (meanfield.spot[iq,0,...] + 1j * meanfield.spot[iq,1,...]) * \
        pinn[0,...] - meanfield.spot[iq,2,...] * pinn[1,...]
    )

    # Step 3: derivative terms in x
    pswk, pswk2 = cdervx02(grids.dx, pinn, meanfield.bmass[iq,...])

    pout = pout.at[0,...].add(
        -(0.0 + 1j * (0.5 * meanfield.aq[iq,0,...] - sigis[0] * \
        meanfield.wlspot[iq,1,...])) * pswk[0,...] - sigis[0] * \
        meanfield.wlspot[iq,2,...] * pswk[1,...] - pswk2[0,...]
    )

    pout = pout.at[1,...].add(
        -(0.0 + 1j * (0.5 * meanfield.aq[iq,0,...] - sigis[1] * \
        meanfield.wlspot[iq,1,...])) * pswk[1,...] - sigis[1] * \
        meanfield.wlspot[iq,2,...] * pswk[0,...] - pswk2[1,...]
    )

    pswk2 = pswk2.at[0,...].set(
        (0.0 - 1j * 0.5) * (meanfield.aq[iq,0,...] - meanfield.wlspot[iq,1,...]) * \
        pinn[0,...] - 0.5 * meanfield.wlspot[iq,2,...] * pinn[1,...]
    )

    pswk2 = pswk2.at[1,...].set(
        (0.0 - 1j * 0.5) * (meanfield.aq[iq,0,...] + meanfield.wlspot[iq,1,...]) * \
        pinn[1,...] + 0.5 * meanfield.wlspot[iq,2,...] * pinn[0,...]
    )

    pswk = cdervx00(grids.dx, pswk2)

    pout = pout.at[...].add(pswk)

    # Step 4: derivative terms in y
    pswk, pswk2 = cdervy02(grids.dy, pinn, meanfield.bmass[iq,...])

    pout = pout.at[0,...].add(
        -(0.0 + 1j * (0.5 * meanfield.aq[iq,1,...] + sigis[0] * \
        meanfield.wlspot[iq,0,...])) * pswk[0,...] + (0.0 + 1j * (0.5 * \
        meanfield.wlspot[iq,2,...])) * pswk[1,...] - pswk2[0,...]
    )

    pout = pout.at[1,...].add(
        -(0.0 + 1j * (0.5 * meanfield.aq[iq,1,...] + sigis[1] * \
        meanfield.wlspot[iq,0,...])) * pswk[1,...] + (0.0 + 1j * (0.5 * \
        meanfield.wlspot[iq,2,...])) * pswk[0,...] - pswk2[1,...]
    )

    pswk2 = pswk2.at[0,...].set(
        (0.0 - 1j * 0.5) * (meanfield.aq[iq,1,...] + meanfield.wlspot[iq,0,...]) * \
        pinn[0,...] + (0.0 + 1j * 0.5) * meanfield.wlspot[iq,2,...] * pinn[1,...]
    )

    pswk2 = pswk2.at[1,...].set(
        (0.0 - 1j * 0.5) * (meanfield.aq[iq,1,...] - meanfield.wlspot[iq,0,...]) * \
        pinn[1,...] + (0.0 + 1j * 0.5 * meanfield.wlspot[iq,2,...]) * pinn[0,...]
    )

    pswk = cdervy00(grids.dy, pswk2)

    pout = pout.at[...].add(pswk)

    # Step 5: derivative terms in z
    pswk, pswk2 = cdervz02(grids.dz, pinn, meanfield.bmass[iq,...])

    pout = pout.at[0,...].add(
        -(0.0 + 1j * (0.5 * meanfield.aq[iq,2,...])) * pswk[0,...] + \
        (sigis[0] * meanfield.wlspot[iq,0,...] - 1j * (0.5 * \
        meanfield.wlspot[iq,1,...])) * pswk[1,...] - pswk2[0,...]
    )

    pout = pout.at[1,...].add(
        -(0.0 + 1j * (0.5 * meanfield.aq[iq,2,...])) * pswk[1,...] + \
        (sigis[1] * meanfield.wlspot[iq,0,...] - 1j * (0.5 * \
        meanfield.wlspot[iq,1,...])) * pswk[0,...] - pswk2[1,...]
    )

    pswk2 = pswk2.at[0,...].set(
        (0.0 - 1j * 0.5) * meanfield.aq[iq,2,...] * pinn[0,...] + \
        (0.5 * meanfield.wlspot[iq,0,...] - 1j * (0.5 * meanfield.wlspot[iq,1,...])) * pinn[1,...]
    )

    pswk2 = pswk2.at[1,...].set(
        (0.0 - 1j * 0.5) * meanfield.aq[iq,2,...] * pinn[1,...] + \
        (-0.5 * meanfield.wlspot[iq,0,...] - 1j * (0.5 * meanfield.wlspot[iq,1,...])) * pinn[0,...]
    )

    pswk = cdervz00(grids.dz, pswk2)

    pout = pout.at[...].add(pswk)

    # Step 6: multiply weight and single-particle
    # Hamiltonian, then add pairing part to pout
    pout_mf = jnp.copy(pout)

    pout = pout.at[0,...].set(
        weight * pout[0,...] - weightuv * meanfield.v_pair[iq,...] * pinn[0,...]
    )

    pout = pout.at[1,...].set(
        weight * pout[1,...] - weightuv * meanfield.v_pair[iq,...] * pinn[1,...]
    )

    return pout, pout_mf


def hpsi01(grids, meanfield, iq, weight, weightuv, pinn):
    sigis = jnp.array([0.5, -0.5])

    # Step 1: non-derivative parts not involving spin
    pout = jnp.multiply(pinn, meanfield.upot[iq,...])

    # Step 2: the spin-current coupling
    pout = pout.at[0,...].add(
        (meanfield.spot[iq,0,...] - 1j * meanfield.spot[iq,1,...]) * \
        pinn[1,...] + meanfield.spot[iq,2,...] * pinn[0,...]
    )

    pout = pout.at[1,...].add(
        (meanfield.spot[iq,0,...] + 1j * meanfield.spot[iq,1,...]) * \
        pinn[0,...] - meanfield.spot[iq,2,...] * pinn[1,...]
    )

    # Step 3: derivative terms in x
    pswk, pswk2 = cdervx02(grids.dx, pinn, meanfield.bmass[iq,...])

    pout = pout.at[0,...].add(
        -(0.0 + 1j * (0.5 * meanfield.aq[iq,0,...] - sigis[0] * \
        meanfield.wlspot[iq,1,...])) * pswk[0,...] - sigis[0] * \
        meanfield.wlspot[iq,2,...] * pswk[1,...] - pswk2[0,...]
    )

    pout = pout.at[1,...].add(
        -(0.0 + 1j * (0.5 * meanfield.aq[iq,0,...] - sigis[1] * \
        meanfield.wlspot[iq,1,...])) * pswk[1,...] - sigis[1] * \
        meanfield.wlspot[iq,2,...] * pswk[0,...] - pswk2[1,...]
    )

    pswk2 = pswk2.at[0,...].set(
        (0.0 - 1j * 0.5) * (meanfield.aq[iq,0,...] - meanfield.wlspot[iq,1,...]) * \
        pinn[0,...] - 0.5 * meanfield.wlspot[iq,2,...] * pinn[1,...]
    )

    pswk2 = pswk2.at[1,...].set(
        (0.0 - 1j * 0.5) * (meanfield.aq[iq,0,...] + meanfield.wlspot[iq,1,...]) * \
        pinn[1,...] + 0.5 * meanfield.wlspot[iq,2,...] * pinn[0,...]
    )

    pswk = cdervx00(grids.dx, pswk2)

    pout = pout.at[...].add(pswk)

    # Step 4: derivative terms in y
    pswk, pswk2 = cdervy02(grids.dy, pinn, meanfield.bmass[iq,...])

    pout = pout.at[0,...].add(
        -(0.0 + 1j * (0.5 * meanfield.aq[iq,1,...] + sigis[0] * \
        meanfield.wlspot[iq,0,...])) * pswk[0,...] + (0.0 + 1j * (0.5 * \
        meanfield.wlspot[iq,2,...])) * pswk[1,...] - pswk2[0,...]
    )

    pout = pout.at[1,...].add(
        -(0.0 + 1j * (0.5 * meanfield.aq[iq,1,...] + sigis[1] * \
        meanfield.wlspot[iq,0,...])) * pswk[1,...] + (0.0 + 1j * (0.5 * \
        meanfield.wlspot[iq,2,...])) * pswk[0,...] - pswk2[1,...]
    )

    pswk2 = pswk2.at[0,...].set(
        (0.0 - 1j * 0.5) * (meanfield.aq[iq,1,...] + meanfield.wlspot[iq,0,...]) * \
        pinn[0,...] + (0.0 + 1j * 0.5) * meanfield.wlspot[iq,2,...] * pinn[1,...]
    )

    pswk2 = pswk2.at[1,...].set(
        (0.0 - 1j * 0.5) * (meanfield.aq[iq,1,...] - meanfield.wlspot[iq,0,...]) * \
        pinn[1,...] + (0.0 + 1j * 0.5 * meanfield.wlspot[iq,2,...]) * pinn[0,...]
    )

    pswk = cdervy00(grids.dy, pswk2)

    pout = pout.at[...].add(pswk)

    # Step 5: derivative terms in z
    pswk, pswk2 = cdervz02(grids.dz, pinn, meanfield.bmass[iq,...])

    pout = pout.at[0,...].add(
        -(0.0 + 1j * (0.5 * meanfield.aq[iq,2,...])) * pswk[0,...] + \
        (sigis[0] * meanfield.wlspot[iq,0,...] - 1j * (0.5 * \
        meanfield.wlspot[iq,1,...])) * pswk[1,...] - pswk2[0,...]
    )

    pout = pout.at[1,...].add(
        -(0.0 + 1j * (0.5 * meanfield.aq[iq,2,...])) * pswk[1,...] + \
        (sigis[1] * meanfield.wlspot[iq,0,...] - 1j * (0.5 * \
        meanfield.wlspot[iq,1,...])) * pswk[0,...] - pswk2[1,...]
    )

    pswk2 = pswk2.at[0,...].set(
        (0.0 - 1j * 0.5) * meanfield.aq[iq,2,...] * pinn[0,...] + \
        (0.5 * meanfield.wlspot[iq,0,...] - 1j * (0.5 * meanfield.wlspot[iq,1,...])) * pinn[1,...]
    )

    pswk2 = pswk2.at[1,...].set(
        (0.0 - 1j * 0.5) * meanfield.aq[iq,2,...] * pinn[1,...] + \
        (-0.5 * meanfield.wlspot[iq,0,...] - 1j * (0.5 * meanfield.wlspot[iq,1,...])) * pinn[0,...]
    )

    pswk = cdervz00(grids.dz, pswk2)

    pout = pout.at[...].add(pswk)

    # Step 6: multiply weight and single-particle
    # Hamiltonian, then add pairing part to pout
    pout_mf = jnp.copy(pout)

    pout_del = jnp.multiply(pinn, meanfield.v_pair[iq,...])

    pout = pout.at[...].set(
        weight * pout - weightuv * pout_del
    )

    return pout, pout_mf, pout_del


@jax.jit
def skyrme(coulomb, densities, forces, grids, meanfield, params, static):
    # Step 1: 3-body contribution to upot.
    for iq in range(2):
        ic = 1 if iq == 0 else 0
        meanfield.upot = meanfield.upot.at[iq,...].set(
            (densities.rho[0,:,:,:] + densities.rho[1,:,:,:]) ** forces.power *
            (
                (forces.b3 * (forces.power + 2.0) / 3.0 - 2.0 * forces.b3p / 3.0) *
                densities.rho[iq,:,:,:] + forces.b3 * (forces.power + 2.0) / 3.0 *
                densities.rho[ic,:,:,:] - (forces.b3p * forces.power / 3.0) *
                (densities.rho[0,:,:,:] ** 2 + densities.rho[1,:,:,:] ** 2) /
                (densities.rho[0,:,:,:] + densities.rho[1,:,:,:] + 1.0e-25)
            )
        )

    # Step 2: add divergence of spin-orbit current to upot
    workden = jnp.zeros(densities.rho.shape, dtype=jnp.float64)

    for iq in range(2):
        workden = workden.at[iq,...].set(jnp.einsum('ij,jkl->ikl', grids.der1x, densities.sodens[iq,0,...]))
        workden = workden.at[iq,...].add(jnp.einsum('jl,ilk->ijk', grids.der1y, densities.sodens[iq,1,...]))
        workden = workden.at[iq,...].add(jnp.einsum('kl,ijl->ijk', grids.der1z, densities.sodens[iq,2,...]))

    for iq in range(2):
        ic = 1 if iq == 0 else 0
        meanfield.upot = meanfield.upot.at[iq,...].add(
            -(forces.b4 + forces.b4p) * workden[iq,...] - forces.b4 * workden[ic,...]
        )

    # Step 3: Coulomb potential
    if (params.tcoul):
        coulomb.wcoul = coulomb.wcoul.at[...].set(
            poisson(grids, params, coulomb, densities.rho)
        )
        meanfield.upot = meanfield.upot.at[1,...].add(coulomb.wcoul)
        if (forces.ex != 0):
            meanfield.upot = meanfield.upot.at[1,...].add(
                -forces.slate * densities.rho[1,...] ** (1.0 / 3.0)
            )

    # Step 4: remaining terms of upot
    workden = workden.at[...].set(0.0)

    for iq in range(2):
        workden = workden.at[iq,...].set(jnp.einsum('ij,jkl->ikl', grids.der2x, densities.rho[iq,...]))
        workden = workden.at[iq,...].add(jnp.einsum('jl,ilk->ijk', grids.der2y, densities.rho[iq,...]))
        workden = workden.at[iq,...].add(jnp.einsum('kl,ijl->ijk', grids.der2z, densities.rho[iq,...]))

    workvec = jnp.zeros(densities.sodens.shape, dtype=jnp.float64)

    for iq in range(2):
        ic = 1 if iq == 0 else 0
        meanfield.upot = meanfield.upot.at[iq,...].add(
            (forces.b0 - forces.b0p) * densities.rho[iq,...] + forces.b0 *
            densities.rho[ic,...] + (forces.b1 - forces.b1p) * densities.tau[iq,...] +
            forces.b1 * densities.tau[ic,...] - (forces.b2 - forces.b2p) *
            workden[iq,...] - forces.b2 * workden[ic,...]
        )

        # Step 5: effective mass
        meanfield.bmass = meanfield.bmass.at[iq,...].set(
            forces.h2m[iq] + (forces.b1 - forces.b1p) * densities.rho[iq,...] +
            forces.b1 * densities.rho[ic,...]
        )

        # Step 6: calculate grad(rho) and wlspot
        workvec = workvec.at[iq,0,...].set(jnp.einsum('ij,jkl->ikl', grids.der1x, densities.rho[iq,...]))
        workvec = workvec.at[iq,1,...].set(jnp.einsum('jl,ilk->ijk', grids.der1y, densities.rho[iq,...]))
        workvec = workvec.at[iq,2,...].set(jnp.einsum('kl,ijl->ijk', grids.der1z, densities.rho[iq,...]))

    for iq in range(2):
        ic = 1 if iq == 0 else 0
        meanfield.wlspot = meanfield.wlspot.at[iq,...].set(
            (forces.b4 + forces.b4p) * workvec[iq,...] + forces.b4 * workvec[ic,...]
        )

    for iq in range(2):
        workvec = workvec.at[iq,0,...].set(jnp.einsum('jl,ilk->ijk', grids.der1y, densities.sdens[iq,2,...]))
        workvec = workvec.at[iq,0,...].add(-jnp.einsum('kl,ijl->ijk', grids.der1z, densities.sdens[iq,1,...]))
        workvec = workvec.at[iq,1,...].set(jnp.einsum('kl,ijl->ijk', grids.der1z, densities.sdens[iq,0,...]))
        workvec = workvec.at[iq,1,...].add(-jnp.einsum('ij,jkl->ikl', grids.der1x, densities.sdens[iq,2,...]))
        workvec = workvec.at[iq,2,...].set(jnp.einsum('ij,jkl->ikl', grids.der1x, densities.sdens[iq,1,...]))
        workvec = workvec.at[iq,2,...].add(-jnp.einsum('kl,ijl->ijk', grids.der1y, densities.sdens[iq,0,...]))

    # Step 8: calculate A_q vector
    for iq in range(2):
        ic = 1 if iq == 0 else 0
        meanfield.aq = meanfield.aq.at[iq,...].set(
            -2.0 * (forces.b1 - forces.b1p) * densities.current[iq,...] - 2.0 *
            forces.b1 * densities.current[ic,...] - (forces.b4 + forces.b4p) *
            workvec[iq,...] -forces.b4 * workvec[ic,...]
        )

    # Step 9: calculate the curl of the current density, store in spot
    for iq in range(2):
        meanfield.spot = meanfield.spot.at[iq,0,...].set(
            jnp.einsum('jl,ilk->ijk', grids.der1y, densities.current[iq,2,...])
        )
        meanfield.spot = meanfield.spot.at[iq,0,...].add(
            -jnp.einsum('kl,ijl->ijk', grids.der1z, densities.current[iq,1,...])
        )
        meanfield.spot = meanfield.spot.at[iq,1,...].set(
            jnp.einsum('kl,ijl->ijk', grids.der1z, densities.current[iq,0,...])
        )
        meanfield.spot = meanfield.spot.at[iq,1,...].add(
            -jnp.einsum('ij,jkl->ikl', grids.der1x, densities.current[iq,2,...])
        )
        meanfield.spot = meanfield.spot.at[iq,2,...].set(
            jnp.einsum('ij,jkl->ikl', grids.der1x, densities.current[iq,1,...])
        )
        meanfield.spot = meanfield.spot.at[iq,2,...].add(
            -jnp.einsum('jl,ilk->ijk', grids.der1y, densities.current[iq,0,...])
        )

    # Step 10: combine isospin contributions
    new_spot_0 = -(forces.b4 + forces.b4p) * meanfield.spot[0,...] - forces.b4 * meanfield.spot[1,...]
    new_spot_1 = -(forces.b4 + forces.b4p) * meanfield.spot[1,...] - forces.b4 * meanfield.spot[0,...]

    meanfield.spot = meanfield.spot.at[0,...].set(new_spot_0)
    meanfield.spot = meanfield.spot.at[1,...].set(new_spot_1)

    # Step 11: calculate divergence of aq in divaq
    for iq in range(2):
        meanfield.divaq = meanfield.divaq.at[iq,...].set(
            jnp.einsum('ij,jkl->ikl', grids.der1x, meanfield.aq[iq,0,...])
        )
        meanfield.divaq = meanfield.divaq.at[iq,...].add(
            jnp.einsum('jl,ilk->ijk', grids.der1y, meanfield.aq[iq,1,...])
        )
        meanfield.divaq = meanfield.divaq.at[iq,...].add(
            jnp.einsum('kl,ijl->ijk', grids.der1z, meanfield.aq[iq,2,...])
        )

    # Step 12: calculate the gradient of the effective mass in dbmass
    for iq in range(2):
        meanfield.dbmass = meanfield.dbmass.at[iq,0,...].set(
            jnp.einsum('ij,jkl->ikl', grids.der1x, meanfield.bmass[iq,...])
        )
        meanfield.dbmass = meanfield.dbmass.at[iq,1,...].add(
            jnp.einsum('jl,ilk->ijk', grids.der1y, meanfield.bmass[iq,...])
        )
        meanfield.dbmass = meanfield.dbmass.at[iq,2,...].add(
            jnp.einsum('kl,ijl->ijk', grids.der1z, meanfield.bmass[iq,...])
        )

    # Step 13: calculate the pairing potential, store in v_pair, and add pairing part to upot
    for iq in range(2):
        ic = 1 if iq == 0 else 0
        if iq == 2:
            v0act, v0other, = forces.v0prot, forces.v0neut
        else:
            v0act, v0other = forces.v0neut, forces.v0prot

        if forces.ipair == 6:
            raise ValueError(f"{forces.ipair=}")
        else:
            meanfield.v_pair = meanfield.v_pair.at[iq,...].set(
                v0act * densities.chi[iq,...]
            )
            meanfield.ecorrp = 0.0

    if static.outertype != 'N':
        raise ValueError(f"{static.outertype=}")

    return meanfield, coulomb
