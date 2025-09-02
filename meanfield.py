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

@jax.jit
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

    workden = jnp.zeros_like(densities.rho)
    # The shape for workvec in Python (2, 3, nx, ny, nz) matches the logic for aq, spot etc.
    workvec_shape = (2, 3, grids.nx, grids.ny, grids.nz)
    workvec = jnp.zeros(workvec_shape, dtype=jnp.float64)

    upot_calc = jnp.zeros_like(meanfield.upot)

    epsilon = 1.0e-25

    # ==============================================================================
    # Step 1: 3-body contribution to upot 
    # ==============================================================================
    rho_tot = densities.rho[0] + densities.rho[1]
    rho_tot_pow = rho_tot ** forces.power

    for iq in range(2):
        ic = 1 - iq  # 0-based indexing for the 'opposite' isospin
         
        three_body_term = (
            (forces.b3 * (forces.power + 2.0) / 3.0 - 2.0 * forces.b3p / 3.0) * densities.rho[iq] +
            forces.b3 * (forces.power + 2.0) / 3.0 * densities.rho[ic] -
            (forces.b3p * forces.power / 3.0) * (densities.rho[0]**2 + densities.rho[1]**2) /
            (rho_tot + epsilon)
        )
        upot_calc = upot_calc.at[iq].set(rho_tot_pow * three_body_term)

    # ==============================================================================
    # Step 2: Add divergence of spin-orbit current to upot 
    # ==============================================================================
    for iq in range(2):
        # Calculate divergence of sodens: div_j = d/dx(sodens_x) + d/dy(sodens_y) + d/dz(sodens_z)
        div_sodens_x = jnp.einsum('ij,jkl->ikl', grids.der1x, densities.sodens[iq, 0])
        div_sodens_y = jnp.einsum('jl,ilk->ijk', grids.der1y, densities.sodens[iq, 1])
        div_sodens_z = jnp.einsum('kl,ijl->ijk', grids.der1z, densities.sodens[iq, 2])
        workden = workden.at[iq].set(div_sodens_x + div_sodens_y + div_sodens_z)

    for iq in range(2):
        ic = 1 - iq
        upot_calc = upot_calc.at[iq].add(
            -(forces.b4 + forces.b4p) * workden[iq] - forces.b4 * workden[ic]
        )

    # ==============================================================================
    # Step 3: Coulomb potential 
    # ==============================================================================
    if params.tcoul:
        # The poisson solver returns the coulomb potential, wcoul
        wcoul = poisson(grids, params, coulomb, densities.rho)
        coulomb.wcoul = coulomb.wcoul.at[...].set(wcoul)
        
        # Add to proton potential (iq=1)
        upot_calc = upot_calc.at[1].add(coulomb.wcoul)

        if forces.ex != 0:
            slater_term = -forces.slate * densities.rho[1] ** (1.0 / 3.0)
            upot_calc = upot_calc.at[1].add(slater_term)
    
    # ==============================================================================
    # Step 4: Remaining (standard) terms of upot 
    # ==============================================================================
    # First, calculate laplacian of densities and store in workden 
    for iq in range(2):
        lap_rho_x = jnp.einsum('ij,jkl->ikl', grids.der2x, densities.rho[iq])
        lap_rho_y = jnp.einsum('jl,ilk->ijk', grids.der2y, densities.rho[iq])
        lap_rho_z = jnp.einsum('kl,ijl->ijk', grids.der2z, densities.rho[iq])
        workden = workden.at[iq].set(lap_rho_x + lap_rho_y + lap_rho_z)

    # Now add the standard terms to upot
    for iq in range(2):
        ic = 1 - iq
        standard_terms = (
            (forces.b0 - forces.b0p) * densities.rho[iq] + forces.b0 * densities.rho[ic] +
            (forces.b1 - forces.b1p) * densities.tau[iq] + forces.b1 * densities.tau[ic] -
            (forces.b2 - forces.b2p) * workden[iq] - forces.b2 * workden[ic]
        )
        upot_calc = upot_calc.at[iq].add(standard_terms)

    # Assign final calculated upot to the meanfield object
    meanfield.upot = upot_calc

    # ==============================================================================
    # Step 5: Effective mass 
    # ==============================================================================
    for iq in range(2):
        ic = 1 - iq
        bmass_val = forces.h2m[iq] + (forces.b1 - forces.b1p) * densities.rho[iq] + forces.b1 * densities.rho[ic]
        meanfield.bmass = meanfield.bmass.at[iq].set(bmass_val)

    # ==============================================================================
    # Step 6: Calculate grad(rho) and wlspot 
    # ==============================================================================
    # First, calculate grad(rho) and store in workvec
    for iq in range(2):
        grad_rho_x = jnp.einsum('ij,jkl->ikl', grids.der1x, densities.rho[iq])
        grad_rho_y = jnp.einsum('jl,ilk->ijk', grids.der1y, densities.rho[iq])
        grad_rho_z = jnp.einsum('kl,ijl->ijk', grids.der1z, densities.rho[iq])
        workvec = workvec.at[iq, 0].set(grad_rho_x)
        workvec = workvec.at[iq, 1].set(grad_rho_y)
        workvec = workvec.at[iq, 2].set(grad_rho_z)

    # Now construct wlspot from grad(rho) 
    for iq in range(2):
        ic = 1 - iq
        wlspot_val = (forces.b4 + forces.b4p) * workvec[iq] + forces.b4 * workvec[ic]
        meanfield.wlspot = meanfield.wlspot.at[iq].set(wlspot_val)

    # ==============================================================================
    # Step 7: Calculate curl of spin density vector, store in workvec 
    # ==============================================================================
    for iq in range(2):
        # x-component of curl: d/dy(sdens_z) - d/dz(sdens_y)
        curl_x = (jnp.einsum('jl,ilk->ijk', grids.der1y, densities.sdens[iq, 2]) -
                  jnp.einsum('kl,ijl->ijk', grids.der1z, densities.sdens[iq, 1]))
        # y-component of curl: d/dz(sdens_x) - d/dx(sdens_z)
        curl_y = (jnp.einsum('kl,ijl->ijk', grids.der1z, densities.sdens[iq, 0]) -
                  jnp.einsum('ij,jkl->ikl', grids.der1x, densities.sdens[iq, 2]))
        # z-component of curl: d/dx(sdens_y) - d/dy(sdens_x)
        curl_z = (jnp.einsum('ij,jkl->ikl', grids.der1x, densities.sdens[iq, 1]) -
                  jnp.einsum('jl,ilk->ijk', grids.der1y, densities.sdens[iq, 0]))
        workvec = workvec.at[iq].set(jnp.stack([curl_x, curl_y, curl_z], axis=0))
        
    # ==============================================================================
    # Step 8: Calculate A_q vector 
    # ==============================================================================
    for iq in range(2):
        ic = 1 - iq
        aq_val = (-2.0 * (forces.b1 - forces.b1p) * densities.current[iq]
                  -2.0 * forces.b1 * densities.current[ic]
                  -(forces.b4 + forces.b4p) * workvec[iq] - forces.b4 * workvec[ic])
        meanfield.aq = meanfield.aq.at[iq].set(aq_val)

    # ==============================================================================
    # Step 9: Calculate curl of the current density, store in spot 
    # ==============================================================================
    for iq in range(2):
        # x-component of curl: d/dy(j_z) - d/dz(j_y)
        curl_x = (jnp.einsum('jl,ilk->ijk', grids.der1y, densities.current[iq, 2]) -
                  jnp.einsum('kl,ijl->ijk', grids.der1z, densities.current[iq, 1]))
        # y-component of curl: d/dz(j_x) - d/dx(j_z)
        curl_y = (jnp.einsum('kl,ijl->ijk', grids.der1z, densities.current[iq, 0]) -
                  jnp.einsum('ij,jkl->ikl', grids.der1x, densities.current[iq, 2]))
        # z-component of curl: d/dx(j_y) - d/dy(j_x)
        curl_z = (jnp.einsum('ij,jkl->ikl', grids.der1x, densities.current[iq, 1]) -
                  jnp.einsum('jl,ilk->ijk', grids.der1y, densities.current[iq, 0]))
        meanfield.spot = meanfield.spot.at[iq].set(jnp.stack([curl_x, curl_y, curl_z], axis=0))

    # ==============================================================================
    # Step 10: Combine isospin contributions for spot 
    # ==============================================================================
    # Save the result of step 9 before overwriting
    spot_temp = jnp.copy(meanfield.spot)
    for iq in range(2):
        ic = 1 - iq
        spot_combined = -(forces.b4 + forces.b4p) * spot_temp[iq] - forces.b4 * spot_temp[ic]
        meanfield.spot = meanfield.spot.at[iq].set(spot_combined)

    # ==============================================================================
    # Step 11: Calculate divergence of aq in divaq 
    # ==============================================================================
    for iq in range(2):
        div_aq_x = jnp.einsum('ij,jkl->ikl', grids.der1x, meanfield.aq[iq, 0])
        div_aq_y = jnp.einsum('jl,ilk->ijk', grids.der1y, meanfield.aq[iq, 1])
        div_aq_z = jnp.einsum('kl,ijl->ijk', grids.der1z, meanfield.aq[iq, 2])
        meanfield.divaq = meanfield.divaq.at[iq].set(div_aq_x + div_aq_y + div_aq_z)

    # ==============================================================================
    # Step 12: Calculate gradient of the effective mass in dbmass 
    # ==============================================================================
    for iq in range(2):
        grad_bmass_x = jnp.einsum('ij,jkl->ikl', grids.der1x, meanfield.bmass[iq])
        grad_bmass_y = jnp.einsum('jl,ilk->ijk', grids.der1y, meanfield.bmass[iq])
        grad_bmass_z = jnp.einsum('kl,ijl->ijk', grids.der1z, meanfield.bmass[iq])
        meanfield.dbmass = meanfield.dbmass.at[iq, 0].set(grad_bmass_x)
        meanfield.dbmass = meanfield.dbmass.at[iq, 1].set(grad_bmass_y)
        meanfield.dbmass = meanfield.dbmass.at[iq, 2].set(grad_bmass_z)

# ==============================================================================
    # Step 13: Calculate pairing potential and DDDI rearrangement term
    # ==============================================================================
    
    # This block handles the DDDI-specific rearrangement terms.
    # It must be executed before the final v_pair calculation.
    if forces.ipair == 6:
        # For DDDI, a rearrangement potential must be added to upot.
        # This potential is the same for neutrons and protons.
        rearrangement_potential = (forces.v0neut / forces.rho0pr) * (densities.chi[0]**2) + \
                                (forces.v0prot / forces.rho0pr) * (densities.chi[1]**2)
        
        # Add this potential to the upot calculated from previous steps
        upot_calc = upot_calc.at[0].add(rearrangement_potential)
        upot_calc = upot_calc.at[1].add(rearrangement_potential)
        
        # This scalar energy is subtracted from the total energy later.
        rho_total = densities.rho[0] + densities.rho[1]
        meanfield.ecorrp = -jnp.sum(rho_total * rearrangement_potential) * grids.wxyz / 2.0
    else:
        meanfield.ecorrp = 0.0

    # This block calculates the pairing potential (v_pair) for each isospin,
    # using the correct formula for either VDI or DDDI.
    for iq in range(2):
        v0act = forces.v0prot if iq == 1 else forces.v0neut
        
        if forces.ipair == 6: # DDDI Pairing
            # The pairing potential is scaled by the total density
            rho_total = densities.rho[0] + densities.rho[1]
            density_factor = 1.0 - (rho_total / forces.rho0pr)
            v_pair_iq = v0act * densities.chi[iq] * density_factor
            meanfield.v_pair = meanfield.v_pair.at[iq].set(v_pair_iq)

        else: # VDI Pairing (ipair=5)
            # This is the simple volume-delta interaction
            meanfield.v_pair = meanfield.v_pair.at[iq].set(v0act * densities.chi[iq])

    # After the loop, the final upot can be assigned to the meanfield object
    meanfield.upot = upot_calc

    # External guiding potential logic (not implemented, as in original Python file)
    if static.outertype != 'N':
        raise NotImplementedError(f"outertype '{static.outertype}' not implemented.")

    return meanfield, coulomb