import jax
import jax.numpy as jnp
from typing import Tuple, Any
from levels import cdervx00, cdervx02, cdervy00, cdervy02, cdervz00, cdervz02

class HPSI01Implementations:
    
    @staticmethod
    def hpsi01_original(grids, meanfield, iq, weight, weightuv, pinn):
        """OLD HPSI01"""
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

        # Step 6: multiply weight and add outputs
        pout_mf = pout
        pout_del = jnp.multiply(pinn, meanfield.v_pair[iq,...])
        pout = weight * pout - weightuv * pout_del

        return pout, pout_mf, pout_del

    @staticmethod
    @jax.jit
    def hpsi01_optimized(grids, meanfield, iq, weight, weightuv, pinn):
        """NEW HPSI01"""
        sigis = jnp.array([0.5, -0.5])
        
        # Pre-compute common terms
        spot_iq = meanfield.spot[iq]
        wlspot_iq = meanfield.wlspot[iq]
        aq_iq = meanfield.aq[iq]
        bmass_iq = meanfield.bmass[iq]
        
        # Step 1: Non-derivative parts not involving spin
        pout = jnp.multiply(pinn, meanfield.upot[iq,...])
        
        # Step 2: Spin-current coupling with preserved ordering
        spin_coupling = jnp.stack([
            (spot_iq[0] - 1j * spot_iq[1]) * pinn[1] + spot_iq[2] * pinn[0],
            (spot_iq[0] + 1j * spot_iq[1]) * pinn[0] - spot_iq[2] * pinn[1]
        ])
        pout = pout + spin_coupling

        # Process x direction with preserved precision
        pswk_x, pswk2_x = cdervx02(grids.dx, pinn, bmass_iq)
        dx_terms = jnp.stack([
            -(1j * (0.5 * aq_iq[0] - sigis[0] * wlspot_iq[1])) * pswk_x[0] - 
            sigis[0] * wlspot_iq[2] * pswk_x[1] - pswk2_x[0],
            
            -(1j * (0.5 * aq_iq[0] - sigis[1] * wlspot_iq[1])) * pswk_x[1] - 
            sigis[1] * wlspot_iq[2] * pswk_x[0] - pswk2_x[1]
        ])
        pout = pout + dx_terms

        # X direction corrections
        pswk2_x = jnp.stack([
            (-0.5j) * (aq_iq[0] - wlspot_iq[1]) * pinn[0] - 
            0.5 * wlspot_iq[2] * pinn[1],
            
            (-0.5j) * (aq_iq[0] + wlspot_iq[1]) * pinn[1] + 
            0.5 * wlspot_iq[2] * pinn[0]
        ])
        pout = pout + cdervx00(grids.dx, pswk2_x)

        # Process y direction with preserved precision
        pswk_y, pswk2_y = cdervy02(grids.dy, pinn, bmass_iq)
        dy_terms = jnp.stack([
            -(1j * (0.5 * aq_iq[1] + sigis[0] * wlspot_iq[0])) * pswk_y[0] + 
            (1j * 0.5 * wlspot_iq[2]) * pswk_y[1] - pswk2_y[0],
            
            -(1j * (0.5 * aq_iq[1] + sigis[1] * wlspot_iq[0])) * pswk_y[1] + 
            (1j * 0.5 * wlspot_iq[2]) * pswk_y[0] - pswk2_y[1]
        ])
        pout = pout + dy_terms

        # Y direction corrections
        pswk2_y = jnp.stack([
            (-0.5j) * (aq_iq[1] + wlspot_iq[0]) * pinn[0] + 
            (0.5j) * wlspot_iq[2] * pinn[1],
            
            (-0.5j) * (aq_iq[1] - wlspot_iq[0]) * pinn[1] + 
            (0.5j) * wlspot_iq[2] * pinn[0]
        ])
        pout = pout + cdervy00(grids.dy, pswk2_y)

        # Process z direction with preserved precision
        pswk_z, pswk2_z = cdervz02(grids.dz, pinn, bmass_iq)
        dz_terms = jnp.stack([
            -(1j * 0.5 * aq_iq[2]) * pswk_z[0] + 
            (sigis[0] * wlspot_iq[0] - 1j * 0.5 * wlspot_iq[1]) * pswk_z[1] - 
            pswk2_z[0],
            
            -(1j * 0.5 * aq_iq[2]) * pswk_z[1] + 
            (sigis[1] * wlspot_iq[0] - 1j * 0.5 * wlspot_iq[1]) * pswk_z[0] - 
            pswk2_z[1]
        ])
        pout = pout + dz_terms

        # Z direction corrections
        pswk2_z = jnp.stack([
            (-0.5j) * aq_iq[2] * pinn[0] + 
            (0.5 * wlspot_iq[0] - 0.5j * wlspot_iq[1]) * pinn[1],
            
            (-0.5j) * aq_iq[2] * pinn[1] + 
            (-0.5 * wlspot_iq[0] - 0.5j * wlspot_iq[1]) * pinn[0]
        ])
        pout = pout + cdervz00(grids.dz, pswk2_z)

        # Final outputs
        pout_mf = pout
        pout_del = jnp.multiply(pinn, meanfield.v_pair[iq,...])
        pout = weight * pout - weightuv * pout_del

        return pout, pout_mf, pout_del