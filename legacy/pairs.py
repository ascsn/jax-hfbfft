import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq
from dataclasses import dataclass, field

from levels import Levels
from meanfield import Meanfield
from grids import Grids
from forces import Forces
from params import Params

@jax.tree_util.register_dataclass
@dataclass
class Pairs:
    eferm: jax.Array = field(default_factory=lambda: jnp.zeros(2, dtype=jnp.float64))
    epair: jax.Array = field(default_factory=lambda: jnp.zeros(2, dtype=jnp.float64))
    avdelt: jax.Array = field(default_factory=lambda: jnp.zeros(2, dtype=jnp.float64))
    avdeltv2: jax.Array = field(default_factory=lambda: jnp.zeros(2, dtype=jnp.float64))
    avg: jax.Array = field(default_factory=lambda: jnp.zeros(2, dtype=jnp.float64))
    # This flag helps replicate the logic of calling pairgap only once if needed.
    firstcall: bool = True


def _bcs_occupation(efermi_trial, sp_energy, deltaf, wstates):
    """
    Calculates the particle number for a given trial Fermi energy.
    """
    edif = sp_energy - efermi_trial
    equasi = jnp.sqrt(edif**2 + deltaf**2)
    
    # Add a small epsilon to the denominator to avoid division by zero
    equasi_safe = jnp.maximum(equasi, 1.0e-20)
    
    # BCS expression for occupation probability v_k^2
    wocc_trial = 0.5 * (1.0 - edif / equasi_safe)
    
    #Clip values to avoid exact 0 or 1, as in FORTRAN
    smal = 1.0e-10
    wocc_trial = jnp.clip(wocc_trial, smal, 1.0 - smal)
    
    #Sum the occupations to get the total particle number
    particle_number = jnp.sum(wocc_trial * wstates)
    
    return particle_number

def _find_eferm_scipy(target_particle_number, sp_energy, deltaf, wstates):
    def objective_func(eferm_trial):
        num = _bcs_occupation(
            eferm_trial, 
            np.asarray(sp_energy), 
            np.asarray(deltaf), 
            np.asarray(wstates)
        )
        return float(num) - target_particle_number

    # Use Brent's method to find the root. The search interval [-100, 100]
    #eferm_found = brentq(objective_func, a=-100.0, b=100.0, xtol=1e-14, rtol=1e-16)
    eferm_found = brentq(objective_func, a=-100.0, b=100.0, xtol=1e-14)

    return eferm_found

def _soft_cutoff(sp_energy, ecut, cutwid):
    return 1.0 / (1.0 + jnp.exp((sp_energy - ecut) / cutwid))

def _pairdn(iq: int, particle_number: float, levels: Levels, forces: Forces, pairs_data: Pairs):
    """
    Solves the pairing problem for a single isospin.
    """
    # Create a mask for the current isospin
    mask = (levels.isospin == iq)
    
    # Get the relevant subset of states for this isospin
    sp_energy_iq = levels.sp_energy[mask]
    deltaf_iq = levels.deltaf[mask]
    wstates_iq = levels.wstates[mask]
    
    # Determine the true Fermi energy by finding the root
    eferm_val = _find_eferm_scipy(particle_number, sp_energy_iq, deltaf_iq, wstates_iq)
    
    # --- Update soft cutoffs for pairwg and wstates if applicable ---
    pairwg_new = levels.pairwg
    wstates_new = levels.wstates
    
    # Create a variable for the wstates slice that will be used in calculations
    wstates_iq_for_calc = levels.wstates[mask]

    if forces.pair_cutoff[iq] > 0.0:
        ecut = eferm_val + forces.pair_cutoff[iq]
        width = forces.softcut_range * forces.pair_cutoff[iq]
        softcut = _soft_cutoff(sp_energy_iq, ecut, width)
        pairwg_new = pairwg_new.at[mask].set(softcut)
        
    if forces.state_cutoff[iq] > 0.0:
        ecut = eferm_val + forces.state_cutoff[iq]
        width = forces.softcut_range * forces.state_cutoff[iq]
        softcut = _soft_cutoff(sp_energy_iq, ecut, width)
        wstates_new = wstates_new.at[mask].set(softcut)
        wstates_iq_for_calc = softcut  # Use the newly calculated cutoff values

    # --- Final calculation of occupations and properties with correct Fermi level ---
    edif = sp_energy_iq - eferm_val
    equasi = jnp.sqrt(edif**2 + deltaf_iq**2)
    equasi_safe = jnp.maximum(equasi, 1.0e-20)
    
    v2 = 0.5 * (1.0 - edif / equasi_safe) # v_k^2
    
    #Calculate uv = sqrt(v^2 * (1 - v^2))
    uv = jnp.sqrt(jnp.maximum(v2 - v2**2, 1.0e-6)) # smallp = 1e-6 in FORTRAN

    wocc_final = jnp.clip(v2, 1e-10, 1.0 - 1e-10)
    wguv_final = uv
    
    # --- Calculate summary statistics for printing ---
    # Use the potentially updated wstates slice
    vol = 0.5 * wguv_final * wstates_iq_for_calc
    sumuv = jnp.sum(vol)
    sumduv = jnp.sum(vol * deltaf_iq)
    sumv2 = jnp.sum(wocc_final * wstates_iq_for_calc)
    sumdv2 = jnp.sum(deltaf_iq * wocc_final * wstates_iq_for_calc)

    sumuv_safe = jnp.maximum(sumuv, 1.0e-20)
    sumv2_safe = jnp.maximum(sumv2, 1.0e-20)
    
    pairs_data.eferm = pairs_data.eferm.at[iq].set(eferm_val)
    pairs_data.epair = pairs_data.epair.at[iq].set(sumduv)
    pairs_data.avdelt = pairs_data.avdelt.at[iq].set(sumduv / sumuv_safe)
    pairs_data.avdeltv2 = pairs_data.avdeltv2.at[iq].set(sumdv2 / sumv2_safe) # Custom definition based on avdelt
    pairs_data.avg = pairs_data.avg.at[iq].set(sumduv / sumuv_safe**2) 

    return wocc_final, wguv_final, pairwg_new, wstates_new

def _pairgap(levels: Levels, meanfield: Meanfield, params: Params, grids: Grids, firstcall: bool):
    #For initial iterations, use a constant gap
    if params.iteration <= 10:
        deltaf = 11.2 / jnp.sqrt(levels.mass_number) * jnp.ones_like(levels.deltaf)
        return deltaf

    # If this is not the first call, deltaf is assumed to be correct from diagstep
    if not firstcall:
        return levels.deltaf

    # In general, calculate from the expectation value of the pairing field
    # Sum over spin components s: |phi_s|^2 = |phi_up|^2 + |phi_down|^2
    psi_sq = jnp.real(levels.psi * jnp.conjugate(levels.psi)) # |psi|^2 for all components
    density_per_state = jnp.sum(psi_sq, axis=1) # Sum over spin -> shape (nst, nx, ny, nz)

    # Select the correct v_pair for each state based on its isospin
    v_pair_n = meanfield.v_pair[0][None, ...] # Add axis for broadcasting
    v_pair_p = meanfield.v_pair[1][None, ...]
    v_pair_for_states = jnp.where(levels.isospin[:, None, None, None] == 0, v_pair_n, v_pair_p)
                        
    #Calculate expectation value: integral(V_pair * |phi|^2) * pairwg
    integrand = v_pair_for_states * density_per_state
    deltaf = jnp.sum(integrand, axis=(1, 2, 3)) * grids.wxyz * levels.pairwg
    
    return deltaf

# In pairs.py

def pair(levels: Levels, meanfield: Meanfield, forces: Forces, params: Params, grids: Grids, pairs_data: Pairs):
    # 1. Calculate pairing gaps for all states.
    # This direct attribute assignment is the required style.
    levels.deltaf = _pairgap(levels, meanfield, params, grids, pairs_data.firstcall)

    # The 'firstcall' flag is a simple boolean and can be mutated directly.
    pairs_data.firstcall = False

    # --- Neutron Calculation (iq=0) ---
    iq_n = 0
    particle_number_n = float(levels.nneut)
    
    # Run the pairing routine for neutrons. It will use the newly calculated levels.deltaf.
    wocc_n, wguv_n, pairwg_n_full, wstates_n_full = _pairdn(
        iq_n, particle_number_n, levels, forces, pairs_data
    )

    # Apply the results from the neutron calculation to the levels object.
    # These mutations ensure the subsequent proton calculation uses the correct, updated state.
    mask_n = (levels.isospin == iq_n)
    levels.wocc = levels.wocc.at[mask_n].set(wocc_n)
    levels.wguv = levels.wguv.at[mask_n].set(wguv_n)
    
    # The _pairdn function returns the *entire* updated array for wstates and pairwg.
    # We must assign the full array back to the levels object to capture the changes.
    if forces.pair_cutoff[iq_n] > 0.0:
        levels.pairwg = pairwg_n_full
    if forces.state_cutoff[iq_n] > 0.0:
        levels.wstates = wstates_n_full

    # --- Proton Calculation (iq=1) ---
    # Now, the 'levels' object contains the correct wocc, wguv, wstates, and pairwg
    # from the neutron step.
    iq_p = 1
    particle_number_p = float(levels.nprot)
    
    # Run the pairing routine for protons using the updated levels object.
    wocc_p, wguv_p, pairwg_p_full, wstates_p_full = _pairdn(
        iq_p, particle_number_p, levels, forces, pairs_data
    )

    # Apply the results from the proton calculation.
    mask_p = (levels.isospin == iq_p)
    levels.wocc = levels.wocc.at[mask_p].set(wocc_p)
    levels.wguv = levels.wguv.at[mask_p].set(wguv_p)
    if forces.pair_cutoff[iq_p] > 0.0:
        levels.pairwg = pairwg_p_full
    if forces.state_cutoff[iq_p] > 0.0:
        levels.wstates = wstates_p_full

    # Print summary information (matches FORTRAN output style).
    print("\n--- Pairing Information ---")
    print('          e_ferm      e_pair     <uv delta>   <v2 delta>    aver_force ')
    for iq in range(2):
        isospin_label = "Protons" if iq == 1 else "Neutrons"
        print(f"{isospin_label:<9s} {pairs_data.eferm[iq]:11.4f} {pairs_data.epair[iq]:11.4f} "
              f"{pairs_data.avdelt[iq]:11.4f} {pairs_data.avdeltv2[iq]:11.4f} {pairs_data.avg[iq]:11.4f}")

    # The 'levels' and 'pairs_data' objects have been modified and are returned.
    return levels, pairs_data
