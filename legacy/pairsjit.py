import jax
import jax.numpy as jnp
from jax import jit
from dataclasses import dataclass, field, replace

from levels import Levels
from meanfield import Meanfield
from grids import Grids
from forces import Forces
from params import Params

# Note: The original Pairs dataclass is already JIT-compatible
# thanks to `jax.tree_util.register_dataclass`. No changes are needed.
@jax.tree_util.register_dataclass
@dataclass
class Pairs:
    eferm: jax.Array = field(default_factory=lambda: jnp.zeros(2, dtype=jnp.float64))
    epair: jax.Array = field(default_factory=lambda: jnp.zeros(2, dtype=jnp.float64))
    avdelt: jax.Array = field(default_factory=lambda: jnp.zeros(2, dtype=jnp.float64))
    avdeltv2: jax.Array = field(default_factory=lambda: jnp.zeros(2, dtype=jnp.float64))
    avg: jax.Array = field(default_factory=lambda: jnp.zeros(2, dtype=jnp.float64))

# --- JAX-native Helper Functions ---

def _bcs_occupation(efermi_trial, sp_energy, deltaf, wstates, mask):
    """Calculates the particle number for a given trial Fermi energy."""
    edif = sp_energy - efermi_trial
    equasi = jnp.sqrt(edif**2 + deltaf**2)
    equasi_safe = jnp.maximum(equasi, 1.0e-20)
    wocc_trial = 0.5 * (1.0 - edif / equasi_safe)
    smal = 1.0e-10
    wocc_trial = jnp.clip(wocc_trial, smal, 1.0 - smal)
    # **FIXED**: Multiply by mask before summing to only count relevant particles
    return jnp.sum(wocc_trial * wstates * mask)

# **MODIFIED**: Added 'mask' argument to pass through
def _bisection_search_eferm(target_particle_number, sp_energy, deltaf, wstates, mask):
    """JAX-native bisection method to find the Fermi energy."""
    def objective_func(eferm_trial):
        # Pass the mask to the calculation
        return _bcs_occupation(eferm_trial, sp_energy, deltaf, wstates, mask) - target_particle_number

    def loop_cond(state):
        low, high, _, _ = state
        return (high - low) > 1e-14

    def loop_body(state):
        low, high, f_low, _ = state
        mid = 0.5 * (low + high)
        f_mid = objective_func(mid)
        new_low, new_high, new_f_low = jax.lax.cond(
            jnp.sign(f_mid) == jnp.sign(f_low),
            lambda: (mid, high, f_mid),
            lambda: (low, mid, f_low)
        )
        return new_low, new_high, new_f_low, f_mid

    low, high = -100.0, 100.0
    f_low, f_high = objective_func(low), objective_func(high)
    final_low, final_high, _, _ = jax.lax.while_loop(
        loop_cond, loop_body, (low, high, f_low, f_high)
    )
    return 0.5 * (final_low + final_high)


def _soft_cutoff(sp_energy, ecut, cutwid):
    """Calculates a soft cutoff function."""
    return 1.0 / (1.0 + jnp.exp((sp_energy - ecut) / cutwid))


def _pairgap(levels: Levels, meanfield: Meanfield, params: Params, grids: Grids):
    """Calculates the pairing gap deltaf for all states."""

    def constant_gap_case(operands):
        """Case for early iterations."""
        levels, *_ = operands
        return 11.2 / jnp.sqrt(levels.mass_number) * jnp.ones_like(levels.deltaf)

    def calculated_gap_case(operands):
        """Case for later iterations, calculating gap from fields."""
        levels, meanfield, grids = operands
        psi_sq = jnp.real(levels.psi * jnp.conjugate(levels.psi))
        density_per_state = jnp.sum(psi_sq, axis=1)
        v_pair_n = meanfield.v_pair[0][None, ...]
        v_pair_p = meanfield.v_pair[1][None, ...]
        v_pair_for_states = jnp.where(levels.isospin[:, None, None, None] == 0, v_pair_n, v_pair_p)
        integrand = v_pair_for_states * density_per_state
        return jnp.sum(integrand, axis=(1, 2, 3)) * grids.wxyz * levels.pairwg

    # Use lax.cond to handle control flow based on iteration number
    return jax.lax.cond(
        params.iteration <= 10,
        constant_gap_case,
        calculated_gap_case,
        (levels, meanfield, grids)
    )


# --- The Main JIT-Compiled Function ---

@jax.jit
def pair(levels: Levels, meanfield: Meanfield, forces: Forces, params: Params, grids: Grids, pairs_data: Pairs):
    deltaf_new = _pairgap(levels, meanfield, params, grids)
    levels = replace(levels, deltaf=deltaf_new)

    def loop_body(iq, state):
        levels, pairs_data = state
        particle_number = jnp.where(iq == 0, levels.nneut, levels.nprot)
        mask = (levels.isospin == iq)

        # **FIXED**: Pass full arrays and the mask to the solver, DON'T slice them.
        eferm_val = _bisection_search_eferm(
            particle_number, levels.sp_energy, levels.deltaf, levels.wstates, mask
        )

        def update_cutoffs(operands):
            # Operate on full arrays
            pairwg, wstates, sp_energy, eferm, pair_cut, state_cut, softcut_range = operands
            ecut_p = eferm + pair_cut
            width_p = softcut_range * pair_cut
            softcut_p = _soft_cutoff(sp_energy, ecut_p, width_p)
            # **FIXED**: Use jnp.where for conditional updates
            pairwg = jnp.where(mask, softcut_p, pairwg)
            
            ecut_s = eferm + state_cut
            width_s = softcut_range * state_cut
            softcut_s = _soft_cutoff(sp_energy, ecut_s, width_s)
            wstates = jnp.where(mask, softcut_s, wstates)
            return pairwg, wstates

        def no_update(operands):
            pairwg, wstates, *_ = operands
            return pairwg, wstates

        pairwg_new, wstates_new = jax.lax.cond(
            (forces.pair_cutoff[iq] > 0.0) | (forces.state_cutoff[iq] > 0.0),
            update_cutoffs,
            no_update,
            (levels.pairwg, levels.wstates, levels.sp_energy, eferm_val,
             forces.pair_cutoff[iq], forces.state_cutoff[iq], forces.softcut_range)
        )
        levels = replace(levels, pairwg=pairwg_new, wstates=wstates_new)
        
        # **FIXED**: Perform calculations on full-sized arrays
        edif = levels.sp_energy - eferm_val
        equasi = jnp.sqrt(edif**2 + levels.deltaf**2)
        equasi_safe = jnp.maximum(equasi, 1.0e-20)
        v2 = 0.5 * (1.0 - edif / equasi_safe)
        # **FIXED**: Clip v2 *before* calculating uv to prevent sqrt of a negative.
        wocc_final = jnp.clip(v2, 1e-10, 1.0 - 1e-10)
        # Now calculate uv from the guaranteed-safe wocc_final.
        # This is equivalent to sqrt(v2*(1-v2)) but safer.
        uv = jnp.sqrt(wocc_final - wocc_final**2)

        new_wocc = jnp.where(mask, wocc_final, levels.wocc)
        new_wguv = jnp.where(mask, uv, levels.wguv)
        levels = replace(levels, wocc=new_wocc, wguv=new_wguv)
        

        # **FIXED**: Use jnp.where to update only the relevant part of the arrays
        new_wocc = jnp.where(mask, wocc_final, levels.wocc)
        new_wguv = jnp.where(mask, uv, levels.wguv)
        levels = replace(levels, wocc=new_wocc, wguv=new_wguv)
        
        # **FIXED**: Use the mask in summations for summary statistics
        vol = 0.5 * levels.wguv * levels.wstates
        sumuv = jnp.sum(vol * mask)
        sumduv = jnp.sum(vol * levels.deltaf * mask)
        sumv2 = jnp.sum(levels.wocc * levels.wstates * mask)
        sumdv2 = jnp.sum(levels.wocc * levels.deltaf * levels.wstates * mask)
        
        sumuv_safe = jnp.maximum(sumuv, 1.0e-20)
        sumv2_safe = jnp.maximum(sumv2, 1.0e-20)
        
        pairs_data = replace(
            pairs_data,
            eferm=pairs_data.eferm.at[iq].set(eferm_val),
            epair=pairs_data.epair.at[iq].set(sumduv),
            avdelt=pairs_data.avdelt.at[iq].set(sumduv / sumuv_safe),
            avdeltv2=pairs_data.avdeltv2.at[iq].set(sumdv2 / sumv2_safe),
            avg=pairs_data.avg.at[iq].set(sumduv / sumuv_safe**2)
        )
        return levels, pairs_data

    initial_state = (levels, pairs_data)
    levels, pairs_data = jax.lax.fori_loop(0, 2, loop_body, initial_state)

    return levels, pairs_data
