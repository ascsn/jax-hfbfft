import jax
import jax.numpy as jnp
from dataclasses import dataclass, field
# Removed: from dataclasses import replace

from levels import Levels
from meanfield import Meanfield
from grids import Grids
from forces import Forces
from params import Params
from jax.tree_util import register_dataclass

@register_dataclass
@dataclass
class Pairs:
    eferm: jax.Array = field(default_factory=lambda: jnp.zeros(2, dtype=jnp.float64))
    epair: jax.Array = field(default_factory=lambda: jnp.zeros(2, dtype=jnp.float64))
    avdelt: jax.Array = field(default_factory=lambda: jnp.zeros(2, dtype=jnp.float64))
    avdeltv2: jax.Array = field(default_factory=lambda: jnp.zeros(2, dtype=jnp.float64))
    avg: jax.Array = field(default_factory=lambda: jnp.zeros(2, dtype=jnp.float64))
    firstcall: jax.Array = field(default_factory=lambda: jnp.array(True, dtype=bool))

def _bcs_occupation(efermi_trial, sp_energy, deltaf, wstates):
    edif = sp_energy - efermi_trial
    equasi = jnp.sqrt(edif**2 + deltaf**2)
    equasi_safe = jnp.maximum(equasi, 1.0e-20)
    wocc_trial = 0.5 * (1.0 - edif / equasi_safe)
    smal = 1.0e-10
    wocc_trial = jnp.clip(wocc_trial, smal, 1.0 - smal)
    particle_number = jnp.sum(wocc_trial * wstates)
    return particle_number

def _find_eferm_jax(target_particle_number, sp_energy, deltaf, wstates):
    low = -100.0
    high = 100.0
    init_state = (0, low, high)

    def cond_fun(state):
        i, l, h = state
        return (i < 100) & ((h - l) > 1e-14)

    def body_fun(state):
        i, l, h = state
        mid = (l + h) / 2.0
        num = _bcs_occupation(mid, sp_energy, deltaf, wstates)
        diff = num - target_particle_number
        new_h = jnp.where(diff > 0, mid, h)
        new_l = jnp.where(diff <= 0, mid, l)
        return (i + 1, new_l, new_h)

    final_state = jax.lax.while_loop(cond_fun, body_fun, init_state)
    _, final_l, final_h = final_state
    return (final_l + final_h) / 2.0

def _soft_cutoff(sp_energy, ecut, cutwid):
    return 1.0 / (1.0 + jnp.exp((sp_energy - ecut) / cutwid))

def _pairdn(iq: int, particle_number: float, levels: Levels, forces: Forces, pairs_data: Pairs):
    mask = (levels.isospin == iq)
    wstates_masked = levels.wstates * mask 
    
    eferm_val = _find_eferm_jax(particle_number, levels.sp_energy, levels.deltaf, wstates_masked)
    
    # --- Branchless Cutoff Updates ---
    ecut_pair = eferm_val + forces.pair_cutoff[iq]
    width_pair = forces.softcut_range * forces.pair_cutoff[iq]
    softcut_pair = _soft_cutoff(levels.sp_energy, ecut_pair, width_pair)
    
    apply_pair_cut = (forces.pair_cutoff[iq] > 0.0)
    pairwg_new = jnp.where(apply_pair_cut & mask, softcut_pair, levels.pairwg)

    ecut_state = eferm_val + forces.state_cutoff[iq]
    width_state = forces.softcut_range * forces.state_cutoff[iq]
    softcut_state = _soft_cutoff(levels.sp_energy, ecut_state, width_state)
    
    apply_state_cut = (forces.state_cutoff[iq] > 0.0)
    wstates_new = jnp.where(apply_state_cut & mask, softcut_state, levels.wstates)
    
    wstates_for_calc = jnp.where(mask, wstates_new, 0.0)

    # --- Occupation Calculation ---
    edif = levels.sp_energy - eferm_val
    equasi = jnp.sqrt(edif**2 + levels.deltaf**2)
    equasi_safe = jnp.maximum(equasi, 1.0e-20)
    
    v2 = 0.5 * (1.0 - edif / equasi_safe)
    uv = jnp.sqrt(jnp.maximum(v2 - v2**2, 1.0e-6))
    
    wocc_calc = jnp.clip(v2, 1e-10, 1.0 - 1e-10)
    wguv_calc = uv
    
    wocc_final = jnp.where(mask, wocc_calc, levels.wocc)
    wguv_final = jnp.where(mask, wguv_calc, levels.wguv)
    
    # --- Statistics ---
    vol = 0.5 * wguv_final * wstates_for_calc
    sumuv = jnp.sum(vol)
    sumduv = jnp.sum(vol * levels.deltaf)
    sumv2 = jnp.sum(wocc_final * wstates_for_calc)
    sumdv2 = jnp.sum(levels.deltaf * wocc_final * wstates_for_calc)
    
    sumuv_safe = jnp.maximum(sumuv, 1.0e-20)
    sumv2_safe = jnp.maximum(sumv2, 1.0e-20)
    
    new_eferm = pairs_data.eferm.at[iq].set(eferm_val)
    new_epair = pairs_data.epair.at[iq].set(sumduv)
    new_avdelt = pairs_data.avdelt.at[iq].set(sumduv / sumuv_safe)
    new_avdeltv2 = pairs_data.avdeltv2.at[iq].set(sumdv2 / sumv2_safe)
    new_avg = pairs_data.avg.at[iq].set(sumduv / sumuv_safe**2)

    # Manual re-instantiation of Pairs (Since we know all fields)
    pairs_data_new = Pairs(
        eferm=new_eferm, 
        epair=new_epair, 
        avdelt=new_avdelt,
        avdeltv2=new_avdeltv2, 
        avg=new_avg, 
        firstcall=pairs_data.firstcall
    )
    
    return wocc_final, wguv_final, pairwg_new, wstates_new, pairs_data_new

def _calc_gap_field(levels, meanfield, grids):
    psi_sq = jnp.real(levels.psi * jnp.conjugate(levels.psi)) 
    density_per_state = jnp.sum(psi_sq, axis=1) 
    
    v_pair_n = meanfield.v_pair[0][None, ...]
    v_pair_p = meanfield.v_pair[1][None, ...]
    v_pair_for_states = jnp.where(levels.isospin[:, None, None, None] == 0, v_pair_n, v_pair_p)
    
    integrand = v_pair_for_states * density_per_state
    deltaf = jnp.sum(integrand, axis=(1, 2, 3)) * grids.wxyz * levels.pairwg
    return deltaf

def _pairgap(levels: Levels, meanfield: Meanfield, params: Params, grids: Grids, firstcall: jax.Array):
    def initial_gap(_):
        return 11.2 / jnp.sqrt(levels.mass_number) * jnp.ones_like(levels.deltaf)

    def standard_logic(_):
        return jax.lax.cond(
            jnp.logical_not(firstcall),
            lambda: levels.deltaf,
            lambda: _calc_gap_field(levels, meanfield, grids)
        )

    deltaf = jax.lax.cond(
        params.iteration <= 10,
        initial_gap,
        standard_logic,
        operand=None
    )
    return deltaf

@jax.jit
def pair(levels: Levels, meanfield: Meanfield, forces: Forces, params: Params, grids: Grids, pairs_data: Pairs):
    # 1. Update Delta F
    new_deltaf = _pairgap(levels, meanfield, params, grids, pairs_data.firstcall)
    
    # Update levels: Unpack current vars, update specific key, pass to constructor
    levels_kwargs = vars(levels).copy()
    levels_kwargs['deltaf'] = new_deltaf
    levels = Levels(**levels_kwargs)
    
    # Update firstcall manually
    pairs_kwargs = vars(pairs_data).copy()
    pairs_kwargs['firstcall'] = jnp.array(False, dtype=bool)
    pairs_data = Pairs(**pairs_kwargs)

    # --- Neutron Calculation ---
    wocc_n, wguv_n, pairwg_n, wstates_n, pairs_data = _pairdn(
        0, float(levels.nneut), levels, forces, pairs_data
    )
    
    # Update levels for Neutrons
    levels_kwargs = vars(levels).copy()
    levels_kwargs.update({
        'wocc': wocc_n,
        'wguv': wguv_n,
        'pairwg': pairwg_n,
        'wstates': wstates_n
    })
    levels = Levels(**levels_kwargs)

    # --- Proton Calculation ---
    wocc_p, wguv_p, pairwg_p, wstates_p, pairs_data = _pairdn(
        1, float(levels.nprot), levels, forces, pairs_data
    )

    # Update levels for Protons
    levels_kwargs = vars(levels).copy()
    levels_kwargs.update({
        'wocc': wocc_p,
        'wguv': wguv_p,
        'pairwg': pairwg_p,
        'wstates': wstates_p
    })
    levels = Levels(**levels_kwargs)

    return levels, pairs_data