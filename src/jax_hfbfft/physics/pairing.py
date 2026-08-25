"""
Pairing calculations for HFB.

This module implements BCS pairing with:
- Volume-delta interaction (VDI)
- Density-dependent delta interaction (DDDI)
- Fermi energy search via Brent's method
- Soft cutoff functions for pairing windows
"""

import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Tuple, Dict, Any

from jax_hfbfft.jax_config import get_dtypes
from jax_hfbfft.core.force import Force


@jax.tree_util.register_dataclass
@dataclass
class Pairing:
    """
    Container for pairing results.
    
    Attributes:
        eferm: Fermi energy per isospin (2,)
        epair: Pairing energy per isospin (2,)
        avdelt: Average pairing gap weighted by uv (2,)
        avdeltv2: Average pairing gap weighted by v^2 (2,)
        avg: Average pairing force (2,)
    """
    eferm: jax.Array    # Fermi energies
    epair: jax.Array    # Pairing energies
    avdelt: jax.Array   # <uv*Delta>/<uv>
    avdeltv2: jax.Array # <v^2*Delta>/<v^2>
    avg: jax.Array      # Average force
    
    @classmethod
    def zeros(cls) -> "Pairing":
        """Create zero-initialized pairing data."""
        dtypes = get_dtypes()
        return cls(
            eferm=jnp.zeros(2, dtype=dtypes.float),
            epair=jnp.zeros(2, dtype=dtypes.float),
            avdelt=jnp.zeros(2, dtype=dtypes.float),
            avdeltv2=jnp.zeros(2, dtype=dtypes.float),
            avg=jnp.zeros(2, dtype=dtypes.float),
        )


@jax.jit
def find_fermi_energy_jax(
    target_n: float,
    sp_energy: jax.Array,
    deltaf: jax.Array,
    wstates: jax.Array,
    emin: float = -200.0,
    emax: float = 200.0,
    max_iter: int = 40,
) -> float:
    """
    Find Fermi energy using JAX-native bisection search.
    
    This is much faster than scipy's brentq as it avoids CPU-GPU synchronization.
    """
    def body_fun(state):
        low, high, i = state
        mid = (low + high) / 2.0
        n = bcs_occupation(mid, sp_energy, deltaf, wstates)
        low = jnp.where(n < target_n, mid, low)
        high = jnp.where(n >= target_n, mid, high)
        return low, high, i + 1

    # Initial brackets - check if expansion is needed
    f_low = bcs_occupation(emin, sp_energy, deltaf, wstates) - target_n
    f_high = bcs_occupation(emax, sp_energy, deltaf, wstates) - target_n
    
    # Simple bracket expansion
    emin = jnp.where(f_low > 0, emin - 100.0, emin)
    emax = jnp.where(f_high < 0, emax + 100.0, emax)

    # Bisection loop
    low, high, _ = jax.lax.while_loop(
        lambda s: s[2] < max_iter,
        body_fun,
        (emin, emax, 0)
    )
    
    return (low + high) / 2.0


def bcs_occupation(
    eferm: jax.Array,
    sp_energy: jax.Array,
    deltaf: jax.Array,
    wstates: jax.Array,
) -> jax.Array:
    """
    Calculate particle number for given Fermi energy.
    
    BCS occupation: v^2 = 0.5 * (1 - (e - ef) / sqrt((e - ef)^2 + Delta^2))
    """
    edif = sp_energy - eferm
    equasi = jnp.sqrt(edif**2 + deltaf**2)
    equasi_safe = jnp.maximum(equasi, 1.0e-20)
    
    # BCS v^2
    v2 = 0.5 * (1.0 - edif / equasi_safe)
    v2 = jnp.clip(v2, 1.0e-10, 1.0 - 1.0e-10)
    
    return jnp.sum(v2 * wstates)


def find_fermi_energy(
    target_n: float,
    sp_energy: jax.Array,
    deltaf: jax.Array,
    wstates: jax.Array,
    emin: float = -200.0,
    emax: float = 200.0,
) -> float:
    """
    Find Fermi energy that gives target particle number.
    
    Args:
        target_n: Target particle number (N or Z)
        sp_energy: Single-particle energies
        deltaf: Pairing gaps
        wstates: State degeneracy weights
        emin, emax: Search bounds
        
    Returns:
        Fermi energy
    """
    return find_fermi_energy_jax(target_n, sp_energy, deltaf, wstates, emin, emax)


def soft_cutoff(energy: jax.Array, cutoff: float, width: float) -> jax.Array:
    """
    Fermi-function soft cutoff.
    
    f(e) = 1 / (1 + exp((e - cutoff) / width))
    
    Args:
        energy: Single-particle energies
        cutoff: Cutoff energy
        width: Cutoff width
        
    Returns:
        Cutoff factors in [0, 1]
    """
    return 1.0 / (1.0 + jnp.exp((energy - cutoff) / width))


def compute_bcs_occupations(
    sp_energy: jax.Array,
    deltaf: jax.Array,
    eferm: float,
) -> Tuple[jax.Array, jax.Array]:
    """
    Compute BCS occupation factors v^2 and uv.
    
    Args:
        sp_energy: Single-particle energies
        deltaf: Pairing gaps
        eferm: Fermi energy
        
    Returns:
        (v2, uv) - occupation and anomalous occupation
    """
    edif = sp_energy - eferm
    equasi = jnp.sqrt(edif**2 + deltaf**2)
    equasi_safe = jnp.maximum(equasi, 1.0e-20)
    
    v2 = 0.5 * (1.0 - edif / equasi_safe)
    v2 = jnp.clip(v2, 1.0e-10, 1.0 - 1.0e-10)
    
    # uv = sqrt(v^2 * (1 - v^2))
    uv = jnp.sqrt(jnp.maximum(v2 * (1.0 - v2), 1.0e-10))
    
    return v2, uv


def solve_pairing_isospin(
    iq: int,
    particle_number: float,
    sp_energy: jax.Array,
    deltaf: jax.Array,
    wstates: jax.Array,
    pairwg: jax.Array,
    pair_cutoff: float = 0.0,
    state_cutoff: float = 0.0,
    softcut_range: float = 0.1,
):
    smallp = 1.0e-6

    # FORTRAN convention (pairs.f90): pairwg is applied to the gap exactly once,
    # in pairgap/diagstep. The BCS solve does NOT reapply it — see pairs.f90:223,
    # "there is already pairwg in deltaf and chi, so no pairwg here." The incoming
    # deltaf already carries pairwg, so use it as-is.
    deltaf_eff = deltaf

    # Find Fermi energy using effective gaps (pairwg baked in)
    eferm = find_fermi_energy(
        particle_number, sp_energy, deltaf_eff, wstates,
        emin=-500.0, emax=500.0,
    )

    # Update cutoffs
    pairwg = jax.lax.cond(
        pair_cutoff > 0.0,
        lambda _: soft_cutoff(
            sp_energy, 
            eferm + pair_cutoff, 
            jnp.maximum(softcut_range * pair_cutoff, 1e-6)
        ),
        lambda _: pairwg,  # The fallback value if pair_cutoff <= 0.0
        operand=None
    )

    # 2. Update wstates
    wstates = jax.lax.cond(
        state_cutoff > 0.0,
        lambda _: soft_cutoff(
            sp_energy, 
            eferm + state_cutoff, 
            jnp.maximum(softcut_range * state_cutoff, 1e-6)
        ),
        lambda _: wstates,  # The fallback value if state_cutoff <= 0.0
        operand=None
    )

    # Final occupations with effective gaps
    edif = sp_energy - eferm
    equasi = jnp.sqrt(edif**2 + deltaf_eff**2)
    equasi_safe = jnp.maximum(equasi, 1.0e-20)

    v2 = 0.5 * (1.0 - edif / equasi_safe)
    v2 = jnp.clip(v2, smallp, 1.0 - smallp)

    # wguv = sqrt(v2*(1-v2)), no pairwg here (already in deltaf_eff)
    wguv = jnp.sqrt(jnp.maximum(v2 * (1.0 - v2), smallp))

    # Statistics — vol uses wstates only, NOT pairwg (FORTRAN comment)
    vol = 0.5 * wguv * wstates
    sumuv = jnp.maximum(jnp.sum(vol), 1.0e-20)
    sumduv = jnp.sum(vol * deltaf_eff)   # deltaf_eff consistent with gaps
    sumv2 = jnp.sum(v2 * wstates)
    sumdv2 = jnp.sum(deltaf_eff * v2 * wstates)

    stats = {
        'eferm':   eferm,
        'epair':   sumduv,
        'avdelt':  sumduv / sumuv,
        'avdeltv2': sumdv2 / jnp.maximum(sumv2, 1.0e-20),
        'avg':     sumduv / sumuv**2,
    }
    return eferm, v2, wguv, pairwg, wstates, stats

def compute_pairing_gaps(
    psi: jax.Array,
    v_pair: jax.Array,
    isospin: jax.Array,
    pairwg: jax.Array,
    wxyz: float,
    iteration: int,
    mass_number: int,
) -> jax.Array:
    """
    Compute pairing gaps from wavefunctions and pairing field.
    
    For early iterations, uses a simple estimate. Otherwise computes:
    Delta_i = pairwg_i * integral(v_pair(r) * |psi_i(r)|^2)
    
    Args:
        psi: Wavefunctions (nstates, 2, nx, ny, nz)
        v_pair: Pairing potential (2, nx, ny, nz)
        isospin: Isospin indices (nstates,)
        pairwg: Pairing cutoff weights (nstates,)
        wxyz: Integration weight
        iteration: Current iteration number
        mass_number: Total mass number
        
    Returns:
        Pairing gaps (nstates,)
    """
    nstates = psi.shape[0]
    dtypes = get_dtypes()
    
    # For early iterations, use constant gap estimate
    gap_estimate = 11.2 / jnp.sqrt(jnp.maximum(1.0, mass_number))
    initial_gaps = gap_estimate * jnp.ones(nstates, dtype=dtypes.float)
    
    # Compute |psi|^2 summed over spin
    psi_sq = jnp.real(psi * jnp.conjugate(psi))
    density = jnp.sum(psi_sq, axis=1)  # (nstates, nx, ny, nz)
    
    # Compute gaps using vectorized operations instead of vmap for speed
    # Select v_pair for each state (nstates, nx, ny, nz)
    # isospin is (nstates,)
    v_pair_states = jnp.where(isospin[:, None, None, None] == 0, v_pair[0], v_pair[1])
    
    # Gap_i = wxyz * sum_r (v_pair(r) * density_i(r))
    calculated_gaps = wxyz * jnp.sum(v_pair_states * density, axis=(1, 2, 3))
    
    calculated_gaps = calculated_gaps * pairwg
    
    # Switch between initial estimate and calculated gaps
    return jnp.where(iteration <= 10, initial_gaps, calculated_gaps)


def solve_pairing(
    sp_energy: jax.Array,
    deltaf: jax.Array,
    wstates: jax.Array,
    pairwg: jax.Array,
    isospin: jax.Array,
    nstates_n: int,
    nstates_p: int,
    target_n: int,
    target_p: int,
    force: Force,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, Pairing]:
    """
    Solve BCS pairing for both isospins.
    
    Args:
        sp_energy: Single-particle energies (nstates,)
        deltaf: Pairing gaps (nstates,)
        wstates: State weights (nstates,)
        pairwg: Pairing cutoff weights (nstates,)
        isospin: Isospin indices (nstates,)
        nstates_n: Number of neutron states
        nstates_p: Number of proton states
        target_n: Target neutron number
        target_p: Target proton number
        force: Force parameters
        
    Returns:
        (wocc, wguv, pairwg_new, wstates_new, pairing)
    """
    dtypes = get_dtypes()
    nstates = sp_energy.shape[0]
    
    # Process each isospin using slices for JIT compatibility
    # This assumes neutrons come first in the state arrays
    sp_n = sp_energy[:nstates_n]
    delta_n = deltaf[:nstates_n]
    wst_n = wstates[:nstates_n]
    pwg_n = pairwg[:nstates_n]
    
    eferm_n, v2_n, uv_n, pwg_new_n, wst_new_n, stats_n = solve_pairing_isospin(
        0, float(target_n), sp_n, delta_n, wst_n, pwg_n,
        pair_cutoff=force.pair_cutoff[0],
        state_cutoff=force.state_cutoff[0],
        softcut_range=force.softcut_range,
    )
    
    sp_p = sp_energy[nstates_n:]
    delta_p = deltaf[nstates_n:]
    wst_p = wstates[nstates_n:]
    pwg_p = pairwg[nstates_n:]
    
    eferm_p, v2_p, uv_p, pwg_new_p, wst_new_p, stats_p = solve_pairing_isospin(
        1, float(target_p), sp_p, delta_p, wst_p, pwg_p,
        pair_cutoff=force.pair_cutoff[1],
        state_cutoff=force.state_cutoff[1],
        softcut_range=force.softcut_range,
    )
    
    # Reassemble arrays
    wocc = jnp.concatenate([v2_n, v2_p])
    wguv = jnp.concatenate([uv_n, uv_p])
    
    # For cutoff weights, conditionally update
    pairwg_new = jnp.concatenate([
        jnp.where(force.pair_cutoff[0] > 0.0, pwg_new_n, pwg_n),
        jnp.where(force.pair_cutoff[1] > 0.0, pwg_new_p, pwg_p)
    ])
    
    wstates_new = jnp.concatenate([
        jnp.where(force.state_cutoff[0] > 0.0, wst_new_n, wst_n),
        jnp.where(force.state_cutoff[1] > 0.0, wst_new_p, wst_p)
    ])
    
    # Create result container
    pairing = Pairing(
        eferm=jnp.array([stats_n['eferm'], stats_p['eferm']]),
        epair=jnp.array([stats_n['epair'], stats_p['epair']]),
        avdelt=jnp.array([stats_n['avdelt'], stats_p['avdelt']]),
        avdeltv2=jnp.array([stats_n['avdeltv2'], stats_p['avdeltv2']]),
        avg=jnp.array([stats_n['avg'], stats_p['avg']])
    )
    
    return wocc, wguv, pairwg_new, wstates_new, pairing
