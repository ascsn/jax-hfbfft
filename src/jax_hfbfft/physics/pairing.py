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
import numpy as np
from dataclasses import dataclass
from typing import Tuple
from scipy.optimize import brentq

from jax_hfbfft.jax_config import get_dtypes


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


def bcs_occupation(
    eferm: float,
    sp_energy: jax.Array,
    deltaf: jax.Array,
    wstates: jax.Array,
) -> float:
    """
    Calculate particle number for given Fermi energy.
    
    BCS occupation: v^2 = 0.5 * (1 - (e - ef) / sqrt((e - ef)^2 + Delta^2))
    
    Args:
        eferm: Trial Fermi energy
        sp_energy: Single-particle energies
        deltaf: Pairing gaps
        wstates: State degeneracy weights
        
    Returns:
        Total particle number
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
    
    Uses Brent's method for root finding.
    
    Args:
        target_n: Target particle number (N or Z)
        sp_energy: Single-particle energies
        deltaf: Pairing gaps
        wstates: State degeneracy weights
        emin, emax: Search bounds
        
    Returns:
        Fermi energy
    """
    # Convert to numpy for scipy
    sp_np = np.asarray(sp_energy)
    delta_np = np.asarray(deltaf)
    wst_np = np.asarray(wstates)
    
    def objective(ef):
        n = bcs_occupation(ef, sp_np, delta_np, wst_np)
        return float(n) - target_n
    
    # Adaptively find bounds where function changes sign
    f_min = objective(emin)
    f_max = objective(emax)
    
    # If bounds don't bracket root, expand them
    expansion_steps = 0
    while f_min * f_max > 0 and expansion_steps < 10:
        emin -= 50.0
        emax += 50.0
        f_min = objective(emin)
        f_max = objective(emax)
        expansion_steps += 1
    
    # If still no bracket, use approximation
    if f_min * f_max > 0:
        # Fall back to mean of sp_energy weighted by occupations
        return float(np.mean(sp_np))
    
    eferm = brentq(objective, emin, emax, xtol=1e-14)
    return eferm


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
) -> Tuple[float, jax.Array, jax.Array, jax.Array, jax.Array, dict]:
    """
    Solve BCS pairing for one isospin.
    
    Args:
        iq: Isospin index (0=neutron, 1=proton)
        particle_number: Target particle number
        sp_energy: Single-particle energies for this isospin
        deltaf: Pairing gaps for this isospin
        wstates: State weights for this isospin
        pairwg: Pairing cutoff weights for this isospin
        pair_cutoff: Pairing space cutoff energy
        state_cutoff: State space cutoff energy
        softcut_range: Relative width of soft cutoff
        
    Returns:
        (eferm, wocc, wguv, pairwg_new, wstates_new, stats)
        where stats contains pairing statistics
    """
    # Find Fermi energy
    eferm = find_fermi_energy(particle_number, sp_energy, deltaf, wstates)
    
    # Update cutoffs if specified
    pairwg_new = pairwg.copy()
    wstates_new = wstates.copy()
    wstates_for_calc = wstates
    
    if pair_cutoff > 0.0:
        ecut = eferm + pair_cutoff
        width = softcut_range * pair_cutoff
        pairwg_new = soft_cutoff(sp_energy, ecut, width)
    
    if state_cutoff > 0.0:
        ecut = eferm + state_cutoff
        width = softcut_range * state_cutoff
        wstates_new = soft_cutoff(sp_energy, ecut, width)
        wstates_for_calc = wstates_new
    
    # Compute BCS occupations
    v2, uv = compute_bcs_occupations(sp_energy, deltaf, eferm)
    
    # Compute statistics
    vol = 0.5 * uv * wstates_for_calc
    sumuv = jnp.sum(vol)
    sumduv = jnp.sum(vol * deltaf)
    sumv2 = jnp.sum(v2 * wstates_for_calc)
    sumdv2 = jnp.sum(deltaf * v2 * wstates_for_calc)
    
    sumuv_safe = jnp.maximum(sumuv, 1.0e-20)
    sumv2_safe = jnp.maximum(sumv2, 1.0e-20)
    
    stats = {
        'eferm': float(eferm),
        'epair': float(sumduv),
        'avdelt': float(sumduv / sumuv_safe),
        'avdeltv2': float(sumdv2 / sumv2_safe),
        'avg': float(sumduv / sumuv_safe**2),
    }
    
    return eferm, v2, uv, pairwg_new, wstates_new, stats


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
    if iteration <= 10:
        gap_estimate = 11.2 / jnp.sqrt(float(mass_number))
        return gap_estimate * jnp.ones(nstates, dtype=dtypes.float)
    
    # Compute |psi|^2 summed over spin
    psi_sq = jnp.real(psi * jnp.conjugate(psi))
    density = jnp.sum(psi_sq, axis=1)  # (nstates, nx, ny, nz)
    
    # Select v_pair for each state
    v_pair_n = v_pair[0]
    v_pair_p = v_pair[1]
    
    def get_gap(i):
        iq = isospin[i]
        vp = jnp.where(iq == 0, v_pair_n, v_pair_p)
        integrand = vp * density[i]
        return jnp.sum(integrand) * wxyz * pairwg[i]
    
    deltaf = jax.vmap(lambda i: get_gap(i))(jnp.arange(nstates))
    return deltaf


def solve_pairing(
    sp_energy: jax.Array,
    deltaf: jax.Array,
    wstates: jax.Array,
    pairwg: jax.Array,
    isospin: jax.Array,
    nneut: int,
    nprot: int,
    force,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, Pairing]:
    """
    Solve BCS pairing for both isospins.
    
    Args:
        sp_energy: Single-particle energies (nstates,)
        deltaf: Pairing gaps (nstates,)
        wstates: State weights (nstates,)
        pairwg: Pairing cutoff weights (nstates,)
        isospin: Isospin indices (nstates,)
        nneut: Neutron number
        nprot: Proton number
        force: Force parameters
        
    Returns:
        (wocc, wguv, pairwg_new, wstates_new, pairing)
    """
    dtypes = get_dtypes()
    nstates = len(sp_energy)
    
    wocc = jnp.zeros(nstates, dtype=dtypes.float)
    wguv = jnp.zeros(nstates, dtype=dtypes.float)
    pairwg_new = pairwg.copy()
    wstates_new = wstates.copy()
    pairing = Pairing.zeros()
    
    # Process each isospin
    for iq, particle_number in [(0, float(nneut)), (1, float(nprot))]:
        mask = (isospin == iq)
        
        sp_iq = sp_energy[mask]
        delta_iq = deltaf[mask]
        wst_iq = wstates[mask]
        pwg_iq = pairwg[mask]
        
        eferm, v2, uv, pwg_new, wst_new, stats = solve_pairing_isospin(
            iq,
            particle_number,
            sp_iq,
            delta_iq,
            wst_iq,
            pwg_iq,
            pair_cutoff=force.pair_cutoff[iq],
            state_cutoff=force.state_cutoff[iq],
            softcut_range=force.softcut_range,
        )
        
        # Update arrays
        wocc = wocc.at[mask].set(v2)
        wguv = wguv.at[mask].set(uv)
        
        if force.pair_cutoff[iq] > 0.0:
            pairwg_new = pairwg_new.at[mask].set(pwg_new)
        if force.state_cutoff[iq] > 0.0:
            wstates_new = wstates_new.at[mask].set(wst_new)
        
        # Update pairing data
        pairing.eferm = pairing.eferm.at[iq].set(stats['eferm'])
        pairing.epair = pairing.epair.at[iq].set(stats['epair'])
        pairing.avdelt = pairing.avdelt.at[iq].set(stats['avdelt'])
        pairing.avdeltv2 = pairing.avdeltv2.at[iq].set(stats['avdeltv2'])
        pairing.avg = pairing.avg.at[iq].set(stats['avg'])
    
    return wocc, wguv, pairwg_new, wstates_new, pairing
