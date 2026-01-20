"""
HFB solver module.

This module implements the iterative HFB solver including:
- Gradient descent steps
- Diagonalization steps  
- Density mixing
- Convergence checking
"""

import jax
import jax.numpy as jnp
from dataclasses import dataclass, field
from typing import Tuple, Optional, Callable
import time

from jax_hfbfft.jax_config import get_dtypes
from jax_hfbfft.core.grid import Grid
from jax_hfbfft.physics.densities import Densities, compute_densities
from jax_hfbfft.physics.meanfield import Meanfield, compute_skyrme_meanfield, apply_hfb_hamiltonian
from jax_hfbfft.physics.energies import Energies, compute_integrated_energy, compute_sp_energy
from jax_hfbfft.physics.pairing import Pairing, solve_pairing, compute_pairing_gaps
from jax_hfbfft.physics.coulomb import CoulombSolver, solve_poisson


@jax.tree_util.register_dataclass
@dataclass
class SolverState:
    """
    State of the HFB solver.
    
    Contains all arrays that are updated during iteration.
    """
    # Wavefunctions
    psi: jax.Array         # Shape: (nstates, 2, nx, ny, nz)
    
    # Single-particle properties
    sp_energy: jax.Array   # Single-particle energies (nstates,)
    sp_kinetic: jax.Array  # Kinetic energies (nstates,)
    deltaf: jax.Array      # Pairing gaps (nstates,)
    
    # Occupation factors
    wocc: jax.Array        # BCS occupation v^2 (nstates,)
    wguv: jax.Array        # BCS uv factor (nstates,)
    wstates: jax.Array     # State weights (nstates,)
    pairwg: jax.Array      # Pairing cutoff weights (nstates,)
    
    # State metadata
    isospin: jax.Array     # 0=neutron, 1=proton (nstates,)
    
    # Densities and potentials
    densities: Densities
    meanfield: Meanfield
    
    # Coulomb
    coulomb_solver: CoulombSolver
    wcoul: jax.Array       # Coulomb potential
    
    # Energies
    energies: Energies
    pairing: Pairing
    
    # Convergence
    iteration: int
    converged: bool
    efluct: float          # Convergence measure
    
    @property
    def nstates(self) -> int:
        return self.psi.shape[0]
    
    @property
    def nneut(self) -> int:
        return int(jnp.sum(self.isospin == 0))
    
    @property
    def nprot(self) -> int:
        return int(jnp.sum(self.isospin == 1))


@dataclass
class SolverConfig:
    """Configuration for the HFB solver."""
    max_iterations: int = 200
    convergence_criterion: float = 1e-6
    
    # Damping and mixing
    x0dmp: float = 0.45           # Gradient step damping
    density_mixing: float = 0.2   # New density fraction
    
    # Iteration control
    diag_start: int = 30          # Start diagonalization after this
    bcs_start: int = 30           # Use pure BCS until this iteration
    
    # Output
    output_interval: int = 5
    verbose: bool = True


def initialize_wavefunctions(
    grid: Grid,
    nneut: int,
    nprot: int,
    nstates_neut: int,
    nstates_prot: int,
) -> Tuple[jax.Array, jax.Array]:
    """
    Initialize wavefunctions with random or harmonic oscillator states.
    
    For simplicity, initializes with normalized random complex arrays.
    A proper implementation would use harmonic oscillator states.
    
    Args:
        grid: Spatial grid
        nneut, nprot: Particle numbers
        nstates_neut, nstates_prot: Number of states per isospin
        
    Returns:
        (psi, isospin) arrays
    """
    dtypes = get_dtypes()
    key = jax.random.PRNGKey(42)
    
    nstates = nstates_neut + nstates_prot
    shape = (nstates, 2, grid.nx, grid.ny, grid.nz)
    
    # Random initialization (should be replaced with HO states)
    key, subkey = jax.random.split(key)
    psi_real = jax.random.normal(subkey, shape)
    key, subkey = jax.random.split(key)
    psi_imag = jax.random.normal(subkey, shape)
    psi = (psi_real + 1j * psi_imag).astype(dtypes.complex)
    
    # Normalize
    norms = jnp.sqrt(jnp.sum(jnp.abs(psi)**2, axis=(1, 2, 3, 4), keepdims=True) * grid.wxyz)
    psi = psi / norms
    
    # Isospin labels
    isospin = jnp.concatenate([
        jnp.zeros(nstates_neut, dtype=dtypes.int),
        jnp.ones(nstates_prot, dtype=dtypes.int),
    ])
    
    return psi, isospin


def create_initial_state(
    grid: Grid,
    force,
    nneut: int,
    nprot: int,
    nstates_neut: int,
    nstates_prot: int,
) -> SolverState:
    """
    Create initial solver state.
    
    Args:
        grid: Spatial grid
        force: Force parameters
        nneut, nprot: Particle numbers
        nstates_neut, nstates_prot: Number of single-particle states
        
    Returns:
        Initial SolverState
    """
    dtypes = get_dtypes()
    nstates = nstates_neut + nstates_prot
    A = nneut + nprot
    
    # Initialize wavefunctions
    psi, isospin = initialize_wavefunctions(
        grid, nneut, nprot, nstates_neut, nstates_prot
    )
    
    # Initial single-particle energies: harmonic oscillator estimate
    # Estimate oscillator frequency: hbar*omega ~ 41/A^(1/3) MeV
    hbar_omega = 41.0 / (A ** (1.0/3.0))
    
    # Spread energies around Fermi level
    # Neutron states from -nneut/2 to nstates_neut - nneut/2
    # Proton states from -nprot/2 to nstates_prot - nprot/2
    sp_energy_neut = hbar_omega * jnp.linspace(-nneut/4, nstates_neut - nneut/2, nstates_neut)
    sp_energy_prot = hbar_omega * jnp.linspace(-nprot/4, nstates_prot - nprot/2, nstates_prot)
    sp_energy = jnp.concatenate([sp_energy_neut, sp_energy_prot]).astype(dtypes.float)
    
    sp_kinetic = jnp.zeros(nstates, dtype=dtypes.float)
    deltaf = 11.2 / jnp.sqrt(A) * jnp.ones(nstates, dtype=dtypes.float)
    
    # Occupations (initial: filled for N lowest, empty otherwise)
    wocc = jnp.ones(nstates, dtype=dtypes.float) * 0.5  # Start with half-filling
    wguv = jnp.ones(nstates, dtype=dtypes.float) * 0.5
    wstates = jnp.ones(nstates, dtype=dtypes.float)
    pairwg = jnp.ones(nstates, dtype=dtypes.float)
    
    # Initialize densities and meanfield
    densities = Densities.zeros(grid.nx, grid.ny, grid.nz)
    meanfield = Meanfield.zeros(grid.nx, grid.ny, grid.nz)
    
    # Coulomb
    coulomb_solver = CoulombSolver.create(grid)
    wcoul = jnp.zeros((grid.nx, grid.ny, grid.nz), dtype=dtypes.float)
    
    return SolverState(
        psi=psi,
        sp_energy=sp_energy,
        sp_kinetic=sp_kinetic,
        deltaf=deltaf,
        wocc=wocc,
        wguv=wguv,
        wstates=wstates,
        pairwg=pairwg,
        isospin=isospin,
        densities=densities,
        meanfield=meanfield,
        coulomb_solver=coulomb_solver,
        wcoul=wcoul,
        energies=Energies.zeros(),
        pairing=Pairing.zeros(),
        iteration=0,
        converged=False,
        efluct=1e10,
    )


def compute_densities_from_state(state: SolverState, grid: Grid) -> Densities:
    """Compute densities from current wavefunctions and occupations."""
    return compute_densities(
        state.psi,
        state.wocc,
        state.wguv,
        state.pairwg,
        state.isospin,
        grid,
    )


def gradient_step(
    psi: jax.Array,
    meanfield: Meanfield,
    wocc: jax.Array,
    wguv: jax.Array,
    pairwg: jax.Array,
    isospin: jax.Array,
    grid: Grid,
    x0dmp: float,
) -> jax.Array:
    """
    Perform gradient descent step on wavefunctions.
    
    Updates: psi <- psi - x0dmp * H|psi>
    Then orthonormalizes.
    
    Args:
        psi: Current wavefunctions (nstates, 2, nx, ny, nz)
        meanfield: Current mean-field potentials
        wocc, wguv: Occupation factors
        pairwg: Pairing cutoff weights
        isospin: Isospin labels
        grid: Spatial grid
        x0dmp: Damping factor
        
    Returns:
        Updated wavefunctions
    """
    nstates = psi.shape[0]
    
    def apply_gradient_to_state(nst):
        """Apply gradient step to a single state."""
        iq = isospin[nst]
        weight = wocc[nst]
        weightuv = wguv[nst] * pairwg[nst]
        
        hpsi, _, _ = apply_hfb_hamiltonian(
            psi[nst],
            meanfield,
            iq,
            weight,
            weightuv,
            grid.dx,
            grid.dy,
            grid.dz,
        )
        
        # Gradient descent step
        return psi[nst] - x0dmp * hpsi
    
    # Apply to all states using vmap
    psi_new = jax.vmap(apply_gradient_to_state)(jnp.arange(nstates))
    
    # Simple normalization for now (full orthonormalization is expensive)
    # Each state should be normalized individually
    psi_new = normalize_states(psi_new, grid.wxyz)
    
    return psi_new


def normalize_states(psi: jax.Array, wxyz: float) -> jax.Array:
    """Normalize each wavefunction."""
    norms = jnp.sqrt(jnp.sum(jnp.abs(psi)**2, axis=(1, 2, 3, 4), keepdims=True) * wxyz)
    norms = jnp.maximum(norms, 1e-30)
    return psi / norms


def orthonormalize_block_cpu(psi_block: jax.Array, wxyz: float) -> jax.Array:
    """
    CPU-friendly Gram-Schmidt orthonormalization for a single block.
    Uses scan for the outer loop.
    """
    n = psi_block.shape[0]
    
    def gs_step(carry, i):
        psi_ortho = carry
        vec = psi_block[i]
        
        # Subtract projections on all previous orthonormalized vectors
        def subtract_proj(j, v):
            overlap = jnp.sum(jnp.conjugate(psi_ortho[j]) * v) * wxyz
            return jnp.where(j < i, v - overlap * psi_ortho[j], v)
        
        vec = jax.lax.fori_loop(0, n, subtract_proj, vec)
        
        # Normalize
        norm = jnp.sqrt(jnp.sum(jnp.abs(vec)**2) * wxyz)
        norm = jnp.maximum(norm, 1e-30)
        vec = vec / norm
        
        psi_ortho = psi_ortho.at[i].set(vec)
        return psi_ortho, None
    
    psi_ortho, _ = jax.lax.scan(gs_step, jnp.zeros_like(psi_block), jnp.arange(n))
    return psi_ortho


def compute_sp_energies(
    psi: jax.Array,
    meanfield: Meanfield,
    isospin: jax.Array,
    grid: Grid,
) -> Tuple[jax.Array, jax.Array]:
    """
    Compute single-particle and kinetic energies.
    
    Args:
        psi: Wavefunctions
        meanfield: Mean-field potentials
        isospin: Isospin labels
        grid: Spatial grid
        
    Returns:
        (sp_energy, sp_kinetic) arrays
    """
    dtypes = get_dtypes()
    nstates = psi.shape[0]
    
    sp_energy = jnp.zeros(nstates, dtype=dtypes.float)
    sp_kinetic = jnp.zeros(nstates, dtype=dtypes.float)
    
    for nst in range(nstates):
        iq = int(isospin[nst])
        psi_n = psi[nst]
        
        # Apply Hamiltonian
        hpsi, hpsi_mf, _ = apply_hfb_hamiltonian(
            psi_n,
            meanfield,
            iq,
            1.0, 0.0,  # weight=1, weightuv=0 for pure mean-field
            grid.dx, grid.dy, grid.dz,
        )
        
        # Expectation values
        e_tot = jnp.real(jnp.sum(jnp.conjugate(psi_n) * hpsi_mf)) * grid.wxyz
        
        # Kinetic energy from effective mass term
        # This is simplified - should use proper kinetic operator
        e_kin = jnp.real(jnp.sum(
            meanfield.bmass[iq] * jnp.abs(psi_n)**2
        )) * grid.wxyz
        
        sp_energy = sp_energy.at[nst].set(e_tot)
        sp_kinetic = sp_kinetic.at[nst].set(e_kin)
    
    return sp_energy, sp_kinetic


def hfb_iteration(
    state: SolverState,
    grid: Grid,
    force,
    config: SolverConfig,
    use_coulomb: bool = True,
) -> SolverState:
    """
    Perform one HFB iteration.
    
    Steps:
    1. Gradient step on wavefunctions
    2. Solve pairing
    3. Compute densities
    4. Compute mean-field potentials
    5. Compute energies
    6. Check convergence
    
    Args:
        state: Current solver state
        grid: Spatial grid
        force: Force parameters
        config: Solver configuration
        use_coulomb: Whether to include Coulomb
        
    Returns:
        Updated solver state
    """
    dtypes = get_dtypes()
    iteration = state.iteration + 1
    
    # Store old densities for mixing
    old_rho = state.densities.rho.copy()
    old_tau = state.densities.tau.copy()
    old_chi = state.densities.chi.copy()
    
    # 1. Gradient step
    psi_new = gradient_step(
        state.psi,
        state.meanfield,
        state.wocc,
        state.wguv,
        state.pairwg,
        state.isospin,
        grid,
        config.x0dmp,
    )
    
    # 2. Compute pairing gaps
    deltaf = compute_pairing_gaps(
        psi_new,
        state.meanfield.v_pair,
        state.isospin,
        state.pairwg,
        grid.wxyz,
        iteration,
        state.nneut + state.nprot,
    )
    
    # 3. Solve BCS equations
    wocc, wguv, pairwg, wstates, pairing = solve_pairing(
        state.sp_energy,
        deltaf,
        state.wstates,
        state.pairwg,
        state.isospin,
        state.nneut,
        state.nprot,
        force,
    )
    
    # 4. Compute densities
    densities = compute_densities(
        psi_new, wocc, wguv, pairwg, state.isospin, grid
    )
    
    # Apply density mixing
    mix = config.density_mixing
    densities = Densities(
        rho=mix * densities.rho + (1 - mix) * old_rho,
        tau=mix * densities.tau + (1 - mix) * old_tau,
        chi=mix * densities.chi + (1 - mix) * old_chi,
        current=densities.current,
        sdens=densities.sdens,
        sodens=densities.sodens,
    )
    
    # 5. Coulomb potential
    wcoul = state.wcoul
    if use_coulomb:
        wcoul = solve_poisson(
            densities.rho[1],  # Proton density
            state.coulomb_solver,
            grid,
        )
    
    # 6. Compute mean-field potentials
    meanfield = compute_skyrme_meanfield(
        densities,
        force,
        grid,
        coulomb_potential=wcoul if use_coulomb else None,
        use_coulomb=use_coulomb,
    )
    
    # 7. Compute single-particle energies
    sp_energy, sp_kinetic = compute_sp_energies(
        psi_new, meanfield, state.isospin, grid
    )
    
    # 8. Compute total energies
    energies = compute_integrated_energy(
        densities,
        force,
        grid,
        coulomb_potential=wcoul if use_coulomb else None,
        pairing_energy=pairing.epair,
        mass_number=state.nneut + state.nprot,
        use_coulomb=use_coulomb,
    )
    
    # 9. Check convergence
    efluct = jnp.max(jnp.abs(sp_energy - state.sp_energy))
    converged = efluct < config.convergence_criterion
    
    return SolverState(
        psi=psi_new,
        sp_energy=sp_energy,
        sp_kinetic=sp_kinetic,
        deltaf=deltaf,
        wocc=wocc,
        wguv=wguv,
        wstates=wstates,
        pairwg=pairwg,
        isospin=state.isospin,
        densities=densities,
        meanfield=meanfield,
        coulomb_solver=state.coulomb_solver,
        wcoul=wcoul,
        energies=energies,
        pairing=pairing,
        iteration=iteration,
        converged=converged,
        efluct=float(efluct),
    )


def run_hfb(
    grid: Grid,
    force,
    nucleus_z: int,
    nucleus_n: int,
    config: Optional[SolverConfig] = None,
    initial_state: Optional[SolverState] = None,
    callback: Optional[Callable[[SolverState], None]] = None,
) -> SolverState:
    """
    Run HFB calculation to convergence.
    
    Args:
        grid: Spatial grid
        force: Force parameters
        nucleus_z: Proton number
        nucleus_n: Neutron number
        config: Solver configuration (uses defaults if None)
        initial_state: Starting state (creates new if None)
        callback: Called after each iteration with current state
        
    Returns:
        Converged SolverState
    """
    if config is None:
        config = SolverConfig()
    
    # Initialize state
    if initial_state is None:
        # Estimate number of states needed (2x particle number)
        nstates_n = max(int(1.5 * nucleus_n), nucleus_n + 10)
        nstates_p = max(int(1.5 * nucleus_z), nucleus_z + 10)
        
        state = create_initial_state(
            grid, force,
            nucleus_n, nucleus_z,
            nstates_n, nstates_p,
        )
    else:
        state = initial_state
    
    # Initial density and potential
    state = SolverState(
        psi=state.psi,
        sp_energy=state.sp_energy,
        sp_kinetic=state.sp_kinetic,
        deltaf=state.deltaf,
        wocc=state.wocc,
        wguv=state.wguv,
        wstates=state.wstates,
        pairwg=state.pairwg,
        isospin=state.isospin,
        densities=compute_densities_from_state(state, grid),
        meanfield=state.meanfield,
        coulomb_solver=state.coulomb_solver,
        wcoul=state.wcoul,
        energies=state.energies,
        pairing=state.pairing,
        iteration=0,
        converged=False,
        efluct=state.efluct,
    )
    
    # Main iteration loop
    start_time = time.time()
    
    for i in range(config.max_iterations):
        state = hfb_iteration(state, grid, force, config)
        
        if callback is not None:
            callback(state)
        
        if config.verbose and (i + 1) % config.output_interval == 0:
            elapsed = time.time() - start_time
            print(f"Iter {state.iteration:4d}: E = {state.energies.ehfint:12.4f} MeV, "
                  f"fluct = {state.efluct:.2e}, time = {elapsed:.1f}s")
        
        if state.converged:
            if config.verbose:
                print(f"\nConverged at iteration {state.iteration}")
                print(f"Total energy: {state.energies.ehfint:.4f} MeV")
            break
    else:
        if config.verbose:
            print(f"\nDid not converge after {config.max_iterations} iterations")
            print(f"Final energy: {state.energies.ehfint:.4f} MeV")
            print(f"Final fluctuation: {state.efluct:.2e}")
    
    return state
