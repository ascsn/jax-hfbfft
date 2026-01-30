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
import dataclasses

from jax_hfbfft.jax_config import get_dtypes
from jax_hfbfft.core.grid import Grid
from jax_hfbfft.core.force import Force
from jax_hfbfft.core.constraint import Constraint
from jax_hfbfft.physics.densities import Densities, compute_densities
from jax_hfbfft.physics.meanfield import Meanfield, compute_skyrme_meanfield, apply_hfb_hamiltonian
from jax_hfbfft.physics.energies import Energies, compute_integrated_energy, compute_sp_energy
from jax_hfbfft.physics.pairing import Pairing, solve_pairing, compute_pairing_gaps
from jax_hfbfft.physics.coulomb import CoulombSolver, solve_poisson
from jax_hfbfft.physics.constraints import (
    ConstraintState,
    build_constraint_state,
    compute_constraint_potential,
    update_constraint_state,
)


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

    # Constraints
    constraint_state: ConstraintState
    
    # Convergence
    iteration: int
    converged: bool
    efluct: float          # Convergence measure
    
    @property
    def nstates(self) -> int:
        return self.psi.shape[0]
    
    @property
    def nneut(self) -> jax.Array:
        return jnp.sum(self.isospin == 0)
    
    @property
    def nprot(self) -> jax.Array:
        return jnp.sum(self.isospin == 1)


@jax.tree_util.register_dataclass
@dataclass
class SolverConfig:
    """Configuration for the HFB solver."""
    max_iterations: int = 200
    convergence_criterion: float = 1e-6
    
    # Damping and mixing
    x0dmp: float = 0.45           # Gradient step damping
    e0dmp: float = 20.0           # Preconditioning energy (MeV)
    density_mixing: float = 0.5   # New density fraction
    
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
    npsi_n: int,
    npsi_p: int,
    target_n: int,
    target_p: int,
    constraint_state: Optional[ConstraintState] = None,
) -> SolverState:
    """Create initial HFB state."""
    dtypes = get_dtypes()
    nstates = npsi_n + npsi_p
    A = target_n + target_p
    
    # Initialize wavefunctions
    psi, isospin = initialize_wavefunctions(
        grid, target_n, target_p, npsi_n, npsi_p
    )
    
    # Initial single-particle energies: harmonic oscillator estimate
    # Estimate oscillator frequency: hbar*omega ~ 41/A^(1/3) MeV
    hbar_omega = 41.0 / (A ** (1.0/3.0))
    
    # Spread energies around Fermi level
    # Neutron states from -nneut/2 to nstates_neut - nneut/2
    # Proton states from -nprot/2 to nstates_prot - target_p/2
    sp_energy_neut = hbar_omega * jnp.linspace(-target_n/4, npsi_n - target_n/2, npsi_n)
    sp_energy_prot = hbar_omega * jnp.linspace(-target_p/4, npsi_p - target_p/2, npsi_p)
    sp_energy = jnp.concatenate([sp_energy_neut, sp_energy_prot]).astype(dtypes.float)
    
    sp_kinetic = jnp.zeros(nstates, dtype=dtypes.float)
    deltaf = 11.2 / jnp.sqrt(A) * jnp.ones(nstates, dtype=dtypes.float)
    
    # Set initial occupations (fill only target levels)
    wocc = jnp.zeros(nstates)
    wocc = wocc.at[:target_n].set(1.0)
    wocc = wocc.at[npsi_n:npsi_n+target_p].set(1.0)
    
    # Initialize other arrays
    wguv = jnp.zeros(nstates, dtype=dtypes.float)
    pairwg = jnp.ones(nstates, dtype=dtypes.float)
    wstates = jnp.ones(nstates, dtype=dtypes.float)
    
    # Initialize densities and meanfield
    densities = Densities.zeros(grid.nx, grid.ny, grid.nz)
    meanfield = Meanfield.zeros(grid.nx, grid.ny, grid.nz)
    
    # Coulomb
    coulomb_solver = CoulombSolver.create(grid)
    wcoul = jnp.zeros((grid.nx, grid.ny, grid.nz), dtype=dtypes.float)
    
    if constraint_state is None:
        constraint_state = ConstraintState.disabled(grid)

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
        constraint_state=constraint_state,
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
    sp_energy: jax.Array,
    isospin: jax.Array,
    grid: Grid,
    x0dmp: float,
    e0dmp: float,
    npsi_n: int,
) -> jax.Array:
    """
    Perform preconditioned gradient descent step on wavefunctions.
    
    Updates: psi <- psi - x0dmp * H|psi>
    Then normalizes and orthonormalizes.
    
    Args:
        psi: Current wavefunctions (nstates, 2, nx, ny, nz)
        meanfield: Current mean-field potentials
        wocc, wguv: Occupation factors
        pairwg: Pairing cutoff weights
        sp_energy: Single-particle energies
        isospin: Isospin labels
        grid: Spatial grid
        x0dmp: Damping factor
        e0dmp: Preconditioning energy (MeV)
        npsi_n: Number of neutron states
        
    Returns:
        Updated wavefunctions
    """
    return _gradient_step_jit(
        psi, meanfield, wocc, wguv, pairwg, sp_energy, isospin,
        grid, x0dmp, e0dmp, npsi_n
    )


def orthonormalize_states(psi: jax.Array, npsi_n: int, wxyz: float) -> jax.Array:
    """
    Orthonormalize wavefunctions per isospin using QR decomposition.
    """
    nstates = psi.shape[0]
    
    # Split by known count (neutrons first, then protons)
    psi_n = psi[:npsi_n]
    psi_p = psi[npsi_n:]
    
    def ortho_block(block):
        if block.shape[0] == 0:
            return block
        # Flatten: (n, 2, nx, ny, nz) -> (n, 2*nx*ny*nz)
        n_block = block.shape[0]
        flat = block.reshape(n_block, -1)
        
        # Scale for integration
        flat = flat * jnp.sqrt(wxyz)
        
        # QR decomposition
        q, r = jnp.linalg.qr(flat.T, mode='reduced')
        
        # Ensure consistent phase by making diagonal of R real and positive
        # This prevents wavefunctions from jumping phases and failing convergence
        phases = jnp.sign(jnp.diag(r))
        # Handle zero diagonal elements (unlikely for basis states)
        phases = jnp.where(phases == 0, 1.0, phases)
        q = q * phases
        
        # Q.T has the orthonormal wavefunctions, rescale back
        ortho_flat = q.T / jnp.sqrt(wxyz)
        
        return ortho_flat.reshape(block.shape)

    psi_n_ortho = ortho_block(psi_n)
    psi_p_ortho = ortho_block(psi_p)
    
    return jnp.concatenate([psi_n_ortho, psi_p_ortho], axis=0)


def apply_preconditioner(
    phi: jax.Array,
    e0dmp: float,
    h2ma: float,
    dx: float, dy: float, dz: float
) -> jax.Array:
    """
    Apply preconditioning (inverse kinetic energy operator) in Fourier space.
    
    ps_out = ps_in / (e0dmp + h2m * k^2)
    
    This version handles both single wavefunctions and batched wavefunctions.
    For batched input with shape (nstates, 2, nx, ny, nz), the FFT is applied
    to the last 3 dimensions efficiently.
    """
    nx, ny, nz = phi.shape[-3:]
    
    # FFT to k-space (operates on last 3 dimensions)
    phi_k = jnp.fft.fftn(phi, axes=(-3, -2, -1))
    
    # Wavenumbers - computed once and broadcast
    kx = 2 * jnp.pi * jnp.fft.fftfreq(nx, d=dx)
    ky = 2 * jnp.pi * jnp.fft.fftfreq(ny, d=dy)
    kz = 2 * jnp.pi * jnp.fft.fftfreq(nz, d=dz)
    
    # k^2 grid (broadcasting)
    k2 = (kx[:, jnp.newaxis, jnp.newaxis]**2 + 
          ky[jnp.newaxis, :, jnp.newaxis]**2 + 
          kz[jnp.newaxis, jnp.newaxis, :]**2)
    
    # Denominator: e0dmp + h2m * k^2
    denom = e0dmp + h2ma * k2
    
    # Apply in k-space
    phi_k = phi_k / denom
    
    # IFFT back
    phi_out = jnp.fft.ifftn(phi_k, axes=(-3, -2, -1))
    
    return jnp.real(phi_out) if jnp.isrealobj(phi) else phi_out


@jax.jit(static_argnums=(10,))
def _gradient_step_jit(
    psi: jax.Array,
    meanfield: Meanfield,
    wocc: jax.Array,
    wguv: jax.Array,
    pairwg: jax.Array,
    sp_energy: jax.Array,
    isospin: jax.Array,
    grid: Grid,
    x0dmp: float,
    e0dmp: float,
    npsi_n: int,
) -> jax.Array:
    """JIT-compiled preconditioned gradient step."""
    nstates = psi.shape[0]
    
    # 1. Apply Hamiltonian to all states and subtract s.p. energy
    def apply_h(p, iq, e):
        # We use weight=1 and weightuv=0 for the pure s.p. gradient
        # (h - epsilon) psi
        hpsi, _, _ = apply_hfb_hamiltonian(p, meanfield, iq, 1.0, 0.0, grid)
        return hpsi - e * p
    
    hpsi_all = jax.vmap(apply_h)(psi, isospin, sp_energy)
    
    # 2. Apply Preconditioner - batched version (FFT on all states at once)
    # hpsi_all has shape (nstates, 2, nx, ny, nz)
    h2ma = 20.73
    hpsi_pre = apply_preconditioner(hpsi_all, e0dmp, h2ma, grid.dx, grid.dy, grid.dz)
    
    # 3. Update Wavefunctions
    psi_new = psi - x0dmp * hpsi_pre
    
    # 4. Orthonormalize
    return orthonormalize_states(psi_new, npsi_n, grid.wxyz)


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
    Compute single-particle and kinetic energies using vectorized operations.
    
    Args:
        psi: Wavefunctions (nstates, 2, nx, ny, nz)
        meanfield: Mean-field potentials
        isospin: Isospin labels
        grid: Spatial grid
        
    Returns:
        (sp_energy, sp_kinetic) arrays
    """
    return _compute_sp_energies_vmap(
        psi, meanfield, isospin, grid
    )


@jax.jit
def _compute_sp_energies_vmap(
    psi: jax.Array,
    meanfield: Meanfield,
    isospin: jax.Array,
    grid: Grid,
) -> Tuple[jax.Array, jax.Array]:
    """Vectorized single-particle energy computation."""
    nstates = psi.shape[0]
    
    def single_state_energy(nst):
        """Compute energy for a single state."""
        iq = isospin[nst]  # Keep as traced value - JAX handles indexing
        psi_n = psi[nst]
        
        # Apply Hamiltonian
        # hpsi_mf is h|psi>
        _, hpsi_mf, _ = apply_hfb_hamiltonian(
            psi_n,
            meanfield,
            iq,
            1.0, 0.0,
            grid,
        )
        
        # Expectation value: <psi|h|psi>
        e_tot = jnp.real(jnp.sum(jnp.conjugate(psi_n) * hpsi_mf)) * grid.wxyz
        
        # Kinetic energy part (excluding effective mass scaling for now or matching legacy)
        # Note: sp_kinetic here is used for preconditioning or information
        bmass_iq = meanfield.bmass[iq]
        e_kin = jnp.real(jnp.sum(bmass_iq * jnp.abs(psi_n)**2)) * grid.wxyz
        
        return e_tot, e_kin
    
    # Vectorize over states
    sp_energy, sp_kinetic = jax.vmap(single_state_energy)(jnp.arange(nstates))
    
    return sp_energy, sp_kinetic


@jax.jit(static_argnums=(4, 5, 6, 7, 8, 9))
def hfb_iteration(
    state: SolverState,
    grid: Grid,
    force: Force,
    config: SolverConfig,
    npsi_n: int,
    npsi_p: int,
    nucleus_n: int,
    nucleus_z: int,
    use_coulomb: bool = True,
    compute_energy: bool = True,
) -> SolverState:
    """
    Perform one HFB iteration.
    
    Steps:
    1. Gradient step on wavefunctions
    2. Solve pairing
    3. Compute densities
    4. Compute mean-field potentials
    5. Compute energies (optional - skip for speed when not needed)
    6. Check convergence
    
    Args:
        state: Current solver state
        grid: Spatial grid
        force: Force parameters
        config: Solver configuration
        npsi_n: Number of neutron states
        npsi_p: Number of proton states
        nucleus_n: Target neutron number
        nucleus_z: Target proton number
        use_coulomb: Whether to include Coulomb
        compute_energy: Whether to compute integrated energy (expensive)
        
    Returns:
        Updated solver state
    """
    dtypes = get_dtypes()
    iteration = state.iteration + 1
    
    # Store old densities for mixing
    old_rho = state.densities.rho
    old_tau = state.densities.tau
    old_chi = state.densities.chi
    
    # 1. Gradient step
    psi_new = gradient_step(
        state.psi,
        state.meanfield,
        state.wocc,
        state.wguv,
        state.pairwg,
        state.sp_energy,
        state.isospin,
        grid,
        config.x0dmp,
        config.e0dmp,
        npsi_n,
    )
    
    # 2. Compute pairing gaps
    # Use actual particle number for estimate
    mass_number = nucleus_n + nucleus_z
    deltaf = compute_pairing_gaps(
        psi_new,
        state.meanfield.v_pair,
        state.isospin,
        state.pairwg,
        grid.wxyz,
        iteration,
        mass_number,
    )
    
    # 3. Solve BCS equations
    wocc, wguv, pairwg, wstates, pairing = solve_pairing(
        state.sp_energy,
        deltaf,
        state.wstates,
        state.pairwg,
        state.isospin,
        npsi_n,
        npsi_p,
        nucleus_n,
        nucleus_z,
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

    # Constraint update (if enabled)
    constraint_state = state.constraint_state
    if constraint_state is not None and constraint_state.constr_field.shape[0] > 0:
        constraint_state = update_constraint_state(
            constraint_state,
            densities,
            grid,
            config.e0dmp,
            config.x0dmp,
        )
        constraint_potential = compute_constraint_potential(constraint_state, grid)
    else:
        constraint_potential = None
    
    # 5. Coulomb potential
    wcoul = state.wcoul
    if use_coulomb:
        from jax_hfbfft.physics.coulomb import solve_poisson
        wcoul = solve_poisson(
            densities.rho[1],  # Proton density
            state.coulomb_solver,
            grid,
        )
    
    # 6. Compute mean-field potentials
    from jax_hfbfft.physics.meanfield import compute_skyrme_meanfield
    meanfield = compute_skyrme_meanfield(
        densities,
        force,
        grid,
        coulomb_potential=wcoul,
        constraint_potential=constraint_potential,
        use_coulomb=use_coulomb,
    )
    
    # 7. Compute single-particle energies
    sp_energy, sp_kinetic = compute_sp_energies(
        psi_new, meanfield, state.isospin, grid
    )
    
    # 8. Compute total energies (only if requested - this is expensive)
    # Since compute_energy is a static argument, we can use Python if/else
    # and JAX will compile separate versions for each case
    from jax_hfbfft.physics.energies import compute_integrated_energy
    
    if compute_energy:
        energies = compute_integrated_energy(
            densities,
            force,
            grid,
            coulomb_potential=wcoul,
            pairing_energy=pairing.epair,
            mass_number=state.nneut + state.nprot,
            use_coulomb=use_coulomb,
        )
    else:
        energies = state.energies
    
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
        constraint_state=constraint_state,
        iteration=iteration,
        converged=converged,
        efluct=efluct,
    )


def run_hfb(
    grid: Grid,
    force,
    nucleus_z: int,
    nucleus_n: int,
    npsi_n: int,  # Added
    config: Optional[SolverConfig] = None,
    initial_state: Optional[SolverState] = None,
    callback: Optional[Callable[[SolverState], None]] = None,
    use_coulomb: bool = True,
    constraint: Optional[Constraint] = None,
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

    # Initialize constraint state
    constraint_state = build_constraint_state(
        constraint if constraint is not None else Constraint.spherical(),
        grid,
        mass_number=nucleus_n + nucleus_z,
    )
    
    # Initialize state
    if initial_state is None:
        # Estimate number of states needed (2x particle number)
        nstates_n = max(int(1.5 * nucleus_n), nucleus_n + 10)
        nstates_p = max(int(1.5 * nucleus_z), nucleus_z + 10)
        
        state = create_initial_state(
            grid,
            nstates_n,
            nstates_p,
            nucleus_n,
            nucleus_z,
            constraint_state=constraint_state,
        )
    else:
        state = initial_state
        if state.constraint_state is None:
            state = dataclasses.replace(state, constraint_state=constraint_state)
    
    # Initial density and potential
    initial_densities = compute_densities_from_state(state, grid)
    
    # 1. Coulomb potential (if needed)
    initial_wcoul = jnp.zeros((grid.nx, grid.ny, grid.nz), dtype=jnp.float64)
    if use_coulomb:
        # For first iteration, we can just use 0 or solve once
        from jax_hfbfft.physics.coulomb import solve_poisson
        initial_wcoul = solve_poisson(initial_densities.rho[1], state.coulomb_solver, grid)

    # 2. Mean-field potentials
    from jax_hfbfft.physics.meanfield import compute_skyrme_meanfield
    if state.constraint_state is not None and state.constraint_state.constr_field.shape[0] > 0:
        constraint_state = update_constraint_state(
            state.constraint_state,
            initial_densities,
            grid,
            config.e0dmp,
            config.x0dmp,
        )
        constraint_potential = compute_constraint_potential(constraint_state, grid)
    else:
        constraint_state = state.constraint_state
        constraint_potential = None

    initial_meanfield = compute_skyrme_meanfield(
        initial_densities,
        force,
        grid,
        coulomb_potential=initial_wcoul if use_coulomb else None,
        constraint_potential=constraint_potential,
        use_coulomb=use_coulomb,
    )
    
    # 3. Initial single-particle energies from the mean-field
    initial_sp_energy, initial_sp_kinetic = compute_sp_energies(
        state.psi, initial_meanfield, state.isospin, grid
    )

    state = SolverState(
        psi=state.psi,
        sp_energy=initial_sp_energy,
        sp_kinetic=initial_sp_kinetic,
        deltaf=state.deltaf,
        wocc=state.wocc,
        wguv=state.wguv,
        wstates=state.wstates,
        pairwg=state.pairwg,
        isospin=state.isospin,
        densities=initial_densities,
        meanfield=initial_meanfield,
        coulomb_solver=state.coulomb_solver,
        wcoul=initial_wcoul,
        energies=state.energies,
        pairing=state.pairing,
        constraint_state=constraint_state,
        iteration=0,
        converged=False,
        efluct=state.efluct,
    )
    
    # Main iteration loop
    start_time = time.time()
    npsi = state.psi.shape[0]
    npsi_p = npsi - npsi_n
    
    for i in range(config.max_iterations):
        # Only compute energy when we need it for output or final result
        need_output = config.verbose and (i + 1) % config.output_interval == 0
        is_near_end = i >= config.max_iterations - 1
        compute_energy = need_output or is_near_end
        
        state = hfb_iteration(
            state, grid, force, config, 
            npsi_n, npsi_p, nucleus_n, nucleus_z, 
            use_coulomb=use_coulomb,
            compute_energy=compute_energy,
        )
        
        if callback is not None:
            callback(state)
        
        if need_output:
            elapsed = time.time() - start_time
            print(f"Iter {state.iteration:4d}: E = {state.energies.ehfint:12.4f} MeV, "
                  f"fluct = {state.efluct:.2e}, time = {elapsed:.1f}s")
        
        if state.converged:
            # Make sure we have final energy computed
            if not compute_energy:
                final_energies = compute_integrated_energy(
                    state.densities,
                    force,
                    grid,
                    coulomb_potential=state.wcoul,
                    pairing_energy=state.pairing.epair,
                    mass_number=nucleus_n + nucleus_z,
                    use_coulomb=use_coulomb,
                )
                state = dataclasses.replace(state, energies=final_energies, converged=True)
            if config.verbose:
                print(f"\nConverged at iteration {state.iteration}")
                print(f"Total energy: {state.energies.ehfint:.4f} MeV")
            break
    else:
        if config.verbose:
            print(f"\nDid not converge after {config.max_iterations} iterations")
            print(f"Final energy: {state.energies.ehfint:.4f} MeV")
            print(f"Final fluctuation: {state.efluct:.2e}")
    
    # Final energy computation if not already done
    if not compute_energy:
        final_energies = compute_integrated_energy(
            state.densities,
            force,
            grid,
            coulomb_potential=state.wcoul,
            pairing_energy=state.pairing.epair,
            mass_number=nucleus_n + nucleus_z,
            use_coulomb=use_coulomb,
        )
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
            densities=state.densities,
            meanfield=state.meanfield,
            coulomb_solver=state.coulomb_solver,
            wcoul=state.wcoul,
            energies=final_energies,
            pairing=state.pairing,
            constraint_state=state.constraint_state,
            iteration=state.iteration,
            converged=state.converged,
            efluct=state.efluct,
        )
    
    return state
