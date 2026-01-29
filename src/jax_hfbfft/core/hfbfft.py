"""
HFBFFT: Main class for Hartree-Fock-Bogoliubov calculations.

This module provides the central HFBFFT class which encapsulates all state
and methods for performing nuclear structure calculations using the
Hartree-Fock-Bogoliubov method with Fast Fourier Transform techniques.
"""

import jax
import jax.numpy as jnp
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, Callable
import dataclasses
import time
import sys
from pathlib import Path

from jax_hfbfft.core.nucleus import Nucleus
from jax_hfbfft.core.force import Force
from jax_hfbfft.core.constraint import Constraint
from jax_hfbfft.core.grid import Grid

# Add legacy code path for imports
_legacy_path = Path(__file__).parent.parent.parent.parent.parent
if str(_legacy_path) not in sys.path:
    sys.path.insert(0, str(_legacy_path))


@dataclass
class HFBFFTState:
    """
    Internal state container for HFBFFT calculations.
    
    This dataclass holds all the internal arrays and data structures
    used during HFB iterations. It is separate from the main HFBFFT class
    to allow for clean serialization and checkpointing.
    """
    
    # Single-particle wavefunctions and related arrays
    psi: Optional[jax.Array] = None           # Wavefunctions [nst, 2, nx, ny, nz]
    hampsi: Optional[jax.Array] = None        # H|psi> 
    lagrange: Optional[jax.Array] = None      # Lagrange multipliers for orthogonality
    hmfpsi: Optional[jax.Array] = None        # Mean-field hamiltonian applied
    delpsi: Optional[jax.Array] = None        # Delta (pairing) field applied
    
    # Densities
    rho: Optional[jax.Array] = None           # Particle density [2, nx, ny, nz]
    chi: Optional[jax.Array] = None           # Pair density
    tau: Optional[jax.Array] = None           # Kinetic density
    current: Optional[jax.Array] = None       # Current density [2, 3, nx, ny, nz]
    sdens: Optional[jax.Array] = None         # Spin density
    sodens: Optional[jax.Array] = None        # Spin-orbit density
    
    # Mean-field potentials
    upot: Optional[jax.Array] = None          # Central potential [2, nx, ny, nz]
    bmass: Optional[jax.Array] = None         # Effective mass
    v_pair: Optional[jax.Array] = None        # Pairing potential
    aq: Optional[jax.Array] = None            # A-field (current coupling)
    spot: Optional[jax.Array] = None          # Spin-orbit potential
    wlspot: Optional[jax.Array] = None        # WLS field
    dbmass: Optional[jax.Array] = None        # Derivative of effective mass
    divaq: Optional[jax.Array] = None         # Divergence of A-field
    
    # Coulomb
    wcoul: Optional[jax.Array] = None         # Coulomb potential
    coulomb_q: Optional[jax.Array] = None     # Coulomb kernel in k-space
    
    # Single-particle properties (per state)
    sp_energy: Optional[jax.Array] = None     # Single-particle energies
    sp_kinetic: Optional[jax.Array] = None    # Kinetic energies
    sp_efluct1: Optional[jax.Array] = None    # Energy fluctuation 1
    sp_efluct2: Optional[jax.Array] = None    # Energy fluctuation 2
    sp_parity: Optional[jax.Array] = None     # Parity
    sp_orbital: Optional[jax.Array] = None    # Orbital angular momentum
    sp_spin: Optional[jax.Array] = None       # Spin
    sp_norm: Optional[jax.Array] = None       # Normalization
    
    # Occupation numbers (for HFB)
    wocc: Optional[jax.Array] = None          # Occupation probabilities
    wguv: Optional[jax.Array] = None          # u*v pairing factors
    pairwg: Optional[jax.Array] = None        # Pairing weights
    wstates: Optional[jax.Array] = None       # State weights
    deltaf: Optional[jax.Array] = None        # Pairing gaps
    
    # Isospin labels
    isospin: Optional[jax.Array] = None       # Isospin of each state
    npsi: Optional[jax.Array] = None          # Number of states per isospin
    npmin: Optional[jax.Array] = None         # Starting index per isospin


@dataclass
class HFBFFTResults:
    """
    Results container for HFBFFT calculations.
    
    This dataclass stores the final results and convergence information
    from an HFB calculation.
    """
    
    # Convergence
    converged: bool = False
    iterations: int = 0
    final_fluctuation: float = 0.0
    convergence_history: Optional[jax.Array] = None
    
    # Energies
    total_energy: float = 0.0
    kinetic_energy: float = 0.0
    potential_energy: float = 0.0
    pairing_energy: float = 0.0
    coulomb_energy: float = 0.0
    rearrangement_energy: float = 0.0
    cm_correction: float = 0.0
    
    # Energy decomposition (Skyrme terms)
    ehf0: float = 0.0   # t0 term
    ehf1: float = 0.0   # current term
    ehf2: float = 0.0   # Laplacian term
    ehf3: float = 0.0   # density-dependent term
    ehfls: float = 0.0  # spin-orbit term
    
    # Pairing properties
    pairing_gap_n: float = 0.0
    pairing_gap_p: float = 0.0
    fermi_energy_n: float = 0.0
    fermi_energy_p: float = 0.0
    
    # Deformation
    beta2: float = 0.0
    gamma: float = 0.0
    q20: float = 0.0
    q22: float = 0.0
    
    # Radii
    rms_radius_n: float = 0.0
    rms_radius_p: float = 0.0
    rms_radius_total: float = 0.0
    charge_radius: float = 0.0
    
    # Angular momentum
    total_angular_momentum: Optional[jax.Array] = None
    orbital_angular_momentum: Optional[jax.Array] = None
    spin_angular_momentum: Optional[jax.Array] = None


class HFBFFT:
    """
    Main class for Hartree-Fock-Bogoliubov calculations with FFT.
    
    This class encapsulates all the state and methods needed to perform
    nuclear structure calculations using the Hartree-Fock-Bogoliubov (HFB)
    method with Skyrme-type effective interactions.
    
    Each HFBFFT instance is independent and can be used for parallel
    calculations with different nuclei, forces, or constraints.
    
    Attributes:
        nucleus (Nucleus): The nuclear system specification
        force (Force): The nuclear interaction
        grid (Grid): Spatial discretization
        constraint (Constraint): Constraint configuration
        state (HFBFFTState): Internal calculation state
        results (HFBFFTResults): Calculation results
        
    Examples:
        >>> # Basic calculation for Sn-132
        >>> from jax_hfbfft import HFBFFT, Nucleus, Force
        >>> 
        >>> calc = HFBFFT(
        ...     nucleus=Nucleus(protons=50, neutrons=82),
        ...     force=Force.from_name("SLy4")
        ... )
        >>> calc.run(max_iterations=1000)
        >>> print(f"E = {calc.results.total_energy:.3f} MeV")
        
        >>> # Multiple calculations in parallel
        >>> nuclei = [
        ...     Nucleus.from_symbol("Ca", 40),
        ...     Nucleus.from_symbol("Ca", 48),
        ...     Nucleus.from_symbol("Sn", 132),
        ... ]
        >>> calcs = [HFBFFT(nucleus=n, force=force) for n in nuclei]
        >>> for calc in calcs:
        ...     calc.run()
    """
    
    def __init__(
        self,
        nucleus: Nucleus,
        force: Optional[Force] = None,
        grid: Optional[Grid] = None,
        constraint: Optional[Constraint] = None,
        *,
        # Grid parameters (if grid not provided)
        nx: int = 32,
        ny: int = 32,
        nz: int = 32,
        dx: float = 0.8,
        dy: float = 0.8,
        dz: float = 0.8,
        # Force parameters (if force not provided)
        force_name: str = "SLy4",
        # Pairing parameters
        ipair: int = 0,
        v0prot: float = 0.0,
        v0neut: float = 0.0,
        tbcs: bool = False,
        # Basis size
        npsi: Optional[Tuple[int, int]] = None,
        # Physical options
        include_coulomb: bool = True,
        use_fft: bool = True,
    ):
        """
        Initialize an HFBFFT calculation.
        
        Args:
            nucleus: Nuclear system specification
            force: Nuclear interaction (if None, uses force_name)
            grid: Spatial grid (if None, creates from nx/ny/nz/dx/dy/dz)
            constraint: Constraint configuration (if None, uses spherical)
            nx, ny, nz: Grid dimensions
            dx, dy, dz: Grid spacing in fm
            force_name: Name of force to load if force is None
            ipair: Pairing type (0=none, 5=VDI, 6=DDDI)
            v0prot, v0neut: Pairing strengths
            tbcs: Use BCS approximation
            npsi: Basis size (neutrons, protons); if None, auto-calculated
            include_coulomb: Whether to include Coulomb interaction
            use_fft: Whether to use FFT-based derivatives
        """
        self.nucleus = nucleus
        
        # Initialize force
        if force is None:
            self.force = Force.from_name(
                force_name,
                ipair=ipair,
                v0prot=v0prot,
                v0neut=v0neut,
                tbcs=tbcs,
            )
        else:
            self.force = force
        
        # Initialize grid
        if grid is None:
            self.grid = Grid.create(
                nx=nx, ny=ny, nz=nz,
                dx=dx, dy=dy, dz=dz,
            )
        else:
            self.grid = grid
        
        # Initialize constraint
        if constraint is None:
            self.constraint = Constraint.spherical()
        else:
            self.constraint = constraint.initialize_arrays(
                self.grid.shape, 
                nucleus.mass_number
            )
        
        # Configuration
        self.include_coulomb = include_coulomb
        self.use_fft = use_fft
        
        # Calculate basis size
        if npsi is None:
            # Check if pairing is disabled
            if self.force.ipair == 0:
                # No pairing: only need exactly N and Z states
                n_basis = nucleus.neutrons
                p_basis = nucleus.protons
            else:
                # With pairing: need extra states for the pairing window
                # Use formula from original code, round to nearest even integer
                def round_even(x):
                    return int(2 * round(x / 2))
                n_basis = max(126, round_even(nucleus.neutrons + int(1.65 * (nucleus.neutrons ** 0.666667))))
                p_basis = max(82, round_even(nucleus.protons + int(1.65 * (nucleus.protons ** 0.666667))))
            self._npsi = (n_basis, p_basis)
        else:
            self._npsi = npsi
            # Warn if npsi is smaller than particle numbers
            if self._npsi[0] < nucleus.neutrons:
                print(f"Warning: npsi[0]={self._npsi[0]} < N={nucleus.neutrons}. "
                      "May not have enough neutron states.")
            if self._npsi[1] < nucleus.protons:
                print(f"Warning: npsi[1]={self._npsi[1]} < Z={nucleus.protons}. "
                      "May not have enough proton states.")
        
        self._nstmax = self._npsi[0] + self._npsi[1]
        
        # Initialize internal state
        self.state = HFBFFTState()
        self._initialize_state()
        
        # Results container
        self.results = HFBFFTResults()
        
        # Iteration tracking
        self._iteration = 0
        self._converged = False
        
        # Callbacks for monitoring
        self._callbacks: list = []
    
    def _initialize_state(self):
        """Initialize all internal arrays to their starting values."""
        nx, ny, nz = self.grid.shape
        nstmax = self._nstmax
        
        # Wavefunction arrays
        shape5d = (nstmax, 2, nx, ny, nz)
        self.state.psi = jnp.zeros(shape5d, dtype=jnp.complex128)
        self.state.hampsi = jnp.zeros(shape5d, dtype=jnp.complex128)
        self.state.lagrange = jnp.zeros(shape5d, dtype=jnp.complex128)
        self.state.hmfpsi = jnp.zeros(shape5d, dtype=jnp.complex128)
        self.state.delpsi = jnp.zeros(shape5d, dtype=jnp.complex128)
        
        # Density arrays
        shape4d = (2, nx, ny, nz)
        shape5d_vec = (2, 3, nx, ny, nz)
        self.state.rho = jnp.zeros(shape4d, dtype=jnp.float64)
        self.state.chi = jnp.zeros(shape4d, dtype=jnp.float64)
        self.state.tau = jnp.zeros(shape4d, dtype=jnp.float64)
        self.state.current = jnp.zeros(shape5d_vec, dtype=jnp.float64)
        self.state.sdens = jnp.zeros(shape5d_vec, dtype=jnp.float64)
        self.state.sodens = jnp.zeros(shape5d_vec, dtype=jnp.float64)
        
        # Mean-field arrays
        self.state.upot = jnp.zeros(shape4d, dtype=jnp.float64)
        self.state.bmass = jnp.zeros(shape4d, dtype=jnp.float64)
        self.state.v_pair = jnp.zeros(shape4d, dtype=jnp.float64)
        self.state.aq = jnp.zeros(shape5d_vec, dtype=jnp.float64)
        self.state.spot = jnp.zeros(shape5d_vec, dtype=jnp.float64)
        self.state.wlspot = jnp.zeros(shape5d_vec, dtype=jnp.float64)
        self.state.dbmass = jnp.zeros(shape5d_vec, dtype=jnp.float64)
        self.state.divaq = jnp.zeros(shape4d, dtype=jnp.float64)
        
        # Coulomb
        self.state.wcoul = jnp.zeros((nx, ny, nz), dtype=jnp.float64)
        
        # Single-particle properties
        self.state.sp_energy = jnp.zeros(nstmax, dtype=jnp.float64)
        self.state.sp_kinetic = jnp.zeros(nstmax, dtype=jnp.float64)
        self.state.sp_efluct1 = jnp.zeros(nstmax, dtype=jnp.float64)
        self.state.sp_efluct2 = jnp.zeros(nstmax, dtype=jnp.float64)
        self.state.sp_parity = jnp.zeros(nstmax, dtype=jnp.float64)
        self.state.sp_orbital = jnp.zeros((nstmax, 3), dtype=jnp.float64)
        self.state.sp_spin = jnp.zeros((nstmax, 3), dtype=jnp.float64)
        self.state.sp_norm = jnp.zeros(nstmax, dtype=jnp.float64)
        
        # Occupation numbers
        self.state.wocc = jnp.zeros(nstmax, dtype=jnp.float64)
        self.state.wguv = jnp.zeros(nstmax, dtype=jnp.float64)
        self.state.pairwg = jnp.ones(nstmax, dtype=jnp.float64)
        self.state.wstates = jnp.ones(nstmax, dtype=jnp.float64)
        self.state.deltaf = jnp.zeros(nstmax, dtype=jnp.float64)
        
        # Isospin assignments
        self.state.isospin = jnp.zeros(nstmax, dtype=jnp.int32)
        self.state.isospin = self.state.isospin.at[self._npsi[0]:].set(1)
        
        # Basis indices
        self.state.npsi = jnp.array(self._npsi)
        self.state.npmin = jnp.array([0, self._npsi[0]])
        
        # Initial occupations
        self.state.wocc = self.state.wocc.at[:self.nucleus.neutrons].set(1.0)
        self.state.wocc = self.state.wocc.at[
            self._npsi[0]:self._npsi[0] + self.nucleus.protons
        ].set(1.0)
    
    def initialize_wavefunctions(
        self,
        method: str = "harmonic_oscillator",
        **kwargs
    ):
        """
        Initialize single-particle wavefunctions.
        
        Args:
            method: Initialization method. Options:
                - "harmonic_oscillator": Use harmonic oscillator basis
                - "woods_saxon": Use Woods-Saxon potential
                - "random": Random initialization with orthogonalization
                - "restart": Load from file
            **kwargs: Method-specific parameters
        """
        if method == "harmonic_oscillator":
            self._init_harmonic_oscillator(**kwargs)
        elif method == "random":
            self._init_random(**kwargs)
        elif method == "restart":
            self._load_from_file(**kwargs)
        else:
            raise ValueError(f"Unknown initialization method: {method}")
            
        # Orthonormalize the initial wavefunctions
        from jax_hfbfft.physics.solver import orthonormalize_states
        self.state.psi = orthonormalize_states(
            self.state.psi, 
            self._npsi[0], 
            self.grid.wxyz
        )
    
    def _init_harmonic_oscillator(
        self,
        radinx: float = 3.0,
        radiny: float = 3.0,
        radinz: float = 3.0,
        seed: int = 42,
    ):
        """
        Initialize wavefunctions using harmonic oscillator basis.
        
        Creates proper shell structure by multiplying Gaussian by polynomials
        x^i * y^j * z^k for each state, matching the FORTRAN legacy code.
        """
        nx, ny, nz = self.grid.shape
        x = self.grid.x
        y = self.grid.y
        z = self.grid.z
        
        # Create base Gaussian on 3D grid
        x_mesh = x[:, jnp.newaxis, jnp.newaxis]
        y_mesh = y[jnp.newaxis, :, jnp.newaxis]
        z_mesh = z[jnp.newaxis, jnp.newaxis, :]
        
        gaussian = jnp.exp(
            -(x_mesh / radinx)**2 - 
             (y_mesh / radiny)**2 - 
             (z_mesh / radinz)**2
        )
        
        # Normalize the base Gaussian
        norm = jnp.sqrt(jnp.sum(gaussian**2) * self.grid.wxyz)
        gaussian = gaussian / norm
        
        # Generate shell quantum numbers for each isospin
        # Pre-allocate quantum number array for the full basis
        nshell = jnp.zeros((3, self._nstmax), dtype=jnp.int32)
        
        nst = 0  # Global state counter
        
        # Loop over isospins
        for iq in range(2):
            nps = int(self._npsi[iq])
            nst_start = nst
            
            # Generate quantum numbers in shell order
            done = False
            for ka in range(nps + 10):  # ka is the shell number
                if done:
                    break
                for k in range(ka + 1):
                    if done:
                        break
                    for j in range(ka + 1):
                        if done:
                            break
                        for i in range(ka + 1):
                            if done:
                                break
                            if ka == i + j + k:  # Valid shell combination
                                for is_spin in range(2):  # Two spin states per spatial state
                                    states_in_this_isospin = nst - nst_start + 1
                                    if states_in_this_isospin > nps:
                                        done = True
                                        break
                                    
                                    if nst < self._nstmax:
                                        nshell = nshell.at[0, nst].set(i)
                                        nshell = nshell.at[1, nst].set(j)
                                        nshell = nshell.at[2, nst].set(k)
                                        nst += 1
                                    else:
                                        done = True
                                        break
        
        # Now initialize all states using shell structure
        for iq in range(2):
            if iq == 0:
                nst_start = 0
                nst_end = int(self._npsi[0])
            else:
                nst_start = int(self._npsi[0])
                nst_end = self._nstmax
            
            for nst in range(nst_start, min(nst_end, self._nstmax)):
                if nst == nst_start:
                    # Lowest state: pure Gaussian in first spin component
                    self.state.psi = self.state.psi.at[nst, 0, :, :, :].set(
                        gaussian.astype(jnp.complex128)
                    )
                    self.state.psi = self.state.psi.at[nst, 1, :, :, :].set(0.0)
                else:
                    # Higher states: Gaussian * polynomial
                    is_component = (nst - nst_start) % 2
                    
                    i_qn = int(nshell[0, nst])
                    j_qn = int(nshell[1, nst])
                    k_qn = int(nshell[2, nst])
                    
                    # Create polynomial factors
                    if i_qn == 0:
                        xx = jnp.ones_like(x)
                    else:
                        xx = x ** i_qn
                    
                    if j_qn == 0:
                        yy = jnp.ones_like(y)
                    else:
                        yy = y ** j_qn
                    
                    if k_qn == 0:
                        zz = jnp.ones_like(z)
                    else:
                        zz = z ** k_qn
                    
                    # Create 3D polynomial
                    polynomial = (xx[:, jnp.newaxis, jnp.newaxis] * 
                                  yy[jnp.newaxis, :, jnp.newaxis] * 
                                  zz[jnp.newaxis, jnp.newaxis, :])
                    
                    # Create wavefunction: Gaussian * polynomial
                    wave_func = gaussian * polynomial
                    
                    # Set in appropriate spin component
                    self.state.psi = self.state.psi.at[nst, is_component, :, :, :].set(
                        wave_func.astype(jnp.complex128)
                    )
                    self.state.psi = self.state.psi.at[nst, 1 - is_component, :, :, :].set(0.0)
                
                # Normalize
                psi_norm = jnp.sqrt(
                    jnp.sum(jnp.abs(self.state.psi[nst])**2) * self.grid.wxyz
                )
                if psi_norm > 1e-12:
                    self.state.psi = self.state.psi.at[nst].set(
                        self.state.psi[nst] / psi_norm
                    )
    
    def _init_random(self, seed: int = 42):
        """Initialize wavefunctions with random values."""
        key = jax.random.PRNGKey(seed)
        
        nx, ny, nz = self.grid.shape
        nstmax = self._nstmax
        
        # Generate random complex wavefunctions
        key, subkey = jax.random.split(key)
        psi_real = jax.random.normal(subkey, (nstmax, 2, nx, ny, nz))
        key, subkey = jax.random.split(key)
        psi_imag = jax.random.normal(subkey, (nstmax, 2, nx, ny, nz))
        
        psi = psi_real + 1j * psi_imag
        
        # Normalize each state
        for nst in range(nstmax):
            norm = jnp.sqrt(jnp.sum(jnp.abs(psi[nst])**2) * self.grid.wxyz)
            psi = psi.at[nst].set(psi[nst] / norm)
        
        self.state.psi = psi
    
    def _load_from_file(self, filename: str):
        """Load wavefunctions from a restart file."""
        # Placeholder for restart functionality
        raise NotImplementedError("Restart from file not yet implemented")
    
    def run(
        self,
        max_iterations: int = 1000,
        convergence_threshold: float = 1e-6,
        print_interval: int = 10,
        checkpoint_interval: int = 0,
        checkpoint_file: Optional[str] = None,
        use_legacy: bool = False,
    ) -> HFBFFTResults:
        """
        Run the HFB calculation.
        
        This is the main entry point for performing a self-consistent
        HFB calculation. It will iterate until convergence or until
        the maximum number of iterations is reached.
        
        Args:
            max_iterations: Maximum number of iterations
            convergence_threshold: Convergence criterion for energy fluctuation
            print_interval: Print progress every N iterations
            checkpoint_interval: Save checkpoint every N iterations (0=disabled)
            checkpoint_file: File to save checkpoints to
            use_legacy: Use the legacy implementation (False = modern OOP)
            
        Returns:
            HFBFFTResults with final energies and properties
        """
        if use_legacy:
            return self._run_with_legacy(
                max_iterations=max_iterations,
                convergence_threshold=convergence_threshold,
                print_interval=print_interval,
            )
        
        return self._run_modern(
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
            print_interval=print_interval,
        )
    
    def _run_modern(
        self,
        max_iterations: int = 1000,
        convergence_threshold: float = 1e-6,
        print_interval: int = 10,
    ) -> HFBFFTResults:
        """Run the calculation using the modern OOP implementation."""
        from jax_hfbfft.physics.solver import (
            run_hfb, SolverConfig, create_initial_state
        )
        from jax_hfbfft.physics.coulomb import CoulombSolver
        import dataclasses
        
        start_time = time.time()
        
        # Apply center-of-mass correction to h2m if zpe==0
        # This is the alternative CM correction that scales the effective mass
        # by (A-1)/A following the legacy implementation
        mass_number = self.nucleus.protons + self.nucleus.neutrons
        if self.force.zpe == 0 and mass_number > 1:
            cm_factor = (mass_number - 1.0) / mass_number
            corrected_h2m = self.force.h2m * cm_factor
            force = dataclasses.replace(self.force, h2m=corrected_h2m)
        else:
            force = self.force
        
        print(f"Starting HFB calculation for {self.nucleus}")
        print(f"Force: {self.force.name}")
        print(f"Grid: {self.grid.nx}x{self.grid.ny}x{self.grid.nz}")
        print(f"Basis size: {self._npsi[0]} neutrons, {self._npsi[1]} protons")
        print("-" * 60)
        
        # Create solver configuration
        config = SolverConfig(
            max_iterations=max_iterations,
            convergence_criterion=convergence_threshold,
            output_interval=print_interval,
            verbose=True,
        )
        
        # Prepare initial state if wavefunctions exist
        initial_state = None
        if self.state.psi is not None:
             # We need to wrap the current values in a SolverState
             # HFBFFTState and SolverState are slightly different but share core fields
             from jax_hfbfft.physics.solver import SolverState
             from jax_hfbfft.physics.densities import Densities
             from jax_hfbfft.physics.meanfield import Meanfield
             from jax_hfbfft.physics.pairing import Pairing
             from jax_hfbfft.physics.coulomb import CoulombSolver
             from jax_hfbfft.physics.energies import Energies
             
             # Map occupations to SolverState
             # In initialization, method="harmonic_oscillator" sets wocc correctly
             
             initial_state = SolverState(
                 psi=self.state.psi,
                 sp_energy=jnp.zeros(len(self.state.isospin)),
                 sp_kinetic=jnp.zeros(len(self.state.isospin)),
                 deltaf=jnp.zeros(len(self.state.isospin)),
                 wocc=self.state.wocc,
                 wguv=self.state.wguv,
                 wstates=self.state.wstates,
                 pairwg=self.state.pairwg,
                 isospin=self.state.isospin,
                 densities=Densities.zeros(self.grid.nx, self.grid.ny, self.grid.nz),
                 meanfield=Meanfield.zeros(self.grid.nx, self.grid.ny, self.grid.nz),
                 coulomb_solver=CoulombSolver.create(self.grid),
                 wcoul=jnp.zeros((self.grid.nx, self.grid.ny, self.grid.nz)),
                 energies=Energies.zeros(),
                 pairing=Pairing.zeros(),
                 iteration=0,
                 converged=False,
                 efluct=1e10
             )

        # Run the HFB solver (use 'force' with CM correction applied)
        final_state = run_hfb(
            grid=self.grid,
            force=force,
            nucleus_z=self.nucleus.protons,
            nucleus_n=self.nucleus.neutrons,
            npsi_n=int(self._npsi[0]),
            config=config,
            initial_state=initial_state,  # Added
            use_coulomb=self.include_coulomb,
        )
        
        elapsed = time.time() - start_time
        
        # Store the solver state
        self._solver_state = final_state
        
        # Calculate radii and deformations
        from jax_hfbfft.physics.energies import compute_radii
        radii = compute_radii(final_state.densities, self.grid)
        
        # Convert to HFBFFTResults
        self.results = HFBFFTResults(
            converged=final_state.converged,
            iterations=final_state.iteration,
            final_fluctuation=final_state.efluct,
            total_energy=float(final_state.energies.ehfint),
            kinetic_energy=float(final_state.energies.ehft),
            potential_energy=float(
                final_state.energies.ehf0 + 
                final_state.energies.ehf1 + 
                final_state.energies.ehf2 + 
                final_state.energies.ehf3 + 
                final_state.energies.ehfls
            ),
            pairing_energy=float(jnp.sum(final_state.pairing.epair)),
            coulomb_energy=float(final_state.energies.ehfc),
            rearrangement_energy=float(final_state.energies.e3corr),
            cm_correction=float(final_state.energies.e_zpe),
            ehf0=float(final_state.energies.ehf0),
            ehf1=float(final_state.energies.ehf1),
            ehf2=float(final_state.energies.ehf2),
            ehf3=float(final_state.energies.ehf3),
            ehfls=float(final_state.energies.ehfls),
            pairing_gap_n=float(final_state.pairing.avdelt[0]),
            pairing_gap_p=float(final_state.pairing.avdelt[1]),
            fermi_energy_n=float(final_state.pairing.eferm[0]),
            fermi_energy_p=float(final_state.pairing.eferm[1]),
            # Radii and deformation
            rms_radius_n=radii.rms_n,
            rms_radius_p=radii.rms_p,
            rms_radius_total=radii.rms_tot,
            charge_radius=radii.charge,
            beta2=radii.beta2,
            gamma=radii.gamma,
            q20=radii.q20,
            q22=radii.q22,
        )
        
        self._iteration = final_state.iteration
        self._converged = final_state.converged
        
        print("-" * 60)
        print(f"Calculation {'converged' if self._converged else 'completed'}")
        print(f"Total energy: {self.results.total_energy:.5f} MeV")
        print(f"Time elapsed: {elapsed:.2f} seconds")
        
        return self.results
    
    def _run_with_legacy(
        self,
        max_iterations: int = 1000,
        convergence_threshold: float = 1e-6,
        print_interval: int = 10,
    ) -> HFBFFTResults:
        """Run the calculation using the legacy implementation."""
        import sys
        import os
        
        # Find the legacy modules directory (workspace root)
        # Try to find it relative to this package
        package_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        
        # Add to path if not already there
        if package_dir not in sys.path:
            sys.path.insert(0, package_dir)
        
        # Also add legacy subdirectory
        legacy_dir = os.path.join(package_dir, 'legacy')
        if os.path.isdir(legacy_dir) and legacy_dir not in sys.path:
            sys.path.insert(0, legacy_dir)
        
        # Import legacy modules
        try:
            from legacy.params import init_params
            from legacy.grids import init_grids
            from legacy.forces import init_forces
            from legacy.levels import init_levels
            from legacy.densities import init_densities
            from legacy.meanfield import init_meanfield
            from legacy.energies import init_energies
            from legacy.coulomb import init_coulomb
            from legacy.static import init_static, statichf, harmosc
            from legacy.pairs import Pairs
            from legacy.moment import init_moment
            import jax.numpy as jnp
        except ImportError:
            # Fall back to non-prefixed import (for old installations)
            try:
                from params import init_params
                from grids import init_grids
                from forces import init_forces
                from levels import init_levels
                from densities import init_densities
                from meanfield import init_meanfield
                from energies import init_energies
                from coulomb import init_coulomb
                from static import init_static, statichf, harmosc
                from pairs import Pairs
                from moment import init_moment
                import jax.numpy as jnp
            except ImportError as e:
                raise ImportError(
                    f"Could not import legacy modules: {e}. "
                    f"Tried adding '{package_dir}' and '{legacy_dir}' to Python path. "
                    "Make sure the legacy code is in the 'legacy/' folder."
                )
        
        start_time = time.time()
        
        print(f"Starting HFB calculation for {self.nucleus}")
        print(f"Force: {self.force.name}")
        print(f"Grid: {self.grid.nx}x{self.grid.ny}x{self.grid.nz}")
        print(f"Basis size: {self._npsi[0]} neutrons, {self._npsi[1]} protons")
        print("-" * 60)
        
        # Build configuration for legacy code
        params_config = {
            'imode': 1,
            'mprint': print_interval,
            'tcoul': self.include_coulomb,
            'tfft': self.use_fft,
            'nof': 0,  # Number of force iterations
        }
        
        grids_config = {
            'nx': self.grid.nx,
            'ny': self.grid.ny,
            'nz': self.grid.nz,
            'dx': self.grid.dx,
            'dy': self.grid.dy,
            'dz': self.grid.dz,
        }
        
        force_config = {
            'name': self.force.name,
            'ipair': self.force.ipair,
            'v0prot': self.force.v0prot,
            'v0neut': self.force.v0neut,
            'tbcs': self.force.tbcs,
        }
        
        levels_config = {
            'nneut': self.nucleus.neutrons,
            'nprot': self.nucleus.protons,
            'npsi': list(self._npsi),
        }
        
        static_config = {
            'maxiter': max_iterations,
            'serr': convergence_threshold,
            'x0dmp': 0.45,
            'e0dmp': 100.0,
            'tvaryx_0': True,
            'radinx': 6.0,  # Initial radius for harmonic oscillator (fm)
            'radiny': 6.0,
            'radinz': 6.0,
        }
        
        # Initialize legacy objects
        params = init_params(**params_config)
        grids = init_grids(params, **grids_config)
        forces = init_forces(params, **force_config)
        levels = init_levels(grids, **levels_config)
        densities = init_densities(grids)
        meanfield = init_meanfield(grids)
        energies = init_energies()
        coulomb = init_coulomb(grids)
        forces, static = init_static(forces, levels, **static_config)
        pairs = Pairs()
        moment = init_moment(jnp.zeros(3))
        
        # Initialize wavefunctions using harmonic oscillator
        levels = harmosc(grids, levels, params, static)
        
        # Run the static HFB iteration
        result = statichf(
            coulomb, densities, energies, forces, grids, 
            levels, meanfield, moment, params, static, pairs
        )
        
        # Unpack results (statichf returns 10 values, pairs is passed in-place)
        (coulomb, densities, energies, forces, grids, 
         levels, meanfield, moment, params, static) = result
        
        elapsed = time.time() - start_time
        
        # Store legacy objects for access
        self._legacy_objects = {
            'coulomb': coulomb,
            'densities': densities,
            'energies': energies,
            'forces': forces,
            'grids': grids,
            'levels': levels,
            'meanfield': meanfield,
            'moment': moment,
            'params': params,
            'static': static,
            'pairs': pairs,
        }
        
        # Convert to HFBFFTResults
        self.results = HFBFFTResults(
            converged=float(energies.efluct1[0]) < convergence_threshold,
            iterations=int(params.iteration),
            final_fluctuation=float(energies.efluct1[0]),
            total_energy=float(energies.ehf),
            kinetic_energy=float(energies.ehft),
            potential_energy=float(energies.ehf0 + energies.ehf3),
            pairing_energy=float(pairs.epair[0] + pairs.epair[1]),
            coulomb_energy=float(energies.ehfc),
            rearrangement_energy=float(energies.e3corr),
            cm_correction=float(energies.e_zpe),
            ehf0=float(energies.ehf0),
            ehf1=float(energies.ehf1),
            ehf2=float(energies.ehf2),
            ehf3=float(energies.ehf3),
            ehfls=float(energies.ehfls),
            pairing_gap_n=float(pairs.avdelt[0]),
            pairing_gap_p=float(pairs.avdelt[1]),
            fermi_energy_n=float(pairs.eferm[0]),
            fermi_energy_p=float(pairs.eferm[1]),
        )
        
        self._iteration = int(params.iteration)
        self._converged = self.results.converged
        
        print("-" * 60)
        print(f"Calculation {'converged' if self._converged else 'completed'}")
        print(f"Total energy: {self.results.total_energy:.5f} MeV")
        print(f"Time elapsed: {elapsed:.2f} seconds")
        
        return self.results
    
    def _iterate(self) -> float:
        """
        Perform a single HFB iteration.
        
        Returns:
            Energy fluctuation for convergence check
        """
        # This is a placeholder that shows the structure
        # Real implementation would call the static.py routines
        
        # 1. Calculate densities from wavefunctions
        # self._calculate_densities()
        
        # 2. Calculate mean-field potentials
        # self._calculate_meanfield()
        
        # 3. Apply Hamiltonian and update wavefunctions
        # self._gradient_step()
        
        # 4. Orthogonalize wavefunctions
        # self._orthogonalize()
        
        # 5. Calculate pairing (if enabled)
        # self._calculate_pairing()
        
        # 6. Calculate energies
        # self._calculate_energies()
        
        # 7. Return fluctuation
        return 1.0  # Placeholder
    
    def _finalize_results(self):
        """Calculate final observables after convergence."""
        # Calculate radii
        # Calculate deformation parameters
        # Calculate angular momenta
        pass
    
    @property
    def total_energy(self) -> float:
        """Return the current total energy."""
        return self.results.total_energy
    
    @property
    def iteration(self) -> int:
        """Return the current iteration number."""
        return self._iteration
    
    @property
    def converged(self) -> bool:
        """Return whether the calculation has converged."""
        return self._converged
    
    def add_callback(self, callback: Callable):
        """
        Add a callback function to be called after each iteration.
        
        Args:
            callback: Function with signature (hfbfft, iteration, fluctuation)
        """
        self._callbacks.append(callback)
    
    def save_checkpoint(self, filename: str):
        """Save current state to a checkpoint file."""
        # Placeholder for checkpointing
        pass
    
    def load_checkpoint(self, filename: str):
        """Load state from a checkpoint file."""
        # Placeholder for checkpointing
        pass
    
    def get_density(self, isospin: int = -1) -> jax.Array:
        """
        Get the particle density.
        
        Args:
            isospin: 0 for neutrons, 1 for protons, -1 for total
            
        Returns:
            Density array with shape (nx, ny, nz)
        """
        if isospin == -1:
            return self.state.rho[0] + self.state.rho[1]
        return self.state.rho[isospin]
    
    def get_wavefunction(self, state_index: int) -> jax.Array:
        """
        Get a single-particle wavefunction.
        
        Args:
            state_index: Index of the state
            
        Returns:
            Wavefunction array with shape (2, nx, ny, nz)
        """
        return self.state.psi[state_index]
    
    def copy(self) -> "HFBFFT":
        """
        Create a deep copy of this HFBFFT instance.
        
        Returns:
            New HFBFFT instance with copied state
        """
        new_calc = HFBFFT(
            nucleus=self.nucleus,
            force=self.force,
            grid=self.grid,
            constraint=self.constraint,
            npsi=self._npsi,
            include_coulomb=self.include_coulomb,
            use_fft=self.use_fft,
        )
        
        # Deep copy state
        new_calc.state = dataclasses.replace(
            self.state,
            psi=self.state.psi.copy() if self.state.psi is not None else None,
            rho=self.state.rho.copy() if self.state.rho is not None else None,
            # ... copy other arrays ...
        )
        
        return new_calc
    
    def __repr__(self) -> str:
        return (
            f"HFBFFT(nucleus={self.nucleus}, force={self.force.name}, "
            f"grid={self.grid.shape}, iteration={self._iteration})"
        )
