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


def angular_momentum_letter(l: int) -> str:
    """
    Convert orbital angular momentum quantum number to spectroscopic letter.
    
    Args:
        l: Orbital angular momentum quantum number
        
    Returns:
        Spectroscopic letter (s, p, d, f, g, h, ...)
    """
    letters = ['s', 'p', 'd', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o']
    if l < len(letters):
        return letters[l]
    # For l >= 13, use alphabetic continuation
    return chr(ord('a') + l)


def make_spectroscopic_label(n: int, l: int, j: float) -> str:
    """
    Create spectroscopic notation for quantum numbers.
    
    Format: {n}{letter}{j_num}/{j_denom}
    Example: n=1, l=0, j=0.5 -> '1s1/2'
             n=1, l=1, j=1.5 -> '1p3/2'
    
    Args:
        n: Radial quantum number (principal quantum number for the shell)
        l: Orbital angular momentum quantum number
        j: Total angular momentum
        
    Returns:
        Spectroscopic label string
    """
    letter = angular_momentum_letter(l)
    
    # Convert j to fraction
    j_times_2 = int(round(2 * j))
    
    return f"{n}{letter}{j_times_2}/2"


def cartesian_to_spherical_qn(i: int, j: int, k: int, is_spin: int) -> Tuple[int, int, float, float]:
    """
    Convert Cartesian harmonic oscillator quantum numbers to approximate spherical quantum numbers.
    
    For Cartesian quantum numbers (i, j, k) with principal shell N = i + j + k:
    - The state contains a mixture of angular momentum values
    - We assign the maximum angular momentum l_max = N
    - Radial quantum number n = 0 (lowest radial excitation for this shell)
    - Total angular momentum j depends on l and spin
    - Parity π = (-1)^N
    
    Args:
        i, j, k: Cartesian quantum numbers
        is_spin: Spin component (0 or 1)
        
    Returns:
        (n, l, j, parity) tuple where:
            n: Radial quantum number
            l: Orbital angular momentum  
            j: Total angular momentum
            parity: Spatial parity (-1)^l
    """
    # Principal shell number
    N = i + j + k
    
    # For Cartesian states, we approximate with maximum angular momentum
    # In reality, the state contains l = N, N-2, N-4, ..., (0 or 1)
    l = N
    
    # Radial quantum number (for lowest radial excitation)
    n = 0
    
    # Total angular momentum: j = l ± 1/2
    # For l=0, only j=1/2 is possible
    if l == 0:
        j = 0.5
    else:
        # Assign j based on spin component
        # is_spin=0 -> j = l + 1/2 (stretch coupling)
        # is_spin=1 -> j = l - 1/2 (anti-stretch coupling)
        if is_spin == 0:
            j = l + 0.5
        else:
            j = l - 0.5
    
    # Parity
    parity = (-1.0) ** l
    
    return n, l, j, parity


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
    
    # Quantum numbers
    sp_n: Optional[jax.Array] = None          # Radial quantum number
    sp_l: Optional[jax.Array] = None          # Orbital angular momentum quantum number
    sp_j: Optional[jax.Array] = None          # Total angular momentum
    sp_labels: Optional[list] = None          # Spectroscopic labels
    
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
        
        # Initialize constraint (arrays built at solver runtime)
        if constraint is None:
            self.constraint = Constraint.spherical()
        else:
            self.constraint = constraint
        
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

        # Solver configuration parameters (match legacy defaults)
        self.x0dmp: float = 0.45
        self.e0dmp: float = 100.0
        self.density_mixing: float = 0.2
        self.diag_start: int = 0
        self.bcs_start: int = 0
        self.tvaryx_0: bool = False

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
        
        # Quantum numbers
        self.state.sp_n = jnp.zeros(nstmax, dtype=jnp.int32)
        self.state.sp_l = jnp.zeros(nstmax, dtype=jnp.int32)
        self.state.sp_j = jnp.full(nstmax, 0.5, dtype=jnp.float64)
        self.state.sp_labels = None
        
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
        
        # Also initialize quantum number and parity arrays
        sp_n_array = jnp.zeros(self._nstmax, dtype=jnp.int32)
        sp_l_array = jnp.zeros(self._nstmax, dtype=jnp.int32)
        sp_j_array = jnp.zeros(self._nstmax, dtype=jnp.float64)
        sp_parity_array = jnp.zeros(self._nstmax, dtype=jnp.float64)
        sp_labels_list = []
        
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
                                        
                                        # Calculate spectroscopic quantum numbers
                                        n_qn, l_qn, j_qn, parity = cartesian_to_spherical_qn(i, j, k, is_spin)
                                        sp_n_array = sp_n_array.at[nst].set(n_qn)
                                        sp_l_array = sp_l_array.at[nst].set(l_qn)
                                        sp_j_array = sp_j_array.at[nst].set(j_qn)
                                        sp_parity_array = sp_parity_array.at[nst].set(parity)
                                        
                                        # Generate spectroscopic label
                                        label = make_spectroscopic_label(n_qn, l_qn, j_qn)
                                        sp_labels_list.append(label)
                                        
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
        
        # Store quantum numbers and labels in state
        self.state.sp_n = sp_n_array
        self.state.sp_l = sp_l_array
        self.state.sp_j = sp_j_array
        self.state.sp_parity = sp_parity_array
        self.state.sp_labels = sp_labels_list
    
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
        seed: int = 42,
        save_dir: Optional[str] = None,
        save_interval: int = 5,
    ) -> HFBFFTResults:
        """
        Run the HFB calculation.

        Args:
            max_iterations: Maximum number of iterations
            convergence_threshold: Convergence criterion for energy fluctuation
            print_interval: Print progress every N iterations
            checkpoint_interval: Save checkpoint every N iterations (0=disabled)
            checkpoint_file: File to save checkpoints to
            use_legacy: Use the legacy implementation (False = modern OOP)
            seed: Random seed for reproducibility (default: 42)
            save_dir: Directory to write progress.txt checkpoints (None = disabled)
            save_interval: Write checkpoint every N iterations (default 5)

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
            seed=seed,
            save_dir=save_dir,
            save_interval=save_interval,
        )
    
    def _run_modern(
        self,
        max_iterations: int = 1000,
        convergence_threshold: float = 1e-6,
        print_interval: int = 10,
        seed: int = 42,
        save_dir: Optional[str] = None,
        save_interval: int = 5,
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
            x0dmp=self.x0dmp,
            e0dmp=self.e0dmp,
            density_mixing=self.density_mixing,
            diag_start=self.diag_start,
            bcs_start=self.bcs_start,
            tvaryx_0=self.tvaryx_0,
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
             from jax_hfbfft.physics.constraints import build_constraint_state
             
             # Map occupations to SolverState
             # In initialization, method="harmonic_oscillator" sets wocc correctly
             
             initial_state = SolverState(
                 psi=self.state.psi,
                 lagrange=self.state.lagrange,
                 sp_energy=jnp.zeros(len(self.state.isospin)),
                 sp_kinetic=jnp.zeros(len(self.state.isospin)),
                 deltaf=jnp.zeros(len(self.state.isospin)),
                 sp_n=self.state.sp_n if self.state.sp_n is not None else jnp.zeros(len(self.state.isospin), dtype=jnp.int32),
                 sp_l=self.state.sp_l if self.state.sp_l is not None else jnp.zeros(len(self.state.isospin), dtype=jnp.int32),
                 sp_j=self.state.sp_j if self.state.sp_j is not None else jnp.full(len(self.state.isospin), 0.5),
                 wocc=self.state.wocc,
                 wguv=self.state.wguv,
                 wstates=self.state.wstates,
                 pairwg=self.state.pairwg,
                 isospin=self.state.isospin,
                 sp_parity=self.state.sp_parity if self.state.sp_parity is not None else jnp.ones(len(self.state.isospin)),
                 densities=Densities.zeros(self.grid.nx, self.grid.ny, self.grid.nz),
                 meanfield=Meanfield.zeros(self.grid.nx, self.grid.ny, self.grid.nz),
                 coulomb_solver=CoulombSolver.create(self.grid),
                 wcoul=jnp.zeros((self.grid.nx, self.grid.ny, self.grid.nz)),
                 energies=Energies.zeros(),
                 pairing=Pairing.zeros(),
                 constraint_state=build_constraint_state(
                     self.constraint,
                     self.grid,
                     mass_number=self.nucleus.mass_number,
                 ),
                 iteration=0,
                 converged=False,
                 efluct=1e10
             )

        # Build checkpoint writer if save_dir is provided
        if save_dir is not None:
            from pathlib import Path
            import math as _math

            _sd = Path(save_dir)
            _save_path    = _sd / "progress.txt"
            _conver_path  = _sd / "conver.res"
            _energ_path   = _sd / "energies.res"
            _mono_path    = _sd / "monopoles.res"
            _quad_path    = _sd / "quadrupoles.res"
            _dip_path     = _sd / "dipoles.res"
            _spin_path    = _sd / "spin.res"
            _mom_path     = _sd / "momenta.res"

            with open(_save_path, 'w') as _f:
                _f.write(f"{'iter':>6}  {'E_total':>14}  {'tke':>12}  "
                         f"{'gap_n':>8}  {'gap_p':>8}  {'efluct':>10}\n")

            with open(_conver_path, 'w') as _f:
                _f.write(
                    "# Iter   Energy  d_Energy    sp_fluct    max(lam-)  "
                    "   rms(lam-)      rms    beta2  gamma      x_0  "
                    "   e_pair(1)    e_pair(2)\n"
                )

            with open(_energ_path, 'w') as _f:
                _f.write(
                    "# Iter    N(n)    N(p)       E(sum)         E(integ)"
                    "       Ekin         E_Coul         ehfCrho0  "
                    "     ehfCrho1       ehfCdrho0      ehfCdrho1"
                    "     ehfCtau0       ehfCtau1       ehfCdJ0  "
                    "      ehfCdJ1       e_pair(1)    e_pair(2)      e_zpe\n"
                )

            with open(_mono_path, 'w') as _f:
                _f.write(
                    "# Iter    RMS_n     RMS_p     RMS_tot   RMS_diff"
                    "   N_dens    Z_dens    A_dens\n"
                )

            with open(_quad_path, 'w') as _f:
                _f.write(
                    "# Iter    Q20_n     Q20_p     Q20_tot   Q22_tot"
                    "   <x²>_tot  <y²>_tot  <z²>_tot  Beta      Gamma\n"
                )

            with open(_dip_path, 'w') as _f:
                _f.write(
                    "# Iter    c.m. x-y-z"
                    "                                  Isovector dipoles x-y-z\n"
                )

            with open(_spin_path, 'w') as _f:
                _f.write(
                    "# Iter      Lx        Ly        Lz"
                    "        Sx        Sy        Sz        Jx        Jy        Jz\n"
                )

            with open(_mom_path, 'w') as _f:
                _f.write(
                    "# Iter    Px_n      Py_n      Pz_n"
                    "      Px_p      Py_p      Pz_p      Px_tot    Py_tot    Pz_tot\n"
                )

            _grid = self.grid
            _N = self.nucleus.neutrons
            _Z = self.nucleus.protons
            _A = _N + _Z
            _prev_ehf = [None]

            def _density_moments(state):
                rho_n  = state.densities.rho[0]
                rho_p  = state.densities.rho[1]
                rho    = rho_n + rho_p
                X = _grid.x[:, None, None]
                Y = _grid.y[None, :, None]
                Z = _grid.z[None, None, :]
                r2 = X**2 + Y**2 + Z**2
                wxyz = _grid.wxyz

                # Particle numbers from density integrals
                N_d = float(jnp.sum(rho_n) * wxyz)
                Z_d = float(jnp.sum(rho_p) * wxyz)
                A_d = float(jnp.sum(rho)   * wxyz)
                A_use = A_d if A_d > 1.0 else _A

                # RMS radii
                rms_n   = float(jnp.sqrt(jnp.maximum(jnp.sum(rho_n * r2) * wxyz / max(N_d, 1e-10), 0.0)))
                rms_p   = float(jnp.sqrt(jnp.maximum(jnp.sum(rho_p * r2) * wxyz / max(Z_d, 1e-10), 0.0)))
                rms_tot = float(jnp.sqrt(jnp.maximum(jnp.sum(rho   * r2) * wxyz / A_use,           0.0)))
                rms_diff = rms_n - rms_p

                # Quadrupole moments
                Q20_n   = float(jnp.sum(rho_n * (2*Z**2 - X**2 - Y**2)) * wxyz)
                Q20_p   = float(jnp.sum(rho_p * (2*Z**2 - X**2 - Y**2)) * wxyz)
                Q20_tot = Q20_n + Q20_p
                Q22_tot = float(jnp.sum(rho   * (X**2 - Y**2)) * wxyz)
                x2      = float(jnp.sum(rho * X**2) * wxyz / A_use)
                y2      = float(jnp.sum(rho * Y**2) * wxyz / A_use)
                z2      = float(jnp.sum(rho * Z**2) * wxyz / A_use)

                # Deformation
                r0   = 1.2
                norm = 4.0 * _math.pi / (3.0 * _A * (r0 * _A**(1.0/3.0))**2)
                beta2 = norm * _math.sqrt(Q20_tot**2 + 2.0*Q22_tot**2)
                gamma = _math.degrees(_math.atan2(_math.sqrt(2.0)*Q22_tot, Q20_tot)) % 60.0

                # Dipole moments
                # c.m. = isoscalar (rho weighted), Isovector = (rho_n - rho_p) weighted
                cm_x = float(jnp.sum(rho * X) * wxyz / A_use)
                cm_y = float(jnp.sum(rho * Y) * wxyz / A_use)
                cm_z = float(jnp.sum(rho * Z) * wxyz / A_use)
                iv_x = float(jnp.sum((rho_n - rho_p) * X) * wxyz)
                iv_y = float(jnp.sum((rho_n - rho_p) * Y) * wxyz)
                iv_z = float(jnp.sum((rho_n - rho_p) * Z) * wxyz)

                # Angular momenta from current j and spin density s
                jc_n = state.densities.current[0]  # (3, nx, ny, nz)
                jc_p = state.densities.current[1]
                jc   = jc_n + jc_p
                # Orbital L = integral(r × j)
                Lx = float(jnp.sum(Y * jc[2] - Z * jc[1]) * wxyz)
                Ly = float(jnp.sum(Z * jc[0] - X * jc[2]) * wxyz)
                Lz = float(jnp.sum(X * jc[1] - Y * jc[0]) * wxyz)
                # Spin S = integral(s)
                s_tot = state.densities.sdens[0] + state.densities.sdens[1]
                Sx = float(jnp.sum(s_tot[0]) * wxyz)
                Sy = float(jnp.sum(s_tot[1]) * wxyz)
                Sz = float(jnp.sum(s_tot[2]) * wxyz)

                # Linear momenta P = integral(j)
                Px_n = float(jnp.sum(jc_n[0]) * wxyz)
                Py_n = float(jnp.sum(jc_n[1]) * wxyz)
                Pz_n = float(jnp.sum(jc_n[2]) * wxyz)
                Px_p = float(jnp.sum(jc_p[0]) * wxyz)
                Py_p = float(jnp.sum(jc_p[1]) * wxyz)
                Pz_p = float(jnp.sum(jc_p[2]) * wxyz)

                return dict(
                    N_d=N_d, Z_d=Z_d, A_d=A_d,
                    rms_n=rms_n, rms_p=rms_p, rms_tot=rms_tot, rms_diff=rms_diff,
                    Q20_n=Q20_n, Q20_p=Q20_p, Q20_tot=Q20_tot, Q22_tot=Q22_tot,
                    x2=x2, y2=y2, z2=z2, beta2=beta2, gamma=gamma,
                    cm_x=cm_x, cm_y=cm_y, cm_z=cm_z,
                    iv_x=iv_x, iv_y=iv_y, iv_z=iv_z,
                    Lx=Lx, Ly=Ly, Lz=Lz, Sx=Sx, Sy=Sy, Sz=Sz,
                    Px_n=Px_n, Py_n=Py_n, Pz_n=Pz_n,
                    Px_p=Px_p, Py_p=Py_p, Pz_p=Pz_p,
                )

            def _bcs_particle_numbers(state):
                n_n = float(jnp.sum(jnp.where(state.isospin == 0,
                                               state.wocc * state.wstates, 0.0)))
                n_p = float(jnp.sum(jnp.where(state.isospin == 1,
                                               state.wocc * state.wstates, 0.0)))
                return n_n, n_p

            def write_checkpoint(state):
                try:
                    ehf  = float(state.energies.ehf)
                    d_ehf = (ehf - _prev_ehf[0]) if _prev_ehf[0] is not None else 0.0
                    _prev_ehf[0] = ehf
                    n_n, n_p = _bcs_particle_numbers(state)
                    m = _density_moments(state)
                    it = state.iteration

                    with open(_save_path, 'a') as _f:
                        _f.write(
                            f"{it:6d}  "
                            f"{ehf:14.4f}  "
                            f"{float(state.energies.tke):12.4f}  "
                            f"{float(state.pairing.avdelt[0]):8.4f}  "
                            f"{float(state.pairing.avdelt[1]):8.4f}  "
                            f"{float(state.efluct):.3e}\n"
                        )

                    with open(_conver_path, 'a') as _f:
                        _f.write(
                            f"{it:6d}"
                            f"  {ehf:8.2f}"
                            f"  {d_ehf:8.2f}"
                            f"  {float(state.energies.efluct1[0]):11.3G}"
                            f"  {float(state.efluct):11.3E}"
                            f"  {float(state.energies.efluct2[0]):11.3E}"
                            f"  {m['rms_tot']:8.3f}"
                            f"  {m['beta2']:7.4f}"
                            f"  {m['gamma']:6.1f}"
                            f"  {float(state.x0dmp):6.3f}"
                            f"  {float(state.pairing.epair[0]):9.3f}"
                            f"  {float(state.pairing.epair[1]):9.3f}\n"
                        )

                    with open(_energ_path, 'a') as _f:
                        e = state.energies
                        _f.write(
                            f"{it:6d}"
                            f"  {n_n:7.3f}"
                            f"  {n_p:7.3f}"
                            f"  {ehf:15.7f}"
                            f"  {float(e.ehfint):15.7f}"
                            f"  {float(e.tke):15.7f}"
                            f"  {float(e.ehfc):15.7f}"
                            f"  {float(e.ehfCrho0):15.7f}"
                            f"  {float(e.ehfCrho1):15.7f}"
                            f"  {float(e.ehfCdrho0):15.7f}"
                            f"  {float(e.ehfCdrho1):15.7f}"
                            f"  {float(e.ehfCtau0):15.7f}"
                            f"  {float(e.ehfCtau1):15.7f}"
                            f"  {float(e.ehfCdJ0):15.7f}"
                            f"  {float(e.ehfCdJ1):15.7f}"
                            f"  {float(e.epair[0]):13.7f}"
                            f"  {float(e.epair[1]):13.7f}"
                            f"  {float(e.e_zpe):13.7f}\n"
                        )

                    with open(_mono_path, 'a') as _f:
                        _f.write(
                            f"{it:6d}"
                            f"  {m['rms_n']:9.6f}"
                            f"  {m['rms_p']:9.6f}"
                            f"  {m['rms_tot']:9.6f}"
                            f"  {m['rms_diff']:9.6f}"
                            f"   {m['N_d']:8.3f}"
                            f"   {m['Z_d']:8.3f}"
                            f"   {m['A_d']:8.3f}\n"
                        )

                    with open(_quad_path, 'a') as _f:
                        _f.write(
                            f"{it:6d}"
                            f"  {m['Q20_n']:9.6f}"
                            f"  {m['Q20_p']:9.6f}"
                            f"  {m['Q20_tot']:9.6f}"
                            f"  {m['Q22_tot']:9.6f}"
                            f"  {m['x2']:9.6f}"
                            f"  {m['y2']:9.6f}"
                            f"  {m['z2']:9.6f}"
                            f"  {m['beta2']:8.6f}"
                            f"  {m['gamma']:8.2f}\n"
                        )

                    with open(_dip_path, 'a') as _f:
                        _f.write(
                            f"{it:6d}"
                            f"  {m['cm_x']:12.7f}"
                            f"  {m['cm_y']:12.7f}"
                            f"  {m['cm_z']:12.7f}"
                            f"  {m['iv_x']:12.7f}"
                            f"  {m['iv_y']:12.7f}"
                            f"  {m['iv_z']:12.7f}\n"
                        )

                    with open(_spin_path, 'a') as _f:
                        Jx = m['Lx'] + m['Sx']
                        Jy = m['Ly'] + m['Sy']
                        Jz = m['Lz'] + m['Sz']
                        _f.write(
                            f"{it:6d}"
                            f"  {m['Lx']:9.6f}"
                            f"  {m['Ly']:9.6f}"
                            f"  {m['Lz']:9.6f}"
                            f"  {m['Sx']:9.6f}"
                            f"  {m['Sy']:9.6f}"
                            f"  {m['Sz']:9.6f}"
                            f"  {Jx:9.6f}"
                            f"  {Jy:9.6f}"
                            f"  {Jz:9.6f}\n"
                        )

                    with open(_mom_path, 'a') as _f:
                        _f.write(
                            f"{it:6d}"
                            f"  {m['Px_n']:9.6f}"
                            f"  {m['Py_n']:9.6f}"
                            f"  {m['Pz_n']:9.6f}"
                            f"  {m['Px_p']:9.6f}"
                            f"  {m['Py_p']:9.6f}"
                            f"  {m['Pz_p']:9.6f}"
                            f"  {m['Px_n']+m['Px_p']:9.6f}"
                            f"  {m['Py_n']+m['Py_p']:9.6f}"
                            f"  {m['Pz_n']+m['Pz_p']:9.6f}\n"
                        )

                except Exception as ex:
                    print(f"Checkpoint write error: {ex}")

            def write_final(state, config):
                """Write once-at-end summary files."""
                try:
                    with open(_sd / "energies.txt", 'w') as _f:
                        _f.write(f"Total energy: {float(state.energies.ehf):.6f} MeV\n")
                        _f.write(f"Iterations: {state.iteration}\n")
                        _f.write(f"Convergence: {state.efluct:.6e}\n")
                        _f.write(f"Target convergence: {config.convergence_criterion:.6e}\n")
                except Exception as ex:
                    print(f"energies.txt write error: {ex}")

                try:
                    with open(_sd / "sp_energies.txt", 'w') as _f:
                        _f.write("# idx  isospin  energy(MeV)  occupation  sp_norm\n")
                        nst = state.psi.shape[0]
                        norms = jnp.sum(jnp.real(state.psi * jnp.conj(state.psi)),
                                        axis=(1, 2, 3, 4)) * _grid.wxyz
                        for i in range(nst):
                            _f.write(
                                f"{i:5d}  {int(state.isospin[i]):1d}"
                                f"  {float(state.sp_energy[i]):12.6f}"
                                f"  {float(state.wocc[i]):10.6f}"
                                f"  {float(norms[i]):10.6f}\n"
                            )
                except Exception as ex:
                    print(f"sp_energies.txt write error: {ex}")

            preloop_cb = write_checkpoint
        else:
            write_checkpoint = None
            write_final = None
            preloop_cb = None

        # Create a wrapper callback that invokes all registered callbacks and periodic saves
        def iteration_callback(state):
            """Wrapper that calls all registered callbacks."""
            for cb in self._callbacks:
                try:
                    cb(
                        state.iteration,
                        float(state.energies.ehf),
                        float(state.efluct)
                    )
                except Exception as e:
                    print(f"Callback error: {e}")
            if write_checkpoint is not None and state.iteration % save_interval == 0:
                write_checkpoint(state)

        # Run the HFB solver (use 'force' with CM correction applied)
        final_state = run_hfb(
            grid=self.grid,
            force=force,
            nucleus_z=self.nucleus.protons,
            nucleus_n=self.nucleus.neutrons,
            npsi_n=int(self._npsi[0]),
            config=config,
            initial_state=initial_state,  # Added
            callback=iteration_callback if (self._callbacks or write_checkpoint is not None) else None,
            preloop_callback=preloop_cb,
            use_coulomb=self.include_coulomb,
            constraint=self.constraint,
            seed=seed,
        )
        
        elapsed = time.time() - start_time

        if write_final is not None:
            write_final(final_state, config)

        # Store the solver state
        self._solver_state = final_state
        
        # Calculate radii and deformations
        from jax_hfbfft.physics.energies import compute_radii
        from jax_hfbfft.physics.constraints import compute_constraint_expectations
        radii = compute_radii(final_state.densities, self.grid)
        
        # Convert to HFBFFTResults
        self.results = HFBFFTResults(
            converged=final_state.converged,
            iterations=final_state.iteration,
            final_fluctuation=final_state.efluct,
            total_energy=float(final_state.energies.ehf),
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

        # If constrained, report Q20/Q22 and beta2 from constraint expectations
        if self.constraint and self.constraint.tconstraint and getattr(final_state, "constraint_state", None) is not None:
            expectations = compute_constraint_expectations(final_state.constraint_state, final_state.densities, self.grid)
            if expectations.shape[0] > 0:
                multipoles = [mp for mp, _ in self.constraint.get_multipole_list()]
                if (2, 0) in multipoles:
                    q20_index = multipoles.index((2, 0))
                    q20_val = float(expectations[q20_index])
                    A = float(self.nucleus.mass_number)
                    R0 = 1.2 * (A ** (1.0 / 3.0))
                    beta2 = (jnp.sqrt(5 * jnp.pi) / (3 * A * R0**2 + 1e-10)) * q20_val
                    self.results.q20 = q20_val
                    self.results.beta2 = float(beta2)
                if (2, 2) in multipoles:
                    q22_index = multipoles.index((2, 2))
                    q22_val = float(expectations[q22_index])
                    self.results.q22 = q22_val
                    self.results.gamma = float(jnp.arctan2(jnp.sqrt(3.0) * q22_val, self.results.q20) * 180.0 / jnp.pi)
        
        self._iteration = final_state.iteration
        self._converged = final_state.converged
        
        # Store the final densities in self.state for later access (e.g., visualization)
        self.state.rho = final_state.densities.rho
        self.state.tau = final_state.densities.tau
        self.state.psi = final_state.psi
        self.state.sp_energy = final_state.sp_energy
        self.state.wocc = final_state.wocc
        
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
            'tvaryx_0': False,
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
