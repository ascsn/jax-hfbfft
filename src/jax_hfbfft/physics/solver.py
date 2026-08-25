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
from functools import partial

from jax_hfbfft.jax_config import get_dtypes
from jax_hfbfft.core.grid import Grid
from jax_hfbfft.core.force import Force
from jax_hfbfft.core.constraint import Constraint
from jax_hfbfft.physics.densities import Densities, compute_densities
from jax_hfbfft.physics.meanfield import Meanfield, compute_skyrme_meanfield, apply_hfb_hamiltonian
from jax_hfbfft.physics.energies import Energies, compute_integrated_energy, compute_sp_energy, print_sinfo
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
    lagrange: jax.Array    # Shape: (nstates, 2, nx, ny, nz) — off-diagonal H term
    
    # Single-particle properties
    sp_energy: jax.Array   # Single-particle energies (nstates,)
    sp_kinetic: jax.Array  # Kinetic energies (nstates,)
    deltaf: jax.Array      # Pairing gaps (nstates,)
    
    # Quantum numbers
    sp_n: jax.Array        # Radial quantum number (nstates,)
    sp_l: jax.Array        # Orbital angular momentum (nstates,)
    sp_j: jax.Array        # Total angular momentum (nstates,)
    
    # Occupation factors
    wocc: jax.Array        # BCS occupation v^2 (nstates,)
    wguv: jax.Array        # BCS uv factor (nstates,)
    wstates: jax.Array     # State weights (nstates,)
    pairwg: jax.Array      # Pairing cutoff weights (nstates,)
    
    # State metadata
    isospin: jax.Array     # 0=neutron, 1=proton (nstates,)
    sp_parity: jax.Array   # Parity (-1)^l (nstates,)
    
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
    x0dmp: float = 0.45   # Current gradient step damping (tracks tvaryx_0 adaptation)
    
    @property
    def nstates(self) -> int:
        return self.psi.shape[0]
    
    @property
    def nneut(self) -> jax.Array:
        return jnp.sum(self.isospin == 0)
    
    @property
    def nprot(self) -> jax.Array:
        return jnp.sum(self.isospin == 1)
    
    def get_spectroscopic_labels(self) -> list:
        """
        Generate spectroscopic labels from quantum numbers.
        
        Returns:
            List of spectroscopic labels (e.g., '1s1/2', '1p3/2')
        """
        # Import locally to avoid circular dependency
        from jax_hfbfft.core.hfbfft import make_spectroscopic_label
        
        labels = []
        for i in range(self.nstates):
            n = int(self.sp_n[i])
            l = int(self.sp_l[i])
            j = float(self.sp_j[i])
            label = make_spectroscopic_label(n, l, j)
            labels.append(label)
        return labels


@jax.tree_util.register_dataclass
@dataclass
class SolverConfig:
    """Configuration for the HFB solver."""
    max_iterations: int = 200
    convergence_criterion: float = 1e-6

    # Damping and mixing
    x0dmp: float = 0.45          # Gradient step damping
    e0dmp: float = 100.0           # Preconditioning energy (MeV)
    density_mixing: float = 0.2   # New density fraction

    # Iteration control
    diag_start: int = 30          # Start diagonalization after this
    bcs_start: int = 30           # Use pure BCS until this iteration

    # Pairing annealing (FORTRAN static.f90:563-572): for the first `iteranneal`
    # iterations the pairing strengths V0 are enhanced by a factor decaying from
    # (1 + pairenhance) at iter 0 down to 1 at iter=iteranneal, then held at 1.
    # This over-pairs early to prevent premature collapse to the trivial gap.
    # iteranneal = 0 disables it (FORTRAN default), giving no behavioral change.
    iteranneal: int = 0
    pairenhance: float = 0.0

    # Adaptive step (tvaryx_0): triple x0dmp before main loop, adapt each iteration
    # Matches legacy static.tvaryx_0 behavior
    tvaryx_0: bool = field(default=False, metadata=dict(static=True))

    # Output
    output_interval: int = 5
    sinfo_interval: int = field(default=50, metadata=dict(static=True))
    verbose: bool = True


def initialize_wavefunctions(
    grid: Grid,
    nneut: int,
    nprot: int,
    nstates_neut: int,
    nstates_prot: int,
    radinx: float = 3.1,
    radiny: float = 3.1,
    radinz: float = 3.1,
    seed: int = 42,
) -> Tuple[jax.Array, ...]:
    """
    Initialize wavefunctions with harmonic oscillator states.

    Follows FORTRAN harmosc: lowest state is a Gaussian, higher states
    multiply the Gaussian by monomials x^i * y^j * z^k ordered by shell
    ka = i+j+k, with two spin components per (i,j,k) triplet.

    Args:
        grid: Spatial grid
        nneut, nprot: Particle numbers
        nstates_neut, nstates_prot: Number of basis states per isospin
        radinx, radiny, radinz: Oscillator widths in each direction (fm)
        seed: Unused (kept for API compatibility)

    Returns:
        (psi, isospin, sp_n, sp_l, sp_j, sp_parity, wocc, wguv)
    """
    dtypes = get_dtypes()
    nstates = nstates_neut + nstates_prot

    # --- Build shell quantum numbers (i, j, k) ---
    # FORTRAN: for each isospin, loop ka=0,1,2,... and assign pairs of
    # spin states to each (i,j,k) triplet where i+j+k == ka.
    def build_nshell(nps: int) -> list:
        """Return list of (i,j,k) shell indices, length nps."""
        shells = []
        for ka in range(nps + 10):  # upper bound safely exceeds nps
            for k in range(ka + 1):
                for j in range(ka + 1):
                    for i in range(ka + 1):
                        if i + j + k == ka:
                            # Two spin states per (i,j,k) triplet
                            shells.append((i, j, k))
                            shells.append((i, j, k))
                            if len(shells) >= nps:
                                return shells[:nps]
        return shells[:nps]

    nshell_neut = build_nshell(nstates_neut)
    nshell_prot = build_nshell(nstates_prot)

    # --- Build the base Gaussian (shared for all states in an isospin) ---
    x = grid.x[:, None, None]   # (nx, 1, 1)
    y = grid.y[None, :, None]   # (1, ny, 1)
    z = grid.z[None, None, :]   # (1, 1, nz)

    gaussian = jnp.exp(-((x / radinx)**2 + (y / radiny)**2 + (z / radinz)**2))
    # Normalize: norm = sum(|psi|^2) * wxyz over all spinors
    anorm = jnp.sum(gaussian**2) * 2.0 * grid.wxyz  # factor 2 for two spinor components
    gaussian = gaussian / jnp.sqrt(anorm)

    # --- Build each wavefunction: psi[nst, spinor, nx, ny, nz] ---
    # FORTRAN: spin component is determined by position within the isospin block
    #   is = mod(nst - npmin(iq), 2) + 1   (1-indexed)
    # i.e. within each isospin: state 0 -> spinor 0, state 1 -> spinor 1, repeating.
    psi_list = []

    for block_idx, (nshell_block, nps, offset) in enumerate([
        (nshell_neut, nstates_neut, 0),
        (nshell_prot, nstates_prot, nstates_neut),
    ]):
        for local_idx, (si, sj, sk) in enumerate(nshell_block):
            spinor = local_idx % 2  # 0 or 1, matching FORTRAN is=mod(...,2)+1

            # Polynomial factor: x^si * y^sj * z^sk (1 if exponent is 0)
            poly_x = x**si if si != 0 else jnp.ones_like(x)
            poly_y = y**sj if sj != 0 else jnp.ones_like(y)
            poly_z = z**sk if sk != 0 else jnp.ones_like(z)
            poly   = poly_x * poly_y * poly_z  # broadcasts to (nx, ny, nz)

            state = jnp.zeros((2, grid.nx, grid.ny, grid.nz), dtype=dtypes.complex)
            raw   = (gaussian * poly).astype(dtypes.complex)

            # Normalize this state before inserting
            norm  = jnp.sqrt(jnp.sum(jnp.abs(raw)**2) * grid.wxyz)
            raw   = raw / norm

            state = state.at[spinor].set(raw)
            psi_list.append(state)

    psi = jnp.stack(psi_list, axis=0)  # (nstates, 2, nx, ny, nz)

    # --- Isospin labels ---
    isospin = jnp.concatenate([
        jnp.zeros(nstates_neut, dtype=dtypes.int),
        jnp.ones(nstates_prot,  dtype=dtypes.int),
    ])

    # --- Occupations: lowest nneut neutron and nprot proton states occupied ---
    wocc = jnp.zeros(nstates, dtype=dtypes.float)
    wocc = wocc.at[:nneut].set(1.0)
    wocc = wocc.at[nstates_neut : nstates_neut + nprot].set(1.0)
    wguv = jnp.zeros(nstates, dtype=dtypes.float)

    # --- Spectroscopic quantum numbers (n, l, j, parity) ---
    # The FORTRAN code carries no n/l/j quantum numbers: coordinate-space states
    # are not eigenstates of angular momentum. These are approximate labels used
    # only for display (e.g. "1p3/2"), assigned with the same Cartesian->spherical
    # mapping as the HFBFFT class so both code paths produce identical labels.
    from jax_hfbfft.core.hfbfft import cartesian_to_spherical_qn  # local: avoid circular import

    sp_n_list, sp_l_list, sp_j_list, sp_parity_list = [], [], [], []
    for nshell_block in (nshell_neut, nshell_prot):
        for local_idx, (si, sj, sk) in enumerate(nshell_block):
            is_spin = local_idx % 2  # matches wavefunction spin assignment above
            n_qn, l_qn, j_qn, parity = cartesian_to_spherical_qn(si, sj, sk, is_spin)
            sp_n_list.append(n_qn)
            sp_l_list.append(l_qn)
            sp_j_list.append(j_qn)
            sp_parity_list.append(parity)

    sp_n      = jnp.array(sp_n_list, dtype=dtypes.int)
    sp_l      = jnp.array(sp_l_list, dtype=dtypes.int)
    sp_j      = jnp.array(sp_j_list, dtype=dtypes.float)
    sp_parity = jnp.array(sp_parity_list, dtype=dtypes.float)

    return psi, isospin, sp_n, sp_l, sp_j, sp_parity, wocc, wguv

def create_initial_state(
    grid: Grid,
    npsi_n: int,
    npsi_p: int,
    target_n: int,
    target_p: int,
    constraint_state: Optional[ConstraintState] = None,
    seed: int = 42,
) -> SolverState:
    """
    Create initial HFB state.
    
    Args:
        grid: Spatial grid
        npsi_n, npsi_p: Number of states per isospin
        target_n, target_p: Target particle numbers
        constraint_state: Constraint configuration
        seed: Random seed for reproducibility (default: 42)
        
    Returns:
        Initial solver state
    """
    dtypes = get_dtypes()
    nstates = npsi_n + npsi_p
    A = target_n + target_p
    
    # Initialize wavefunctions
    psi, isospin, sp_n, sp_l, sp_j, sp_parity, wocc, wguv = initialize_wavefunctions(
        grid, target_n, target_p, npsi_n, npsi_p, seed=seed
    )
    
    # FORTRAN statichf Step 3: sp_energy zeroed before initial grstep
    # grstep will compute the real values; don't prefill with HO estimates.
    sp_energy  = jnp.zeros(nstates, dtype=dtypes.float)
    sp_kinetic = jnp.zeros(nstates, dtype=dtypes.float)

    # deltaf also starts at zero; pair() will populate it after the first grstep
    deltaf = jnp.zeros(nstates, dtype=dtypes.float)

    pairwg  = jnp.ones(nstates,  dtype=dtypes.float)
    wstates = jnp.ones(nstates,  dtype=dtypes.float)

    densities      = Densities.zeros(grid.nx, grid.ny, grid.nz)
    meanfield      = Meanfield.zeros(grid.nx, grid.ny, grid.nz)
    coulomb_solver = CoulombSolver.create(grid)
    wcoul          = jnp.zeros((grid.nx, grid.ny, grid.nz), dtype=dtypes.float)
    
    lagrange = jnp.zeros_like(psi)

    if constraint_state is None:
        constraint_state = ConstraintState.disabled(grid)

    return SolverState(
        psi=psi,
        lagrange = lagrange,
        sp_energy=sp_energy,
        sp_kinetic=sp_kinetic,
        deltaf=deltaf,
        sp_n=sp_n,
        sp_l=sp_l,
        sp_j=sp_j,
        wocc=wocc,
        wguv=wguv,
        wstates=wstates,
        pairwg=pairwg,
        isospin=isospin,
        sp_parity=sp_parity,
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
        x0dmp=0.45,
    )


def compute_densities_from_state(state: SolverState, grid: Grid) -> Densities:
    """Compute densities from current wavefunctions and occupations."""
    return compute_densities(
        state.psi,
        state.wocc,
        state.wguv,
        state.pairwg,
        state.wstates,
        state.isospin,
        grid,
    )


def gradient_step(
    psi: jax.Array,
    meanfield: Meanfield,
    wocc: jax.Array,
    wguv: jax.Array,
    pairwg: jax.Array,
    wstates: jax.Array,
    sp_energy: jax.Array,
    isospin: jax.Array,
    lagrange: jax.Array,    # (nstates, 2, nx, ny, nz)
    grid: Grid,
    x0dmp: float,
    e0dmp: float,
    h2ma: float,
    npsi_n: int,
    use_pairing: bool = True,
    tbcs: bool = True,
    use_lagrange: bool = False,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    return _gradient_step_jit(
        psi, meanfield, wocc, wguv, pairwg, wstates, sp_energy,
        isospin, lagrange, grid, x0dmp, e0dmp, h2ma,
        npsi_n, use_pairing, tbcs, use_lagrange,
    )


@partial(jax.jit, static_argnums=(13, 14, 15, 16))
def _gradient_step_jit(
    psi: jax.Array,
    meanfield: Meanfield,
    wocc: jax.Array,
    wguv: jax.Array,
    pairwg: jax.Array,
    wstates: jax.Array,
    sp_energy: jax.Array,
    isospin: jax.Array,
    lagrange: jax.Array,    # (nstates, 2, nx, ny, nz)
    grid: Grid,
    x0dmp: float,
    e0dmp: float,
    h2ma: float,
    npsi_n: int,            # static 12
    use_pairing: bool,      # static 13
    tbcs: bool,             # static 14
    use_lagrange: bool,     # static 15
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """JIT-compiled preconditioned gradient step."""
    nstates = psi.shape[0]

    # ── 1. Weights: BCS/tbcs → uniform; HFB → occupation-weighted ────────────
    if tbcs or not use_pairing:
        # FORTRAN grstep: IF(tbcs) CALL hpsi(..., 1D0, 0D0, ...)
        w_rho = jnp.ones(nstates)
        w_uv  = jnp.zeros(nstates)
    else:
        w_rho = wstates * wocc
        w_uv  = wstates * wguv * pairwg

    # ── 2. Gradient: (H - lagrange)|ψ⟩ or ps1 - h_exp|ψ⟩ ──────────────────
    # Legacy e0dmp_gt_zero (static.py:96-102): the shift h_exp = Re<ψ|ps1> is
    # computed dynamically from the WEIGHTED ps1 = w_rho*H_MF*ψ - w_uv*Δ*ψ.
    # Because h_exp carries the same weights as ps1, the gradient vanishes for
    # states with w_rho ≈ 0 (far above Fermi) instead of becoming -ε*ψ, and at
    # initialization (stored sp_energy = 0) the shift is still applied.
    def apply_h(p, iq, wr, wuv, lag):
        hpsi, _, _ = apply_hfb_hamiltonian(p, meanfield, iq, wr, wuv, grid)
        if use_lagrange:
            return hpsi - lag
        else:
            h_exp = jnp.real(jnp.sum(jnp.conj(p) * hpsi)) * grid.wxyz
            return hpsi - h_exp * p

    hpsi_all = jax.vmap(apply_h)(psi, isospin, w_rho, w_uv, lagrange)

    # ── 3. Preconditioner ────────────────────────────────────────────────────
    # FORTRAN grstep:
    #   tbcs → CALL laplace(1D0, 0D0, ps1, ps2, 0D0, e0inv=e0dmp)
    #   else → CALL laplace(wocc(nst), weightuv, ps1, ps2, MAXVAL(v_pair), e0inv=e0dmp)
    if tbcs or not use_pairing:
        # Simple BCS denominator: e0dmp + h2ma * k²  (weight=1, no pairing term)
        phi_k   = jnp.fft.fftn(hpsi_all, axes=(-3, -2, -1))
        denom   = e0dmp + h2ma * grid.k2                 # (nx, ny, nz), broadcasts
        hpsi_pre = jnp.fft.ifftn(phi_k / denom, axes=(-3, -2, -1))
    else:
        # Full HFB denominator: weight*(e0dmp + h2ma*k²) + 0.5*weightuv*v_pairmax
        # Legacy laplace (levels.py:399-401) floors BOTH weights at 0.1, so the
        # denominator is at least 0.1*e0dmp even for empty states.
        # v_pairmax is the SIGNED maximum (FORTRAN MAXVAL), not max(abs).
        weight      = jnp.maximum(wocc,          0.1)           # (nstates,) — no wstates, matches FORTRAN
        weightuv    = jnp.maximum(wguv * pairwg, 0.1)           # (nstates,)
        v_pairmax   = jnp.array([
            jnp.max(meanfield.v_pair[0]),
            jnp.max(meanfield.v_pair[1]),
        ])
        vpmax = v_pairmax[isospin]                              # (nstates,)

        w  = weight  [:, None, None, None, None]
        wu = weightuv[:, None, None, None, None]
        vp = vpmax   [:, None, None, None, None]

        phi_k   = jnp.fft.fftn(hpsi_all, axes=(-3, -2, -1))
        denom   = w * (e0dmp + h2ma * grid.k2) + 0.5 * wu * vp
        hpsi_pre = jnp.fft.ifftn(phi_k / denom, axes=(-3, -2, -1))

    # ── 4. Update wavefunctions ───────────────────────────────────────────────
    psi_new = psi - x0dmp * hpsi_pre

    # ── 5. Recompute sp_energy on *old* psi for diagnostics (matches legacy spe_mf_new) ─
    # hpsi here is H_MF*psi_old = legacy levels.hampsi (grstep step-1 psi_mf),
    # which the legacy diagstep uses for the diagonalization unitary when tdiag.
    def apply_h_fresh(p, iq):
        hpsi, _, _ = apply_hfb_hamiltonian(p, meanfield, iq, 1.0, 0.0, grid)
        spe_mf   = jnp.real(jnp.sum(jnp.conjugate(p) * hpsi)) * grid.wxyz
        residual = hpsi - spe_mf * p
        return residual, spe_mf, hpsi

    residuals, spe_mf_all, hampsi_old = jax.vmap(apply_h_fresh)(psi, isospin)
    sp_efluct = jnp.sqrt(jnp.sum(jnp.abs(residuals)**2,  axis=(1, 2, 3, 4)) * grid.wxyz)
    ps2_norm  = jnp.sqrt(jnp.sum(jnp.abs(hpsi_pre)**2, axis=(1, 2, 3, 4)) * grid.wxyz)

    # ── 6. Recompute H|psi_new> — matches legacy grstep Step 4 ──────────────────
    # Legacy grstep_helper (static.py lines 194-208): after the gradient update,
    # recomputes H on the NEW psi and stores it in levels.hmfpsi.  diagstep reads
    # levels.hmfpsi, so it uses H|psi_new>, NOT H|psi_old>.  The earlier
    # hmfpsi_before approach was wrong: it fed <psi_new|H|psi_old> into the H matrix
    # instead of the correct <psi_new|H|psi_new>.
    def apply_h_on_new(p, iq):
        # Return plain H_MF * p (second return), matching legacy levels.hmfpsi
        # which is the 2nd return value of hpsi01 regardless of tbcs/HFB weights.
        # For BCS (w_rho=1, w_uv=0) the old code was accidentally correct since
        # hpsi == hpsi_mf.  For HFB (w_rho=wstates*wocc) hpsi ≠ hpsi_mf, causing
        # the 35 MeV sp_diff in the proton block.
        _, hpsi_mf, _ = apply_hfb_hamiltonian(p, meanfield, iq, 1.0, 0.0, grid)
        return hpsi_mf
    hmfpsi_after = jax.vmap(apply_h_on_new)(psi_new, isospin)

    return psi_new, jnp.real(spe_mf_all), hmfpsi_after, hampsi_old


def orthonormalize_states(psi: jax.Array, npsi_n: int, wxyz: float) -> jax.Array:
    """
    Orthonormalize wavefunctions per isospin using Loewdin symmetric orthogonalization.
    Replaces QR decomposition to match legacy physics.
    """
    psi_n = psi[:npsi_n]
    psi_p = psi[npsi_n:]
    
    def loewdin_block(block):
        if block.shape[0] == 0:
            return block
            
        nst = block.shape[0]
        
        # Flatten spatial and spin dimensions to shape (2*nx*ny*nz, nst)
        psi_2d = jnp.reshape(
            jnp.transpose(block, axes=(2, 3, 4, 1, 0)),
            shape=(-1, nst),
            order='F'
        )
        
        # Calculate the overlap matrix S = psi_dagger * psi
        rhomatr_lin = jnp.dot(jnp.conjugate(psi_2d.T), psi_2d) * wxyz
        
        # Eigendecomposition: S * v = w * v
        w, v = jnp.linalg.eigh(rhomatr_lin, symmetrize_input=False)
        
        # Regularize small eigenvalues to prevent division by zero instability
        machine_epsilon = jnp.finfo(jnp.float64).eps
        eigenvalue_threshold = jnp.maximum(1e-12, machine_epsilon * jnp.max(w) * nst)
        w_safe = jnp.maximum(w, eigenvalue_threshold)
        
        # Construct the transformation matrix U = V * W^(-1/2) * V_dagger
        w_inv_sqrt = 1.0 / jnp.sqrt(w_safe)
        v_scaled = v * w_inv_sqrt[None, :]
        unitary_rho = jnp.dot(v, jnp.conjugate(v_scaled.T))
        
        # Apply the symmetric transformation
        transformed_psi = jnp.dot(psi_2d, unitary_rho)
        
        # Reshape back to the original 5D block shape: (nst, 2, nx, ny, nz)
        ortho_block = jnp.transpose(
            jnp.reshape(
                transformed_psi,
                shape=(block.shape[2], block.shape[3], block.shape[4], 2, nst),
                order='F'
            ),
            axes=(4, 3, 0, 1, 2)
        )
        
        return ortho_block

    psi_n_ortho = loewdin_block(psi_n)
    psi_p_ortho = loewdin_block(psi_p)
    
    return jnp.concatenate([psi_n_ortho, psi_p_ortho], axis=0)

def apply_preconditioner(
    phi: jax.Array,
    wocc: jax.Array,
    wguv_pairwg: jax.Array,
    isospin: jax.Array,
    v_pairmax: jax.Array,
    e0dmp: float,
    h2ma: float,
    k2: jax.Array,
    tbcs: bool = False,
) -> jax.Array:
    """
    Apply preconditioning (inverse kinetic energy operator) in Fourier space.
    
    ps_out = ps_in / (e0dmp + h2m * k^2)
    
    This version handles both single wavefunctions and batched wavefunctions.
    For batched input with shape (nstates, 2, nx, ny, nz), the FFT is applied
    to the last 3 dimensions efficiently.
    
    Args:
        phi: Wavefunction(s) to precondition
        e0dmp: Damping energy scale (MeV)
        h2ma: Kinetic energy coefficient (MeV*fm^2)
        k2: Precomputed k^2 grid with shape (nx, ny, nz)
        
    Returns:
        Preconditioned wavefunction with same shape as phi
    """

    if tbcs or not use_pairing:
        weight   = jnp.ones(nstates)
        weightuv = jnp.zeros(nstates)
        vpmax    = jnp.zeros(nstates)
    else:
        weight   = jnp.maximum(wocc, weightmin)
        weightuv = jnp.maximum(wguv_pairwg, weightmin)
        vpmax    = v_pairmax[isospin]

    # Broadcast shapes: (nstates, 1, nx, ny, nz)
    w  = weight[:, None, None, None, None]
    wu = weightuv[:, None, None, None, None]
    vp = vpmax[:, None, None, None, None]


    # FFT to k-space (operates on last 3 dimensions)
    phi_k = jnp.fft.fftn(phi, axes=(-3, -2, -1))
    denom = w * (e0dmp + h2ma * k2) + 0.5 * wu * vp   # broadcasts over spin+spatial
    phi_k = phi_k / denom
    return jnp.fft.ifftn(phi_k, axes=(-3, -2, -1))


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
    h2m: jax.Array,
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
        psi, meanfield, isospin, grid, h2m
    )


@jax.jit
def _compute_sp_energies_vmap(
    psi: jax.Array,
    meanfield: Meanfield,
    isospin: jax.Array,
    grid: Grid,
    h2m: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """Vectorized single-particle energy computation."""
    nstates = psi.shape[0]

    def single_state_energy(nst):
        iq = isospin[nst]
        psi_n = psi[nst]   # (2, nx, ny, nz)

        # Mean-field expectation value: <psi|h_mf|psi>
        _, hpsi_mf, _ = apply_hfb_hamiltonian(
            psi_n, meanfield, iq, 1.0, 0.0, grid,
        )
        e_tot = jnp.real(jnp.sum(jnp.conj(psi_n) * hpsi_mf)) * grid.wxyz

        # ── FFT Laplacian (matches FORTRAN cdervx/y/z with TFFT=True) ──────
        # ∇²ψ(k) = −k² ψ(k)  →  kin = −⟨ψ|∇²|ψ⟩ = ⟨ψ|k²|ψ⟩
        psi_k   = jnp.fft.fftn(psi_n, axes=(-3, -2, -1))
        lap_psi = jnp.fft.ifftn(
            -grid.k2[None, :, :, :] * psi_k, axes=(-3, -2, -1)
        )
        kin   = -jnp.real(jnp.sum(jnp.conj(psi_n) * lap_psi)) * grid.wxyz
        e_kin = h2m[iq] * kin

        return e_tot, e_kin

    sp_energy, sp_kinetic = jax.vmap(single_state_energy)(jnp.arange(nstates))
    return sp_energy, sp_kinetic




@partial(jax.jit,static_argnums=(4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14))
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
    use_pairing: bool = True,
    tbcs: bool = True,
    tdiag: bool = True,
    use_lagrange: bool = True,
    print_sinfo_flag: bool = False,
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

    # ── 1. Gradient step ──────────────────────────────────────────────────────
    psi_new, new_sp_energies, hmfpsi_new, hampsi_old = gradient_step(
        state.psi,
        state.meanfield,
        state.wocc,
        state.wguv,
        state.pairwg,
        state.wstates,
        state.sp_energy,
        state.isospin,
        state.lagrange,         # ← Lagrange multiplier from previous diagstep
        grid,
        config.x0dmp,
        config.e0dmp,
        force.h2ma,
        npsi_n,
        use_pairing=use_pairing,
        tbcs=tbcs,
        use_lagrange=use_lagrange,
    )

    # ── 2. Diagstep: Löwdin + optional diagonalization ───────────────────────
    # Pass hmfpsi_new (H|psi_new>, recomputed on the post-grstep psi).
    # Matches legacy grstep Step 4 which recomputes H|psi> on the new wavefunction
    # and stores it in levels.hmfpsi for diagstep to consume.
    psi_new, sp_energy_for_pairing, deltaf_from_diag, lagrange_new, efluct1q_new, efluct2q_new = diagstep(
        psi_new,
        state.meanfield,
        state.isospin,
        grid,
        npsi_n,
        tdiag,                  # diagonalize
        state.wocc,
        state.wguv,
        state.pairwg,
        state.wstates,
        use_pairing,
        tbcs,
        hmfpsi_new,             # H|psi_new> — matches legacy levels.hmfpsi after grstep Step 4
        hampsi_old,             # H|psi_old> — matches legacy levels.hampsi from grstep Step 1
    )

    _, sp_kinetic = compute_sp_energies(
        psi_new, state.meanfield, state.isospin, grid, force.h2m
    )

    # ── 3. Pairing ────────────────────────────────────────────────────────────
    if use_pairing:
        # deltaf: diagstep's gapmatrix diagonal is the canonical source.
        # For very early iters (before gapmatrix stabilizes), compute_pairing_gaps
        # provides the constant-gap fallback — replicated via iteration <= itrsin
        # inside compute_pairing_gaps. After that, deltaf_from_diag is used.
        itrsin = 10   # FORTRAN constant: first 10 iters use constant gap
        mass_number = nucleus_n + nucleus_z
        deltaf_v_pair = compute_pairing_gaps(
            psi_new,
            state.meanfield.v_pair,
            state.isospin,
            state.pairwg,
            grid.wxyz,
            iteration,
            mass_number,
        )
        # Use v_pair integral for early iters; gapmatrix diagonal after
        deltaf = jax.lax.cond(
            iteration <= itrsin,
            lambda: deltaf_v_pair,
            lambda: deltaf_from_diag,
        )

        wocc, wguv, pairwg, wstates, pairing = solve_pairing(
            sp_energy_for_pairing,
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
    else:
        deltaf  = jnp.zeros_like(state.deltaf)
        wocc    = state.wocc
        wguv    = jnp.zeros_like(state.wguv)
        pairwg  = state.pairwg
        wstates = state.wstates
        pairing = state.pairing
    

    # 4. Compute densities
    densities = compute_densities(
        psi_new, wocc, wguv, pairwg, wstates, state.isospin, grid
    )

    state = dataclasses.replace(state, densities=densities)

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

    # Pairing annealing (FORTRAN static.f90:563-572): enhance the pairing
    # strengths V0 for the first `iteranneal` iterations. v0neut/v0prot enter
    # only the pairing field v_pair (meanfield.py), so this leaves the Skyrme
    # mean field untouched. With iteranneal = 0 the factor is exactly 1.0.
    anneal_factor = jnp.where(
        (config.iteranneal > 0) & (iteration < config.iteranneal),
        1.0 + config.pairenhance
        * (config.iteranneal - iteration) / jnp.maximum(config.iteranneal, 1),
        1.0,
    )
    force_annealed = dataclasses.replace(
        force,
        v0neut=force.v0neut * anneal_factor,
        v0prot=force.v0prot * anneal_factor,
    )

    meanfield = compute_skyrme_meanfield(
        densities,
        force_annealed,
        grid,
        coulomb_potential=wcoul,
        constraint_potential=constraint_potential,
        use_coulomb=use_coulomb,
    )

    state = dataclasses.replace(state, meanfield=meanfield, wcoul=wcoul)


    sp_energy = sp_energy_for_pairing

    if compute_energy:
        energies = compute_integrated_energy(
            densities,
            force,
            grid,
            coulomb_potential=wcoul,
            pairing_energy=pairing.epair,
            mass_number=state.nneut + state.nprot,
            use_coulomb=use_coulomb,
            wocc=wocc,
            wstates=wstates,
            sp_kinetic=sp_kinetic,
            sp_energy=sp_energy,
            # Legacy reads meanfield.ecorrp (DDDI pairing rearrangement, set by
            # skyrme); it was wrongly read from pairing (no such attr → 0.0).
            # Keep as traced JAX scalar: hfb_iteration is jitted.
            ecorrp=meanfield.ecorrp,
        )
    else:
        energies = state.energies


    # 9. Check convergence — use efluct1 (max Lagrange asymmetry) matching legacy
    efluct = jnp.max(efluct1q_new)
    converged = efluct < config.convergence_criterion

    # Legacy convergence measures from the lambda matrix (construct_hfb_matrices):
    # efluct1 = max over isospin of max|lambda_asym|, efluct2 = average of rms.
    # These drive the tvaryx_0 adaptive x0dmp exactly as in the legacy.
    energies = dataclasses.replace(
        energies,
        efluct1=jnp.reshape(jnp.max(efluct1q_new), energies.efluct1.shape).astype(energies.efluct1.dtype),
        efluct1q=efluct1q_new.astype(energies.efluct1q.dtype),
        efluct2=jnp.reshape(jnp.mean(efluct2q_new), energies.efluct2.shape).astype(energies.efluct2.dtype),
        efluct2q=efluct2q_new.astype(energies.efluct2q.dtype),
    )
    if print_sinfo_flag and compute_energy:
        print_sinfo(energies, iteration=iteration, pairing=pairing)

    return SolverState(
        psi=psi_new,
        lagrange=lagrange_new,
        sp_energy=sp_energy,
        sp_kinetic=sp_kinetic,
        deltaf=deltaf,
        sp_n=state.sp_n,
        sp_l=state.sp_l,
        sp_j=state.sp_j,
        wocc=wocc,
        wguv=wguv,
        wstates=wstates,
        pairwg=pairwg,
        isospin=state.isospin,
        sp_parity=state.sp_parity,
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
    preloop_callback: Optional[Callable[[SolverState], None]] = None,
    use_coulomb: bool = True,
    constraint: Optional[Constraint] = None,
    seed: int = 42,
    hook=None,
) -> SolverState:
    """
    Run HFB calculation to convergence.
    
    Args:
        grid: Spatial grid
        force: Force parameters
        nucleus_z: Proton number
        nucleus_n: Neutron number
        npsi_n: Number of neutron states
        config: Solver configuration (uses defaults if None)
        initial_state: Starting state (creates new if None)
        callback: Called after each iteration with current state
        use_coulomb: Whether to include Coulomb interaction
        constraint: Constraint configuration
        seed: Random seed for reproducibility (default: 42)
        
    Returns:
        Converged SolverState
    """
    if config is None:
        config = SolverConfig()

    use_pairing = force.ipair != 0
    
    tbcs_init = not use_pairing

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
            seed=seed,
        )
    else:
        state = initial_state
        if state.constraint_state is None:
            state = dataclasses.replace(state, constraint_state=constraint_state)

    
    npsi = state.psi.shape[0]
    npsi_p = npsi - npsi_n

    psi_new, new_sp_energy, _deltaf_init, lagrange_init, _e1q_init, _e2q_init = diagstep(
        state.psi, state.meanfield, state.isospin, grid, npsi_n,
        False,                  # diagonalize=False
        state.wocc, state.wguv, state.pairwg, state.wstates,
        use_pairing,            # use_pairing
        tbcs_init,                   # tbcs=True for init
    )

    state = dataclasses.replace(
        state, psi=psi_new, sp_energy=new_sp_energy, lagrange=lagrange_init,
    )


    # ── Step 2: densities + mean field ───────────────────────────────────────
    initial_densities = compute_densities_from_state(state, grid)
    state = dataclasses.replace(state, densities=initial_densities)


    initial_wcoul = jnp.zeros((grid.nx, grid.ny, grid.nz), dtype=jnp.float64)
    if use_coulomb:
        initial_wcoul = solve_poisson(initial_densities.rho[1], state.coulomb_solver, grid)

    constraint_state = state.constraint_state
    constraint_potential = None
    if constraint_state is not None and constraint_state.constr_field.shape[0] > 0:
        constraint_state = update_constraint_state(
            constraint_state, initial_densities, grid, config.e0dmp, config.x0dmp
        )
        constraint_potential = compute_constraint_potential(constraint_state, grid)

    initial_meanfield = compute_skyrme_meanfield(
        initial_densities, force, grid,
        coulomb_potential=initial_wcoul if use_coulomb else None,
        constraint_potential=constraint_potential,
        use_coulomb=use_coulomb,
    )
    state = dataclasses.replace(
        state, meanfield=initial_meanfield, wcoul=initial_wcoul,
        constraint_state=constraint_state,
    )



    # ── Step 3: initial gradient step ────────────────────────────────────────
    # sp_energy is still zero here — grstep fills it in, exactly as FORTRAN does.
    psi_new, sp_energy_from_grstep, hmfpsi_init, _hampsi_old_init = gradient_step(
        state.psi, state.meanfield,
        state.wocc, state.wguv, state.pairwg, state.wstates,
        state.sp_energy,
        state.isospin,
        state.lagrange,         # zeros at this point
        grid,
        config.x0dmp, config.e0dmp, force.h2ma, npsi_n,
        use_pairing=use_pairing,
        tbcs=tbcs_init,              # always BCS for init
        use_lagrange=False,     # no lagrange on first grstep
    )
    state = dataclasses.replace(
        state, psi=psi_new, sp_energy = sp_energy_from_grstep
    )

    # ── Step 3 (continued): pairing on initial sp_energy ────────────────────
    # FORTRAN: IF(ipair/=0) CALL pair  (after initial grstep, before 2nd diagstep)
    initial_deltaf = compute_pairing_gaps(
        state.psi, state.meanfield.v_pair,
        state.isospin, state.pairwg, grid.wxyz,
        iteration=0, mass_number=nucleus_n + nucleus_z,
    )

    wocc, wguv, pairwg, wstates, initial_pairing = solve_pairing(
        state.sp_energy, initial_deltaf, state.wstates, state.pairwg,
        state.isospin, npsi_n, npsi_p, nucleus_n, nucleus_z, force,
    )

    state = dataclasses.replace(
        state, deltaf=initial_deltaf,
        wocc=wocc, wguv=wguv, pairwg=pairwg, wstates=wstates,
        pairing=initial_pairing,
    )



    # ── Step 3 (continued): second orthogonalization ─────────────────────────
    # FORTRAN: CALL diagstep(iq, .FALSE.)  (second call, after pair)

    psi_new, new_sp_energy, _deltaf_init2, lagrange_init2, _e1q_init2, _e2q_init2 = diagstep(
        psi_new, state.meanfield, state.isospin, grid, npsi_n,
        False,                  # diagonalize=False
        state.wocc, state.wguv, state.pairwg, state.wstates,
        use_pairing,
        tbcs_init,                   # tbcs=True for init
        hmfpsi_init,            # H|psi_new> from initial grstep (Step 4 recompute)
    )


    recomputed_sp_energy, recomputed_sp_kinetic = compute_sp_energies(
        psi_new, state.meanfield, state.isospin, grid, force.h2m
    )

    state = dataclasses.replace(
        state, psi=psi_new, sp_energy=new_sp_energy, lagrange=lagrange_init2,
    )


    # Pre-loop ehfint: computed from initial densities, before any iteration.
    # This is what the FORTRAN prints as "[iter=0 (pre-loop)] ehfint=..."
    # 1. Compute single-particle properties FIRST to populate kinetic energy
    initial_sp_energy, initial_sp_kinetic = compute_sp_energies(
        psi_new, state.meanfield, state.isospin, grid, force.h2m
    )
    
    # 2. Update state with the new SP properties
    state = dataclasses.replace(
        state, 
        psi=psi_new,
        sp_energy=initial_sp_energy, 
        sp_kinetic=initial_sp_kinetic,
    )

    # 3. NOW compute pre-loop integrated energy, using the populated sp_kinetic
    initial_energies = compute_integrated_energy(
        state.densities, force, grid,
        coulomb_potential=state.wcoul,
        pairing_energy=state.pairing.epair,
        mass_number=nucleus_n + nucleus_z,
        use_coulomb=use_coulomb,
        wocc=wocc,
        wstates=wstates,
        sp_kinetic=state.sp_kinetic,       # Now this contains non-zero data
        sp_energy=state.sp_energy,         # the sp_energy_for_pairing array
        ecorrp=float(state.meanfield.ecorrp),
    )

    # 4. Update state with final initial energies
    state = dataclasses.replace(state, energies=initial_energies)


    # ── Main iteration loop ───────────────────────────────────────────────────
    start_time = time.time()

    print_sinfo(state.energies, iteration=0, pairing=state.pairing)

    current_x0dmp = config.x0dmp
    state = dataclasses.replace(state, x0dmp=current_x0dmp)
    if preloop_callback is not None:
        preloop_callback(state)

    # tvaryx_0: triple x0dmp before main loop for faster convergence (legacy behavior)
    # Also sets up adaptive x0dmp state.
    if config.tvaryx_0:
        current_x0dmp = config.x0dmp * 3.0
        print(f"tvaryx_0: x0dmp set to {current_x0dmp:.4f} (3 × {config.x0dmp:.4f})")
    # Legacy initializes efluct1prev/efluct2prev/ehfprev to 0.0 (energies.py:61-67)
    prev_efluct1 = 0.0
    prev_efluct2 = 0.0
    prev_ehf = 0.0

    for i in range(config.max_iterations):

        python_iter = i + 1   # iteration number as seen inside hfb_iteration

        # Evaluate tbcs: Override to True during initial phase, then revert to base state
        if python_iter <= config.bcs_start:
            tbcs = True
        else:
            tbcs = tbcs_init

        # Evaluate tdiag: False after diag_start, overridden to True if tbcs is active
        if python_iter > config.diag_start:
            tdiag = False
        else:
            tdiag = True

        if tbcs:
            tdiag = True

        use_lagrange = (not tbcs) and (python_iter > 1)

        if python_iter <= 3:
            print(f"ITER {python_iter} START tbcs={tbcs}  tdiag={tdiag}  use_lagrange={use_lagrange}")

        need_output = config.verbose and (i + 1) % config.output_interval == 0
        need_sinfo = config.verbose and (i + 1) % config.sinfo_interval == 0
        is_near_end = i >= config.max_iterations - 1
        # tvaryx_0 needs a fresh ehf every iteration (legacy runs with mprint=1)
        compute_energy = need_output or need_sinfo or is_near_end or config.tvaryx_0

        # Use per-iteration x0dmp (potentially tripled / adapted from tvaryx_0)
        iter_config = dataclasses.replace(config, x0dmp=current_x0dmp)

        if hook is not None:
            hook.pre_iteration(python_iter, state)

        state = hfb_iteration(
            state, grid, force, iter_config,
            npsi_n, npsi_p, nucleus_n, nucleus_z,
            use_coulomb=use_coulomb,
            compute_energy=compute_energy, use_pairing=use_pairing,
            tbcs=tbcs, tdiag=tdiag, use_lagrange=use_lagrange,
            print_sinfo_flag=need_sinfo,
        )

        if hook is not None:
            hook.post_iteration(python_iter, state)

        # tvaryx_0 adaptive x0dmp update (matches legacy static.py lines 1572-1585):
        # improving = (ehf < ehfprev AND efluct1 < efluct1prev*(1-1e-5))
        #             OR efluct2 < efluct2prev*(1-1e-5)
        if config.tvaryx_0:
            curr_efluct1 = float(state.energies.efluct1[0])
            curr_efluct2 = float(state.energies.efluct2[0])
            curr_ehf = float(state.energies.ehf)
            improving = ((curr_ehf < prev_ehf and curr_efluct1 < prev_efluct1 * (1.0 - 1e-5))
                         or curr_efluct2 < prev_efluct2 * (1.0 - 1e-5))
            if improving:
                current_x0dmp = current_x0dmp * 1.005
            else:
                current_x0dmp = current_x0dmp * 0.8
            current_x0dmp = float(jnp.clip(current_x0dmp, config.x0dmp, config.x0dmp * 5.0))
            prev_efluct1 = curr_efluct1
            prev_efluct2 = curr_efluct2
            prev_ehf = curr_ehf
        
        state = dataclasses.replace(state, x0dmp=current_x0dmp)
        if callback is not None:
            callback(state)
        
        if need_output:
            elapsed = time.time() - start_time
            # Compute axis-resolved Q20 to diagnose deformation axis
            rho_tot = state.densities.rho[0] + state.densities.rho[1]
            X = grid.x[:, jnp.newaxis, jnp.newaxis]
            Y = grid.y[jnp.newaxis, :, jnp.newaxis]
            Z = grid.z[jnp.newaxis, jnp.newaxis, :]
            Q_xx = float(jnp.sum(rho_tot * (2*X**2 - Y**2 - Z**2)) * grid.wxyz)
            Q_yy = float(jnp.sum(rho_tot * (2*Y**2 - X**2 - Z**2)) * grid.wxyz)
            Q_zz = float(jnp.sum(rho_tot * (2*Z**2 - X**2 - Y**2)) * grid.wxyz)
            print(f"Iter {state.iteration:4d}: E = {state.energies.ehfint:12.4f} MeV, "
                  f"fluct = {state.efluct:.2e}, time = {elapsed:.1f}s  "
                  f"Q_xx={Q_xx:.3f} Q_yy={Q_yy:.3f} Q_zz={Q_zz:.3f} fm^2")
        
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
                    wocc=wocc,
                    wstates=wstates,
                    sp_kinetic=sp_kinetic,
                    sp_energy=sp_energy,         # the sp_energy_for_pairing array
                    ecorrp=float(state.meanfield.ecorrp),

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

    if hook is not None:
        hook.finalize()

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
            wocc=wocc,
            wstates=wstates,
            sp_kinetic=sp_kinetic,
            sp_energy=sp_energy,         # the sp_energy_for_pairing array
            ecorrp=float(state.meanfield.ecorrp),

        )
        state = SolverState(
            psi=state.psi,
            sp_energy=state.sp_energy,
            sp_kinetic=state.sp_kinetic,
            deltaf=state.deltaf,
            sp_n=state.sp_n,
            sp_l=state.sp_l,
            sp_j=state.sp_j,
            wocc=state.wocc,
            wguv=state.wguv,
            wstates=state.wstates,
            pairwg=state.pairwg,
            isospin=state.isospin,
            sp_parity=state.sp_parity,
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

@partial(jax.jit, static_argnums=(4, 5, 10, 11))
def diagstep(
    psi: jax.Array,          # (nstates, 2, nx, ny, nz)
    meanfield: Meanfield,
    isospin: jax.Array,      # (nstates,)
    grid: Grid,
    npsi_n: int,             # static
    diagonalize: bool,       # static
    wocc: jax.Array,         # (nstates,)
    wguv: jax.Array,         # (nstates,)
    pairwg: jax.Array,       # (nstates,)
    wstates: jax.Array,      # (nstates,)
    use_pairing: bool,       # static
    tbcs: bool,              # static
    hampsi: jax.Array = None,      # (nstates, 2, nx, ny, nz) H_mf*psi_NEW from grstep step 4 (legacy hmfpsi)
    hampsi_old: jax.Array = None,  # (nstates, 2, nx, ny, nz) H_mf*psi_OLD from grstep step 1 (legacy hampsi)
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Matches FORTRAN diagstep: Löwdin orthonormalization + optional H diagonalization.
    Also computes gapmatrix → deltaf and builds the Lagrange multiplier array.

    hampsi (legacy levels.hmfpsi = H*psi_new) builds the H matrix → sp_energy.
    hampsi_old (legacy levels.hampsi = H*psi_old) builds the diagonalization
    unitary when diagonalize=True.

    Returns:
        psi_new:   (nstates, 2, nx, ny, nz)
        sp_energy: (nstates,)
        deltaf:    (nstates,)  — rotated pairing gaps
        lagrange:  (nstates, 2, nx, ny, nz)  — Σ_β λ[β,α] ψ_β
        efluct1q:  (2,) — max|lambda_asym| per isospin (legacy efluct1q)
        efluct2q:  (2,) — rms of lambda_asym per isospin (legacy efluct2q)
    """
    # ── Compute hmfpsi and delpsi for all states ──────────────────────────────
    # Legacy: diagstep uses levels.hampsi = H_mf*psi_before (from previous grstep).
    # OOP: if hampsi provided, use it; else compute fresh H*psi_current.
    def get_hpsi_parts(p, iq):
        _, hmfpsi, delpsi = apply_hfb_hamiltonian(p, meanfield, iq, 1.0, 1.0, grid)
        return hmfpsi, delpsi

    if hampsi is not None:
        # Use pre-computed H*psi_new (matches legacy levels.hmfpsi behavior)
        _, delpsi_all = jax.vmap(get_hpsi_parts)(psi, isospin)
        hmfpsi_all = hampsi
    else:
        hmfpsi_all, delpsi_all = jax.vmap(get_hpsi_parts)(psi, isospin)

    # H*psi_old for the diagonalization unitary (legacy levels.hampsi).
    # Fallback to hmfpsi_all only matters when diagonalize=True without a
    # grstep-provided hampsi_old (not the legacy-matching path).
    hampsi_old_all = hampsi_old if hampsi_old is not None else hmfpsi_all

    def process_block(
        psi_block,      # (nst, 2, nx, ny, nz)
        hmfpsi_block,   # (nst, 2, nx, ny, nz)
        delpsi_block,   # (nst, 2, nx, ny, nz)
        hampsi_old_block,  # (nst, 2, nx, ny, nz)
        wocc_block,     # (nst,)
        wguv_block,     # (nst,)
        pairwg_block,   # (nst,)
        wstates_block,  # (nst,)
        iq: int,        # neutron=0, proton=1
    ):
        nst  = psi_block.shape[0]
        wxyz = grid.wxyz
        nx, ny, nz = psi_block.shape[2], psi_block.shape[3], psi_block.shape[4]

        def to_2d(block):
            return jnp.reshape(
                jnp.transpose(block, axes=(2, 3, 4, 1, 0)),
                shape=(-1, nst), order='F'
            )

        def to_block(mat_2d):
            return jnp.transpose(
                jnp.reshape(mat_2d, shape=(nx, ny, nz, 2, nst), order='F'),
                axes=(4, 3, 0, 1, 2)
            )

        psi_2d    = to_2d(psi_block)
        hmfpsi_2d = to_2d(hmfpsi_block)

        # ── Löwdin: S = ψ†ψ, U_rho = S^{-1/2} ──────────────────────────────
        S = jnp.dot(jnp.conj(psi_2d.T), psi_2d) * wxyz
        w, v = jnp.linalg.eigh(S, symmetrize_input=False)
        eps    = jnp.maximum(1e-12, jnp.finfo(jnp.float64).eps * jnp.max(w) * nst)
        w_safe = jnp.maximum(w, eps)
        U_rho  = jnp.dot(v * (1.0 / jnp.sqrt(w_safe))[None, :], jnp.conj(v.T))

        # ── H matrix in non-orthogonal basis ────────────────────────────────
        H_raw = jnp.dot(jnp.conj(psi_2d.T), hmfpsi_2d) * wxyz  # (nst, nst)

        # ── Diagonalization or Löwdin-only ───────────────────────────────────
        # Legacy diagstep (static.py:300-312): the diagonalization unitary comes
        # from eigh of lambda_lin = <ψ_new|H_MF ψ_old>·wxyz built from the STORED
        # hampsi (H·ψ before the gradient update) — no U_rho sandwich, no
        # symmetrization (eigh reads the lower triangle).  U = U_rho @ U_lam.
        # sp_energy always comes from diag(U† H_raw U) with H_raw built from
        # hmfpsi (H·ψ_new), matching construct_hfb_matrices.
        if diagonalize:
            hampsi_old_2d = to_2d(hampsi_old_block)
            lambda_lin = jnp.dot(jnp.conj(psi_2d.T), hampsi_old_2d) * wxyz
            _, U_lam = jnp.linalg.eigh(lambda_lin, symmetrize_input=False)
            U = jnp.dot(U_rho, U_lam)
        else:
            U = U_rho
        H_ortho = jnp.dot(jnp.conj(U.T), jnp.dot(H_raw, U))
        sp_energy_block = jnp.real(jnp.diag(H_ortho))

        # ── Apply transformation ─────────────────────────────────────────────
        psi_new_2d    = jnp.dot(psi_2d, U)
        psi_new_block = to_block(psi_new_2d)
        
        # ── hmatrix in rotated basis: U† H_raw U ─────────────────────────────
        hmatrix = jnp.dot(jnp.conj(U.T), jnp.dot(H_raw, U))   # (nst, nst)

        # ── gapmatrix and deltaf ─────────────────────────────────────────────
        # FORTRAN: gapmatrix = U† G_raw U,  deltaf = diag(gapmatrix) * pairwg
        delpsi_2d = to_2d(delpsi_block)
        if use_pairing:
            G_raw     = jnp.dot(jnp.conj(psi_2d.T), delpsi_2d) * wxyz  # (nst, nst)
            gapmatrix = jnp.dot(jnp.conj(U.T), jnp.dot(G_raw, U))
            deltaf_block = jnp.real(jnp.diag(gapmatrix)) * pairwg_block
        else:
            gapmatrix    = jnp.zeros((nst, nst), dtype=H_raw.dtype)
            deltaf_block = jnp.zeros(nst, dtype=jnp.float64)


        # ── Lagrange multiplier: FORTRAN calc_lambda + recombine ─────────────
        # calc_lambda: lambda_temp[:, β] = wocc[β]*wstates[β]*H[:,β]
        #                                - wguv[β]*pairwg[β]*wstates[β]*G[:,β]
        #              lambda = 0.5*(lambda_temp + conj(lambda_temp).T)
        # For tbcs: zero gapmatrix contribution (FORTRAN: gapmatrix off-diag=0
        #           before symcond; here we zero entire matrix since diagonal
        #           deltaf was already extracted above)
        if tbcs:
            gap_for_lambda = gapmatrix * jnp.eye(nst, dtype=gapmatrix.dtype)
        else:
            gap_for_lambda = gapmatrix

        w_col   = (wocc_block * wstates_block)                      # (nst,)
        wuv_col = (wguv_block * pairwg_block * wstates_block)       # (nst,)
        lambda_temp = w_col[None, :] * hmatrix - wuv_col[None, :] * gap_for_lambda
        lambda_mat  = 0.5 * (lambda_temp + jnp.conj(lambda_temp.T))

        # Legacy convergence measures (construct_hfb_matrices, static.py:546-551):
        # antisymmetric part of lambda → efluct1q (max), efluct2q (rms)
        lambda_asym = 0.5 * (lambda_temp - jnp.conj(lambda_temp.T))
        efluct1q_block = jnp.max(jnp.abs(lambda_asym))
        efluct2q_block = jnp.sqrt(jnp.sum(jnp.abs(lambda_asym) ** 2) / nst**2)

        # recombine: lagrange_2d = psi_new_2d @ lambda_mat
        lagrange_2d = jnp.dot(psi_new_2d, lambda_mat)                       # (2*nxyz, nst)
        lagrange_block = to_block(lagrange_2d)

        return psi_new_block, sp_energy_block, deltaf_block, lagrange_block, efluct1q_block, efluct2q_block

    # ── Split into neutron / proton blocks ────────────────────────────────────
    psi_n      = psi[:npsi_n];       psi_p      = psi[npsi_n:]
    hmfpsi_n   = hmfpsi_all[:npsi_n]; hmfpsi_p  = hmfpsi_all[npsi_n:]
    delpsi_n   = delpsi_all[:npsi_n]; delpsi_p  = delpsi_all[npsi_n:]
    hampsi_old_n = hampsi_old_all[:npsi_n]; hampsi_old_p = hampsi_old_all[npsi_n:]
    wocc_n     = wocc[:npsi_n];       wocc_p     = wocc[npsi_n:]
    wguv_n     = wguv[:npsi_n];       wguv_p     = wguv[npsi_n:]
    pairwg_n   = pairwg[:npsi_n];     pairwg_p   = pairwg[npsi_n:]
    wstates_n  = wstates[:npsi_n];    wstates_p  = wstates[npsi_n:]

    psi_n_new, spe_n, deltaf_n, lag_n, e1q_n, e2q_n = process_block(
        psi_n, hmfpsi_n, delpsi_n, hampsi_old_n, wocc_n, wguv_n, pairwg_n, wstates_n, 0
    )
    psi_p_new, spe_p, deltaf_p, lag_p, e1q_p, e2q_p = process_block(
        psi_p, hmfpsi_p, delpsi_p, hampsi_old_p, wocc_p, wguv_p, pairwg_p, wstates_p, 1
    )

    psi_new  = jnp.concatenate([psi_n_new, psi_p_new], axis=0)
    sp_energy = jnp.concatenate([spe_n,    spe_p],     axis=0)
    deltaf    = jnp.concatenate([deltaf_n,  deltaf_p],  axis=0)
    lagrange  = jnp.concatenate([lag_n,     lag_p],     axis=0)
    efluct1q  = jnp.stack([e1q_n, e1q_p])
    efluct2q  = jnp.stack([e2q_n, e2q_p])

    return psi_new, sp_energy, deltaf, lagrange, efluct1q, efluct2q
