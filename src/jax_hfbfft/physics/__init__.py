"""
Physics modules for jax-hfbfft.

This subpackage contains the core physics computations:
- densities: Compute nuclear densities from wavefunctions
- meanfield: Compute Skyrme mean-field potentials
- pairing: HFB pairing calculations
- coulomb: Coulomb potential solver
- energies: Energy functional calculations
- solver: Main HFB iteration loop
"""

from jax_hfbfft.physics.densities import compute_densities, Densities
from jax_hfbfft.physics.meanfield import (
    Meanfield,
    compute_skyrme_meanfield,
    apply_hamiltonian,
    apply_hfb_hamiltonian,
)
from jax_hfbfft.physics.energies import (
    Energies,
    compute_integrated_energy,
    compute_sp_energy,
)
from jax_hfbfft.physics.pairing import (
    Pairing,
    solve_pairing,
    compute_pairing_gaps,
    bcs_occupation,
)
from jax_hfbfft.physics.coulomb import (
    CoulombSolver,
    solve_poisson,
    compute_coulomb_energy,
)
from jax_hfbfft.physics.solver import (
    SolverState,
    SolverConfig,
    run_hfb,
    hfb_iteration,
    create_initial_state,
)

__all__ = [
    # Densities
    "Densities",
    "compute_densities",
    # Meanfield
    "Meanfield",
    "compute_skyrme_meanfield",
    "apply_hamiltonian",
    "apply_hfb_hamiltonian",
    # Energies
    "Energies",
    "compute_integrated_energy",
    "compute_sp_energy",
    # Pairing
    "Pairing",
    "solve_pairing",
    "compute_pairing_gaps",
    "bcs_occupation",
    # Coulomb
    "CoulombSolver",
    "solve_poisson",
    "compute_coulomb_energy",
    # Solver
    "SolverState",
    "SolverConfig",
    "run_hfb",
    "hfb_iteration",
    "create_initial_state",
]
