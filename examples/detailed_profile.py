#!/usr/bin/env python3
"""
Detailed profiling of iteration components.
"""

import os
import time

os.environ['JAX_ENABLE_X64'] = 'True'

import jax
import jax.numpy as jnp

from jax_hfbfft import HFBFFT, Nucleus, Force, Grid
from jax_hfbfft.physics.solver import (
    gradient_step, compute_sp_energies, SolverConfig
)
from jax_hfbfft.physics.densities import compute_densities
from jax_hfbfft.physics.meanfield import compute_skyrme_meanfield
from jax_hfbfft.physics.coulomb import solve_poisson


def benchmark(fn, *args, warmup=3, repeats=10, **kwargs):
    """Benchmark a function."""
    # Warmup
    for _ in range(warmup):
        result = fn(*args, **kwargs)
    jax.block_until_ready(result)
    
    # Timed runs
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        jax.block_until_ready(result)
        times.append(time.perf_counter() - start)
    
    return sum(times) / len(times) * 1000  # Return mean in ms


def main():
    print("=" * 60)
    print("Detailed Iteration Component Profiling")
    print("=" * 60)
    
    # Setup
    nucleus = Nucleus(protons=8, neutrons=8)
    force = Force.from_name("SLy4").with_pairing(ipair=0)
    grid = Grid.create(nx=24, ny=24, nz=24, dx=1.0, dy=1.0, dz=1.0)
    
    calc = HFBFFT(nucleus=nucleus, force=force, grid=grid, include_coulomb=True)
    calc.initialize_wavefunctions(method="harmonic_oscillator", radinx=3.0, radiny=3.0, radinz=3.0)
    
    # Run a few iterations to get a valid state
    calc.run(max_iterations=30, convergence_threshold=1e-6, print_interval=100)
    
    state = calc._solver_state
    npsi_n = int(calc._npsi[0])
    npsi_p = int(calc._npsi[1])
    
    print("\nComponent Benchmarks (3 warmup + 10 timed):")
    print("-" * 50)
    
    # 1. Gradient step
    t = benchmark(
        gradient_step,
        state.psi, state.meanfield, state.wocc, state.wguv,
        state.pairwg, state.sp_energy, state.isospin, grid,
        0.45, 20.0, npsi_n
    )
    print(f"gradient_step:            {t:6.2f} ms")
    
    # 2. Compute densities
    t = benchmark(
        compute_densities,
        state.psi, state.wocc, state.wguv, state.pairwg, state.isospin, grid
    )
    print(f"compute_densities:        {t:6.2f} ms")
    
    # 3. Solve Poisson
    t = benchmark(
        solve_poisson,
        state.densities.rho[1], state.coulomb_solver, grid
    )
    print(f"solve_poisson:            {t:6.2f} ms")
    
    # 4. Compute meanfield
    t = benchmark(
        compute_skyrme_meanfield,
        state.densities, force, grid,
        coulomb_potential=state.wcoul, use_coulomb=True
    )
    print(f"compute_skyrme_meanfield: {t:6.2f} ms")
    
    # 5. Compute SP energies
    t = benchmark(
        compute_sp_energies,
        state.psi, state.meanfield, state.isospin, grid
    )
    print(f"compute_sp_energies:      {t:6.2f} ms")
    
    print("-" * 50)
    
    # Breakdown of gradient_step
    print("\nGradient Step Breakdown:")
    print("-" * 50)
    
    from jax_hfbfft.physics.meanfield import apply_hfb_hamiltonian
    from jax_hfbfft.physics.solver import apply_preconditioner, orthonormalize_states
    
    # Time Hamiltonian application
    def apply_h_all():
        def apply_h(p, iq, e):
            hpsi, _, _ = apply_hfb_hamiltonian(p, state.meanfield, iq, 1.0, 0.0, grid)
            return hpsi - e * p
        return jax.vmap(apply_h)(state.psi, state.isospin, state.sp_energy)
    
    t = benchmark(apply_h_all)
    print(f"  vmap(apply_hfb_hamiltonian): {t:6.2f} ms")
    
    # Get hpsi for next benchmarks
    hpsi_all = apply_h_all()
    
    # Time preconditioner
    t = benchmark(
        apply_preconditioner,
        hpsi_all, 20.0, 20.73, grid.dx, grid.dy, grid.dz
    )
    print(f"  apply_preconditioner:        {t:6.2f} ms")
    
    # Apply preconditioner
    hpsi_pre = apply_preconditioner(hpsi_all, 20.0, 20.73, grid.dx, grid.dy, grid.dz)
    psi_new = state.psi - 0.45 * hpsi_pre
    
    # Time orthonormalization
    t = benchmark(orthonormalize_states, psi_new, npsi_n, grid.wxyz)
    print(f"  orthonormalize_states:       {t:6.2f} ms")
    
    print("-" * 50)


if __name__ == "__main__":
    main()
