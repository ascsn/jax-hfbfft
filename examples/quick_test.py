#!/usr/bin/env python3
"""Quick test of the jax-hfbfft OOP API with minimal iterations."""

import os
import sys

# Enable 64-bit precision for JAX
os.environ['JAX_ENABLE_X64'] = 'True'

from jax_hfbfft import HFBFFT, Nucleus, Force, Grid


def main():
    print("=" * 60)
    print("Quick Test: jax-hfbfft OOP API")
    print("=" * 60)
    
    # Create a simple nucleus calculation - O-16 is fast
    nucleus = Nucleus(protons=8, neutrons=8)
    print(f"Nucleus: O-16 (Z={nucleus.protons}, N={nucleus.neutrons})")
    
    # Use SLy4 force with no pairing (simpler/faster)
    force = Force.from_name("SLy4").with_pairing(ipair=0)
    print(f"Force: {force.name} (no pairing)")
    
    # Small grid for speed
    grid = Grid.create(nx=16, ny=16, nz=16, dx=1.0, dy=1.0, dz=1.0)
    print(f"Grid: {grid.nx}x{grid.ny}x{grid.nz}, spacing={grid.dx} fm")
    
    # Create HFBFFT calculation with minimal basis
    calc = HFBFFT(
        nucleus=nucleus,
        force=force,
        grid=grid,
        npsi=(16, 16),  # Minimal basis
        include_coulomb=True,  
    )
    print(f"Basis: 16 neutron states, 16 proton states")
    
    # Initialize wavefunctions
    print("\nInitializing wavefunctions...")
    calc.initialize_wavefunctions(
        method="harmonic_oscillator",
        radinx=3.0, radiny=3.0, radinz=3.0
    )
    
    # Run a few iterations (not to convergence)
    print("\nStarting HFB iteration (10 iterations only)...")
    try:
        results = calc.run(
            max_iterations=10,
            convergence_threshold=1e-6,
            print_interval=1,
        )
        
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Converged: {results.converged}")
        print(f"Iterations: {results.iterations}")
        print(f"Total energy: {results.total_energy:.3f} MeV")
        
        print("\nTest PASSED! The OOP API is working.")
        
    except Exception as e:
        print(f"\nError during calculation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
