"""
Example: Multiple parallel HFB calculations.

This example demonstrates how to create and run multiple independent
HFBFFT calculations, which can be parallelized across nuclei.
"""

from jax_hfbfft import HFBFFT, Nucleus, Force


def main():
    # Define a set of nuclei to calculate
    nuclei = [
        Nucleus.from_symbol("O", 16),
        Nucleus.from_symbol("Ca", 40),
        Nucleus.from_symbol("Ca", 48),
        Nucleus.from_symbol("Ni", 56),
        Nucleus.from_symbol("Ni", 68),
        Nucleus.from_symbol("Sn", 100),
        Nucleus.from_symbol("Sn", 132),
        Nucleus.from_symbol("Pb", 208),
    ]
    
    print(f"Setting up calculations for {len(nuclei)} nuclei:")
    for n in nuclei:
        print(f"  {n.name}: Z={n.Z}, N={n.N}")
    
    # Use SLy4 force for all calculations
    force = Force.from_name("SLy4")
    print(f"\nUsing force: {force.name}")
    
    # Create independent HFBFFT calculations
    calculations = []
    for nucleus in nuclei:
        calc = HFBFFT(
            nucleus=nucleus,
            force=force,
            nx=32, ny=32, nz=32,
            dx=0.8, dy=0.8, dz=0.8,
        )
        calc.initialize_wavefunctions(method="harmonic_oscillator")
        calculations.append(calc)
    
    print(f"\nCreated {len(calculations)} independent calculations")
    
    # Run all calculations
    # In a real scenario, these could be distributed across multiple GPUs
    # or run in parallel using multiprocessing
    print("\nRunning calculations...")
    print("-" * 70)
    
    results_table = []
    
    for i, calc in enumerate(calculations):
        print(f"\n[{i+1}/{len(calculations)}] {calc.nucleus.name}")
        
        results = calc.run(
            max_iterations=500,
            convergence_threshold=1e-5,
            print_interval=0,  # Quiet mode
        )
        
        results_table.append({
            'name': calc.nucleus.name,
            'Z': calc.nucleus.Z,
            'N': calc.nucleus.N,
            'A': calc.nucleus.A,
            'energy': results.total_energy,
            'converged': results.converged,
            'iterations': results.iterations,
        })
    
    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY OF RESULTS")
    print("=" * 70)
    print(f"{'Nucleus':>10} {'Z':>4} {'N':>4} {'A':>4} {'E (MeV)':>12} {'Conv':>6} {'Iter':>6}")
    print("-" * 70)
    
    for r in results_table:
        conv_str = "Yes" if r['converged'] else "No"
        print(f"{r['name']:>10} {r['Z']:>4} {r['N']:>4} {r['A']:>4} "
              f"{r['energy']:>12.3f} {conv_str:>6} {r['iterations']:>6}")
    
    # Calculate binding energy per nucleon
    print("\n" + "-" * 70)
    print("Binding energy per nucleon:")
    for r in results_table:
        be_per_a = abs(r['energy']) / r['A']
        print(f"  {r['name']:>10}: {be_per_a:.3f} MeV/A")


if __name__ == "__main__":
    main()
