"""
Example: Basic HFB calculation for a single nucleus.

This example demonstrates how to perform a basic HFB calculation
for Ca-48 using the SLy4 Skyrme force.
"""

from jax_hfbfft import HFBFFT, Nucleus, Force


def main():
    # Define the nucleus
    print("Setting up calculation for Ca-48...")
    nucleus = Nucleus(protons=20, neutrons=28, name="Ca-48")
    
    # Alternative: use element symbol
    # nucleus = Nucleus.from_symbol("Sn", 132)
    
    print(f"Nucleus: {nucleus}")
    print(f"  Z = {nucleus.Z}")
    print(f"  N = {nucleus.N}")
    print(f"  A = {nucleus.A}")
    
    # Choose the nuclear force
    force = Force.from_name("SLy4")
    print(f"\nForce: {force}")
    
    # Create the HFBFFT calculation
    calc = HFBFFT(
        nucleus=nucleus,
        force=force,
        nx=30, ny=30, nz=30,  # Grid dimensions
        dx=0.8, dy=0.8, dz=0.8,  # Grid spacing in fm
    )
    
    print(f"\nGrid: {calc.grid.nx}x{calc.grid.ny}x{calc.grid.nz}")
    print(f"Grid spacing: {calc.grid.dx} fm")
    print(f"Basis size: {calc._npsi[0]} neutrons, {calc._npsi[1]} protons")
    
    # Initialize wavefunctions
    print("\nInitializing wavefunctions...")
    calc.initialize_wavefunctions(
        method="harmonic_oscillator",
        radinx=3.0, radiny=3.0, radinz=3.0
    )
    
    # Run the calculation
    print("\nStarting HFB iteration...")
    results = calc.run(
        max_iterations=200,
        convergence_threshold=1e-6,
        print_interval=20,
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Converged: {results.converged}")
    print(f"Iterations: {results.iterations}")
    print(f"\nEnergies (MeV):")
    print(f"  Total binding energy: {results.total_energy:.3f}")
    print(f"  Kinetic energy:       {results.kinetic_energy:.3f}")
    print(f"  Potential energy:     {results.potential_energy:.3f}")
    print(f"  Pairing energy:       {results.pairing_energy:.3f}")
    print(f"  Coulomb energy:       {results.coulomb_energy:.3f}")
    
    print(f"\nRadii (fm):")
    print(f"  RMS neutron:  {results.rms_radius_n:.3f}")
    print(f"  RMS proton:   {results.rms_radius_p:.3f}")
    print(f"  RMS total:    {results.rms_radius_total:.3f}")
    print(f"  Charge:       {results.charge_radius:.3f}")
    
    print(f"\nDeformation:")
    print(f"  β₂ = {results.beta2:.3f}")
    print(f"  γ  = {results.gamma:.1f}°")


if __name__ == "__main__":
    main()
