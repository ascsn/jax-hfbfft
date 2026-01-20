"""
Example: Constrained HFB calculations for deformation.

This example demonstrates how to perform constrained HFB calculations
to study nuclear deformation along the potential energy surface.
"""

from jax_hfbfft import HFBFFT, Nucleus, Force, Constraint


def main():
    # Calculate deformation for U-238
    nucleus = Nucleus(protons=92, neutrons=146, name="U-238")
    force = Force.from_name("SLy4")
    
    print(f"Constrained HFB calculation for {nucleus.name}")
    print(f"Force: {force.name}")
    print("=" * 60)
    
    # Define deformation points along the fission path
    beta2_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    print(f"\nCalculating {len(beta2_values)} deformation points...")
    
    results = []
    
    for beta2 in beta2_values:
        print(f"\nβ₂ = {beta2:.2f}")
        
        # Create constraint for this deformation
        if beta2 == 0.0:
            constraint = Constraint.spherical()
        else:
            constraint = Constraint.prolate(beta2=beta2)
        
        print(f"  Constraint: {constraint}")
        
        # Create calculation
        calc = HFBFFT(
            nucleus=nucleus,
            force=force,
            constraint=constraint,
            nx=32, ny=32, nz=32,
            dx=0.8, dy=0.8, dz=0.8,
        )
        
        calc.initialize_wavefunctions(method="harmonic_oscillator")
        
        # Run with more iterations for constrained calculations
        result = calc.run(
            max_iterations=2000,
            convergence_threshold=1e-5,
            print_interval=0,
        )
        
        results.append({
            'beta2_target': beta2,
            'beta2_actual': result.beta2,
            'energy': result.total_energy,
            'converged': result.converged,
        })
        
        print(f"  E = {result.total_energy:.3f} MeV")
        print(f"  β₂(actual) = {result.beta2:.3f}")
    
    # Print potential energy surface
    print("\n" + "=" * 60)
    print("POTENTIAL ENERGY SURFACE")
    print("=" * 60)
    print(f"{'β₂(target)':>12} {'β₂(actual)':>12} {'E (MeV)':>12} {'ΔE (MeV)':>12}")
    print("-" * 60)
    
    e_min = min(r['energy'] for r in results)
    
    for r in results:
        delta_e = r['energy'] - e_min
        print(f"{r['beta2_target']:>12.2f} {r['beta2_actual']:>12.3f} "
              f"{r['energy']:>12.3f} {delta_e:>12.3f}")
    
    # Find equilibrium deformation
    min_result = min(results, key=lambda x: x['energy'])
    print(f"\nEquilibrium deformation: β₂ = {min_result['beta2_actual']:.3f}")
    print(f"Binding energy at equilibrium: {min_result['energy']:.3f} MeV")


def constrained_triaxial():
    """
    Example of triaxial deformation calculations.
    """
    nucleus = Nucleus.from_symbol("Ge", 76)
    force = Force.from_name("SLy4")
    
    print(f"\nTriaxial calculation for {nucleus.name}")
    print("=" * 60)
    
    # Create triaxial constraint
    constraint = Constraint.triaxial(alpha20=0.3, alpha22=0.1)
    
    calc = HFBFFT(
        nucleus=nucleus,
        force=force,
        constraint=constraint,
    )
    
    calc.initialize_wavefunctions()
    result = calc.run(max_iterations=1000)
    
    print(f"\nResults:")
    print(f"  β₂ = {result.beta2:.3f}")
    print(f"  γ  = {result.gamma:.1f}°")
    print(f"  E  = {result.total_energy:.3f} MeV")


if __name__ == "__main__":
    main()
    # Uncomment to run triaxial example:
    # constrained_triaxial()
