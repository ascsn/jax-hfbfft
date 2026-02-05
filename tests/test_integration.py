"""
Integration tests for HFBFFT comparing legacy vs modern JAX solver.

These tests run small calculations and compare results to ensure
numerical consistency between implementations.
"""

import pytest
import jax.numpy as jnp
import numpy as np
from jax_hfbfft import HFBFFT, Nucleus, Force


@pytest.mark.integration
class TestLegacyVsModernConsistency:
    """Compare legacy and modern implementations."""
    
    def test_small_nucleus_energy_consistency(self):
        """Test that legacy and modern give consistent energies for small nucleus."""
        # Very small nucleus and grid for fast test
        nucleus = Nucleus(protons=4, neutrons=4)
        
        # Tiny grid
        calc_modern = HFBFFT(
            nucleus=nucleus,
            force_name="SLy4",
            nx=8, ny=8, nz=8,
            dx=1.0, dy=1.0, dz=1.0,
            npsi=(8, 8),
        )
        
        # Initialize wavefunctions
        calc_modern.initialize_wavefunctions(method="harmonic_oscillator")
        
        # Run modern solver (short iteration for CI speed)
        results_modern = calc_modern.run(
            max_iterations=20,
            convergence_threshold=1e-4,
            print_interval=5,
            use_legacy=False,
        )
        
        # Create a second instance for legacy run
        calc_legacy = HFBFFT(
            nucleus=nucleus,
            force_name="SLy4",
            nx=8, ny=8, nz=8,
            dx=1.0, dy=1.0, dz=1.0,
            npsi=(8, 8),
        )
        
        # Run legacy solver
        results_legacy = calc_legacy.run(
            max_iterations=20,
            convergence_threshold=1e-4,
            print_interval=5,
            use_legacy=True,
        )
        
        # Compare energies (loose tolerance for different algorithms)
        # Legacy and modern may converge to slightly different local minima
        energy_diff = abs(results_modern.total_energy - results_legacy.total_energy)
        energy_avg = (abs(results_modern.total_energy) + abs(results_legacy.total_energy)) / 2
        
        relative_diff = energy_diff / (energy_avg + 1e-10)
        
        # Check relative difference is reasonable (< 5% for 20 iterations)
        assert relative_diff < 0.05, (
            f"Energy difference too large: modern={results_modern.total_energy:.4f}, "
            f"legacy={results_legacy.total_energy:.4f}, rel_diff={relative_diff:.2%}"
        )
        
        print(f"\nEnergy comparison:")
        print(f"  Modern: {results_modern.total_energy:.4f} MeV")
        print(f"  Legacy: {results_legacy.total_energy:.4f} MeV")
        print(f"  Relative difference: {relative_diff:.2%}")
    
    @pytest.mark.slow
    def test_medium_nucleus_convergence(self):
        """Test convergence for slightly larger nucleus (Ca-40)."""
        nucleus = Nucleus(protons=20, neutrons=20)
        
        calc = HFBFFT(
            nucleus=nucleus,
            force_name="SLy4",
            nx=16, ny=16, nz=16,
            dx=0.8, dy=0.8, dz=0.8,
            npsi=(30, 30),
        )
        
        calc.initialize_wavefunctions(method="harmonic_oscillator")
        
        results = calc.run(
            max_iterations=50,
            convergence_threshold=1e-5,
            print_interval=10,
            use_legacy=False,
        )
        
        # Check that energy is reasonable for Ca-40
        # SLy4 should give around -300 to -400 MeV for Ca-40
        assert -500 < results.total_energy < -200, (
            f"Unreasonable energy for Ca-40: {results.total_energy:.2f} MeV"
        )
        
        # Check that RMS radius is reasonable (should be around 3-4 fm)
        assert 2.0 < results.rms_radius_total < 5.0, (
            f"Unreasonable RMS radius: {results.rms_radius_total:.2f} fm"
        )
        
        print(f"\nCa-40 results:")
        print(f"  Energy: {results.total_energy:.2f} MeV")
        print(f"  RMS radius: {results.rms_radius_total:.2f} fm")
        print(f"  Converged: {results.converged}")
        print(f"  Iterations: {results.iterations}")


@pytest.mark.integration
class TestDeterminism:
    """Test that calculations are deterministic with fixed RNG seed."""
    
    def test_repeated_runs_identical(self):
        """Test that repeated runs with same seed give identical results."""
        nucleus = Nucleus(protons=4, neutrons=4)
        
        def run_calc(seed=42):
            calc = HFBFFT(
                nucleus=nucleus,
                force_name="SLy4",
                nx=8, ny=8, nz=8,
                dx=1.0, dy=1.0, dz=1.0,
                npsi=(8, 8),
            )
            calc.initialize_wavefunctions(method="random", seed=seed)
            results = calc.run(
                max_iterations=10,
                convergence_threshold=1e-4,
                print_interval=100,
                use_legacy=False,
            )
            return results
        
        # Run twice with same seed
        results1 = run_calc(seed=42)
        results2 = run_calc(seed=42)
        
        # Energies should be identical
        np.testing.assert_allclose(
            results1.total_energy,
            results2.total_energy,
            rtol=1e-10,
            err_msg="Results not deterministic with same RNG seed"
        )
        
        print(f"\nDeterminism test passed:")
        print(f"  Run 1: {results1.total_energy:.10f} MeV")
        print(f"  Run 2: {results2.total_energy:.10f} MeV")


@pytest.mark.integration
class TestBasicPhysics:
    """Test basic physics properties."""
    
    def test_particle_number_conservation(self):
        """Test that particle numbers are conserved."""
        nucleus = Nucleus(protons=8, neutrons=8)
        
        calc = HFBFFT(
            nucleus=nucleus,
            force_name="SLy4",
            nx=12, ny=12, nz=12,
            dx=1.0, dy=1.0, dz=1.0,
            npsi=(16, 16),
        )
        
        calc.initialize_wavefunctions(method="harmonic_oscillator")
        
        results = calc.run(
            max_iterations=30,
            convergence_threshold=1e-4,
            print_interval=10,
            use_legacy=False,
        )
        
        # Integrate densities to get particle numbers
        grid = calc.grid
        rho_n = calc.state.rho[0]  # Neutron density
        rho_p = calc.state.rho[1]  # Proton density
        
        N = float(jnp.sum(rho_n) * grid.wxyz)
        Z = float(jnp.sum(rho_p) * grid.wxyz)
        
        # Should be close to target values
        np.testing.assert_allclose(N, nucleus.neutrons, rtol=0.05,
            err_msg=f"Neutron number not conserved: {N:.2f} vs {nucleus.neutrons}")
        np.testing.assert_allclose(Z, nucleus.protons, rtol=0.05,
            err_msg=f"Proton number not conserved: {Z:.2f} vs {nucleus.protons}")
        
        print(f"\nParticle number check:")
        print(f"  Target N={nucleus.neutrons}, got {N:.2f}")
        print(f"  Target Z={nucleus.protons}, got {Z:.2f}")
    
    def test_energy_components_reasonable(self):
        """Test that energy components have reasonable magnitudes."""
        nucleus = Nucleus(protons=8, neutrons=8)
        
        calc = HFBFFT(
            nucleus=nucleus,
            force_name="SLy4",
            nx=12, ny=12, nz=12,
            dx=1.0, dy=1.0, dz=1.0,
        )
        
        calc.initialize_wavefunctions(method="harmonic_oscillator")
        
        results = calc.run(
            max_iterations=30,
            convergence_threshold=1e-4,
            print_interval=10,
            use_legacy=False,
        )
        
        # Check energy components
        total = results.total_energy
        kinetic = results.kinetic_energy
        potential = results.potential_energy
        coulomb = results.coulomb_energy
        
        # Kinetic should be positive
        assert kinetic > 0, f"Kinetic energy should be positive: {kinetic:.2f}"
        
        # Coulomb should be positive for protons
        assert coulomb >= 0, f"Coulomb energy should be non-negative: {coulomb:.2f}"
        
        # Total should be sum of components (approximately)
        # Note: there are additional rearrangement terms
        print(f"\nEnergy components:")
        print(f"  Total:     {total:.2f} MeV")
        print(f"  Kinetic:   {kinetic:.2f} MeV")
        print(f"  Potential: {potential:.2f} MeV")
        print(f"  Coulomb:   {coulomb:.2f} MeV")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
