import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from hpsi_implementations import HPSIImplementations

@dataclass
class BenchmarkResult:
    execution_times: Dict[str, List[float]]
    average_times: Dict[str, float]
    speedup: float
    numerical_diff: float
    relative_error: float

class HPSIBenchmark:
    
    def __init__(self, grids, meanfield):
        """Initialize benchmark with grid and meanfield objects"""
        self.grids = grids
        self.meanfield = meanfield
        self.implementations = HPSIImplementations()

    def generate_physical_wavefunction(self) -> jax.Array:
        nx, ny, nz = self.grids.nx, self.grids.ny, self.grids.nz
        
        # Create coordinate grids centered at 0
        x = self.grids.x  # Already defined in grids as centered coordinates
        y = self.grids.y
        z = self.grids.z
        
        # Create meshgrid for vectorized operations
        X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
        
        # Initialize wavefunction array
        psi = jnp.zeros((2, nx, ny, nz), dtype=jnp.complex128)
        
        centers = [
            (0.0, 0.0, 0.0),   # s-like state
            (2.0, 0.0, 0.0),   # p-like state
            (0.0, 2.0, 0.0),
            (0.0, 0.0, 2.0),
            (-2.0, -2.0, 0.0)  # higher angular momentum state
        ]
        
        widths = [1.0, 1.5, 2.0, 1.2, 1.8]  # Different spatial extents
        phases = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]  # Phase differences
        
        # Build up the wavefunction with different contributions
        for i, (cx, cy, cz) in enumerate(centers):
            # Radial distance from center
            r2 = (X-cx)**2 + (Y-cy)**2 + (Z-cz)**2
            width = widths[i]
            phase = phases[i]
            
            # Add angular dependence for higher angular momentum states
            theta = jnp.arctan2(Y-cy, X-cx)
            phi = jnp.arctan2(jnp.sqrt((X-cx)**2 + (Y-cy)**2), Z-cz)
            
            # Different angular momentum contributions for different spin components
            base_function = jnp.exp(-r2/(2*width**2))
            
            # Spin up component: mixture of s and p-like states
            psi = psi.at[0,...].add(
                base_function * (1 + 0.5*jnp.cos(theta)) * 
                jnp.exp(1j * phase)
            )
            
            # Spin down component: different mixture
            psi = psi.at[1,...].add(
                base_function * (1 + 0.5*jnp.sin(2*theta)*jnp.sin(phi)) * 
                jnp.exp(1j * (phase + np.pi/3))
            )

        # Normalize the wavefunction
        norm = jnp.sqrt(jnp.sum(jnp.abs(psi)**2) * self.grids.wxyz)
        psi = psi / norm
        
        # Verify normalization
        final_norm = jnp.sum(jnp.abs(psi)**2) * self.grids.wxyz
        assert jnp.abs(final_norm - 1.0) < 1e-10, f"Normalization error: {final_norm}"
        
        return psi

    def initialize_meanfield(self):
        nx, ny, nz = self.grids.nx, self.grids.ny, self.grids.nz
        
        # Create coordinate grids
        X, Y, Z = jnp.meshgrid(self.grids.x, self.grids.y, self.grids.z, indexing='ij')
        R = jnp.sqrt(X**2 + Y**2 + Z**2)
        
        # Wood-Saxon like potential for upot
        R0 = 5.0  # fm
        a = 0.5   # fm
        V0 = -50  # MeV
        self.meanfield.upot = self.meanfield.upot.at[...].set(
            V0 / (1 + jnp.exp((R - R0)/a))
        )
        
        # Mass term (typically around nucleon mass)
        self.meanfield.bmass = self.meanfield.bmass.at[...].set(
            20.0 * jnp.ones_like(self.meanfield.bmass)
        )
        
        # Spin-orbit potential
        Vso = 5.0  # MeV
        r_vec = jnp.stack([X/R, Y/R, Z/R])
        for i in range(3):
            self.meanfield.wlspot = self.meanfield.wlspot.at[:,i,...].set(
                Vso * r_vec[i] / (1 + jnp.exp((R - R0)/a))
            )
            
        # Vector potential
        self.meanfield.aq = self.meanfield.aq.at[...].set(
            0.1 * jnp.ones_like(self.meanfield.aq)
        )
        
        # Spin-current coupling
        self.meanfield.spot = self.meanfield.spot.at[...].set(
            0.5 * jnp.ones_like(self.meanfield.spot)
        )
        
        # Pairing field
        self.meanfield.v_pair = self.meanfield.v_pair.at[...].set(
            -2.0 * jnp.exp(-R**2/(2*R0**2))
        )
        
        return self.meanfield

    def generate_test_inputs(self) -> Tuple[Any, ...]:
        """Generate test inputs for HPSI functions using physically plausible data"""
        # Initialize meanfield with realistic values
        self.meanfield = self.initialize_meanfield()
        iq = 0  # Test with first isospin value
        weight = 1.0
        weightuv = 0.5
        pinn = self.generate_physical_wavefunction()
        return iq, weight, weightuv, pinn

    def verify_implementations(self, iq: int, weight: float, weightuv: float, pinn: jax.Array) -> Tuple[float, float, float]:
        """Verify that both implementations produce valid, non-zero results"""
        # Run both implementations
        orig_out, orig_mf = self.implementations.hpsi00_original(
            self.grids, self.meanfield, iq, weight, weightuv, pinn
        )
        opt_out, opt_mf = self.implementations.hpsi00_optimized(
            self.grids, self.meanfield, iq, weight, weightuv, pinn
        )

        # Check for non-zero outputs
        orig_magnitude = jnp.mean(jnp.abs(orig_out))
        opt_magnitude = jnp.mean(jnp.abs(opt_out))
        
        if orig_magnitude < 1e-10:
            raise ValueError(f"Original implementation producing near-zero results: {orig_magnitude}")
        if opt_magnitude < 1e-10:
            raise ValueError(f"Optimized implementation producing near-zero results: {opt_magnitude}")
            
        # Check for meaningful operations (output should be different from input)
        input_output_diff = jnp.mean(jnp.abs(orig_out - pinn))
        if input_output_diff < 1e-10:
            raise ValueError("Output is nearly identical to input, suggesting operations may not be occurring")

        # Calculate differences between implementations
        output_diff = jnp.max(jnp.abs(orig_out - opt_out))
        mf_diff = jnp.max(jnp.abs(orig_mf - opt_mf))
        
        # Calculate relative error
        relative_error = (jnp.linalg.norm(orig_out - opt_out) / 
                        jnp.linalg.norm(orig_out))

        # Print detailed verification info
        print("\nVerification Details:")
        print(f"Original implementation magnitude: {orig_magnitude:.2e}")
        print(f"Optimized implementation magnitude: {opt_magnitude:.2e}")
        print(f"Input-output difference: {input_output_diff:.2e}")
        print(f"Implementation difference: {output_diff:.2e}")
        
        return output_diff, mf_diff, relative_error

    def time_implementation(self, func, inputs: Tuple, n_runs: int=10) -> List[float]:
        """Time a single implementation"""
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            result = func(self.grids, self.meanfield, *inputs)
            jax.block_until_ready(result)
            end = time.perf_counter()
            times.append(end - start)
        return times

    def run_benchmark(self, n_runs: int=10, warming_runs: int=2) -> BenchmarkResult:
        """Run complete benchmark of both implementations"""
        # Generate test inputs
        inputs = self.generate_test_inputs()
        
        # Warm up JIT compilation
        for _ in range(warming_runs):
            self.implementations.hpsi00_original(self.grids, self.meanfield, *inputs)
            self.implementations.hpsi00_optimized(self.grids, self.meanfield, *inputs)

        # Time both implementations
        original_times = self.time_implementation(
            self.implementations.hpsi00_original, inputs, n_runs
        )
        optimized_times = self.time_implementation(
            self.implementations.hpsi00_optimized, inputs, n_runs
        )

        # Calculate averages
        avg_original = np.mean(original_times)
        avg_optimized = np.mean(optimized_times)
        
        # Calculate speedup
        speedup = avg_original / avg_optimized

        # Verify numerical accuracy
        output_diff, mf_diff, relative_error = self.verify_implementations(*inputs)

        return BenchmarkResult(
            execution_times={
                'original': original_times,
                'optimized': optimized_times
            },
            average_times={
                'original': avg_original,
                'optimized': avg_optimized
            },
            speedup=speedup,
            numerical_diff=max(output_diff, mf_diff),
            relative_error=relative_error
        )

    def plot_results(self, results: BenchmarkResult, save_path: str=None):
        """Plot benchmark results"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot timing distributions
        ax1.boxplot([results.execution_times['original'], 
                    results.execution_times['optimized']],
                   labels=['Original', 'Optimized'])
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Execution Time Distribution')

        # Plot individual runs
        x = range(1, len(results.execution_times['original']) + 1)
        ax2.plot(x, results.execution_times['original'], 'b-', label='Original')
        ax2.plot(x, results.execution_times['optimized'], 'r-', label='Optimized')
        ax2.set_xlabel('Run number')
        ax2.set_ylabel('Time (seconds)')
        ax2.set_title('Execution Times per Run')
        ax2.legend()

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def print_summary(self, results: BenchmarkResult):
        """Print a summary of benchmark results"""
        print("\n=== HPSI Benchmark Results ===")
        print(f"\nAverage Execution Times:")
        print(f"Original:  {results.average_times['original']:.6f} seconds")
        print(f"Optimized: {results.average_times['optimized']:.6f} seconds")
        print(f"\nSpeedup: {results.speedup:.2f}x")
        print(f"\nNumerical Accuracy:")
        print(f"Maximum Absolute Difference: {results.numerical_diff:.2e}")
        print(f"Relative Error: {results.relative_error:.2e}")
        
        # Additional statistics
        print("\nTiming Statistics:")
        for impl in ['original', 'optimized']:
            times = results.execution_times[impl]
            print(f"\n{impl.capitalize()}:")
            print(f"  Min: {min(times):.6f} seconds")
            print(f"  Max: {max(times):.6f} seconds")
            print(f"  Std Dev: {np.std(times):.6f} seconds")

def run_example_benchmark(grids, meanfield):
    """Run an example benchmark with the provided grids and meanfield"""
    benchmark = HPSIBenchmark(grids, meanfield)
    results = benchmark.run_benchmark(n_runs=50, warming_runs=5)
    benchmark.print_summary(results)
    benchmark.plot_results(results)
    return results

def main():
    """Main function to run benchmarks with different configurations"""
    # Import necessary modules for grid and meanfield setup
    from reader import read_yaml
    from params import init_params
    from forces import init_forces
    from grids import init_grids
    from meanfield import init_meanfield

    # Read configuration
    config = read_yaml('_config.yml')

    # Initialize required objects
    params = init_params(**config.get('params', {}))
    forces = init_forces(params, **config.get('force', {}))
    grids = init_grids(params, **config.get('grids', {}))
    meanfield = init_meanfield(grids)

    print("\nRunning quick benchmark (10 runs)...")
    benchmark = HPSIBenchmark(grids, meanfield)
    quick_results = benchmark.run_benchmark(n_runs=10, warming_runs=2)
    benchmark.print_summary(quick_results)

    print("\nRunning detailed benchmark (100 runs)...")
    detailed_results = benchmark.run_benchmark(n_runs=100, warming_runs=5)
    benchmark.print_summary(detailed_results)
    benchmark.plot_results(detailed_results, save_path="benchmark_results.png")

    # Test with different grid sizes
    print("\nTesting with different grid sizes...")
    grid_sizes = [(32, 32, 32), (48, 48, 48), (64, 64, 64)]
    
    for nx, ny, nz in grid_sizes:
        print(f"\nGrid size: {nx}x{ny}x{nz}")
        # Update grid configuration
        grids_config = config.get('grids', {}).copy()
        grids_config.update({
            'nx': nx,
            'ny': ny,
            'nz': nz
        })
        
        # Reinitialize objects with new grid size
        grids = init_grids(params, **grids_config)
        meanfield = init_meanfield(grids)
        benchmark = HPSIBenchmark(grids, meanfield)
        
        # Run benchmark
        results = benchmark.run_benchmark(n_runs=20, warming_runs=2)
        benchmark.print_summary(results)
        print(f"\nGrid points: {nx * ny * nz}")

if __name__ == "__main__":
    main()