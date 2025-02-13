import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from hpsi01_implementations import HPSI01Implementations

@dataclass
class HPSI01BenchmarkResult:
    execution_times: Dict[str, List[float]]
    average_times: Dict[str, float]
    speedup: float
    numerical_diffs: Dict[str, float]  
    relative_errors: Dict[str, float]  
    peak_memory: Dict[str, float]      

class HPSI01Benchmark:
    
    def __init__(self, grids, meanfield):
        """Initialize benchmark with grid and meanfield objects"""
        self.grids = grids
        self.meanfield = meanfield
        self.implementations = HPSI01Implementations()

    def generate_physical_wavefunction(self) -> jax.Array:
        nx, ny, nz = self.grids.nx, self.grids.ny, self.grids.nz
        
        # Create coordinate grids centered at 0
        x = self.grids.x
        y = self.grids.y
        z = self.grids.z
        
        # Create meshgrid for vectorized operations
        X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
        R = jnp.sqrt(X**2 + Y**2 + Z**2)
        
        # Initialize wavefunction array
        psi = jnp.zeros((2, nx, ny, nz), dtype=jnp.complex128)
        
        # Generate realistic nuclear wavefunction features
        # Woods-Saxon like radial part
        R0 = 5.0  # Nuclear radius (fm)
        a = 0.5   # Surface diffuseness (fm)
        ws_form = 1.0 / (1.0 + jnp.exp((R - R0)/a))
        
        # Angular parts using spherical harmonics
        theta = jnp.arctan2(jnp.sqrt(X**2 + Y**2), Z)
        phi = jnp.arctan2(Y, X)
        
        # Spin up component: s-wave plus d-wave mixture
        psi = psi.at[0,...].set(
            ws_form * (
                0.8 * jnp.ones_like(R) +  # s-wave
                0.2 * (3*jnp.cos(theta)**2 - 1)  # d-wave
            )
        )
        
        # Spin down component: p-wave mixture
        psi = psi.at[1,...].set(
            ws_form * (
                jnp.sin(theta) * (
                    jnp.cos(phi) + 1j * jnp.sin(phi)
                )
            )
        )

        # Normalize the wavefunction
        norm = jnp.sqrt(jnp.sum(jnp.abs(psi)**2) * self.grids.wxyz)
        psi = psi / norm
        
        return psi

    def initialize_meanfield(self):
        nx, ny, nz = self.grids.nx, self.grids.ny, self.grids.nz
        
        # Create coordinate grids
        X, Y, Z = jnp.meshgrid(self.grids.x, self.grids.y, self.grids.z, indexing='ij')
        R = jnp.sqrt(X**2 + Y**2 + Z**2)
        
        # Woods-Saxon parameters
        R0 = 5.0  # Nuclear radius (fm)
        a = 0.5   # Surface diffuseness (fm)
        V0 = -50  # Potential depth (MeV)
        
        # Central potential (Woods-Saxon)
        self.meanfield.upot = self.meanfield.upot.at[...].set(
            V0 / (1 + jnp.exp((R - R0)/a))
        )
        
        # Effective mass (typically around nucleon mass)
        self.meanfield.bmass = self.meanfield.bmass.at[...].set(
            20.0 * jnp.ones_like(self.meanfield.bmass)  # In MeV
        )
        
        # Spin-orbit potential (realistic nuclear magnitude)
        Vso = 5.0  # Spin-orbit strength (MeV)
        r_vec = jnp.stack([X/R, Y/R, Z/R])
        for i in range(3):
            self.meanfield.wlspot = self.meanfield.wlspot.at[:,i,...].set(
                Vso * r_vec[i] / (1 + jnp.exp((R - R0)/a))
            )
            
        # Vector potential (typically small in nuclear systems)
        self.meanfield.aq = self.meanfield.aq.at[...].set(
            0.1 * jnp.ones_like(self.meanfield.aq)
        )
        
        # Spin-current coupling
        self.meanfield.spot = self.meanfield.spot.at[...].set(
            0.5 * jnp.ones_like(self.meanfield.spot)
        )
        
        # Pairing field (surface peaked)
        self.meanfield.v_pair = self.meanfield.v_pair.at[...].set(
            -2.0 * jnp.exp(-((R - R0)/2.0)**2)
        )
        
        return self.meanfield

    def verify_hpsi01_implementations(self, iq: int, weight: float, weightuv: float, 
                                    pinn: jax.Array) -> Tuple[Dict[str, float], Dict[str, float]]:
        
        # Run both implementations
        orig_out, orig_mf, orig_del = self.implementations.hpsi01_original(
            self.grids, self.meanfield, iq, weight, weightuv, pinn
        )
        opt_out, opt_mf, opt_del = self.implementations.hpsi01_optimized(
            self.grids, self.meanfield, iq, weight, weightuv, pinn
        )

        # Calculate differences for each output
        numerical_diffs = {
            'main': jnp.max(jnp.abs(orig_out - opt_out)),
            'meanfield': jnp.max(jnp.abs(orig_mf - opt_mf)),
            'pairing': jnp.max(jnp.abs(orig_del - opt_del))
        }
        
        # Calculate relative errors
        relative_errors = {
            'main': jnp.linalg.norm(orig_out - opt_out) / jnp.linalg.norm(orig_out),
            'meanfield': jnp.linalg.norm(orig_mf - opt_mf) / jnp.linalg.norm(orig_mf),
            'pairing': jnp.linalg.norm(orig_del - opt_del) / jnp.linalg.norm(orig_del)
        }

        return numerical_diffs, relative_errors

    def run_benchmark(self, n_runs: int=50, warming_runs: int=5) -> HPSI01BenchmarkResult:
        """Run complete benchmark comparing original and optimized HPSI01"""
        
        # Initialize test data
        iq = 0
        weight = 1.0
        weightuv = 0.5
        pinn = self.generate_physical_wavefunction()
        self.initialize_meanfield()
        
        # Warm up JIT compilation
        for _ in range(warming_runs):
            self.implementations.hpsi01_original(
                self.grids, self.meanfield, iq, weight, weightuv, pinn
            )
            self.implementations.hpsi01_optimized(
                self.grids, self.meanfield, iq, weight, weightuv, pinn
            )

        # Time both implementations
        original_times = []
        optimized_times = []
        
        for _ in range(n_runs):
            # Time original implementation
            start = time.perf_counter()
            orig_result = self.implementations.hpsi01_original(
                self.grids, self.meanfield, iq, weight, weightuv, pinn
            )
            jax.block_until_ready(orig_result)
            original_times.append(time.perf_counter() - start)
            
            # Time optimized implementation
            start = time.perf_counter()
            opt_result = self.implementations.hpsi01_optimized(
                self.grids, self.meanfield, iq, weight, weightuv, pinn
            )
            jax.block_until_ready(opt_result)
            optimized_times.append(time.perf_counter() - start)

        # Calculate statistics
        avg_original = np.mean(original_times)
        avg_optimized = np.mean(optimized_times)
        speedup = avg_original / avg_optimized
        
        # Verify numerical accuracy
        numerical_diffs, relative_errors = self.verify_hpsi01_implementations(
            iq, weight, weightuv, pinn
        )

        return HPSI01BenchmarkResult(
            execution_times={
                'original': original_times,
                'optimized': optimized_times
            },
            average_times={
                'original': avg_original,
                'optimized': avg_optimized
            },
            speedup=speedup,
            numerical_diffs=numerical_diffs,
            relative_errors=relative_errors,
            peak_memory={
                'original': 0.0,  # Would need memory profiler
                'optimized': 0.0  # Would need memory profiler
            }
        )

    def plot_results(self, results: HPSI01BenchmarkResult):
        """Create detailed visualization of benchmark results"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Execution Time Distribution
        box_data = [results.execution_times['original'], 
                   results.execution_times['optimized']]
        ax1.boxplot(box_data, labels=['Original', 'Optimized'])
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Execution Time Distribution')

        # 2. Individual Run Times
        x = range(1, len(results.execution_times['original']) + 1)
        ax2.plot(x, results.execution_times['original'], 'b-', label='Original')
        ax2.plot(x, results.execution_times['optimized'], 'r-', label='Optimized')
        ax2.set_xlabel('Run number')
        ax2.set_ylabel('Time (seconds)')
        ax2.set_title('Execution Times per Run')
        ax2.legend()

        # 3. Numerical Differences
        differences = list(results.numerical_diffs.values())
        ax3.bar(results.numerical_diffs.keys(), differences)
        ax3.set_yscale('log')
        ax3.set_title('Maximum Absolute Differences')
        ax3.tick_params(axis='x', rotation=45)

        # 4. Relative Errors
        errors = list(results.relative_errors.values())
        ax4.bar(results.relative_errors.keys(), errors)
        ax4.set_yscale('log')
        ax4.set_title('Relative Errors')
        ax4.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

    def print_summary(self, results: HPSI01BenchmarkResult):
        """Print detailed summary of benchmark results"""
        print("\n=== HPSI01 Benchmark Results ===")
        
        print("\nExecution Times:")
        print(f"Original:  {results.average_times['original']:.6f} seconds")
        print(f"Optimized: {results.average_times['optimized']:.6f} seconds")
        print(f"Speedup: {results.speedup:.2f}x")
        
        print("\nNumerical Accuracy:")
        for output, diff in results.numerical_diffs.items():
            print(f"{output} max difference: {diff:.2e}")
        
        print("\nRelative Errors:")
        for output, error in results.relative_errors.items():
            print(f"{output} relative error: {error:.2e}")
        
        print("\nDetailed Statistics:")
        for impl in ['original', 'optimized']:
            times = results.execution_times[impl]
            print(f"\n{impl.capitalize()}:")
            print(f"  Min: {min(times):.6f} seconds")
            print(f"  Max: {max(times):.6f} seconds")
            print(f"  Std Dev: {np.std(times):.6f} seconds")
            print(f"  95th percentile: {np.percentile(times, 95):.6f} seconds")

def main():
    """Main function to run HPSI01 benchmarks with different configurations"""
    import yaml
    from reader import read_yaml
    from params import init_params
    from forces import init_forces
    from grids import init_grids
    from meanfield import init_meanfield

    print("\n=== Starting HPSI01 Benchmark Suite ===\n")

    # Read configuration
    config = read_yaml('_config.yml')

    # Initialize required objects
    params = init_params(**config.get('params', {}))
    forces = init_forces(params, **config.get('force', {}))
    grids = init_grids(params, **config.get('grids', {}))
    meanfield = init_meanfield(grids)

    # Create benchmark instance
    benchmark = HPSI01Benchmark(grids, meanfield)

    # 1. Quick initial test
    print("Running quick benchmark (10 runs)...")
    quick_results = benchmark.run_benchmark(n_runs=10, warming_runs=2)
    benchmark.print_summary(quick_results)
    
    # 2. Detailed benchmark
    print("\nRunning detailed benchmark (100 runs)...")
    detailed_results = benchmark.run_benchmark(n_runs=100, warming_runs=5)
    benchmark.print_summary(detailed_results)
    benchmark.plot_results(detailed_results)

    # 3. Test with different grid sizes
    print("\nTesting with different grid sizes...")
    grid_sizes = [(32, 32, 32), (48, 48, 48), (64, 64, 64)]
    grid_results = {}
    
    for nx, ny, nz in grid_sizes:
        print(f"\nGrid size: {nx}x{ny}x{nz}")
        
        # Update grid configuration
        grids_config = config.get('grids', {}).copy()
        grids_config.update({'nx': nx, 'ny': ny, 'nz': nz})
        
        # Reinitialize objects with new grid size
        grids = init_grids(params, **grids_config)
        meanfield = init_meanfield(grids)
        benchmark = HPSI01Benchmark(grids, meanfield)
        
        # Run benchmark
        results = benchmark.run_benchmark(n_runs=20, warming_runs=2)
        benchmark.print_summary(results)
        grid_results[(nx, ny, nz)] = results

    # 4. Compare speedups across grid sizes
    print("\nSpeedup comparison across grid sizes:")
    print("Grid Size    | Speedup | Original (ms) | Optimized (ms)")
    print("-" * 55)
    for size, result in grid_results.items():
        nx, ny, nz = size
        orig_ms = result.average_times['original'] * 1000
        opt_ms = result.average_times['optimized'] * 1000
        print(f"{nx}x{ny}x{nz:3} | {result.speedup:7.2f}x | {orig_ms:11.2f} | {opt_ms:12.2f}")

    # 5. Plot scaling comparison
    plt.figure(figsize=(10, 6))
    grid_points = [(nx * ny * nz) for nx, ny, nz in grid_sizes]
    speedups = [result.speedup for result in grid_results.values()]
    
    plt.plot(grid_points, speedups, 'bo-')
    plt.xscale('log')
    plt.xlabel('Number of Grid Points')
    plt.ylabel('Speedup Factor')
    plt.title('HPSI01 Optimization Speedup vs Problem Size')
    plt.grid(True)
    plt.savefig('hpsi01_scaling.png')
    plt.close()

    # 6. Numerical accuracy analysis
    print("\nNumerical Accuracy Analysis:")
    print("\nMaximum relative errors across grid sizes:")
    print("Grid Size    | Main Output | Mean Field  | Pairing")
    print("-" * 55)
    for size, result in grid_results.items():
        nx, ny, nz = size
        errors = result.relative_errors
        print(f"{nx}x{ny}x{nz:3} | {errors['main']:.2e} | {errors['meanfield']:.2e} | {errors['pairing']:.2e}")

    # Save detailed results to file
    with open('hpsi01_benchmark_results.txt', 'w') as f:
        # Write summary header
        f.write("=== HPSI01 Benchmark Results ===\n\n")
        
        # Write grid size comparisons
        f.write("Grid Size Performance Comparison\n")
        f.write("-------------------------------\n")
        for size, result in grid_results.items():
            nx, ny, nz = size
            f.write(f"\nGrid size: {nx}x{ny}x{nz}\n")
            f.write(f"Speedup: {result.speedup:.2f}x\n")
            f.write(f"Original: {result.average_times['original']:.6f} seconds\n")
            f.write(f"Optimized: {result.average_times['optimized']:.6f} seconds\n")
            
            # Write numerical accuracy
            f.write("\nNumerical differences:\n")
            for output, diff in result.numerical_diffs.items():
                f.write(f"{output}: {diff:.2e}\n")
            
            f.write("\nRelative errors:\n")
            for output, error in result.relative_errors.items():
                f.write(f"{output}: {error:.2e}\n")
            f.write("\n" + "-"*50 + "\n")

    print("\nBenchmark complete! Detailed results saved to 'hpsi01_benchmark_results.txt'")
    print("Scaling plot saved as 'hpsi01_scaling.png'")

if __name__ == "__main__":
    main()