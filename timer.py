#!/usr/bin/env python3
"""
JAX HFB Solver Benchmark Script - Direct Function Instrumentation

This script benchmarks individual components by directly modifying the core functions
in your static.py and related modules with timing instrumentation.
"""

import jax
import jax.numpy as jnp
import time
import os
from dataclasses import asdict
import functools

# Import all necessary modules
from reader import read_yaml
from params import init_params
from forces import init_forces
from grids import init_grids
from densities import init_densities, add_density
from meanfield import init_meanfield, skyrme
from levels import init_levels
from static import init_static, harmosc
from coulomb import init_coulomb
from moment import init_moment
from energies import init_energies
from pairs import Pairs
from inout import sp_properties


class ComponentTimer:
    """Timer that tracks individual component execution times"""
    
    def __init__(self):
        self.times = {}
        self.calls = {}
        self.active_timers = {}
        
    def start_timer(self, component_name):
        """Start timing a component"""
        self.active_timers[component_name] = time.time()
        
    def end_timer(self, component_name, result):
        """End timing a component and record the result"""
        if component_name in self.active_timers:
            elapsed = time.time() - self.active_timers[component_name]
            del self.active_timers[component_name]
            
            # Ensure JAX computation completes
            self._block_until_ready(result)
            
            if component_name not in self.times:
                self.times[component_name] = 0.0
                self.calls[component_name] = 0
                
            self.times[component_name] += elapsed
            self.calls[component_name] += 1
            
        return result
    
    def _block_until_ready(self, obj):
        """Recursively block on JAX arrays in any structure"""
        if isinstance(obj, tuple):
            for item in obj:
                self._block_until_ready(item)
        elif hasattr(obj, '__dict__'):
            # For dataclass objects, block on all JAX array attributes
            for attr_name in obj.__dict__:
                attr_val = getattr(obj, attr_name)
                if isinstance(attr_val, jnp.ndarray):
                    jax.block_until_ready(attr_val)
        elif isinstance(obj, jnp.ndarray):
            jax.block_until_ready(obj)
    
    def get_stats(self):
        """Get timing statistics"""
        stats = {}
        total_time = sum(self.times.values())
        
        for name in self.times:
            stats[name] = {
                'total_time': self.times[name],
                'avg_time_ms': (self.times[name] / self.calls[name]) * 1000,
                'calls': self.calls[name],
                'percentage': (self.times[name] / total_time) * 100 if total_time > 0 else 0
            }
        return stats
    
    def print_summary(self):
        """Print detailed timing results"""
        stats = self.get_stats()
        total_time = sum(self.times.values())
        total_calls = sum(self.calls.values())
        
        print("\n" + "="*80)
        print("COMPONENT-LEVEL BENCHMARK RESULTS")
        print("="*80)
        print(f"{'Component':<20} {'Total (s)':<10} {'Avg (ms)':<10} {'Calls':<8} {'% Total':<8}")
        print("-"*80)
        
        # Sort by total time
        sorted_components = sorted(stats.keys(), 
                                 key=lambda x: stats[x]['total_time'], 
                                 reverse=True)
        
        for name in sorted_components:
            s = stats[name]
            print(f"{name:<20} {s['total_time']:<10.3f} {s['avg_time_ms']:<10.2f} "
                  f"{s['calls']:<8} {s['percentage']:<8.1f}")
        
        print("-"*80)
        print(f"{'TOTAL':<20} {total_time:<10.3f} {'':<10} {total_calls:<8} {'100.0':<8}")
        print("="*80)
        
        if stats:
            print("\nPERFORMANCE INSIGHTS:")
            print("-"*40)
            bottleneck = max(stats.keys(), key=lambda x: stats[x]['total_time'])
            print(f"Primary bottleneck: {bottleneck} ({stats[bottleneck]['percentage']:.1f}%)")
            print(f"Total computation time: {total_time:.2f} seconds")


def create_timed_function(original_func, timer, component_name):
    """Create a timed version of a function"""
    
    @functools.wraps(original_func)
    def timed_wrapper(*args, **kwargs):
        timer.start_timer(component_name)
        result = original_func(*args, **kwargs)
        return timer.end_timer(component_name, result)
    
    return timed_wrapper


def calculate_total_energy(densities, levels, forces, grids):
    """Calculate total binding energy like test.py"""
    # Kinetic energy
    e_kin = jnp.sum(levels.wocc * levels.wstates * levels.sp_kinetic)
    
    # Single-particle energy sum (includes potential)
    e_sp = jnp.sum(levels.wocc * levels.wstates * levels.sp_energy)
    
    # Total energy is half the sum (avoids double counting)
    total_energy = 0.5 * (e_kin + e_sp)
    
    return total_energy


def create_benchmark_statichf(timer):
    """Create a benchmarking version of statichf that times individual components"""
    
    def benchmark_statichf(coulomb, densities, energies, forces, grids, levels, 
                          meanfield, moment, params, static):
        """Instrumented version of statichf with component timing"""
        
        # Initialize pairs
        pairs = Pairs()
        
        # Initial setup
        timer.start_timer("initialization")
        levels = harmosc(grids, levels, params, static)
        densities = add_density(densities, grids, levels)
        meanfield, coulomb = skyrme(coulomb, densities, forces, grids, meanfield, params, static)
        timer.end_timer("initialization", (levels, densities, meanfield))
        
        # Get convergence parameters
        max_iterations = int(static.maxiter)
        convergence_threshold = float(static.serr)
        converged = False
        
        # Mixing parameters
        addnew = 0.2
        addco = 0.8
        
        print(f"\nRunning HFB calculation (target convergence: {convergence_threshold:.1e})")
        print("-" * 70)
        print(f"{'Iter':<6} {'Energy (MeV)':<15} {'Convergence':<15}")
        print("-" * 70)
        
        # Main iteration loop
        for iteration in range(1, max_iterations + 1):
            
            # Update iteration parameters
            params_dict = asdict(params)
            params_dict['iteration'] = iteration
            params = type(params)(**params_dict)
            
            # Update static parameters
            static_dict = asdict(static)
            static_dict['tdiag'] = (iteration <= static.inidiag)
            static_dict['delesum'] = 0.0
            static_dict['sumflu'] = 0.0
            static = type(static)(**static_dict)
            
            # Update forces
            forces_dict = asdict(forces)
            if forces.ipair != 0:
                forces_dict['tbcs'] = (iteration <= static.inibcs)
            forces = type(forces)(**forces_dict)
            
            # Component 1: Gradient step
            timer.start_timer("grstep")
            from static import grstep
            levels, static = grstep(forces, grids, levels, meanfield, params, static)
            timer.end_timer("grstep", (levels, static))
            
            # Component 2: Diagonalization
            timer.start_timer("diagstep")
            from static import diagstep
            energies, levels, static = diagstep(energies, forces, grids, levels, static, 
                                              static.tdiag, True)
            timer.end_timer("diagstep", (energies, levels, static))
            
            # Component 3: Pairing (if enabled)
            if forces.ipair != 0:
                timer.start_timer("pairing")
                from pairs import pair
                levels, pairs = pair(levels, meanfield, forces, params, grids, pairs)
                timer.end_timer("pairing", (levels, pairs))
            
            # Component 4: Single-particle properties
            timer.start_timer("sp_properties")
            levels = sp_properties(forces, grids, levels, moment)
            timer.end_timer("sp_properties", levels)
            
            # Store old densities for mixing
            old_densities = densities
            
            # Component 5: Density calculation
            timer.start_timer("add_density")
            new_densities = add_density(densities, grids, levels)
            timer.end_timer("add_density", new_densities)
            
            # Density mixing for stability
            if iteration > 1:
                densities_dict = asdict(new_densities)
                densities_dict['rho'] = addnew * new_densities.rho + addco * old_densities.rho
                densities_dict['tau'] = addnew * new_densities.tau + addco * old_densities.tau
                densities_dict['chi'] = addnew * new_densities.chi + addco * old_densities.chi
                densities_dict['current'] = addnew * new_densities.current + addco * old_densities.current
                densities_dict['sdens'] = addnew * new_densities.sdens + addco * old_densities.sdens
                densities_dict['sodens'] = addnew * new_densities.sodens + addco * old_densities.sodens
                densities = type(new_densities)(**densities_dict)
            else:
                densities = new_densities
            
            # Component 6: Mean field calculation
            timer.start_timer("skyrme")
            meanfield, coulomb = skyrme(coulomb, densities, forces, grids, meanfield, params, static)
            timer.end_timer("skyrme", (meanfield, coulomb))
            
            # Calculate total energy properly
            total_energy = calculate_total_energy(densities, levels, forces, grids)
            
            # Update energies object with correct total energy
            energies_dict = asdict(energies)
            energies_dict['ehf'] = float(total_energy)
            energies = type(energies)(**energies_dict)
            
            # Check convergence
            convergence_metric = float(energies.efluct1[0])
            total_energy = float(total_energy)  # Use the calculated energy
            
            print(f"{iteration:<6} {total_energy:<15.6f} {convergence_metric:<15.2e}")
            
            if convergence_metric < convergence_threshold:
                print("-" * 70)
                print(f"Converged after {iteration} iterations!")
                print(f"Final energy: {total_energy:.6f} MeV")
                print(f"Final convergence: {convergence_metric:.2e}")
                converged = True
                break
            
            # Check for numerical problems
            if jnp.isnan(convergence_metric) or jnp.isnan(total_energy):
                print(f"Numerical instability at iteration {iteration}")
                break
            
            # Safety check - avoid infinite loops during benchmark
            if iteration >= 100:
                print("Stopping at 100 iterations for benchmark purposes")
                break
        
        print("-" * 70)
        
        if not converged:
            print(f"Reached maximum iterations without convergence")
            print(f"Final convergence: {convergence_metric:.2e} (target: {convergence_threshold:.2e})")
        
        return (coulomb, densities, energies, forces, grids, levels, 
                meanfield, moment, params, static)
    
    return benchmark_statichf


def create_config():
    """Create configuration for benchmark"""
    try:
        config = read_yaml('_config.yml')
        print("Using configuration from _config.yml")
        
        # Ensure reasonable convergence threshold
        if 'static' not in config:
            config['static'] = {}
        if config['static'].get('serr', 1e-20) < 1e-8:
            config['static']['serr'] = 1e-6
            print(f"  → Adjusting convergence threshold to {config['static']['serr']:.1e}")
        
    except FileNotFoundError:
        print("Using fallback configuration")
        config = {
            'params': {'imode': 1},
            'force': {
                'name': 'SLy4',
                'ipair': 6,      # DDDI pairing interaction
                'v0prot': -300.0, # Proton pairing strength
                'v0neut': -300.0, # Neutron pairing strength
                'tbcs': True
            },
            'grids': {
                'nx': 32, 'ny': 32, 'nz': 32,
                'dx': 0.8, 'dy': 0.8, 'dz': 0.8
            },
            'levels': {
                'nneut': 4,   # Be-8 as shown in your output
                'nprot': 4
            },
            'static': {
                'maxiter': 50,      # Reduced for benchmark
                'serr': 1.0e-6,     # Reasonable convergence
                'x0dmp': 0.45,
                'inibcs': 30,
                'inidiag': 30
            }
        }
    return config


def run_hfb_benchmark():
    """Main benchmark function"""
    
    print("JAX HFB Nuclear Structure Solver - Component Benchmark")
    print("=" * 80)
    
    # Enable 64-bit precision
    jax.config.update('jax_enable_x64', True)
    print("JAX 64-bit precision enabled")
    
    # Load configuration and initialize system
    config = create_config()
    
    print("\nInitializing system...")
    params = init_params(**config.get('params', {}))
    forces = init_forces(params, **config.get('force', {}))
    grids = init_grids(params, **config.get('grids', {}))
    densities = init_densities(grids)
    meanfield = init_meanfield(grids)
    levels = init_levels(grids, **config.get('levels', {}))
    forces, static = init_static(forces, levels, **config.get('static', {}))
    coulomb = init_coulomb(grids)
    energies = init_energies()
    moment = init_moment(jnp.array([0.0, 0.0, 0.0]))
    
    # Print system info
    print(f"\nSystem: Z={levels.nprot}, N={levels.nneut}, A={levels.mass_number}")
    print(f"Grid: {grids.nx}³ points, spacing {grids.dx:.2f} fm")
    print(f"Force: {forces.name}")
    print(f"Pairing: {'Enabled' if forces.ipair != 0 else 'Disabled'}")
    
    # Warm up JAX compilation
    print("\nPerforming JIT warm-up...")
    timer_warmup = ComponentTimer()
    benchmark_statichf_func = create_benchmark_statichf(timer_warmup)
    
    # Run a few iterations for warm-up
    temp_static = asdict(static)
    temp_static['maxiter'] = 3  # Just 3 iterations for warm-up
    temp_static['serr'] = 1.0   # Don't converge during warm-up
    warmup_static = type(static)(**temp_static)
    
    benchmark_statichf_func(coulomb, densities, energies, forces, grids, levels,
                           meanfield, moment, params, warmup_static)
    print("JIT compilation completed")
    
    # Reset system for actual benchmark
    densities = init_densities(grids)
    meanfield = init_meanfield(grids)
    energies = init_energies()
    
    # Create fresh timer for actual benchmark
    timer = ComponentTimer()
    benchmark_statichf_func = create_benchmark_statichf(timer)
    
    # Run the actual benchmark
    print(f"\n{'='*80}")
    print("STARTING COMPONENT BENCHMARK")
    print(f"{'='*80}")
    
    benchmark_start = time.time()
    
    try:
        result = benchmark_statichf_func(
            coulomb, densities, energies, forces, grids, levels,
            meanfield, moment, params, static
        )
        
        total_time = time.time() - benchmark_start
        
        # Extract final results
        (coulomb_f, densities_f, energies_f, forces_f, grids_f, levels_f,
         meanfield_f, moment_f, params_f, static_f) = result
        
        print(f"\nFinal Results:")
        print(f"Total binding energy: {float(energies_f.ehf):.6f} MeV")
        print(f"Final convergence: {float(energies_f.efluct1[0]):.2e}")
        print(f"Iterations completed: {params_f.iteration}")
        print(f"Total benchmark time: {total_time:.2f} seconds")
        
        # Print component timing analysis
        timer.print_summary()
        
        return timer
        
    except Exception as e:
        print(f"\nError during benchmark: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("Starting HFB component-level benchmark...")
    
    try:
        timer = run_hfb_benchmark()
        if timer:
            print("\nBenchmark completed successfully!")
        else:
            print("\nBenchmark failed.")
    
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
    
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("Benchmark finished.")