"""
Performance profiling utilities for jax-hfbfft.

This module provides tools for benchmarking and profiling the HFB calculations:
- Timer context manager and decorator for measuring execution time
- Detailed breakdown profiler for iteration components
- JAX profiler integration for GPU/TPU tracing
- Memory usage tracking
"""

import time
import functools
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
import jax
import jax.numpy as jnp


@dataclass
class TimingStats:
    """Statistics for a single timed operation."""
    name: str
    total_time: float = 0.0
    call_count: int = 0
    min_time: float = float('inf')
    max_time: float = 0.0
    times: List[float] = field(default_factory=list)
    
    @property
    def avg_time(self) -> float:
        return self.total_time / max(self.call_count, 1)
    
    @property
    def std_time(self) -> float:
        if len(self.times) < 2:
            return 0.0
        avg = self.avg_time
        return (sum((t - avg)**2 for t in self.times) / len(self.times)) ** 0.5
    
    def add_timing(self, elapsed: float):
        self.total_time += elapsed
        self.call_count += 1
        self.min_time = min(self.min_time, elapsed)
        self.max_time = max(self.max_time, elapsed)
        self.times.append(elapsed)
    
    def __repr__(self):
        return (f"{self.name}: {self.total_time:.3f}s total, "
                f"{self.avg_time*1000:.2f}ms avg, "
                f"{self.call_count} calls")


class Profiler:
    """
    Global profiler for tracking timing across the HFB calculation.
    
    Usage:
        profiler = Profiler()
        
        with profiler.timer("compute_densities"):
            compute_densities(...)
            
        # Or as decorator
        @profiler.profile("gradient_step")
        def gradient_step(...):
            ...
            
        # Print report
        profiler.report()
    """
    
    _instance: Optional['Profiler'] = None
    
    def __init__(self):
        self.stats: Dict[str, TimingStats] = {}
        self.enabled = True
        self.iteration_times: List[float] = []
        self.start_time: Optional[float] = None
        self._current_iteration_start: Optional[float] = None
        
    @classmethod
    def get_instance(cls) -> 'Profiler':
        """Get or create the global profiler instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset(cls):
        """Reset the global profiler."""
        cls._instance = None
    
    def start_calculation(self):
        """Mark the start of a calculation."""
        self.start_time = time.perf_counter()
        self.stats.clear()
        self.iteration_times.clear()
    
    def start_iteration(self):
        """Mark the start of an iteration."""
        self._current_iteration_start = time.perf_counter()
    
    def end_iteration(self):
        """Mark the end of an iteration."""
        if self._current_iteration_start is not None:
            elapsed = time.perf_counter() - self._current_iteration_start
            self.iteration_times.append(elapsed)
            self._current_iteration_start = None
    
    @contextmanager
    def timer(self, name: str):
        """Context manager for timing a block of code."""
        if not self.enabled:
            yield
            return
            
        # Block until any async operations complete for accurate timing
        jax.block_until_ready(jnp.array(0))
        
        start = time.perf_counter()
        try:
            yield
        finally:
            # Block until computation completes
            jax.block_until_ready(jnp.array(0))
            elapsed = time.perf_counter() - start
            
            if name not in self.stats:
                self.stats[name] = TimingStats(name)
            self.stats[name].add_timing(elapsed)
    
    def profile(self, name: str):
        """Decorator for timing a function."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with self.timer(name):
                    return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def record(self, name: str, elapsed: float):
        """Manually record a timing."""
        if not self.enabled:
            return
        if name not in self.stats:
            self.stats[name] = TimingStats(name)
        self.stats[name].add_timing(elapsed)
    
    def report(self, top_n: int = 20) -> str:
        """Generate a profiling report."""
        if not self.stats:
            return "No timing data collected."
        
        lines = []
        lines.append("=" * 70)
        lines.append("PERFORMANCE PROFILE REPORT")
        lines.append("=" * 70)
        
        total_time = time.perf_counter() - self.start_time if self.start_time else 0
        lines.append(f"\nTotal elapsed time: {total_time:.2f}s")
        
        if self.iteration_times:
            avg_iter = sum(self.iteration_times) / len(self.iteration_times)
            lines.append(f"Number of iterations: {len(self.iteration_times)}")
            lines.append(f"Average iteration time: {avg_iter*1000:.2f}ms")
            # Exclude first iteration (JIT compilation)
            if len(self.iteration_times) > 1:
                avg_iter_no_jit = sum(self.iteration_times[1:]) / len(self.iteration_times[1:])
                lines.append(f"Average iteration time (excluding first): {avg_iter_no_jit*1000:.2f}ms")
        
        lines.append("\n" + "-" * 70)
        lines.append("TIMING BREAKDOWN (sorted by total time)")
        lines.append("-" * 70)
        
        # Sort by total time
        sorted_stats = sorted(self.stats.values(), key=lambda s: s.total_time, reverse=True)
        
        lines.append(f"{'Operation':<35} {'Total':>10} {'Avg':>10} {'Calls':>8} {'%':>6}")
        lines.append("-" * 70)
        
        accounted_time = sum(s.total_time for s in sorted_stats)
        
        for stat in sorted_stats[:top_n]:
            pct = 100 * stat.total_time / max(accounted_time, 0.001)
            lines.append(
                f"{stat.name:<35} {stat.total_time:>9.3f}s {stat.avg_time*1000:>9.2f}ms "
                f"{stat.call_count:>8} {pct:>5.1f}%"
            )
        
        lines.append("-" * 70)
        lines.append(f"{'TOTAL ACCOUNTED':<35} {accounted_time:>9.3f}s")
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get profiling data as a dictionary."""
        return {
            name: {
                "total": stat.total_time,
                "avg": stat.avg_time,
                "calls": stat.call_count,
                "min": stat.min_time if stat.min_time != float('inf') else 0,
                "max": stat.max_time,
            }
            for name, stat in self.stats.items()
        }


# Global profiler instance accessor
def get_profiler() -> Profiler:
    """Get the global profiler instance."""
    return Profiler.get_instance()


@contextmanager
def profile_section(name: str):
    """Convenience context manager using global profiler."""
    with get_profiler().timer(name):
        yield


def jax_profile(output_dir: str = "./jax_profile"):
    """
    Context manager for JAX's built-in profiler (for TensorBoard).
    
    Usage:
        with jax_profile("./profile_output"):
            run_calculation()
        
        # Then view with: tensorboard --logdir=./profile_output
    """
    return jax.profiler.trace(output_dir)


def profile_memory():
    """Print current JAX memory usage."""
    try:
        devices = jax.devices()
        for device in devices:
            stats = device.memory_stats()
            if stats:
                used = stats.get('bytes_in_use', 0) / 1e9
                limit = stats.get('bytes_limit', 0) / 1e9
                print(f"Device {device}: {used:.2f}GB / {limit:.2f}GB")
    except Exception as e:
        print(f"Memory stats not available: {e}")


def benchmark_function(func: Callable, *args, 
                       warmup: int = 3, 
                       repeats: int = 10,
                       block: bool = True,
                       **kwargs) -> Dict[str, float]:
    """
    Benchmark a function with warmup and multiple runs.
    
    Args:
        func: Function to benchmark
        *args: Positional arguments for func
        warmup: Number of warmup calls (for JIT)
        repeats: Number of timed calls
        block: Whether to block on JAX async operations
        **kwargs: Keyword arguments for func
        
    Returns:
        Dictionary with timing statistics
    """
    # Warmup
    for _ in range(warmup):
        result = func(*args, **kwargs)
        if block:
            jax.block_until_ready(result)
    
    # Timed runs
    times = []
    for _ in range(repeats):
        if block:
            jax.block_until_ready(jnp.array(0))
        
        start = time.perf_counter()
        result = func(*args, **kwargs)
        
        if block:
            jax.block_until_ready(result)
        
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    return {
        "mean": sum(times) / len(times),
        "min": min(times),
        "max": max(times),
        "std": (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
        "times": times,
    }


def create_detailed_iteration_profiler():
    """
    Create a profiler configured for detailed HFB iteration analysis.
    
    Returns configured profiler that tracks:
    - gradient_step
    - compute_densities  
    - compute_meanfield
    - solve_poisson (Coulomb)
    - solve_pairing
    - compute_energies
    - orthonormalization
    """
    profiler = Profiler()
    profiler.start_calculation()
    return profiler
