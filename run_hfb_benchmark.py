#!/usr/bin/env python3
"""
Benchmark entry point for jax-hfbfft HFB calculations.

Reads the same config.yml as run_hfb.py, plus the output.benchmark_* keys,
and runs the calculation with a BenchmarkHook that handles JAX XLA profiling
and per-iteration wall-time measurement entirely outside the core solver.

Usage:
    python run_hfb_benchmark.py                  # Use config.yml
    python run_hfb_benchmark.py my_config.yml    # Use custom config file
"""

import os
import sys
import time
import statistics

import jax

# Re-use all setup/construction/output helpers from run_hfb
from run_hfb import (
    load_config,
    validate_config,
    setup_jax,
    setup_output_directory,
    print_banner,
    create_nucleus,
    create_force,
    create_grid,
    create_constraint,
    determine_basis_size,
    save_results,
    print_results,
)

import argparse
from jax_hfbfft import HFBFFT
import jax.numpy as jnp


class BenchmarkHook:
    """
    Iteration hook that profiles a window of HFB iterations.

    Attaches to run_hfb via the hook= parameter.  The solver calls
    pre_iteration / post_iteration around each hfb_iteration call and
    finalize() after the loop exits (including early convergence).
    """

    def __init__(self, benchmark_start: int, benchmark_end: int, logdir: str):
        self.benchmark_start = benchmark_start
        self.benchmark_end = benchmark_end
        self.logdir = logdir

        self._iter_times: list[float] = []
        self._t0: float | None = None
        self._profiler_ctx = None

        os.makedirs(logdir, exist_ok=True)
        print(
            f"\n[benchmark] will profile iterations {benchmark_start}–{benchmark_end}"
            f" → {logdir}"
        )

    def _in_window(self, iteration: int) -> bool:
        return self.benchmark_start <= iteration <= self.benchmark_end

    def pre_iteration(self, iteration: int, state) -> None:
        if not self._in_window(iteration):
            return

        if iteration == self.benchmark_start:
            # Flush any pending XLA work so the profiler window starts clean.
            jax.block_until_ready(state.psi)
            self._profiler_ctx = jax.profiler.trace(
                self.logdir, create_perfetto_link=False
            )
            self._profiler_ctx.__enter__()
            print(f"[benchmark] profiler started at iteration {iteration}")

        self._t0 = time.perf_counter()

    def post_iteration(self, iteration: int, state) -> None:
        if not self._in_window(iteration):
            return

        # Block until all XLA ops finish to get real wall time.
        jax.block_until_ready(state.psi)
        elapsed = time.perf_counter() - self._t0
        self._iter_times.append(elapsed)

        mem = {}
        try:
            mem = jax.devices()[0].memory_stats() or {}
        except Exception:
            pass

        print(
            f"[benchmark] iter {iteration:4d}: {elapsed * 1e3:8.2f} ms"
            + (
                f"  live_bytes={mem.get('bytes_in_use', mem.get('live_bytes', '?'))}"
                if mem
                else ""
            )
        )

        if iteration == self.benchmark_end:
            self._close_profiler()
            self._print_summary()

    def finalize(self) -> None:
        # Called after the loop — handles early convergence before benchmark_end.
        self._close_profiler()
        if self._iter_times:
            self._print_summary()

    def _close_profiler(self) -> None:
        if self._profiler_ctx is not None:
            self._profiler_ctx.__exit__(None, None, None)
            self._profiler_ctx = None

    def _print_summary(self) -> None:
        if not self._iter_times:
            return
        times_ms = [t * 1e3 for t in self._iter_times]
        n = len(times_ms)
        print(
            f"\n[benchmark] summary over {n} iters "
            f"({self.benchmark_start}–{self.benchmark_start + n - 1}):\n"
            f"  mean={statistics.mean(times_ms):.2f} ms  "
            f"min={min(times_ms):.2f} ms  "
            f"max={max(times_ms):.2f} ms"
            + (f"  stdev={statistics.stdev(times_ms):.2f} ms" if n > 1 else "")
            + f"\n  traces written to: {self.logdir}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run a benchmarked HFB calculation with jax-hfbfft",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_hfb_benchmark.py                    # Use config.yml
  python run_hfb_benchmark.py my_config.yml      # Use custom config file
        """,
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="config.yml",
        help="Configuration file (default: config.yml)",
    )
    parser.add_argument(
        "--start",
        type=int,
        required=True,
        metavar="N",
        help="First iteration to profile (must be > 0)",
    )
    parser.add_argument(
        "--end",
        type=int,
        required=True,
        metavar="N",
        help="Last iteration to profile (must be >= --start)",
    )
    parser.add_argument(
        "--logdir",
        default="./benchmark_profile",
        metavar="DIR",
        help="Directory for XLA traces and timing output (default: ./benchmark_profile)",
    )
    args = parser.parse_args()

    benchmark_start = args.start
    benchmark_end = args.end
    benchmark_logdir = args.logdir

    if benchmark_start <= 0 or benchmark_end < benchmark_start:
        print("Error: --start must be > 0 and --end must be >= --start.")
        sys.exit(1)

    print("Loading configuration from:", args.config)
    config = load_config(args.config)
    config = validate_config(config)

    setup_jax(config)

    print_banner(config)

    output_dir = setup_output_directory(config)
    print(f"Output directory: {output_dir}")
    print(f"Benchmark window: iterations {benchmark_start}–{benchmark_end}  logdir: {benchmark_logdir}")
    print()

    print("Creating calculation components...")
    nucleus = create_nucleus(config)
    force = create_force(config)
    grid = create_grid(config, nucleus)
    constraint = create_constraint(config)

    npsi_n, npsi_p = determine_basis_size(config, nucleus, force)
    print(f"Basis size: {npsi_n} neutron states, {npsi_p} proton states")

    if constraint is not None:
        print(f"Constraints: {constraint}")

    calc = HFBFFT(
        nucleus=nucleus,
        force=force,
        grid=grid,
        npsi=(npsi_n, npsi_p),
        include_coulomb=config["physics"]["include_coulomb"],
        constraint=constraint,
    )

    iter_params = config["iteration"]
    calc.x0dmp = float(iter_params["x0dmp"])
    calc.e0dmp = float(iter_params["e0dmp"])
    calc.density_mixing = float(iter_params["density_mixing"])
    calc.diag_start = int(iter_params["diag_start"])
    calc.bcs_start = int(iter_params["bcs_start"])
    calc.tvaryx_0 = bool(iter_params.get("tvaryx_0", False))

    print("\nInitializing wavefunctions...")
    init_config = config["initialization"]
    if init_config["method"] == "harmonic_oscillator":
        calc.initialize_wavefunctions(
            method="harmonic_oscillator",
            radinx=init_config["ho_length_x"],
            radiny=init_config["ho_length_y"],
            radinz=init_config["ho_length_z"],
        )
    else:
        calc.initialize_wavefunctions(method=init_config["method"])

    hook = BenchmarkHook(benchmark_start, benchmark_end, benchmark_logdir)

    print("\nStarting HFB iterations...")
    print("-" * 80)

    iter_config = config["iteration"]
    results = calc.run(
        max_iterations=iter_config["max_iterations"],
        convergence_threshold=iter_config["convergence_threshold"],
        print_interval=output_cfg["print_interval"],
        sinfo_interval=output_cfg.get("sinfo_interval", 50),
        hook=hook,
        save_dir=str(output_dir),
        save_interval=5,
    )

    print("\nSaving results...")
    save_results(calc, results, config, output_dir)

    print_results(calc, results, config)

    print(f"\nResults saved to: {output_dir}")
    print("\nCalculation complete!")

    return 0 if results.converged else 1


if __name__ == "__main__":
    sys.exit(main())
