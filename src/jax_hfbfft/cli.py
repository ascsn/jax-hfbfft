"""
Command-line interface for JAX-HFBFFT.

This module provides a command-line interface for running HFB calculations.
"""

import argparse
import sys
from pathlib import Path


def main():
    """Main entry point for the jax-hfbfft CLI."""
    parser = argparse.ArgumentParser(
        prog="jax-hfbfft",
        description="JAX-based Hartree-Fock-Bogoliubov solver for nuclear structure",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run an HFB calculation")
    run_parser.add_argument(
        "config", 
        type=str, 
        help="Path to YAML configuration file"
    )
    run_parser.add_argument(
        "-o", "--output",
        type=str,
        default="output",
        help="Output directory (default: output)"
    )
    run_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show information")
    info_parser.add_argument(
        "--forces",
        action="store_true",
        help="List available forces"
    )
    info_parser.add_argument(
        "--gpu",
        action="store_true",
        help="Show GPU information"
    )
    
    # Version command
    version_parser = subparsers.add_parser("version", help="Show version")
    
    args = parser.parse_args()
    
    if args.command == "run":
        run_calculation(args)
    elif args.command == "info":
        show_info(args)
    elif args.command == "version":
        show_version()
    else:
        parser.print_help()
        sys.exit(1)


def run_calculation(args):
    """Run an HFB calculation from a configuration file."""
    from jax_hfbfft import Config, HFBFFT
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    
    print(f"Loading configuration from: {args.config}")
    config = Config.from_yaml(args.config)
    
    print("Creating HFBFFT calculation...")
    # Note: HFBFFT.from_config would need to be implemented
    # calc = HFBFFT.from_config(config)
    
    print("This feature is not yet fully implemented.")
    print("Please use the Python API directly.")


def show_info(args):
    """Show information about the package."""
    if args.forces:
        from jax_hfbfft.forces import AVAILABLE_FORCES
        print("Available forces:")
        for force in AVAILABLE_FORCES:
            print(f"  - {force}")
    
    if args.gpu:
        try:
            import jax
            devices = jax.devices()
            print(f"JAX devices: {devices}")
            print(f"Default backend: {jax.default_backend()}")
        except Exception as e:
            print(f"Error getting JAX info: {e}")


def show_version():
    """Show the package version."""
    from jax_hfbfft import __version__
    print(f"jax-hfbfft version {__version__}")


if __name__ == "__main__":
    main()
