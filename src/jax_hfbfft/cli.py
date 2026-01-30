"""
Command-line interface for JAX-HFBFFT.

This module provides a command-line interface for running HFB calculations
using configuration files, similar to traditional HFB codes.
"""

import argparse
import sys
import shutil
from pathlib import Path
from typing import Optional


def main():
    """Main entry point for the hfbfft CLI."""
    parser = argparse.ArgumentParser(
        prog="hfbfft",
        description="Hartree-Fock-Bogoliubov solver for nuclear structure",
        epilog="See documentation at https://github.com/ascsn/jax-hfbfft",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Init command - generate sample config
    init_parser = subparsers.add_parser(
        "init", 
        help="Generate a sample configuration file"
    )
    init_parser.add_argument(
        "-o", "--output",
        type=str,
        default="config.yml",
        help="Output filename (default: config.yml)"
    )
    init_parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="Overwrite existing file"
    )
    
    # Run command
    run_parser = subparsers.add_parser(
        "run", 
        help="Run an HFB calculation from config file"
    )
    run_parser.add_argument(
        "config",
        nargs="?",
        type=str,
        default=None,
        help="Path to YAML configuration file (default: config.yml if it exists)"
    )
    run_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    # Info command
    info_parser = subparsers.add_parser(
        "info", 
        help="Show package information"
    )
    info_parser.add_argument(
        "--forces",
        action="store_true",
        help="List available Skyrme forces"
    )
    info_parser.add_argument(
        "--devices",
        action="store_true",
        help="Show JAX devices (CPU/GPU)"
    )
    info_parser.add_argument(
        "--all",
        action="store_true",
        help="Show all information"
    )
    
    # Version command
    version_parser = subparsers.add_parser(
        "version", 
        help="Show version"
    )
    
    # GUI command
    gui_parser = subparsers.add_parser(
        "gui",
        help="Start the web-based GUI"
    )
    gui_parser.add_argument(
        "-p", "--port",
        type=int,
        default=8080,
        help="Port to run the server on (default: 8080)"
    )
    gui_parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    gui_parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't automatically open browser"
    )
    gui_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    if args.command == "init":
        generate_config(args.output, args.force)
    elif args.command == "run":
        run_calculation(args.config, args.verbose)
    elif args.command == "info":
        show_info(args)
    elif args.command == "version":
        show_version()
    elif args.command == "gui":
        start_gui(args.host, args.port, not args.no_browser, args.debug)
    elif args.command is None:
        # No subcommand - check if config.yml exists
        if Path("config.yml").exists():
            print("Found config.yml - running calculation...")
            print("(Use 'hfbfft run' to be explicit, or 'hfbfft --help' for more options)")
            print()
            run_calculation("config.yml", verbose=False)
        else:
            parser.print_help()
            print()
            print("Quick start:")
            print("  1. hfbfft init           # Generate sample config.yml")
            print("  2. nano config.yml       # Edit configuration")
            print("  3. hfbfft run            # Run calculation")
    else:
        parser.print_help()
        sys.exit(1)


def generate_config(output_path: str, force_overwrite: bool = False):
    """Generate a sample configuration file."""
    output = Path(output_path)
    
    if output.exists() and not force_overwrite:
        print(f"Error: File '{output_path}' already exists.")
        print(f"Use --force to overwrite, or specify a different filename with -o")
        sys.exit(1)
    
    # Get the config.yml from repository root
    # The package is installed at: site-packages/jax_hfbfft/
    # We need to go up to find the repository root
    package_dir = Path(__file__).parent  # .../jax_hfbfft/
    
    # Try multiple locations for the template
    template_locations = [
        package_dir.parent.parent.parent / "config.yml",  # Development install (repo root)
        package_dir.parent.parent / "config.yml",          # One level up
        package_dir / "data" / "config.yml",               # Packaged data (future)
    ]
    
    template_path = None
    for loc in template_locations:
        if loc.exists():
            template_path = loc
            break
    
    if template_path is None:
        print("Error: Could not find config.yml template.")
        print("This might happen if the package was not installed correctly.")
        print()
        print("Attempting to create a basic configuration...")
        
        # Create a minimal functional config as fallback
        minimal_config = """# ============================================================================
# jax-hfbfft Configuration File
# ============================================================================
# For full documentation, see: https://github.com/ascsn/jax-hfbfft
# Or view the complete config.yml template in the repository.

# ----------------------------------------------------------------------------
# NUCLEUS CONFIGURATION
# ----------------------------------------------------------------------------
nucleus:
  protons: 8        # Z (atomic number)
  neutrons: 8       # N (neutron number)
  name: "O16"       # Name for output files

# ----------------------------------------------------------------------------
# FORCE PARAMETERS
# ----------------------------------------------------------------------------
force:
  name: "SLy4"      # Skyrme force: SLy4, SLy5, SkM*, SkP, UNEDF0, etc.
  ipair: 6          # Pairing: 0=none, 5=VDI, 6=DDDI (recommended)
  use_bcs: true     # BCS (true) or HFB (false)

# ----------------------------------------------------------------------------
# GRID CONFIGURATION
# ----------------------------------------------------------------------------
grid:
  nx: 32            # Grid points in x direction
  ny: 32            # Grid points in y direction
  nz: 32            # Grid points in z direction
  dx: 0.8           # Grid spacing in x (fm)
  dy: 0.8           # Grid spacing in y (fm)
  dz: 0.8           # Grid spacing in z (fm)

# ----------------------------------------------------------------------------
# ITERATION CONTROL
# ----------------------------------------------------------------------------
iteration:
  max_iterations: 500
  convergence_threshold: 1.0e-6
  density_mixing: 0.3           # Mixing parameter (lower = more stable)

# ----------------------------------------------------------------------------
# OUTPUT CONFIGURATION
# ----------------------------------------------------------------------------
output:
  directory: "hfb_results"
  save_convergence: true
  save_energies: true
  save_moments: true
  save_radii: true
  save_single_particle: true
  save_pairing: true
  save_densities: false         # Can be large files

# ----------------------------------------------------------------------------
# CONSTRAINT OPTIONS (optional)
# ----------------------------------------------------------------------------
# Uncomment to add constraints for deformed nuclei
#
# constraints:
#   # Option 1: Direct multipole moments (in fm^lambda)
#   multipoles:
#     Q20: 10.0   # Quadrupole (fm²)
#     Q30: 2.0    # Octupole (fm³)
#
#   # Option 2: Beta-gamma parameters (often more intuitive)
#   beta_gamma:
#     beta2: 0.3   # Positive = prolate, negative = oblate
#     gamma: 0     # Triaxiality angle (0° = axially symmetric)
#     # beta3: 0.05  # Octupole (optional)
#     # beta4: 0.02  # Hexadecapole (optional)
"""
        output.write_text(minimal_config)
        print(f"Created basic configuration: {output_path}")
        print()
        print("Note: For a fully documented config with all options, please")
        print("      copy config.yml from the repository root.")
    else:
        # Copy the full documented template
        shutil.copy(template_path, output)
        print(f"✓ Created configuration file: {output_path}")
        print(f"  (Copied from: {template_path.parent.name}/config.yml)")
    
    print()
    print("Next steps:")
    print(f"  1. Edit {output_path} to set your nucleus and parameters")
    print(f"  2. Run: hfbfft run")
    print()
    print("Quick examples:")
    print("  • Change nucleus: Edit 'protons' and 'neutrons'")
    print("  • Add deformation: Uncomment and edit 'constraints' section")
    print("  • Change force: Set 'force.name' to SLy4, SLy5, UNEDF0, etc.")
    print()
    print("For detailed documentation:")
    print("  • hfbfft info --forces    # List available forces")
    print("  • See docs/RUN_INSTRUCTIONS.md")


def run_calculation(config_path: Optional[str], verbose: bool = False):
    """Run an HFB calculation from configuration file."""
    # Import here to ensure JAX configuration happens first
    # (jax_config auto-configures on import)
    import jax_hfbfft.jax_config
    
    # Determine config file
    if config_path is None:
        if Path("config.yml").exists():
            config_path = "config.yml"
        else:
            print("Error: No configuration file specified and config.yml not found.")
            print()
            print("Generate a config file with: hfbfft init")
            sys.exit(1)
    
    config_file = Path(config_path)
    if not config_file.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    # Import the run_hfb logic
    try:
        import yaml
        from datetime import datetime
        import numpy as np
        
        # Import after JAX is configured
        from jax_hfbfft import HFBFFT, Nucleus, Force, Grid, Constraint
        
        print("=" * 70)
        print("HFB-FFT Nuclear Structure Calculator")
        print("=" * 70)
        print()
        
        # Load configuration
        print(f"Loading configuration: {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Create nucleus
        nucleus_config = config['nucleus']
        nucleus = Nucleus(
            protons=nucleus_config['protons'],
            neutrons=nucleus_config['neutrons'],
            name=nucleus_config.get('name', f"Z{nucleus_config['protons']}N{nucleus_config['neutrons']}")
        )
        
        # Create force
        force_config = config.get('force', {})
        force = Force.from_name(
            force_config.get('name', 'SLy4'),
            ipair=force_config.get('ipair', 6),
            tbcs=force_config.get('use_bcs', True),  # Config uses 'use_bcs', Force uses 'tbcs'
            v0neut=force_config.get('v0_neutron', -1.0),  # Config uses v0_neutron, Force uses v0neut
            v0prot=force_config.get('v0_proton', -1.0),   # Config uses v0_proton, Force uses v0prot
        )
        
        # Create grid
        grid_config = config.get('grid', {})
        grid = Grid.create(
            nx=grid_config.get('nx', 32),
            ny=grid_config.get('ny', 32),
            nz=grid_config.get('nz', 32),
            dx=grid_config.get('dx') or 0.8,  # Handle null/None in config
            dy=grid_config.get('dy') or 0.8,
            dz=grid_config.get('dz') or 0.8,
        )
        
        # Create constraint (if specified)
        constraint = None
        if 'constraints' in config and config['constraints']:
            constraints_config = config['constraints']
            mass_number = nucleus.protons + nucleus.neutrons
            
            # Check for beta_gamma
            if 'beta_gamma' in constraints_config and constraints_config['beta_gamma']:
                bg = constraints_config['beta_gamma']
                constraint = Constraint.from_beta_gamma(
                    mass_number=mass_number,
                    beta2=bg.get('beta2'),
                    gamma=bg.get('gamma'),
                    beta3=bg.get('beta3'),
                    beta4=bg.get('beta4'),
                    r0=bg.get('r0', 1.2),
                    principal_axes=constraints_config.get('principal_axes', False)
                )
            # Check for multipoles
            elif 'multipoles' in constraints_config and constraints_config['multipoles']:
                multipoles = {}
                for key, value in constraints_config['multipoles'].items():
                    if isinstance(key, list):
                        multipoles[tuple(key)] = value
                    else:
                        multipoles[key] = value
                constraint = Constraint.from_multipoles(
                    multipoles=multipoles,
                    principal_axes=constraints_config.get('principal_axes', False)
                )
        
        # Print configuration
        print(f"Nucleus:  {nucleus.name} (Z={nucleus.protons}, N={nucleus.neutrons}, A={nucleus.mass_number})")
        print(f"Force:    {force.name}")
        print(f"Grid:     {grid.nx} × {grid.ny} × {grid.nz}")
        if constraint:
            print(f"Constraint: {constraint}")
        print()
        
        # Create calculator
        calc = HFBFFT(
            nucleus=nucleus,
            force=force,
            grid=grid,
            constraint=constraint
        )
        
        # Run calculation
        iteration_config = config.get('iteration', {})
        max_iterations = iteration_config.get('max_iterations', 500)
        convergence = iteration_config.get('convergence_threshold', 1.0e-6)
        
        print(f"Starting HFB iteration (max: {max_iterations}, convergence: {convergence:.1e})...")
        print()
        
        results = calc.run(
            max_iterations=max_iterations,
            convergence_threshold=convergence
        )
        
        # Print results
        print()
        print("=" * 70)
        print("Calculation completed successfully!")
        print("=" * 70)
        print()
        print(f"Total Energy:     {results.total_energy:.3f} MeV")
        print(f"Binding Energy:   {getattr(results, 'binding_energy', 0.0):.3f} MeV")
        print(f"E/A:              {results.total_energy/nucleus.mass_number:.3f} MeV")
        print()
        
        # Save results if output directory specified
        output_config = config.get('output', {})
        if output_config and output_config.get('directory'):
            output_dir = Path(output_config['directory'])
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = output_dir / f"{nucleus.name}_{timestamp}"
            run_dir.mkdir(parents=True, exist_ok=True)
            
            # Save summary
            summary_path = run_dir / "summary.txt"
            with open(summary_path, 'w') as f:
                f.write("=" * 70 + "\n")
                f.write("HFB Calculation Summary\n")
                f.write("=" * 70 + "\n\n")
                f.write(f"Nucleus:  {nucleus.name} (Z={nucleus.protons}, N={nucleus.neutrons})\n")
                f.write(f"Force:    {force.name}\n")
                f.write(f"Total Energy: {results.total_energy:.6f} MeV\n")
                f.write(f"E/A:          {results.total_energy/nucleus.mass_number:.6f} MeV\n")
            
            print(f"Results saved to: {run_dir}")
            print()
        
    except ImportError as e:
        print(f"Error: Missing required dependencies: {e}")
        print("Make sure jax-hfbfft is properly installed.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during calculation: {e}")
        import traceback
        if verbose:
            traceback.print_exc()
        sys.exit(1)


def show_info(args):
    """Show information about the package."""
    show_all = args.all if hasattr(args, 'all') else False
    
    if args.forces or show_all:
        print("Available Skyrme Forces:")
        print("-" * 40)
        forces = [
            "SLy4", "SLy5", "SkM*", "SkP", 
            "SV-min", "UNEDF0", "UNEDF1", "UNEDF2"
        ]
        for force in forces:
            print(f"  • {force}")
        print()
    
    if args.devices or show_all:
        print("JAX Device Information:")
        print("-" * 40)
        try:
            import jax
            devices = jax.devices()
            print(f"Available devices: {len(devices)}")
            for i, dev in enumerate(devices):
                print(f"  [{i}] {dev}")
            print(f"Default backend: {jax.default_backend()}")
        except Exception as e:
            print(f"Error getting JAX info: {e}")
        print()
    
    if not (args.forces or args.devices or show_all):
        # Show basic info by default
        show_version()
        print()
        print("For more information:")
        print("  hfbfft info --forces    # List available forces")
        print("  hfbfft info --devices   # Show GPU/CPU devices")
        print("  hfbfft info --all       # Show everything")


def show_version():
    """Show the package version."""
    try:
        from jax_hfbfft import __version__
        print(f"hfbfft version {__version__}")
    except:
        print("hfbfft (version unknown)")


def start_gui(host: str, port: int, open_browser: bool, debug: bool):
    """Start the web-based GUI server."""
    try:
        from jax_hfbfft.gui import start_server
    except ImportError as e:
        print("Error: GUI dependencies not installed.")
        print()
        print("Install GUI dependencies with:")
        print("  pip install jax-hfbfft[gui]")
        print()
        print(f"Missing: {e}")
        sys.exit(1)
    
    print()
    print("=" * 60)
    print("  HFBFFT Web GUI")
    print("=" * 60)
    print()
    print("Starting GUI server...")
    print()
    
    start_server(
        host=host,
        port=port,
        open_browser=open_browser,
        debug=debug,
    )


if __name__ == "__main__":
    main()
