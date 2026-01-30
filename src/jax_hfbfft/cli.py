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
        force_kwargs = {
            'ipair': force_config.get('ipair', 6),
            'tbcs': force_config.get('use_bcs', True),  # Config uses 'use_bcs', Force uses 'tbcs'
            'v0neut': force_config.get('v0_neutron', -1.0),  # Config uses v0_neutron, Force uses v0neut
            'v0prot': force_config.get('v0_proton', -1.0),   # Config uses v0_proton, Force uses v0prot
        }
        
        # Add pairing cutoff if specified
        if 'pairing_cutoff' in force_config:
            import jax.numpy as jnp
            cutoff = force_config['pairing_cutoff']
            if isinstance(cutoff, list):
                force_kwargs['pair_cutoff'] = jnp.array(cutoff)
            else:
                force_kwargs['pair_cutoff'] = jnp.array([cutoff, cutoff])
        
        # Add pairing density if specified
        if 'rho0_pairing' in force_config:
            force_kwargs['rho0pr'] = force_config['rho0_pairing']
        
        force = Force.from_name(
            force_config.get('name', 'SLy4'),
            **force_kwargs
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
        
        # Get basis configuration
        basis_config = config.get('basis', {})
        npsi_neutron = basis_config.get('npsi_neutron')
        npsi_proton = basis_config.get('npsi_proton')
        
        # Build npsi tuple if either is specified
        npsi = None
        if npsi_neutron is not None and npsi_proton is not None:
            npsi = (npsi_neutron, npsi_proton)
        
        # Create calculator
        calc = HFBFFT(
            nucleus=nucleus,
            force=force,
            grid=grid,
            constraint=constraint,
            npsi=npsi,
        )
        
        # Initialize wavefunctions
        init_config = config.get('initialization', {})
        init_method = init_config.get('method', 'harmonic_oscillator')
        
        if verbose:
            print(f"Initializing wavefunctions using {init_method} method...")
        
        if init_method == 'harmonic_oscillator':
            ho_params = {}
            if 'ho_length_x' in init_config:
                ho_params['radinx'] = init_config['ho_length_x']
            if 'ho_length_y' in init_config:
                ho_params['radiny'] = init_config['ho_length_y']
            if 'ho_length_z' in init_config:
                ho_params['radinz'] = init_config['ho_length_z']
            calc.initialize_wavefunctions(method=init_method, **ho_params)
        else:
            calc.initialize_wavefunctions(method=init_method)
        
        # Get iteration configuration
        iteration_config = config.get('iteration', {})
        max_iterations = iteration_config.get('max_iterations', 500)
        convergence = iteration_config.get('convergence_threshold', 1.0e-6)
        print_interval = iteration_config.get('print_interval', 10)
        
        # Set iteration parameters if provided
        if 'x0dmp' in iteration_config:
            calc.x0dmp = iteration_config['x0dmp']
        if 'e0dmp' in iteration_config:
            calc.e0dmp = iteration_config['e0dmp']
        if 'density_mixing' in iteration_config:
            calc.density_mixing = iteration_config['density_mixing']
        if 'diag_start' in iteration_config:
            calc.diag_start = iteration_config['diag_start']
        if 'bcs_start' in iteration_config:
            calc.bcs_start = iteration_config['bcs_start']
        
        print(f"Starting HFB iteration (max: {max_iterations}, convergence: {convergence:.1e})...")
        print()
        
        results = calc.run(
            max_iterations=max_iterations,
            convergence_threshold=convergence,
            print_interval=print_interval,
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
        output_dir_name = output_config.get('output_dir') or output_config.get('directory')
        
        if output_dir_name:
            output_dir = Path(output_dir_name)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = output_dir / f"{nucleus.name}_{timestamp}"
            run_dir.mkdir(parents=True, exist_ok=True)
            
            # Save configuration file used for this run
            config_path = run_dir / "config.yml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            # Save summary
            _save_summary(calc, results, config, run_dir)
            
            # Save detailed results based on config flags
            if output_config.get('save_convergence', True):
                _save_convergence(results, run_dir)
            
            if output_config.get('save_energies', True):
                _save_energies(calc, results, run_dir)
            
            if output_config.get('save_moments', True):
                _save_moments(results, run_dir)
            
            if output_config.get('save_radii', True):
                _save_radii(results, run_dir)
            
            if output_config.get('save_single_particle', True):
                _save_single_particle(calc, run_dir)
            
            if output_config.get('save_pairing', True) and force.ipair > 0:
                _save_pairing(calc, results, run_dir)
            
            if output_config.get('save_densities', True):
                _save_densities(calc, run_dir)
            
            print()
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


def _save_summary(calc, results, config, output_dir):
    """Save summary file with main results."""
    summary_file = output_dir / "summary.txt"
    
    nucleus = calc.nucleus
    force = calc.force
    
    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(" " * 25 + "jax-hfbfft HFB Calculation\n")
        f.write("=" * 80 + "\n\n")
        
        # Configuration
        f.write(f"Nucleus:  {nucleus.name} (Z={nucleus.protons}, N={nucleus.neutrons}, A={nucleus.mass_number})\n")
        f.write(f"Force:    {force.name}\n")
        
        ipair_names = {0: "No pairing", 5: "VDI", 6: "DDDI"}
        ipair_name = ipair_names.get(force.ipair, f"ipair={force.ipair}")
        f.write(f"Pairing:  {ipair_name}\n")
        f.write(f"Grid:     {calc.grid.nx}×{calc.grid.ny}×{calc.grid.nz} points\n\n")
        
        # Convergence
        f.write("=" * 80 + "\n")
        f.write("CONVERGENCE\n")
        f.write("=" * 80 + "\n")
        f.write(f"Converged:         {results.converged}\n")
        f.write(f"Iterations:        {results.iterations}\n")
        f.write(f"Final fluctuation: {results.final_fluctuation:.3e} MeV\n\n")
        
        # Energies
        f.write("=" * 80 + "\n")
        f.write("ENERGIES (MeV)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total binding energy:      {results.total_energy:12.3f}\n")
        f.write(f"  Kinetic:                 {results.kinetic_energy:12.3f}\n")
        f.write(f"  Potential:               {results.potential_energy:12.3f}\n")
        f.write(f"  Coulomb:                 {results.coulomb_energy:12.3f}\n")
        f.write(f"  Pairing:                 {results.pairing_energy:12.3f}\n")
        f.write(f"  Rearrangement:           {results.rearrangement_energy:12.3f}\n")
        f.write(f"  Center-of-mass:          {results.cm_correction:12.3f}\n\n")
        
        # Binding energy per nucleon
        f.write(f"Binding energy/A:          {results.total_energy/nucleus.mass_number:12.3f} MeV\n\n")
        
        # Radii
        f.write("=" * 80 + "\n")
        f.write("RADII AND DEFORMATION\n")
        f.write("=" * 80 + "\n")
        f.write(f"RMS radius (neutron):      {results.rms_radius_n:12.3f} fm\n")
        f.write(f"RMS radius (proton):       {results.rms_radius_p:12.3f} fm\n")
        f.write(f"RMS radius (total):        {results.rms_radius_total:12.3f} fm\n")
        f.write(f"Charge radius:             {results.charge_radius:12.3f} fm\n\n")
        
        f.write(f"Quadrupole Q20:            {results.q20:12.3f} fm²\n")
        f.write(f"Quadrupole Q22:            {results.q22:12.3f} fm²\n")
        f.write(f"Deformation β₂:            {results.beta2:12.4f}\n")
        f.write(f"Deformation γ:             {results.gamma:12.1f} deg\n\n")
        
        # Pairing
        if force.ipair != 0:
            f.write("=" * 80 + "\n")
            f.write("PAIRING\n")
            f.write("=" * 80 + "\n")
            f.write(f"Neutron Fermi energy:      {results.fermi_energy_n:12.3f} MeV\n")
            f.write(f"Proton Fermi energy:       {results.fermi_energy_p:12.3f} MeV\n")
            f.write(f"Neutron pairing gap:       {results.pairing_gap_n:12.3f} MeV\n")
            f.write(f"Proton pairing gap:        {results.pairing_gap_p:12.3f} MeV\n\n")


def _save_convergence(results, output_dir):
    """Save convergence history."""
    if results.convergence_history is None:
        return
    
    conv_file = output_dir / "convergence.res"
    
    with open(conv_file, 'w') as f:
        f.write("# Iteration    Fluctuation(MeV)\n")
        for i, fluct in enumerate(results.convergence_history):
            f.write(f"{i:6d}    {float(fluct):15.6e}\n")


def _save_energies(calc, results, output_dir):
    """Save detailed energy breakdown."""
    energy_file = output_dir / "energies.res"
    
    with open(energy_file, 'w') as f:
        f.write("# Detailed energy breakdown (all in MeV)\n")
        f.write(f"# Total energy:      {results.total_energy:12.6f}\n")
        f.write(f"# Kinetic energy:    {results.kinetic_energy:12.6f}\n")
        f.write(f"# Potential energy:  {results.potential_energy:12.6f}\n")
        f.write(f"# Coulomb energy:    {results.coulomb_energy:12.6f}\n")
        f.write(f"# Pairing energy:    {results.pairing_energy:12.6f}\n")
        f.write(f"# Rearrangement:     {results.rearrangement_energy:12.6f}\n")
        f.write(f"# CM correction:     {results.cm_correction:12.6f}\n")
        f.write(f"#\n")
        f.write(f"# Skyrme functional terms:\n")
        f.write(f"# ehf0 (t0):         {results.ehf0:12.6f}\n")
        f.write(f"# ehf1 (current):    {results.ehf1:12.6f}\n")
        f.write(f"# ehf2 (Laplacian):  {results.ehf2:12.6f}\n")
        f.write(f"# ehf3 (density):    {results.ehf3:12.6f}\n")
        f.write(f"# ehfls (spin-orb):  {results.ehfls:12.6f}\n")


def _save_moments(results, output_dir):
    """Save multipole moments."""
    moments_file = output_dir / "moments.res"
    
    with open(moments_file, 'w') as f:
        f.write("# Multipole moments\n")
        f.write(f"Q20 (fm²):     {results.q20:12.6f}\n")
        f.write(f"Q22 (fm²):     {results.q22:12.6f}\n")
        f.write(f"beta2:         {results.beta2:12.6f}\n")
        f.write(f"gamma (deg):   {results.gamma:12.2f}\n")


def _save_radii(results, output_dir):
    """Save radii and deformation parameters."""
    radii_file = output_dir / "radii.res"
    
    with open(radii_file, 'w') as f:
        f.write("# Radii (fm) and deformation parameters\n")
        f.write(f"RMS radius (neutron):  {results.rms_radius_n:12.6f}\n")
        f.write(f"RMS radius (proton):   {results.rms_radius_p:12.6f}\n")
        f.write(f"RMS radius (total):    {results.rms_radius_total:12.6f}\n")
        f.write(f"Charge radius:         {results.charge_radius:12.6f}\n")
        f.write(f"Beta2:                 {results.beta2:12.6f}\n")
        f.write(f"Gamma (deg):           {results.gamma:12.2f}\n")


def _save_single_particle(calc, output_dir):
    """Save single-particle spectrum."""
    sp_file = output_dir / "single_particle.res"
    
    state = calc._solver_state
    import jax.numpy as jnp
    
    # Separate neutrons and protons
    neutron_mask = state.isospin == 0
    proton_mask = state.isospin == 1
    
    with open(sp_file, 'w') as f:
        f.write("# Single-particle spectrum\n")
        f.write("#\n")
        f.write("# Neutrons:\n")
        f.write("# Index    Energy(MeV)    Occupation\n")
        
        neutron_indices = jnp.where(neutron_mask)[0]
        for idx in neutron_indices:
            i = int(idx)
            energy = float(state.sp_energy[i])
            occupation = float(state.wocc[i])
            f.write(f"{i:5d}    {energy:12.6f}    {occupation:8.6f}\n")
        
        f.write("#\n")
        f.write("# Protons:\n")
        f.write("# Index    Energy(MeV)    Occupation\n")
        
        proton_indices = jnp.where(proton_mask)[0]
        for idx in proton_indices:
            i = int(idx)
            energy = float(state.sp_energy[i])
            occupation = float(state.wocc[i])
            f.write(f"{i:5d}    {energy:12.6f}    {occupation:8.6f}\n")


def _save_pairing(calc, results, output_dir):
    """Save pairing properties."""
    pairing_file = output_dir / "pairing.res"
    
    with open(pairing_file, 'w') as f:
        f.write("# Pairing properties\n")
        f.write(f"Neutron Fermi energy (MeV):  {results.fermi_energy_n:12.6f}\n")
        f.write(f"Proton Fermi energy (MeV):   {results.fermi_energy_p:12.6f}\n")
        f.write(f"Neutron pairing gap (MeV):   {results.pairing_gap_n:12.6f}\n")
        f.write(f"Proton pairing gap (MeV):    {results.pairing_gap_p:12.6f}\n")
        f.write(f"Neutron pairing energy (MeV): {results.pairing_energy:12.6f}\n")


def _save_densities(calc, output_dir):
    """Save density distributions."""
    densities_file = output_dir / "densities.npz"
    
    state = calc._solver_state
    import numpy as np
    
    # Save densities and grid information
    np.savez(
        densities_file,
        rho=np.array(state.densities.rho),
        tau=np.array(state.densities.tau),
        chi=np.array(state.densities.chi),
        current=np.array(state.densities.current),
        sdens=np.array(state.densities.sdens),
        sodens=np.array(state.densities.sodens),
        x=np.array(calc.grid.x),
        y=np.array(calc.grid.y),
        z=np.array(calc.grid.z),
    )


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
