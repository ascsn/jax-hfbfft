#!/usr/bin/env python3
"""
Main run script for jax-hfbfft HFB calculations.

This provides the classic HFB code user experience:
1. Edit config.yml with your desired parameters
2. Run: python run_hfb.py [optional_config_file.yml]
3. Results are written to output directory and stdout

Usage:
    python run_hfb.py                  # Use config.yml
    python run_hfb.py my_config.yml    # Use custom config file
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from datetime import datetime
import jax.numpy as jnp

# Configure JAX before importing jax_hfbfft
def setup_jax(config):
    """Configure JAX environment based on config."""
    precision = config.get('advanced', {}).get('precision', 64)
    if precision == 64:
        os.environ['JAX_ENABLE_X64'] = 'True'
    
    device = config.get('advanced', {}).get('device')
    if device:
        os.environ['JAX_PLATFORM_NAME'] = device


def load_config(config_file):
    """Load configuration from YAML file."""
    if not os.path.exists(config_file):
        print(f"Error: Configuration file '{config_file}' not found.")
        print("\nPlease create a config.yml file or specify a valid config file.")
        sys.exit(1)
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def validate_config(config):
    """Validate configuration and fill in defaults."""
    # Required sections
    required = ['nucleus', 'force', 'grid']
    for section in required:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    # Set defaults for optional sections
    if 'basis' not in config:
        config['basis'] = {}
    if 'initialization' not in config:
        config['initialization'] = {}
    if 'iteration' not in config:
        config['iteration'] = {}
    if 'physics' not in config:
        config['physics'] = {}
    if 'output' not in config:
        config['output'] = {}
    if 'advanced' not in config:
        config['advanced'] = {}
    
    # Nucleus defaults
    if 'name' not in config['nucleus']:
        Z = config['nucleus']['protons']
        N = config['nucleus']['neutrons']
        config['nucleus']['name'] = f"Z{Z}N{N}"
    
    # Grid defaults
    grid = config['grid']
    for dim in ['nx', 'ny', 'nz']:
        if dim not in grid:
            grid[dim] = 32
    
    # Iteration defaults
    iter_defaults = {
        'max_iterations': 500,
        'convergence_threshold': 1e-6,
        'x0dmp': 0.45,
        'e0dmp': 20.0,
        'density_mixing': 0.5,
        'diag_start': 30,
        'bcs_start': 30,
    }
    for key, val in iter_defaults.items():
        if key not in config['iteration']:
            config['iteration'][key] = val
    
    # Physics defaults
    physics_defaults = {
        'include_coulomb': True,
        'time_reversal': True,
        'time_odd': False,
    }
    for key, val in physics_defaults.items():
        if key not in config['physics']:
            config['physics'][key] = val
    
    # Output defaults
    output_defaults = {
        'output_dir': 'hfb_results',
        'print_interval': 50,
        'save_convergence': True,
        'save_energies': True,
        'save_densities': True,
        'save_moments': True,
        'save_radii': True,
        'save_single_particle': True,
        'save_pairing': True,
        'save_wavefunctions': False,
        'verbose': True,
        'make_plots': False,
    }
    for key, val in output_defaults.items():
        if key not in config['output']:
            config['output'][key] = val
    
    # Initialization defaults
    init_defaults = {
        'method': 'harmonic_oscillator',
        'ho_length_x': 3.0,
        'ho_length_y': 3.0,
        'ho_length_z': 3.0,
    }
    for key, val in init_defaults.items():
        if key not in config['initialization']:
            config['initialization'][key] = val
    
    # Force defaults
    force_defaults = {
        'ipair': 6,
        'use_bcs': True,
        'v0_neutron': -1.0,
        'v0_proton': -1.0,
        'rho0_pairing': 0.16,
        'pairing_cutoff': [60.0, 60.0],
        'include_cm_correction': True,
    }
    for key, val in force_defaults.items():
        if key not in config['force']:
            config['force'][key] = val
    
    # Advanced defaults
    advanced_defaults = {
        'precision': 64,
        'device': None,
        'use_jit': True,
        'random_seed': 42,
        'restart': False,
        'restart_file': None,
    }
    for key, val in advanced_defaults.items():
        if key not in config['advanced']:
            config['advanced'][key] = val
    
    return config


def setup_output_directory(config):
    """Create output directory structure."""
    base_dir = config['output']['output_dir']
    nucleus_name = config['nucleus']['name']
    
    # Create timestamped directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / f"{nucleus_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save copy of config file
    config_copy = output_dir / "config.yml"
    with open(config_copy, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    return output_dir


def print_banner(config):
    """Print startup banner with configuration summary."""
    print("=" * 80)
    print(" " * 25 + "jax-hfbfft HFB Calculation")
    print("=" * 80)
    print()
    
    nucleus = config['nucleus']
    print(f"Nucleus:  {nucleus['name']} (Z={nucleus['protons']}, N={nucleus['neutrons']}, A={nucleus['protons']+nucleus['neutrons']})")
    
    force = config['force']
    print(f"Force:    {force['name']}")
    
    ipair_names = {0: "No pairing", 5: "VDI", 6: "DDDI"}
    ipair_name = ipair_names.get(force['ipair'], f"ipair={force['ipair']}")
    pairing_method = 'BCS' if force['use_bcs'] else 'HFB'
    print(f"Pairing:  {ipair_name} ({pairing_method})")
    
    grid = config['grid']
    print(f"Grid:     {grid['nx']}×{grid['ny']}×{grid['nz']} points")
    
    iteration = config['iteration']
    print(f"Target:   Convergence < {iteration['convergence_threshold']:.1e} MeV "
          f"(max {iteration['max_iterations']} iterations)")
    
    print()


def create_nucleus(config):
    """Create Nucleus object from config."""
    from jax_hfbfft import Nucleus
    
    return Nucleus(
        protons=config['nucleus']['protons'],
        neutrons=config['nucleus']['neutrons']
    )


def create_force(config):
    """Create Force object from config."""
    from jax_hfbfft import Force
    
    force_config = config['force']
    
    # Load base force
    force = Force.from_name(force_config['name'])
    
    # Configure pairing
    force = force.with_pairing(
        ipair=force_config['ipair'],
        v0neut=force_config['v0_neutron'],
        v0prot=force_config['v0_proton'],
        rho0pr=force_config['rho0_pairing'],
    )
    
    # Set BCS flag
    if 'use_bcs' in force_config:
        force.tbcs = force_config['use_bcs']

    # Set pairing energy window cutoff (pair_cutoff: which states enter the pairing sum)
    if 'pairing_cutoff' in force_config:
        cutoff = force_config['pairing_cutoff']
        force.pair_cutoff = jnp.array(cutoff if isinstance(cutoff, list) else [cutoff, cutoff])

    # Set wstates soft cutoff (state_cutoff: weight suppression above Fermi level)
    if 'state_cutoff' in force_config:
        sc = force_config['state_cutoff']
        force.state_cutoff = jnp.array(sc if isinstance(sc, list) else [sc, sc])

    # Center-of-mass correction
    if not force_config.get('include_cm_correction', True):
        force.zpe = 0
    
    return force


def create_grid(config, nucleus):
    """Create Grid object from config."""
    from jax_hfbfft import Grid
    
    grid_config = config['grid']
    
    # Determine grid spacing
    dx = grid_config.get('dx')
    dy = grid_config.get('dy')
    dz = grid_config.get('dz')
    
    # Auto-determine spacing if not specified
    if dx is None or dy is None or dz is None:
        # Estimate based on nucleus size
        A = nucleus.mass_number
        r0 = 1.2  # fm
        R = r0 * A**(1/3)  # Nuclear radius
        
        # Box should be ~4-5× nuclear radius
        box_size = 5.0 * R
        
        # Round up to have reasonable spacing
        nx = grid_config.get('nx', 32)
        default_spacing = box_size / nx
        
        # Round to nearest 0.1 fm
        default_spacing = round(default_spacing * 10) / 10
        
        dx = dx or default_spacing
        dy = dy or default_spacing
        dz = dz or default_spacing
    
    return Grid.create(
        nx=grid_config['nx'],
        ny=grid_config['ny'],
        nz=grid_config['nz'],
        dx=dx,
        dy=dy,
        dz=dz
    )


def determine_basis_size(config, nucleus, force):
    """Determine number of basis states."""
    basis = config['basis']
    
    npsi_n = basis.get('npsi_neutron')
    npsi_p = basis.get('npsi_proton')
    
    # Auto-determine if not specified
    if npsi_n is None:
        if force.ipair == 0:  # No pairing
            npsi_n = nucleus.neutrons
        else:  # With pairing
            npsi_n = max(2 * nucleus.neutrons, 20)
    
    if npsi_p is None:
        if force.ipair == 0:  # No pairing
            npsi_p = nucleus.protons
        else:  # With pairing
            npsi_p = max(2 * nucleus.protons, 20)
    
    return npsi_n, npsi_p


def create_constraint(config):
    """Create Constraint object from config."""
    from jax_hfbfft import Constraint
    
    if 'constraints' not in config or config['constraints'] is None:
        return None
    
    constraints_config = config['constraints']
    
    # Check if beta_gamma parameters are specified
    if 'beta_gamma' in constraints_config and constraints_config['beta_gamma']:
        beta_gamma_config = constraints_config['beta_gamma']
        
        # Get mass number
        mass_number = config['nucleus']['protons'] + config['nucleus']['neutrons']
        
        # Get principal axes flag
        principal_axes = constraints_config.get('principal_axes', False)
        
        # Get algorithm parameters (if specified)
        kwargs = {}
        for param in ['c0constr', 'qepsconstr', 'dampgamma', 'damprad']:
            if param in constraints_config:
                kwargs[param] = constraints_config[param]
        
        # Create constraint from beta-gamma parameters
        return Constraint.from_beta_gamma(
            mass_number=mass_number,
            beta2=beta_gamma_config.get('beta2'),
            gamma=beta_gamma_config.get('gamma'),
            beta3=beta_gamma_config.get('beta3'),
            beta4=beta_gamma_config.get('beta4'),
            r0=beta_gamma_config.get('r0', 1.2),
            principal_axes=principal_axes,
            **kwargs
        )
    
    # Check if multipoles are specified
    multipoles = constraints_config.get('multipoles', {})
    if not multipoles:
        # No constraints specified
        return None
    
    # Convert multipole specifications
    # Handle both string keys (Q20, Q30) and list keys ([2, 0], [3, 0])
    parsed_multipoles = {}
    for key, value in multipoles.items():
        if isinstance(key, list):
            # Convert list to tuple for Constraint class
            parsed_multipoles[tuple(key)] = value
        else:
            # String name like 'Q20'
            parsed_multipoles[key] = value
    
    # Get principal axes flag
    principal_axes = constraints_config.get('principal_axes', False)
    
    # Get algorithm parameters (if specified)
    kwargs = {}
    for param in ['c0constr', 'qepsconstr', 'dampgamma', 'damprad']:
        if param in constraints_config:
            kwargs[param] = constraints_config[param]
    
    return Constraint.from_multipoles(
        multipoles=parsed_multipoles,
        principal_axes=principal_axes,
        **kwargs
    )


def save_results(calc, results, config, output_dir):
    """Save results to output files."""
    output_config = config['output']
    
    # Always save summary
    save_summary(calc, results, config, output_dir)
    
    if output_config['save_convergence']:
        save_convergence(calc, results, output_dir)
    
    if output_config['save_energies']:
        save_energies(calc, results, output_dir)
    
    if output_config['save_moments']:
        save_moments(calc, results, output_dir)
    
    if output_config['save_radii']:
        save_radii(calc, results, output_dir)
    
    if output_config['save_single_particle']:
        save_single_particle(calc, results, output_dir)
    
    if output_config['save_pairing']:
        save_pairing(calc, results, output_dir)
    
    if output_config['save_densities']:
        save_densities(calc, results, output_dir)


def save_summary(calc, results, config, output_dir):
    """Save summary file with main results."""
    summary_file = output_dir / "summary.txt"
    
    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(" " * 25 + "jax-hfbfft HFB Calculation\n")
        f.write("=" * 80 + "\n\n")
        
        # Configuration
        nucleus = config['nucleus']
        f.write(f"Nucleus:  {nucleus['name']} (Z={nucleus['protons']}, N={nucleus['neutrons']}, A={nucleus['protons']+nucleus['neutrons']})\n")
        f.write(f"Force:    {config['force']['name']}\n")
        
        ipair_names = {0: "No pairing", 5: "VDI", 6: "DDDI"}
        ipair = config['force']['ipair']
        ipair_name = ipair_names.get(ipair, f"ipair={ipair}")
        f.write(f"Pairing:  {ipair_name}\n")
        
        grid = config['grid']
        f.write(f"Grid:     {grid['nx']}×{grid['ny']}×{grid['nz']} points\n\n")
        
        # Convergence
        f.write("=" * 80 + "\n")
        f.write("CONVERGENCE\n")
        f.write("=" * 80 + "\n")
        f.write(f"Converged:        {results.converged}\n")
        f.write(f"Iterations:       {results.iterations}\n")
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
        A = nucleus['protons'] + nucleus['neutrons']
        f.write(f"Binding energy/A:          {results.total_energy/A:12.3f} MeV\n\n")
        
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
        if ipair != 0:
            f.write("=" * 80 + "\n")
            f.write("PAIRING\n")
            f.write("=" * 80 + "\n")
            f.write(f"Neutron Fermi energy:      {results.fermi_energy_n:12.3f} MeV\n")
            f.write(f"Proton Fermi energy:       {results.fermi_energy_p:12.3f} MeV\n")
            f.write(f"Neutron pairing gap:       {results.pairing_gap_n:12.3f} MeV\n")
            f.write(f"Proton pairing gap:        {results.pairing_gap_p:12.3f} MeV\n\n")


def save_convergence(calc, results, output_dir):
    """Save convergence history."""
    if results.convergence_history is None:
        return
    
    conv_file = output_dir / "convergence.res"
    
    with open(conv_file, 'w') as f:
        f.write("# Iteration    Fluctuation(MeV)\n")
        for i, fluct in enumerate(results.convergence_history):
            f.write(f"{i:6d}    {float(fluct):15.6e}\n")


def save_energies(calc, results, output_dir):
    """Save detailed energy breakdown."""
    energy_file = output_dir / "energies.res"
    
    state = calc._solver_state
    energies = state.energies
    pairing = state.pairing
    nucleus = calc.nucleus
    
    with open(energy_file, 'w') as f:
        # Header
        f.write("# Detailed energy breakdown (all in MeV)\n")
        f.write("# Format: Iter  N(n)  N(p)  E_total  E_kin  E_Coul  ehf0  ehf1  ehf2  ehf3  ehfls  ")
        f.write("Crho0  Crho1  Cdrho0  Cdrho1  Ctau0  Ctau1  CdJ0  CdJ1  e_pair(n)  e_pair(p)  e_zpe\n")
        
        # Data
        f.write(f"{results.iterations:5d}  ")
        f.write(f"{nucleus.neutrons:4d}  {nucleus.protons:4d}  ")
        f.write(f"{float(energies.ehfint):12.6f}  ")
        f.write(f"{float(energies.ehft):12.6f}  ")
        f.write(f"{float(energies.ehfc):12.6f}  ")
        f.write(f"{float(energies.ehf0):12.6f}  ")
        f.write(f"{float(energies.ehf1):12.6f}  ")
        f.write(f"{float(energies.ehf2):12.6f}  ")
        f.write(f"{float(energies.ehf3):12.6f}  ")
        f.write(f"{float(energies.ehfls):12.6f}  ")
        f.write(f"{float(energies.ehfCrho0):12.6f}  ")
        f.write(f"{float(energies.ehfCrho1):12.6f}  ")
        f.write(f"{float(energies.ehfCdrho0):12.6f}  ")
        f.write(f"{float(energies.ehfCdrho1):12.6f}  ")
        f.write(f"{float(energies.ehfCtau0):12.6f}  ")
        f.write(f"{float(energies.ehfCtau1):12.6f}  ")
        f.write(f"{float(energies.ehfCdJ0):12.6f}  ")
        f.write(f"{float(energies.ehfCdJ1):12.6f}  ")
        f.write(f"{float(pairing.epair[0]):12.6f}  ")
        f.write(f"{float(pairing.epair[1]):12.6f}  ")
        f.write(f"{float(energies.e_zpe):12.6f}\n")


def save_moments(calc, results, output_dir):
    """Save multipole moments."""
    moments_file = output_dir / "moments.res"
    
    with open(moments_file, 'w') as f:
        f.write("# Multipole moments\n")
        f.write(f"# Q20 (fm²):     {results.q20:12.6f}\n")
        f.write(f"# Q22 (fm²):     {results.q22:12.6f}\n")
        f.write(f"# beta2:         {results.beta2:12.6f}\n")
        f.write(f"# gamma (deg):   {results.gamma:12.6f}\n")


def save_radii(calc, results, output_dir):
    """Save radii information."""
    radii_file = output_dir / "radii.res"
    
    with open(radii_file, 'w') as f:
        f.write("# RMS radii (fm)\n")
        f.write(f"# Neutron:       {results.rms_radius_n:12.6f}\n")
        f.write(f"# Proton:        {results.rms_radius_p:12.6f}\n")
        f.write(f"# Total:         {results.rms_radius_total:12.6f}\n")
        f.write(f"# Charge:        {results.charge_radius:12.6f}\n")


def save_single_particle(calc, results, output_dir):
    """Save single-particle spectrum."""
    sp_file = output_dir / "single_particle.res"
    
    state = calc._solver_state
    sp_energies = state.sp_energy
    sp_occ = state.wocc
    isospin = state.isospin
    
    with open(sp_file, 'w') as f:
        f.write("# Single-particle spectrum (MeV)\n")
        f.write("# Format: State  Isospin  Energy  Occupation\n\n")
        
        # Separate by isospin
        f.write("# Neutron states (isospin=0)\n")
        neutron_mask = isospin == 0
        n_energies = sp_energies[neutron_mask]
        n_occ = sp_occ[neutron_mask]
        n_sorted = jnp.argsort(n_energies)
        
        for i, idx in enumerate(n_sorted):
            f.write(f"{i+1:5d}  0  {float(n_energies[idx]):12.6f}  {float(n_occ[idx]):10.6f}\n")
        
        f.write("\n# Proton states (isospin=1)\n")
        proton_mask = isospin == 1
        p_energies = sp_energies[proton_mask]
        p_occ = sp_occ[proton_mask]
        p_sorted = jnp.argsort(p_energies)
        
        for i, idx in enumerate(p_sorted):
            f.write(f"{i+1:5d}  1  {float(p_energies[idx]):12.6f}  {float(p_occ[idx]):10.6f}\n")


def save_pairing(calc, results, output_dir):
    """Save pairing properties."""
    pairing_file = output_dir / "pairing.res"
    
    state = calc._solver_state
    pairing = state.pairing
    
    with open(pairing_file, 'w') as f:
        f.write("# Pairing properties\n")
        f.write(f"# Neutron Fermi energy (MeV):    {float(pairing.eferm[0]):12.6f}\n")
        f.write(f"# Proton Fermi energy (MeV):     {float(pairing.eferm[1]):12.6f}\n")
        f.write(f"# Neutron avg gap uv (MeV):      {float(pairing.avdelt[0]):12.6f}\n")
        f.write(f"# Proton avg gap uv (MeV):       {float(pairing.avdelt[1]):12.6f}\n")
        f.write(f"# Neutron avg gap v2 (MeV):      {float(pairing.avdeltv2[0]):12.6f}\n")
        f.write(f"# Proton avg gap v2 (MeV):       {float(pairing.avdeltv2[1]):12.6f}\n")
        f.write(f"# Neutron pairing energy (MeV):  {float(pairing.epair[0]):12.6f}\n")
        f.write(f"# Proton pairing energy (MeV):   {float(pairing.epair[1]):12.6f}\n")


def save_densities(calc, results, output_dir):
    """Save density distributions (3D grids)."""
    densities_file = output_dir / "densities.npz"
    
    state = calc._solver_state
    densities = state.densities
    grid = calc.grid
    
    # Save as numpy arrays
    import numpy as np
    np.savez(
        densities_file,
        x=np.array(grid.x),
        y=np.array(grid.y),
        z=np.array(grid.z),
        rho_neutron=np.array(densities.rho[0]),
        rho_proton=np.array(densities.rho[1]),
        tau_neutron=np.array(densities.tau[0]),
        tau_proton=np.array(densities.tau[1]),
    )


def print_results(calc, results, config):
    """Print results to stdout."""
    if not config['output']['verbose']:
        return
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    
    print(f"\nConvergence: {results.converged}")
    print(f"Iterations:  {results.iterations}")
    print(f"Fluctuation: {results.final_fluctuation:.3e} MeV")
    
    print("\n" + "-" * 80)
    print("ENERGIES")
    print("-" * 80)
    print(f"Total binding energy:  {results.total_energy:12.3f} MeV")
    print(f"  Kinetic:             {results.kinetic_energy:12.3f} MeV")
    print(f"  Potential:           {results.potential_energy:12.3f} MeV")
    print(f"  Coulomb:             {results.coulomb_energy:12.3f} MeV")
    print(f"  Pairing:             {results.pairing_energy:12.3f} MeV")
    print(f"  Rearrangement:       {results.rearrangement_energy:12.3f} MeV")
    print(f"  CM correction:       {results.cm_correction:12.3f} MeV")
    
    A = config['nucleus']['protons'] + config['nucleus']['neutrons']
    print(f"\nBinding energy/A:      {results.total_energy/A:12.3f} MeV")
    
    print("\n" + "-" * 80)
    print("RADII AND DEFORMATION")
    print("-" * 80)
    print(f"RMS radius (n):        {results.rms_radius_n:12.3f} fm")
    print(f"RMS radius (p):        {results.rms_radius_p:12.3f} fm")
    print(f"RMS radius (total):    {results.rms_radius_total:12.3f} fm")
    print(f"Charge radius:         {results.charge_radius:12.3f} fm")
    print(f"Q20:                   {results.q20:12.3f} fm²")
    print(f"Q22:                   {results.q22:12.3f} fm²")
    print(f"β₂:                    {results.beta2:12.4f}")
    print(f"γ:                     {results.gamma:12.1f} deg")
    
    if config['force']['ipair'] != 0:
        print("\n" + "-" * 80)
        print("PAIRING")
        print("-" * 80)
        print(f"Neutron Fermi energy:  {results.fermi_energy_n:12.3f} MeV")
        print(f"Proton Fermi energy:   {results.fermi_energy_p:12.3f} MeV")
        print(f"Neutron pairing gap:   {results.pairing_gap_n:12.3f} MeV")
        print(f"Proton pairing gap:    {results.pairing_gap_p:12.3f} MeV")
    
    print("\n" + "=" * 80)


def main():
    """Main execution function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run HFB calculation with jax-hfbfft',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_hfb.py                    # Use config.yml
  python run_hfb.py my_config.yml      # Use custom config file
        """
    )
    parser.add_argument('config', nargs='?', default='config.yml',
                       help='Configuration file (default: config.yml)')
    args = parser.parse_args()
    
    # Load and validate configuration
    print("Loading configuration from:", args.config)
    config = load_config(args.config)
    config = validate_config(config)
    
    # Setup JAX
    setup_jax(config)
    
    # Now import jax_hfbfft (after JAX configuration)
    from jax_hfbfft import HFBFFT
    
    # Print banner
    print_banner(config)
    
    # Setup output directory
    output_dir = setup_output_directory(config)
    print(f"Output directory: {output_dir}")
    print()
    
    # Create components
    print("Creating calculation components...")
    nucleus = create_nucleus(config)
    force = create_force(config)
    grid = create_grid(config, nucleus)
    constraint = create_constraint(config)
    
    # Determine basis size
    npsi_n, npsi_p = determine_basis_size(config, nucleus, force)
    print(f"Basis size: {npsi_n} neutron states, {npsi_p} proton states")
    
    # Print constraint info if present
    if constraint is not None:
        print(f"Constraints: {constraint}")
    
    # Create HFBFFT calculation
    calc = HFBFFT(
        nucleus=nucleus,
        force=force,
        grid=grid,
        npsi=(npsi_n, npsi_p),
        include_coulomb=config['physics']['include_coulomb'],
        constraint=constraint,
    )

    # Wire iteration parameters from config.yml into the solver.
    # HFBFFT.__init__ hardcodes defaults (bcs_start=0, diag_start=0,
    # tvaryx_0=False) that would otherwise silently override the config file.
    iter_params = config['iteration']
    calc.x0dmp = float(iter_params['x0dmp'])
    calc.e0dmp = float(iter_params['e0dmp'])
    calc.density_mixing = float(iter_params['density_mixing'])
    calc.diag_start = int(iter_params['diag_start'])
    calc.bcs_start = int(iter_params['bcs_start'])
    calc.tvaryx_0 = bool(iter_params.get('tvaryx_0', False))
    print(f"Iteration params: x0dmp={calc.x0dmp}  e0dmp={calc.e0dmp}  "
          f"density_mixing={calc.density_mixing}  diag_start={calc.diag_start}  "
          f"bcs_start={calc.bcs_start}  tvaryx_0={calc.tvaryx_0}")

    # Initialize wavefunctions
    print("\nInitializing wavefunctions...")
    init_config = config['initialization']
    if init_config['method'] == 'harmonic_oscillator':
        calc.initialize_wavefunctions(
            method='harmonic_oscillator',
            radinx=init_config['ho_length_x'],
            radiny=init_config['ho_length_y'],
            radinz=init_config['ho_length_z'],
        )
    else:
        calc.initialize_wavefunctions(method=init_config['method'])
    
    # Run calculation
    print("\nStarting HFB iterations...")
    print("-" * 80)
    
    iter_config = config['iteration']
    results = calc.run(
        max_iterations=iter_config['max_iterations'],
        convergence_threshold=iter_config['convergence_threshold'],
        print_interval=config['output']['print_interval'],
        save_dir=str(output_dir),
        save_interval=5,
    )
    
    # Save results
    print("\nSaving results...")
    save_results(calc, results, config, output_dir)
    
    # Print summary
    print_results(calc, results, config)
    
    print(f"\nResults saved to: {output_dir}")
    print("\nCalculation complete!")
    
    return 0 if results.converged else 1


if __name__ == "__main__":
    sys.exit(main())
