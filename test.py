import jax
import jax.numpy as jnp
import os
import time
import traceback
import numpy as np
import matplotlib.pyplot as plt
from reader import read_yaml
from params import init_params
from forces import init_forces
from grids import init_grids
from densities import init_densities
from meanfield import init_meanfield
from levels import init_levels
from static import init_static, statichf, harmosc, statichf_with_benchmark, statichf_with_detailed_benchmark
from coulomb import init_coulomb
from moment import init_moment
from energies import init_energies
from pairs import Pairs
from dataclasses import asdict
from functools import partial
from output import FortranOutputWriter, complete_sinfo_with_fortran_output

#os.environ['JAX_PLATFORMS'] = 'cpu'
#os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Create output directory
output_dir = "hfb_results_40Ca_detailed"
os.makedirs(output_dir, exist_ok=True)

# Add hash method to make classes hashable for JAX transformations
def safe_hash_dataclass(self):
    """A safer hash implementation for dataclasses containing JAX arrays"""
    return id(self)

# Apply to relevant classes
from levels import Levels
from densities import Densities
from meanfield import Meanfield
from static import Static

for cls in [Levels, Densities, Meanfield, Static]:
    if not hasattr(cls, '__hash__') or cls.__hash__ is None:
        cls.__hash__ = safe_hash_dataclass

def calculate_total_energy(densities, meanfield, forces, grids, levels):
    """Calculate the total HFB energy"""
    # Kinetic energy
    e_kin = jnp.sum(levels.wocc * levels.wstates * levels.sp_kinetic)
    
    # Potential energy
    e_pot = 0.5 * grids.wxyz * jnp.sum(
        jnp.real(densities.rho[0,...] * meanfield.upot[0,...] + 
                 densities.rho[1,...] * meanfield.upot[1,...])
    )
    
    # Pairing energy
    e_pair = 0.5 * grids.wxyz * jnp.sum(
        jnp.real(densities.chi[0,...] * meanfield.v_pair[0,...] + 
                 densities.chi[1,...] * meanfield.v_pair[1,...])
    )
    
    total_energy = e_kin + e_pot + e_pair
    return total_energy

def plot_density_profiles(output_dir, grids, densities):
    """Create and save density profiles"""
    # Create directory for plots
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Convert JAX arrays to NumPy for plotting
    x = np.array(grids.x)
    y = np.array(grids.y)
    z = np.array(grids.z)
    
    # Get central slices of densities
    rho_neutron = np.array(densities.rho[0, :, grids.ny//2, grids.nz//2])
    rho_proton = np.array(densities.rho[1, :, grids.ny//2, grids.nz//2])
    
    # Plot density profiles
    plt.figure(figsize=(10, 6))
    plt.plot(x, rho_neutron, label='Neutron', linewidth=2)
    plt.plot(x, rho_proton, label='Proton', linewidth=2)
    plt.plot(x, rho_neutron + rho_proton, label='Total', linewidth=2, linestyle='--')
    plt.xlabel('x (fm)', fontsize=14)
    plt.ylabel('Density (fm$^{-3}$)', fontsize=14)
    plt.title('Nuclear Density Profiles', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(plots_dir, 'density_profiles.png'), dpi=300)
    plt.close()

def save_results_to_files(output_dir, densities, energies, forces, grids, levels, params, static, total_energy):
    """Save calculation results to files"""
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save energy components
    with open(os.path.join(output_dir, "energies.txt"), "w") as f:
        f.write(f"Total energy: {total_energy:.6f} MeV\n")
        f.write(f"Iterations: {params.iteration}\n")
        f.write(f"Convergence: {energies.efluct1[0]:.6e}\n")
        f.write(f"Target convergence: {static.serr:.6e}\n")
    
    # Save single-particle energies
    with open(os.path.join(output_dir, "sp_energies.txt"), "w") as f:
        f.write("# idx  isospin  energy(MeV)  occupation  sp_norm\n")
        for i in range(levels.nstmax):
            isospin = int(levels.isospin[i])
            energy = float(levels.sp_energy[i])
            occ = float(levels.wocc[i])
            norm = float(levels.sp_norm[i])
            f.write(f"{i:3d}  {isospin:1d}  {energy:12.6f}  {occ:8.6f}  {norm:8.6f}\n")
    
    print(f"Results saved to {output_dir}/")

def run_hfb(config_file='_config.yml', force_name='SLy4', enable_pairing=True, save_results=True):
    """Run a full convergent HFB calculation"""
    try:
        print(f"Starting HFB calculation with {force_name} force")
        print(f"Pairing is {'enabled' if enable_pairing else 'disabled'}")
        
        # Read configuration
        config = read_yaml(config_file)
        
        # Modify configuration for this run
        if 'force' not in config:
            config['force'] = {}
        
        config['force']['name'] = force_name
        
        # Enable or disable pairing
        if enable_pairing:
            config['force']['ipair'] = 6  # Density-dependent delta interaction
            config['force']['tbcs'] = True
        else:
            config['force']['ipair'] = 0  # No pairing
            config['force']['tbcs'] = False
        
        # Initialize all components
        params = init_params(**config.get('params', {}))
        forces = init_forces(params, **config.get('force', {}))
        grids = init_grids(params, **config.get('grids', {}))
        densities = init_densities(grids)
        meanfield = init_meanfield(grids)
        levels = init_levels(grids, **config.get('levels', {}))
        forces, static = init_static(forces, levels, **config.get('static', {}))
        coulomb = init_coulomb(grids)
        energies = init_energies()
        moment = init_moment(jnp.array([0, 0, 0]))
        pairs = Pairs()
        
        # Print calculation parameters
        print(f"Nucleus: Z={levels.nprot}, N={levels.nneut}, A={levels.mass_number}")
        print(f"Grid: {grids.nx}x{grids.ny}x{grids.nz}, dx={grids.dx:.2f} fm")
        print(f"Force: {forces.name}, h2m={forces.h2m}")
        if enable_pairing:
            print(f"Pairing strengths: v0prot={forces.v0prot:.2f}, v0neut={forces.v0neut:.2f}")
        
        # Initialize wavefunctions with harmonic oscillator basis
        print("Initializing wavefunctions with harmonic oscillator basis...")
        levels = harmosc(grids, levels, params, static)
        
        # Run to convergence
        print("Starting self-consistent calculation...")
        start_time = time.time()
        
        output_writer = FortranOutputWriter(output_dir)

        try:
            coulomb, densities, energies, forces, grids, levels, meanfield, moment, params, static = statichf_with_benchmark(coulomb,densities,energies,forces,grids,levels,meanfield,moment,params,static,pairs,output_writer)

            # Ensure all JAX arrays are fully computed
            jax.block_until_ready(coulomb)
            jax.block_until_ready(densities)
            jax.block_until_ready(energies)
            jax.block_until_ready(forces)
            jax.block_until_ready(grids)
            jax.block_until_ready(levels)
            jax.block_until_ready(meanfield)
            jax.block_until_ready(moment)
            jax.block_until_ready(params)
            jax.block_until_ready(static)
        except Exception as e:
            print(f"Error in self-consistent calculation: {str(e)}")
            traceback.print_exc()
            raise
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Print final results
        print(f"\nResults after {params.iteration} iterations ({elapsed:.2f} seconds):")
        print(f"Convergence metric: {energies.efluct1[0]:.6e} (target: {static.serr:.6e})")
        
        # Calculate and print observables
        total_energy = calculate_total_energy(densities, meanfield, forces, grids, levels)
        print(f"Total energy: {total_energy:.6f} MeV")
        
        # Print single-particle energies
        print("\nSingle-particle energies (neutrons):")
        for i in range(min(10, levels.nneut)):
            print(f"  n{i+1}: {levels.sp_energy[i]:.4f} MeV")
        
        print("\nSingle-particle energies (protons):")
        for i in range(min(10, levels.nprot)):
            idx = i + levels.nneut
            print(f"  p{i+1}: {levels.sp_energy[idx]:.4f} MeV")
        
        # Save results if requested
        if save_results:
            save_results_to_files(output_dir, densities, energies, forces, grids, levels, params, static, total_energy)
            plot_density_profiles(output_dir, grids, densities)
        
        return coulomb, densities, energies, forces, grids, levels, meanfield, moment, params, static, total_energy
    
    except Exception as e:
        print(f"Error running HFB calculation: {str(e)}")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    # Set JAX to use 64-bit precision
    jax.config.update('jax_enable_x64', True)
    jax.profiler.start_trace("logs")
    try:
        # Run a calculation for Sn-132 with SLy4 force
        run_hfb(force_name='SLy4', enable_pairing=True)
    finally:
        jax.profiler.stop_trace()
        #print(f"Error in main execution: {str(e)}")
        #traceback.print_exc()
