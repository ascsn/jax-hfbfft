# test_res_output.py - Complete test script that writes .res files

import jax
import jax.numpy as jnp
import os
from reader import read_yaml
from params import init_params
from forces import init_forces
from grids import init_grids
from densities import init_densities
from meanfield import init_meanfield
from levels import init_levels
from static import init_static, harmosc
from coulomb import init_coulomb
from moment import init_moment
from energies import init_energies
from pairs import Pairs
from output import FortranOutputWriter, complete_sinfo_with_fortran_output

# Import existing calculation functions
from static import statichf
from densities import add_density
from meanfield import skyrme
from static import grstep, diagstep
from inout import sp_properties
from energies import integ_energy, sum_energy

jax.config.update('jax_enable_x64', True)

def test_res_file_output():
    """Test script that runs a few iterations and writes .res files"""
    
    print("=== TESTING .RES FILE OUTPUT ===\n")
    
    # 1. Setup calculation
    config = read_yaml('_config.yml')
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
    pairs = Pairs()  # Initialize with default values
    
    print(f"Initialized calculation: N={levels.nneut}, Z={levels.nprot}")
    
    # 2. Create output directory and writer
    output_dir = "res_test_output"
    os.makedirs(output_dir, exist_ok=True)
    output_writer = FortranOutputWriter(output_dir)
    print(f"Created output directory: {output_dir}")
    
    # 3. Initialize wavefunctions
    print("Initializing wavefunctions...")
    levels = harmosc(grids, levels, params, static)
    
    # 4. Run a few test iterations with .res output
    print("Starting test iterations with .res output...\n")
    
    # Set to run only a few iterations for testing
    original_maxiter = static.maxiter
    static.maxiter = min(5000, static.maxiter)
    
    try:
        # Run the main calculation with integrated .res output
        results = run_calculation_with_res_output(
            coulomb, densities, energies, forces, grids, levels, 
            meanfield, moment, params, static, pairs, output_writer
        )
        
        print(f"\n=== TEST COMPLETED SUCCESSFULLY ===")
        print(f"Ran {params.iteration} iterations")
        print(f"Final energy: {energies.ehf:.6f} MeV")
        print(f"Output files written to: {output_dir}/")
        
        # List the created files
        res_files = [f for f in os.listdir(output_dir) if f.endswith('.res')]
        print(f"Created .res files:")
        for file in sorted(res_files):
            file_path = os.path.join(output_dir, file)
            file_size = os.path.getsize(file_path)
            print(f"  {file} ({file_size} bytes)")
            
        return True
        
    except Exception as e:
        print(f"\n=== TEST FAILED ===")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        static.maxiter = original_maxiter  # Restore original value


def run_calculation_with_res_output(coulomb, densities, energies, forces, grids, levels, 
                                   meanfield, moment, params, static, pairs, output_writer):
    """Modified version of statichf that writes .res files at each iteration"""
    
    firstiter = 1
    addnew = 0.2
    addco = 1.0 - addnew
    taddnew = True

    if params.trestart:
        firstiter = params.iteration + 1
    else:
        params.iteration = 0
        energies, levels, static = diagstep(energies, forces, grids, levels, static, False, True)

    # Initial densities and fields
    densities = add_density(densities, grids, levels)
    meanfield, coulomb = skyrme(coulomb, densities, forces, grids, meanfield, params, static)
    
    # Initial gradient step and properties calculation
    levels, static = grstep(forces, grids, levels, meanfield, params, static)
    energies, levels, static = diagstep(energies, forces, grids, levels, static, False, True)
    
    levels = sp_properties(forces, grids, levels, moment)
    energies = integ_energy(coulomb, densities, energies, forces, grids, levels, params, pairs)
    energies = sum_energy(energies, levels, meanfield, pairs)
    
    # Write iteration 0 output
    energies, observables = complete_sinfo_with_fortran_output(
        coulomb, densities, energies, forces, grids, levels, 
        meanfield, moment, params, static, pairs, output_writer
    )
    
    print(f"Iteration 0: Energy={energies.ehf:.6f} MeV, RMS={observables.rmstot:.4f} fm")

    # Save pairing strengths for annealing
    v0protsav = forces.v0prot
    v0neutsav = forces.v0neut
    tbcssav = forces.tbcs

    # Main iteration loop
    for i in range(firstiter, static.maxiter + 1):
        params.iteration = i
        
        print(f"\n--- Iteration {i} ---")

        # Set BCS/HFB mode
        if i <= static.inibcs:
            forces.tbcs = True
        else:
            forces.tbcs = tbcssav

        # Set diagonalization mode
        if i > static.inidiag:
            static.tdiag = False
        else:
            static.tdiag = True

        # Pairing annealing
        if static.iteranneal > 0:
            if i < static.iteranneal:
                forces.v0prot = v0protsav + v0protsav * static.pairenhance * (static.iteranneal - i) / (1.0 * static.iteranneal)
                forces.v0neut = v0neutsav + v0neutsav * static.pairenhance * (static.iteranneal - i) / (1.0 * static.iteranneal)
            else:
                forces.v0prot = v0protsav
                forces.v0neut = v0neutsav

        # Gradient step
        levels, static = grstep(forces, grids, levels, meanfield, params, static)

        if forces.tbcs:
            static.tdiag = True

        # Diagonalization
        energies, levels, static = diagstep(energies, forces, grids, levels, static, static.tdiag, True)

        # Density mixing
        if taddnew:
            meanfield.upot = meanfield.upot.at[...].set(densities.rho)
            meanfield.bmass = meanfield.bmass.at[...].set(densities.tau)
            meanfield.v_pair = meanfield.v_pair.at[...].set(densities.chi)

        # Calculate new densities
        densities = add_density(densities, grids, levels)

        # Mix densities for stability
        if taddnew:
            densities.rho = densities.rho.at[...].set(addco * meanfield.upot + addnew * densities.rho)
            densities.tau = densities.tau.at[...].set(addco * meanfield.bmass + addnew * densities.tau)
            densities.chi = densities.chi.at[...].set(addco * meanfield.v_pair + addnew * densities.chi)

        # Calculate new mean fields
        meanfield, coulomb = skyrme(coulomb, densities, forces, grids, meanfield, params, static)


        levels = sp_properties(forces, grids, levels, moment)
        energies = integ_energy(coulomb, densities, energies, forces, grids, levels, params, pairs)
        energies = sum_energy(energies, levels, meanfield, pairs)
        
        if (i % 5 == 0):
            # Write .res files for this iteration
            energies, observables = complete_sinfo_with_fortran_output(
                coulomb, densities, energies, forces, grids, levels, 
                meanfield, moment, params, static, pairs, output_writer
            )
            
            print(f"Iteration {i}: Energy={energies.ehf:.6f} MeV, RMS={observables.rmstot:.4f} fm, Beta={observables.beta:.4f}")

        # Check convergence
        if energies.efluct1[0] < static.serr:
            print(f"*** CONVERGED at iteration {i} ***")
            print(f"Energy fluctuation {energies.efluct1[0]:.2e} < {static.serr:.2e}")
            break

    return coulomb, densities, energies, forces, grids, levels, meanfield, moment, params, static


def examine_res_files(output_dir="res_test_output"):
    """Examine the contents of generated .res files"""
    
    print(f"\n=== EXAMINING .RES FILES ===")
    
    res_files = ['conver.res', 'energies.res', 'dipoles.res', 'spin.res', 
                 'monopoles.res', 'quadrupoles.res', 'momenta.res']
    
    for filename in res_files:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            print(f"\n--- {filename} ---")
            with open(filepath, 'r') as f:
                lines = f.readlines()
                print(f"Header: {lines[0].strip()}")
                if len(lines) > 1:
                    print(f"First data line: {lines[1].strip()}")
                    if len(lines) > 2:
                        print(f"Last data line: {lines[-1].strip()}")
                print(f"Total lines: {len(lines)}")
        else:
            print(f"\n{filename}: NOT FOUND")


if __name__ == "__main__":
    # Run the test
    success = test_res_file_output()
    
    if success:
        # Examine the results
        examine_res_files()
        
        print(f"\n=== NEXT STEPS ===")
        print("1. Check the .res files in res_test_output/")
        print("2. Compare with FORTRAN reference files")
        print("3. Fix any issues in sp_properties or calculate_complete_observables")
        print("4. Once working, integrate into your main calculation")
    else:
        print("\n=== DEBUGGING NEEDED ===")
        print("1. Check if sp_properties is working correctly")
        print("2. Verify calculate_complete_observables is complete")
        print("3. Check that all required energy fields exist")
        print("4. Run with smaller system or fewer iterations to isolate issues")