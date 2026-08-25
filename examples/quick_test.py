#!/usr/bin/env python3
"""Quick test of the jax-hfbfft OOP API with minimal iterations."""

import os
import sys

# Enable 64-bit precision for JAX
os.environ['JAX_ENABLE_X64'] = 'True'

import jax.numpy as jnp
from jax_hfbfft import HFBFFT, Nucleus, Force, Grid


def main():
    print("=" * 60)
    print("Quick Test: jax-hfbfft OOP API")
    print("=" * 60)
    
    # Create a simple nucleus calculation - O-16
    nucleus = Nucleus(protons=8, neutrons=8)
    print(f"Nucleus: O-16 (Z={nucleus.protons}, N={nucleus.neutrons})")
    
    # Use SLy4 force with no pairing (simpler/faster)
    force = Force.from_name("SLy4").with_pairing(ipair=0)
    print(f"Force: {force.name} (no pairing)")
    
    # Small grid for speed
    grid = Grid.create(nx=32, ny=32, nz=32, dx=0.8, dy=0.8, dz=0.8)
    print(f"Grid: {grid.nx}x{grid.ny}x{grid.nz}, spacing={grid.dx} fm")
    
    # Create HFBFFT calculation
    # With ipair=0 (no pairing), npsi is automatically set to (N, Z)
    calc = HFBFFT(
        nucleus=nucleus,
        force=force,
        grid=grid,
        include_coulomb=True,  
    )
    print(f"Basis: {calc._npsi[0]} neutron states, {calc._npsi[1]} proton states")
    
    # Initialize wavefunctions
    print("\nInitializing wavefunctions...")
    calc.initialize_wavefunctions(
        method="harmonic_oscillator",
        radinx=3.0, radiny=3.0, radinz=3.0
    )
    
    # Run a few iterations (not to convergence)
    print("\nStarting HFB iteration (500 iterations)...")
    try:
        results = calc.run(
            max_iterations=500,
            convergence_threshold=1e-6,
            print_interval=50,
        )
        
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Converged: {results.converged}")
        print(f"Iterations: {results.iterations}")
        print(f"Total energy: {results.total_energy:.3f} MeV")
        
        # Print detailed energy breakdown
        print("\n" + "=" * 60)
        print("ENERGY BREAKDOWN - INTERACTION TERMS")
        print("=" * 60)
        final_state = calc._solver_state
        energies = final_state.energies
        print(f"  ehft (kinetic):    {float(energies.ehft):12.3f} MeV")
        print(f"  ehf0 (t0 central): {float(energies.ehf0):12.3f} MeV")
        print(f"  ehf1 (t1/t2 mom):  {float(energies.ehf1):12.3f} MeV")
        print(f"  ehf2 (gradient):   {float(energies.ehf2):12.3f} MeV")
        print(f"  ehf3 (t3 dens):    {float(energies.ehf3):12.3f} MeV")
        print(f"  ehfls (spin-orb):  {float(energies.ehfls):12.3f} MeV")
        print(f"  ehfc (Coulomb):    {float(energies.ehfc):12.3f} MeV")
        print(f"  e3corr (rearr):    {float(energies.e3corr):12.3f} MeV")
        print(f"  e_zpe (cm corr):   {float(energies.e_zpe):12.3f} MeV")
        print(f"  epair (pairing):   {float(final_state.pairing.epair.sum()):12.3f} MeV")
        print(f"  ---------------------------------")
        print(f"  ehfint (total):    {float(energies.ehfint):12.3f} MeV")
        
        # Print EDF coupling constant contributions
        print("\n" + "=" * 60)
        print("ENERGY BREAKDOWN - EDF COUPLING CONSTANTS")
        print("=" * 60)
        print(f"  Crho0 (isoscalar dens):    {float(energies.ehfCrho0):12.3f} MeV")
        print(f"  Crho1 (isovector dens):    {float(energies.ehfCrho1):12.3f} MeV")
        print(f"  Cdrho0 (isoscalar grad):   {float(energies.ehfCdrho0):12.3f} MeV")
        print(f"  Cdrho1 (isovector grad):   {float(energies.ehfCdrho1):12.3f} MeV")
        print(f"  Ctau0 (isoscalar kin):     {float(energies.ehfCtau0):12.3f} MeV")
        print(f"  Ctau1 (isovector kin):     {float(energies.ehfCtau1):12.3f} MeV")
        print(f"  CdJ0 (isoscalar spin-orb): {float(energies.ehfCdJ0):12.3f} MeV")
        print(f"  CdJ1 (isovector spin-orb): {float(energies.ehfCdJ1):12.3f} MeV")
        print(f"  Cj0 (isoscalar current):   {float(energies.ehfCj0):12.3f} MeV")
        print(f"  Cj1 (isovector current):   {float(energies.ehfCj1):12.3f} MeV")
        
        # Additional useful energy information
        print("\n" + "=" * 60)
        print("ADDITIONAL ENERGY INFO")
        print("=" * 60)
        print(f"  Time-odd spin-orbit:  {float(energies.ehflsodd):12.3f} MeV")
        print(f"  Exchange correlation: {float(energies.ecorc):12.3f} MeV")
        print(f"  Total kinetic (tke):  {float(energies.tke):12.3f} MeV")
        print(f"  From s.p. levels:     {float(energies.ehf):12.3f} MeV")
        
        # Also print densities for sanity check
        print("\n" + "=" * 60)
        print("DENSITY INTEGRALS")
        print("=" * 60)
        dens = final_state.densities
        wxyz = calc.grid.wxyz
        import jax.numpy as jnp
        n_integral = float(wxyz * jnp.sum(dens.rho[0]))
        p_integral = float(wxyz * jnp.sum(dens.rho[1]))
        print(f"  Neutrons: {n_integral:.3f} (target: {nucleus.neutrons})")
        print(f"  Protons:  {p_integral:.3f} (target: {nucleus.protons})")
        print(f"  Total:    {n_integral + p_integral:.3f} (target: {nucleus.mass_number})")
        
        # Print radii and deformation
        print("\n" + "=" * 60)
        print("RADII AND DEFORMATION")
        print("=" * 60)
        print(f"  RMS radius (n): {results.rms_radius_n:.3f} fm")
        print(f"  RMS radius (p): {results.rms_radius_p:.3f} fm")
        print(f"  RMS radius:     {results.rms_radius_total:.3f} fm")
        print(f"  Charge radius:  {results.charge_radius:.3f} fm")
        print(f"  beta2:          {results.beta2:.4f}")
        print(f"  gamma:          {results.gamma:.1f} deg")
        print(f"  Q20:            {results.q20:.3f} fm^2")
        print(f"  Q22:            {results.q22:.3f} fm^2")
        
        # Print single-particle spectrum
        print("\n" + "=" * 60)
        print("SINGLE-PARTICLE SPECTRUM")
        print("=" * 60)
        sp_energies = final_state.sp_energy
        sp_occ = final_state.wocc
        isospin = final_state.isospin
        
        # Separate neutron and proton states
        neutron_mask = isospin == 0
        proton_mask = isospin == 1
        
        neutron_energies = sp_energies[neutron_mask]
        proton_energies = sp_energies[proton_mask]
        neutron_occ = sp_occ[neutron_mask]
        proton_occ = sp_occ[proton_mask]
        
        # Sort by energy
        n_sorted_idx = jnp.argsort(neutron_energies)
        p_sorted_idx = jnp.argsort(proton_energies)
        
        print("\nNeutron states:")
        print("  State    Energy (MeV)    Occupation")
        print("  " + "-" * 40)
        for i, idx in enumerate(n_sorted_idx):
            if i < 20:  # Print first 20 states
                print(f"  {i+1:3d}      {float(neutron_energies[idx]):10.4f}      {float(neutron_occ[idx]):8.5f}")
            elif i == 20:
                print(f"  ...  (showing first 20 of {len(neutron_energies)} states)")
                break
        
        print("\nProton states:")
        print("  State    Energy (MeV)    Occupation")
        print("  " + "-" * 40)
        for i, idx in enumerate(p_sorted_idx):
            if i < 20:  # Print first 20 states
                print(f"  {i+1:3d}      {float(proton_energies[idx]):10.4f}      {float(proton_occ[idx]):8.5f}")
            elif i == 20:
                print(f"  ...  (showing first 20 of {len(proton_energies)} states)")
                break
        
        # Print Fermi energies
        print("\n" + "=" * 60)
        print("FERMI ENERGIES")
        print("=" * 60)
        print(f"  Neutron lambda: {float(final_state.pairing.eferm[0]):10.4f} MeV")
        print(f"  Proton lambda:  {float(final_state.pairing.eferm[1]):10.4f} MeV")
        
        # Print pairing gaps
        print("\n" + "=" * 60)
        print("PAIRING PROPERTIES")
        print("=" * 60)
        print(f"  Neutron avg gap (uv):  {float(final_state.pairing.avdelt[0]):10.4f} MeV")
        print(f"  Proton avg gap (uv):   {float(final_state.pairing.avdelt[1]):10.4f} MeV")
        print(f"  Neutron avg gap (v2):  {float(final_state.pairing.avdeltv2[0]):10.4f} MeV")
        print(f"  Proton avg gap (v2):   {float(final_state.pairing.avdeltv2[1]):10.4f} MeV")
        print(f"  Neutron epair:         {float(final_state.pairing.epair[0]):10.4f} MeV")
        print(f"  Proton epair:          {float(final_state.pairing.epair[1]):10.4f} MeV")
        print(f"  Neutron avg force:     {float(final_state.pairing.avg[0]):10.4f} MeV")
        print(f"  Proton avg force:      {float(final_state.pairing.avg[1]):10.4f} MeV")
        
        print("\nTest PASSED! The OOP API is working.")
        
    except Exception as e:
        print(f"\nError during calculation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
