import jax
import jax.numpy as jnp
import numpy as np
import os
from dataclasses import dataclass, field
from typing import Dict, Any

@jax.tree_util.register_dataclass
@dataclass
class CompleteNuclearObservables:
    
    # Basic particle numbers
    pnr: jax.Array = field(default_factory=lambda: jnp.zeros(2))  # [neutrons, protons]
    pnrtot: float = 0.0
    
    # Center of mass and dipole moments
    cm: jax.Array = field(default_factory=lambda: jnp.zeros((3, 2)))     # [x,y,z] × [n,p]
    cmtot: jax.Array = field(default_factory=lambda: jnp.zeros(3))       # Total CM
    dipole_isovector: jax.Array = field(default_factory=lambda: jnp.zeros(3))  # Isovector dipoles
    
    # RMS radii and moments
    rms: jax.Array = field(default_factory=lambda: jnp.zeros(2))         # [neutron, proton]
    rmstot: float = 0.0
    x2m: jax.Array = field(default_factory=lambda: jnp.zeros((3, 2)))    # Second moments
    x2mtot: jax.Array = field(default_factory=lambda: jnp.zeros(3))
    
    # Quadrupole moments
    q20: jax.Array = field(default_factory=lambda: jnp.zeros(2))         # Spherical Q20 [n,p]
    q22: jax.Array = field(default_factory=lambda: jnp.zeros(2))         # Spherical Q22 [n,p]
    q20tot: float = 0.0
    q22tot: float = 0.0
    q20T1: float = 0.0    # Isovector Q20
    
    # Deformation parameters
    beta20tot: float = 0.0
    beta22tot: float = 0.0
    beta: float = 0.0
    gamma: float = 0.0
    
    # Angular momentum (L, S, J)
    orbital_momentum: jax.Array = field(default_factory=lambda: jnp.zeros(3))    # Lx, Ly, Lz
    spin_momentum: jax.Array = field(default_factory=lambda: jnp.zeros(3))       # Sx, Sy, Sz
    total_momentum: jax.Array = field(default_factory=lambda: jnp.zeros(3))      # Jx, Jy, Jz
    
    # Momentum components (for dynamic case)
    pcm: jax.Array = field(default_factory=lambda: jnp.zeros((3, 2)))            # Integrated momentum


class FortranOutputWriter:
    """Handles writing all FORTRAN-compatible output files"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # File names matching FORTRAN exactly
        self.files = {
            'conver': os.path.join(output_dir, 'conver.res'),
            'energies': os.path.join(output_dir, 'energies.res'),
            'dipoles': os.path.join(output_dir, 'dipoles.res'),
            'spin': os.path.join(output_dir, 'spin.res'),
            'monopoles': os.path.join(output_dir, 'monopoles.res'),
            'quadrupoles': os.path.join(output_dir, 'quadrupoles.res'),
            'momenta': os.path.join(output_dir, 'momenta.res')
        }
        
        # Initialize files with headers
        self._initialize_files()
    
    def _initialize_files(self):
        """Initialize all output files with proper headers"""
        
        # conver.res - convergence information
        with open(self.files['conver'], 'w') as f:
            f.write('# Iter   Energy  d_Energy    sp_fluct    max(lam-)     rms(lam-)      rms    '
                   'beta2  gamma      x_0     e_pair(1)    e_pair(2)\n')
        
        # energies.res - detailed energy components
        with open(self.files['energies'], 'w') as f:
            f.write('# Iter    N(n)    N(p)       E(sum)         E(integ)       Ekin         '
                   'E_Coul         ehfCrho0       ehfCrho1       ehfCdrho0      ehfCdrho1     '
                   'ehfCtau0       ehfCtau1       ehfCdJ0        ehfCdJ1       e_pair(1)    '
                   'e_pair(2)      e_zpe\n')
        
        # dipoles.res - center of mass and dipole moments
        with open(self.files['dipoles'], 'w') as f:
            f.write('# Iter    c.m. x-y-z                                  Isovector '
                   'dipoles x-y-z\n')
        
        # spin.res - angular momentum components
        with open(self.files['spin'], 'w') as f:
            f.write('# Iter      Lx        Ly        Lz        Sx        Sy        '
                   'Sz        Jx        Jy        Jz\n')
        
        # monopoles.res - RMS radii and monopole moments
        with open(self.files['monopoles'], 'w') as f:
            f.write('# Iter    RMS_n     RMS_p     RMS_tot   RMS_diff   '
                   'N_dens    Z_dens    A_dens\n')
        
        # quadrupoles.res - quadrupole moments and deformation
        with open(self.files['quadrupoles'], 'w') as f:
            f.write('# Iter    Q20_n     Q20_p     Q20_tot   Q22_tot   '
                   '<x²>_tot  <y²>_tot  <z²>_tot  Beta      Gamma\n')
        
        # momenta.res - momentum components
        with open(self.files['momenta'], 'w') as f:
            f.write('# Iter    Px_n      Py_n      Pz_n      Px_p      Py_p      Pz_p      '
                   'Px_tot    Py_tot    Pz_tot\n')
    
    def write_iteration_data(self, iteration: int, energies, observables, static, pairs):
        """Write data for one iteration to all files"""
        
        # Calculate energy change
        energy_change = 0.0
        if hasattr(energies, 'ehfprev') and energies.ehfprev != 0.0 and iteration > 0:
            energy_change = energies.ehf - energies.ehfprev
        
        # Write conver.res - convergence data
        with open(self.files['conver'], 'a') as f:
            f.write(f'{iteration:4d} {energies.ehf:12.6f} {energy_change:12.6f} '
                   f'{energies.efluct1[0]:12.6e} {energies.efluct1[0]:12.6e} '
                   f'{energies.efluct2[0]:12.6e} {observables.rmstot:8.4f} '
                   f'{observables.beta:8.4f} {observables.gamma:8.2f} '
                   f'{static.x0dmp:8.4f} {pairs.epair[0]:12.6f} {pairs.epair[1]:12.6f}\n')
        
        # Write energies.res - detailed energy breakdown
        with open(self.files['energies'], 'a') as f:
            f.write(f'{iteration:4d} {observables.pnr[0]:8.2f} {observables.pnr[1]:8.2f} '
                   f'{energies.ehf:14.6f} {energies.ehfint:14.6f} {energies.tke:12.6f} '
                   f'{energies.ehfc:12.6f} {energies.ehfCrho0:12.6f} {energies.ehfCrho1:12.6f} '
                   f'{energies.ehfCdrho0:12.6f} {energies.ehfCdrho1:12.6f} '
                   f'{energies.ehfCtau0:12.6f} {energies.ehfCtau1:12.6f} '
                   f'{energies.ehfCdJ0:12.6f} {energies.ehfCdJ1:12.6f} '
                   f'{pairs.epair[0]:12.6f} {pairs.epair[1]:12.6f} {energies.e_zpe:12.6f}\n')
        
        # Write dipoles.res - center of mass and dipoles
        with open(self.files['dipoles'], 'a') as f:
            f.write(f'{iteration:4d} {observables.cmtot[0]:12.6f} {observables.cmtot[1]:12.6f} '
                   f'{observables.cmtot[2]:12.6f} '
                   f'{observables.dipole_isovector[0]:12.6f} '
                   f'{observables.dipole_isovector[1]:12.6f} '
                   f'{observables.dipole_isovector[2]:12.6f}\n')
        
        # Write spin.res - angular momentum
        with open(self.files['spin'], 'a') as f:
            f.write(f'{iteration:4d} {observables.orbital_momentum[0]:10.6f} '
                   f'{observables.orbital_momentum[1]:10.6f} {observables.orbital_momentum[2]:10.6f} '
                   f'{observables.spin_momentum[0]:10.6f} {observables.spin_momentum[1]:10.6f} '
                   f'{observables.spin_momentum[2]:10.6f} {observables.total_momentum[0]:10.6f} '
                   f'{observables.total_momentum[1]:10.6f} {observables.total_momentum[2]:10.6f}\n')
        
        # Write monopoles.res - RMS radii
        rms_diff = observables.rms[0] - observables.rms[1]  # neutron - proton difference
        with open(self.files['monopoles'], 'a') as f:
            f.write(f'{iteration:4d} {observables.rms[0]:9.6f} {observables.rms[1]:9.6f} '
                   f'{observables.rmstot:9.6f} {rms_diff:9.6f} '
                   f'{observables.pnr[0]:9.3f} {observables.pnr[1]:9.3f} '
                   f'{observables.pnrtot:9.3f}\n')
        
        # Write quadrupoles.res - quadrupole moments and deformation
        with open(self.files['quadrupoles'], 'a') as f:
            f.write(f'{iteration:4d} {observables.q20[0]:9.6f} {observables.q20[1]:9.6f} '
                   f'{observables.q20tot:9.6f} {observables.q22tot:9.6f} '
                   f'{observables.x2mtot[0]:9.6f} {observables.x2mtot[1]:9.6f} '
                   f'{observables.x2mtot[2]:9.6f} {observables.beta:9.6f} '
                   f'{observables.gamma:9.2f}\n')
        
        # Write momenta.res - momentum components
        pcm_tot = observables.pcm[:, 0] + observables.pcm[:, 1]
        with open(self.files['momenta'], 'a') as f:
            f.write(f'{iteration:4d} {observables.pcm[0,0]:10.6f} {observables.pcm[1,0]:10.6f} '
                   f'{observables.pcm[2,0]:10.6f} {observables.pcm[0,1]:10.6f} '
                   f'{observables.pcm[1,1]:10.6f} {observables.pcm[2,1]:10.6f} '
                   f'{pcm_tot[0]:10.6f} {pcm_tot[1]:10.6f} {pcm_tot[2]:10.6f}\n')


def calculate_complete_observables(densities, levels, grids, params, current_densities=None):
    """Calculate all nuclear observables to match FORTRAN exactly"""
    
    print("Calculating complete nuclear structure observables...")
    
    # Step 1: Basic quantities (particle numbers, center of mass)
    pnr = jnp.zeros(2)
    cm = jnp.zeros((3, 2))
    pcm = jnp.zeros((3, 2))  # Momentum components
    
    for iq in range(2):
        # Particle number
        pnr = pnr.at[iq].set(jnp.sum(densities.rho[iq,...]) * grids.wxyz)
        
        # Center of mass
        for idim in range(3):
            if idim == 0:
                coord = grids.x[:, None, None]
            elif idim == 1:
                coord = grids.y[None, :, None]
            else:
                coord = grids.z[None, None, :]
            
            cm_comp = jnp.sum(coord * densities.rho[iq,...]) * grids.wxyz / pnr[iq]
            cm = cm.at[idim, iq].set(cm_comp)
        
        # Momentum components (for dynamic calculations)
        if current_densities is not None:
            for idim in range(3):
                pcm_comp = jnp.sum(current_densities[iq, idim, ...]) * grids.wxyz
                pcm = pcm.at[idim, iq].set(pcm_comp)
    
    pnrtot = pnr[0] + pnr[1]
    cmtot = (pnr[0] * cm[:, 0] + pnr[1] * cm[:, 1]) / pnrtot
    
    # Step 2: RMS radii and second moments
    rms = jnp.zeros(2)
    x2m = jnp.zeros((3, 2))
    qmat = jnp.zeros((3, 3, 2))
    
    for iq in range(2):
        # Coordinate grids relative to center of mass
        xx = grids.x[:, None, None] - cm[0, iq]
        yy = grids.y[None, :, None] - cm[1, iq]
        zz = grids.z[None, None, :] - cm[2, iq]
        
        r2 = xx**2 + yy**2 + zz**2
        
        # RMS radius
        rms = rms.at[iq].set(jnp.sqrt(jnp.sum(r2 * densities.rho[iq,...]) * grids.wxyz / pnr[iq]))
        
        # Second moments
        x2m = x2m.at[0, iq].set(jnp.sum(xx**2 * densities.rho[iq,...]) * grids.wxyz / pnr[iq])
        x2m = x2m.at[1, iq].set(jnp.sum(yy**2 * densities.rho[iq,...]) * grids.wxyz / pnr[iq])
        x2m = x2m.at[2, iq].set(jnp.sum(zz**2 * densities.rho[iq,...]) * grids.wxyz / pnr[iq])
        
        # Quadrupole tensor
        coords = [xx, yy, zz]
        for i in range(3):
            for j in range(3):
                if i == j:
                    # Diagonal: 3x_i^2 - r^2
                    qij = jnp.sum((3 * coords[i]**2 - r2) * densities.rho[iq,...]) * grids.wxyz
                else:
                    # Off-diagonal: 3x_i x_j
                    qij = jnp.sum(3 * coords[i] * coords[j] * densities.rho[iq,...]) * grids.wxyz
                qmat = qmat.at[i, j, iq].set(qij)
    
    # Total quantities
    rmstot = jnp.sqrt((pnr[0] * rms[0]**2 + pnr[1] * rms[1]**2) / pnrtot)
    x2mtot = (pnr[0] * x2m[:, 0] + pnr[1] * x2m[:, 1]) / pnrtot
    qmtot = qmat[:, :, 0] + qmat[:, :, 1]
    
    # Step 3: Quadrupole moments and deformation
    q20 = jnp.zeros(2)
    q22 = jnp.zeros(2)
    
    for iq in range(2):
        eigvals, _ = jnp.linalg.eigh(qmat[:, :, iq])
        q20 = q20.at[iq].set(jnp.sqrt(5.0 / (16.0 * jnp.pi)) * eigvals[2])
        q22 = q22.at[iq].set(jnp.sqrt(5.0 / (96.0 * jnp.pi)) * (eigvals[1] - eigvals[0]))
    
    eigvals_tot, _ = jnp.linalg.eigh(qmtot)
    q20tot = jnp.sqrt(5.0 / (16.0 * jnp.pi)) * eigvals_tot[2]
    q22tot = jnp.sqrt(5.0 / (96.0 * jnp.pi)) * (eigvals_tot[1] - eigvals_tot[0])
    
    # Isovector quadrupole
    q20T1 = -q20[0]/levels.nneut + q20[1]/levels.nprot
    
    # Deformation parameters
    r0 = params.r0 * jnp.sqrt(0.6)
    radius = rmstot
    beta20tot = q20tot * (4.0 * jnp.pi / (5.0 * radius**2 * pnrtot))
    beta22tot = q22tot * (4.0 * jnp.pi / (5.0 * radius**2 * pnrtot))
    beta = jnp.sqrt(beta20tot**2 + 2.0 * beta22tot**2)
    
    gamma_rad = jnp.abs(jnp.arctan2(jnp.sqrt(2.0) * beta22tot, beta20tot))
    gamma = gamma_rad * 180.0 / jnp.pi
    gamma = jnp.where(gamma > 120.0, gamma - 120.0, gamma)
    gamma = jnp.where((gamma <= 120.0) & (gamma > 60.0), 120.0 - gamma, gamma)
    
    # Step 4: Angular momentum (requires single-particle properties)
    orbital_momentum, spin_momentum = calculate_angular_momentum(levels, grids)
    total_momentum = orbital_momentum + spin_momentum
    
    # Step 5: Dipole moments
    dipole_isovector = calculate_isovector_dipoles(densities, grids, cm, pnr)
    
    return CompleteNuclearObservables(
        pnr=pnr, pnrtot=pnrtot,
        cm=cm, cmtot=cmtot, dipole_isovector=dipole_isovector,
        rms=rms, rmstot=rmstot, x2m=x2m, x2mtot=x2mtot,
        q20=q20, q22=q22, q20tot=q20tot, q22tot=q22tot, q20T1=q20T1,
        beta20tot=beta20tot, beta22tot=beta22tot, beta=beta, gamma=gamma,
        orbital_momentum=orbital_momentum, spin_momentum=spin_momentum, total_momentum=total_momentum,
        pcm=pcm
    )


def calculate_angular_momentum(levels, grids):
    """Safe version that handles both shapes"""
    import jax.numpy as jnp
    
    orbital = jnp.zeros(3)
    spin = jnp.zeros(3)
    
    if hasattr(levels, 'sp_orbital') and hasattr(levels, 'sp_spin'):
        # Auto-detect shape and handle accordingly
        if levels.sp_orbital.shape[0] == 3:
            # Shape (3, nstmax) - FORTRAN style
            for i in range(levels.nstmax):
                if levels.wocc[i] > 1e-10:
                    weight = levels.wocc[i] * levels.wstates[i]
                    orbital += weight * levels.sp_orbital[:, i]
                    spin += weight * levels.sp_spin[:, i]
        else:
            # Shape (nstmax, 3) - Current JAX style  
            for i in range(levels.nstmax):
                if levels.wocc[i] > 1e-10:
                    weight = levels.wocc[i] * levels.wstates[i]
                    orbital += weight * levels.sp_orbital[i, :]
                    spin += weight * levels.sp_spin[i, :]
    
    return orbital, spin


def calculate_isovector_dipoles(densities, grids, cm, pnr):
    """Calculate isovector dipole moments"""
    
    dipoles = jnp.zeros(3)
    
    for idim in range(3):
        if idim == 0:
            coord = grids.x[:, None, None]
        elif idim == 1:
            coord = grids.y[None, :, None]
        else:
            coord = grids.z[None, None, :]
        
        # Isovector dipole: ∫ r (ρ_n/N - ρ_p/Z) d³r
        dipole_density = densities.rho[0,...]/pnr[0] - densities.rho[1,...]/pnr[1]
        dipoles = dipoles.at[idim].set(jnp.sum(coord * dipole_density) * grids.wxyz)
    
    return dipoles


def complete_sinfo_with_fortran_output(coulomb, densities, energies, forces, grids, levels, 
                                     meanfield, moment, params, static, pairs, output_writer):
    """Enhanced sinfo that writes all FORTRAN-compatible output files"""
    
    # Calculate complete observables
    observables = calculate_complete_observables(densities, levels, grids, params)
    
    # Update moment object with calculated values
    moment.cmtot = observables.cmtot
    
    # Calculate integrated energy (if not already done)
    if not hasattr(energies, 'ehfint'):
        energies.ehfint = energies.ehf  # Placeholder
    
    # Write all output files
    output_writer.write_iteration_data(params.iteration, energies, observables, static, pairs)
    
    # Print summary (matching FORTRAN output)
    if params.iteration % params.mprint == 0:
        print_fortran_style_summary(params.iteration, energies, observables, static)
    
    return energies, observables


def print_fortran_style_summary(iteration, energies, observables, static):
    """Print iteration summary in FORTRAN style"""
    
    print(f"\nIteration {iteration:3d}: Energy={energies.ehf:12.6f} MeV,  "
          f"sp_fluct={energies.efluct1[0]:.2e} MeV")
    
    print(f"Beta20: {observables.beta20tot:8.4f} Beta22: {observables.beta22tot:8.4f} "
          f"Beta: {observables.beta:8.4f} Gamma: {observables.gamma:8.2f}")
    
    print(f"RMS radii: n={observables.rms[0]:.4f} fm, p={observables.rms[1]:.4f} fm, "
          f"total={observables.rmstot:.4f} fm")