# inout.py

import jax
import jax.numpy as jnp
from levels import cdervx01, cdervy01, cdervz01 
import math

@jax.jit
def sp_properties_for_single_state(pst, dx, dy, dz, xx, yy, zz,i):
    """
    Calculates properties for a single wavefunction slice of shape (spin, nx, ny, nz).
    """
    # 1. Get FIRST derivatives (ignore the second return value from cderv*01)
    psx, _ = cdervx01(dx, pst)
    psy, _ = cdervy01(dy, pst)
    psz, _ = cdervz01(dz, pst)

    # 2. calculate the Laplacian (∇²ψ)
    
    # Get k-space factors
    nx, ny, nz = pst.shape[1], pst.shape[2], pst.shape[3]
    kfacx = (jnp.pi * 2) / (dx * nx)
    kfacy = (jnp.pi * 2) / (dy * ny)
    kfacz = (jnp.pi * 2) / (dz * nz)

    # Construct k² arrays for each dimension
    k2facx = jnp.concatenate((-(jnp.arange(0, nx//2) * kfacx)**2, -(jnp.arange(nx//2, 0, -1) * kfacx)**2))
    k2facy = jnp.concatenate((-(jnp.arange(0, ny//2) * kfacy)**2, -(jnp.arange(ny//2, 0, -1) * kfacy)**2))
    k2facz = jnp.concatenate((-(jnp.arange(0, nz//2) * kfacz)**2, -(jnp.arange(nz//2, 0, -1) * kfacz)**2))

    # Reshape for broadcasting
    k2_total = (k2facx[jnp.newaxis, :, jnp.newaxis, jnp.newaxis] +
                k2facy[jnp.newaxis, jnp.newaxis, :, jnp.newaxis] +
                k2facz[jnp.newaxis, jnp.newaxis, jnp.newaxis, :])
    
    # Apply the Laplacian operator in Fourier space
    pst_k = jnp.fft.fftn(pst, axes=(1, 2, 3))
    lap_psi_k = pst_k * k2_total
    psw = jnp.fft.ifftn(lap_psi_k, axes=(1, 2, 3)) # This is the correct Laplacian

    # Orbital angular momentum calculation with correctly shaped meshes
    cc = jnp.array([
        jnp.sum(jnp.real(pst) * (yy * jnp.imag(psz) - zz * jnp.imag(psy)) + jnp.imag(pst) * (zz * jnp.real(psy) - yy * jnp.real(psz))),
        jnp.sum(jnp.real(pst) * (zz * jnp.imag(psx) - xx * jnp.imag(psz)) + jnp.imag(pst) * (xx * jnp.real(psz) - zz * jnp.real(psx))),
        jnp.sum(jnp.real(pst) * (xx * jnp.imag(psy) - yy * jnp.imag(psx)) + jnp.imag(pst) * (yy * jnp.real(psx) - xx * jnp.real(psy)))
    ])

    kin_raw = -jnp.sum(jnp.real(jnp.conjugate(pst) * psw))

    

    flipped_pst = pst[:, ::-1, ::-1, ::-1]
    xpar = jnp.sum(jnp.real(pst) * jnp.real(flipped_pst) + jnp.imag(pst) * jnp.imag(flipped_pst))

    ss = jnp.array([
        jnp.sum(jnp.real(jnp.conjugate(pst[0,...]) * pst[1,...]) + jnp.real(jnp.conjugate(pst[1,...]) * pst[0,...])),
        jnp.sum(jnp.real(jnp.conjugate(pst[0,...]) * pst[1,...] * (0.0 - 1.0j)) + jnp.real(jnp.conjugate(pst[1,...]) * pst[0,...] * (0.0 + 1.0j))),
        jnp.sum(jnp.real(jnp.conjugate(pst[0,...]) * pst[0,...]) - jnp.real(jnp.conjugate(pst[1,...]) * pst[1,...]))
    ])
    

    return ss, cc, kin_raw, xpar

@jax.jit
def sp_properties(forces, grids, levels, moment):
    xx = (grids.x - moment.cmtot[0])[jnp.newaxis, :, jnp.newaxis, jnp.newaxis]
    yy = (grids.y - moment.cmtot[1])[jnp.newaxis, jnp.newaxis, :, jnp.newaxis]
    zz = (grids.z - moment.cmtot[2])[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]
    
    def calculate_laplacian_fortran_style(pst, dx, dy, dz):
        """Calculate Laplacian using method that matches FORTRAN"""
        nx, ny, nz = pst.shape[1], pst.shape[2], pst.shape[3]
        
        kfacx = (jnp.pi * 2) / (dx * nx)
        kfacy = (jnp.pi * 2) / (dy * ny)  
        kfacz = (jnp.pi * 2) / (dz * nz)
        
        k2facx = jnp.concatenate((-(jnp.arange(0, nx//2) * kfacx)**2, 
                                 -(jnp.arange(nx//2, 0, -1) * kfacx)**2))
        k2facy = jnp.concatenate((-(jnp.arange(0, ny//2) * kfacy)**2, 
                                 -(jnp.arange(ny//2, 0, -1) * kfacy)**2))
        k2facz = jnp.concatenate((-(jnp.arange(0, nz//2) * kfacz)**2, 
                                 -(jnp.arange(nz//2, 0, -1) * kfacz)**2))
        
        k2_total = (k2facx[jnp.newaxis, :, jnp.newaxis, jnp.newaxis] +
                   k2facy[jnp.newaxis, jnp.newaxis, :, jnp.newaxis] +
                   k2facz[jnp.newaxis, jnp.newaxis, jnp.newaxis, :])
        
        pst_k = jnp.fft.fftn(pst, axes=(1, 2, 3))
        lap_psi_k = pst_k * k2_total
        return jnp.fft.ifftn(lap_psi_k, axes=(1, 2, 3))
    
    all_sp_orbital = []
    all_sp_spin = []
    all_sp_kinetic = []
    all_sp_parity = []
    
    for i in range(levels.nstmax):
        pst = levels.psi[i]  # Shape: (2, nx, ny, nz)
        
        # Get derivatives 
        psx, _ = cdervx01(grids.dx, pst)
        psy, _ = cdervy01(grids.dy, pst) 
        psz, _ = cdervz01(grids.dz, pst)
        
        # Calculate Laplacian
        psw = calculate_laplacian_fortran_style(pst, grids.dx, grids.dy, grids.dz)
        
        # L = r × p = r × (-i∇)
        orbital_x = jnp.sum(jnp.real(pst) * (yy * jnp.imag(psz) - zz * jnp.imag(psy)) + 
                           jnp.imag(pst) * (zz * jnp.real(psy) - yy * jnp.real(psz)))
        orbital_y = jnp.sum(jnp.real(pst) * (zz * jnp.imag(psx) - xx * jnp.imag(psz)) + 
                           jnp.imag(pst) * (xx * jnp.real(psz) - zz * jnp.real(psx)))
        orbital_z = jnp.sum(jnp.real(pst) * (xx * jnp.imag(psy) - yy * jnp.imag(psx)) + 
                           jnp.imag(pst) * (yy * jnp.real(psx) - xx * jnp.real(psy)))
        
        orbital = jnp.array([orbital_x, orbital_y, orbital_z])
        
        # S = (1/2) * σ, where σ are Pauli matrices
        spin_x = jnp.sum(jnp.real(jnp.conjugate(pst[0]) * pst[1]) + 
                        jnp.real(jnp.conjugate(pst[1]) * pst[0]))
        spin_y = jnp.sum(jnp.real(jnp.conjugate(pst[0]) * pst[1] * (-1j)) + 
                        jnp.real(jnp.conjugate(pst[1]) * pst[0] * (1j)))
        spin_z = jnp.sum(jnp.real(jnp.conjugate(pst[0]) * pst[0]) - 
                        jnp.real(jnp.conjugate(pst[1]) * pst[1]))
        
        spin = 0.5 * jnp.array([spin_x, spin_y, spin_z])
        
        # T = -ħ²/(2m) * ∫ ψ* ∇²ψ dr
        kinetic_raw = -jnp.sum(jnp.real(jnp.conjugate(pst) * psw))
        h2m_factor = jnp.where(levels.isospin[i] == 0, forces.h2m[0], forces.h2m[1])
        kinetic = grids.wxyz * h2m_factor * kinetic_raw
        
        # P = ∫ ψ*(r) ψ(-r) dr
        flipped_pst = pst[:, ::-1, ::-1, ::-1]
        parity = jnp.sum(jnp.real(pst) * jnp.real(flipped_pst) + 
                        jnp.imag(pst) * jnp.imag(flipped_pst))
        
        # Store results
        all_sp_orbital.append(grids.wxyz * orbital)
        all_sp_spin.append(grids.wxyz * spin)
        all_sp_kinetic.append(kinetic)
        all_sp_parity.append(grids.wxyz * parity)
    
    levels.sp_orbital = jnp.stack(all_sp_orbital)     # Shape: (nstmax, 3)
    levels.sp_spin = jnp.stack(all_sp_spin)           # Shape: (nstmax, 3)  
    levels.sp_kinetic = jnp.array(all_sp_kinetic)     # Shape: (nstmax,)
    levels.sp_parity = jnp.array(all_sp_parity)       # Shape: (nstmax,)
    
    return levels