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


    # Use the full psi array, treating the first dimension as a batch dimension
    pst = levels.psi  # Shape: (nstmax, 2, nx, ny, nz)
    
    # Coordinate grids will broadcast automatically to the shape of pst
    xx = (grids.x - moment.cmtot[0])[jnp.newaxis, jnp.newaxis, :, jnp.newaxis, jnp.newaxis]
    yy = (grids.y - moment.cmtot[1])[jnp.newaxis, jnp.newaxis, jnp.newaxis, :, jnp.newaxis]
    zz = (grids.z - moment.cmtot[2])[jnp.newaxis, jnp.newaxis, jnp.newaxis, jnp.newaxis, :]
    
    # 1. Get FIRST derivatives for all wavefunctions at once
    psx, _ = batched_cdervx01(grids.dx, pst)
    psy, _ = batched_cdervy01(grids.dy, pst)
    psz, _ = batched_cdervz01(grids.dz, pst)
    
    # 2. Calculate the Laplacian (∇²ψ) in a vectorized way
    nx, ny, nz = pst.shape[2], pst.shape[3], pst.shape[4]
    
    kfacx = (jnp.pi * 2) / (grids.dx * nx)
    kfacy = (jnp.pi * 2) / (grids.dy * ny)
    kfacz = (jnp.pi * 2) / (grids.dz * nz)
    
    k2facx = jnp.concatenate((-(jnp.arange(0, nx//2) * kfacx)**2, -(jnp.arange(nx//2, 0, -1) * kfacx)**2))
    k2facy = jnp.concatenate((-(jnp.arange(0, ny//2) * kfacy)**2, -(jnp.arange(ny//2, 0, -1) * kfacy)**2))
    k2facz = jnp.concatenate((-(jnp.arange(0, nz//2) * kfacz)**2, -(jnp.arange(nz//2, 0, -1) * kfacz)**2))
    
    # Reshape k² for broadcasting over the (nstmax, spin) dimensions of pst
    k2_total = (k2facx[jnp.newaxis, jnp.newaxis, :, jnp.newaxis, jnp.newaxis] +
                k2facy[jnp.newaxis, jnp.newaxis, jnp.newaxis, :, jnp.newaxis] +
                k2facz[jnp.newaxis, jnp.newaxis, jnp.newaxis, jnp.newaxis, :])
                
    # Apply FFT along spatial axes (2, 3, 4)
    pst_k = jnp.fft.fftn(pst, axes=(2, 3, 4))
    lap_psi_k = pst_k * k2_total
    psw = jnp.fft.ifftn(lap_psi_k, axes=(2, 3, 4)) # psw has shape (nstmax, 2, nx, ny, nz)

    # Define axes to sum over, keeping the first (nstmax) dimension
    spatial_spin_axes = (1, 2, 3, 4)
    spatial_axes = (1, 2, 3)

    # 3. Orbital angular momentum L = r × p
    orbital_x = jnp.sum(jnp.real(pst) * (yy * jnp.imag(psz) - zz * jnp.imag(psy)) + jnp.imag(pst) * (zz * jnp.real(psy) - yy * jnp.real(psz)), axis=spatial_spin_axes)
    orbital_y = jnp.sum(jnp.real(pst) * (zz * jnp.imag(psx) - xx * jnp.imag(psz)) + jnp.imag(pst) * (xx * jnp.real(psz) - zz * jnp.real(psx)), axis=spatial_spin_axes)
    orbital_z = jnp.sum(jnp.real(pst) * (xx * jnp.imag(psy) - yy * jnp.imag(psx)) + jnp.imag(pst) * (yy * jnp.real(psx) - xx * jnp.real(psy)), axis=spatial_spin_axes)
    
    orbital = jnp.stack([orbital_x, orbital_y, orbital_z], axis=1) # Shape: (nstmax, 3)
    
    # 4. Spin expectation value S
    pst_up = pst[:, 0]   # Shape: (nstmax, nx, ny, nz)
    pst_down = pst[:, 1] # Shape: (nstmax, nx, ny, nz)

    spin_x = jnp.sum(jnp.real(jnp.conjugate(pst_up) * pst_down) + jnp.real(jnp.conjugate(pst_down) * pst_up), axis=spatial_axes)
    spin_y = jnp.sum(jnp.real(jnp.conjugate(pst_up) * pst_down * (-1j)) + jnp.real(jnp.conjugate(pst_down) * pst_up * (1j)), axis=spatial_axes)
    spin_z = jnp.sum(jnp.real(jnp.conjugate(pst_up) * pst_up) - jnp.real(jnp.conjugate(pst_down) * pst_down), axis=spatial_axes)
    
    spin = 0.5 * jnp.stack([spin_x, spin_y, spin_z], axis=1) # Shape: (nstmax, 3)

    # 5. Kinetic energy T
    kinetic_raw = -jnp.sum(jnp.real(jnp.conjugate(pst) * psw), axis=spatial_spin_axes) # Shape: (nstmax,)
    # Vectorize the conditional logic for the h2m factor
    h2m_factors = jnp.where(levels.isospin == 0, forces.h2m[0], forces.h2m[1]) # Shape: (nstmax,)
    kinetic = grids.wxyz * h2m_factors * kinetic_raw

    # 6. Parity P
    flipped_pst = pst[:, :, ::-1, ::-1, ::-1]
    parity = jnp.sum(jnp.real(pst) * jnp.real(flipped_pst) + jnp.imag(pst) * jnp.imag(flipped_pst), axis=spatial_spin_axes)

    # Store results directly into the levels object
    levels.sp_orbital = grids.wxyz * orbital
    levels.sp_spin = grids.wxyz * spin
    levels.sp_kinetic = kinetic
    levels.sp_parity = grids.wxyz * parity
    
    return levels

def batched_cdervx01(dx, pst_batch):
    vmapped_derivative_func = jax.vmap(cdervx01, in_axes=(None, 0))
    return vmapped_derivative_func(dx, pst_batch)

def batched_cdervy01(dy, pst_batch):
    vmapped_derivative_func = jax.vmap(cdervy01, in_axes=(None, 0))
    return vmapped_derivative_func(dy, pst_batch)

def batched_cdervz01(dz, pst_batch):
    vmapped_derivative_func = jax.vmap(cdervz01, in_axes=(None, 0))
    return vmapped_derivative_func(dz, pst_batch)