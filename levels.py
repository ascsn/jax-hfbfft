import jax
from jax import numpy as jnp
from dataclasses import dataclass, field
from trivial import rpsnorm

@jax.tree_util.register_dataclass
@dataclass
class Levels:
    nstmax: int = field(metadata=dict(static=True))
    nneut: int = field(metadata=dict(static=True))
    nprot: int = field(metadata=dict(static=True))
    npmin: jax.Array
    npsi: jax.Array
    charge_number: float
    mass_number: float
    psi: jax.Array
    hampsi: jax.Array
    lagrange: jax.Array
    hmfpsi: jax.Array
    delpsi: jax.Array
    sp_orbital: jax.Array
    sp_spin: jax.Array
    isospin: jax.Array
    sp_energy: jax.Array
    sp_efluct1: jax.Array
    sp_kinetic: jax.Array
    sp_norm: jax.Array
    sp_efluct2: jax.Array
    sp_parity: jax.Array
    wocc: jax.Array
    wguv: jax.Array
    pairwg: jax.Array
    wstates: jax.Array
    deltaf: jax.Array


def init_levels(grids, **kwargs):
    # Get actual particle numbers from config
    nneut = kwargs.get('nneut', 82)  # From _config.yml: nneut: 82
    nprot = kwargs.get('nprot', 50)  # From _config.yml: nprot: 50
    
    
    # Calculate basis sizes using FORTRAN formula for pairing calculations
    if 'npsi' in kwargs and kwargs['npsi'] is not None:
        npsi_array = jnp.array(kwargs['npsi'])
    else:
        neutron_basis = max(126, nneut + int(1.65 * (nneut**0.666667)))
        proton_basis = max(82, nprot + int(1.65 * (nprot**0.666667)))
        npsi_array = jnp.array([neutron_basis, proton_basis])
    
    nstmax = int(npsi_array[0] + npsi_array[1])
    
    # Starting indices for each isospin in the global state array
    npmin = jnp.array([0, int(npsi_array[0])])
    
    # Physical nucleus properties
    charge_number = nprot
    mass_number = nneut + nprot
    
    # Array shapes based on TOTAL basis size (not just occupied states!)
    shape2d = (nstmax, 3)
    shape5d = (nstmax, 2, grids.nx, grids.ny, grids.nz)
    
    # Initialize all arrays for the FULL basis
    psi = jnp.zeros(shape5d, dtype=jnp.complex128)
    hampsi = jnp.zeros(shape5d, dtype=jnp.complex128)
    lagrange = jnp.zeros(shape5d, dtype=jnp.complex128)
    hmfpsi = jnp.zeros(shape5d, dtype=jnp.complex128)
    delpsi = jnp.zeros(shape5d, dtype=jnp.complex128)
    
    # Single-particle properties for ALL basis states
    sp_orbital = jnp.zeros(shape2d, dtype=jnp.float64)
    sp_spin = jnp.zeros(shape2d, dtype=jnp.float64)
    sp_energy = jnp.zeros(nstmax, dtype=jnp.float64)
    sp_efluct1 = jnp.zeros(nstmax, dtype=jnp.float64)
    sp_kinetic = jnp.zeros(nstmax, dtype=jnp.float64)
    sp_norm = jnp.zeros(nstmax, dtype=jnp.float64)
    sp_efluct2 = jnp.zeros(nstmax, dtype=jnp.float64)
    sp_parity = jnp.zeros(nstmax, dtype=jnp.float64)
    deltaf = jnp.zeros(nstmax, dtype=jnp.float64)
    
    # States 0 to npsi[0]-1 are neutrons (isospin=0)
    # States npsi[0] to nstmax-1 are protons (isospin=1)
    isospin = jnp.zeros(nstmax, dtype=jnp.int32)
    proton_start = int(npsi_array[0])
    isospin = isospin.at[proton_start:].set(1)  # All proton states have isospin=1
    wocc = jnp.zeros(nstmax, dtype=jnp.float64)
    
    # Occupy first nneut neutron states (indices 0 to nneut-1)
    wocc = wocc.at[:nneut].set(1.0)
    
    # Occupy first nprot proton states (indices npmin[1] to npmin[1]+nprot-1)
    proton_end = proton_start + nprot
    wocc = wocc.at[proton_start:proton_end].set(1.0)
       
    # Initialize other arrays
    wguv = jnp.zeros(nstmax, dtype=jnp.float64)
    pairwg = jnp.ones(nstmax, dtype=jnp.float64)
    wstates = jnp.ones(nstmax, dtype=jnp.float64)
    
    

    return Levels(
        nstmax=nstmax,
        nneut=nneut,
        nprot=nprot,
        npmin=npmin,
        npsi=npsi_array,
        charge_number=charge_number,
        mass_number=mass_number,
        psi=psi,
        hampsi=hampsi,
        lagrange=lagrange,
        hmfpsi=hmfpsi,
        delpsi=delpsi,
        sp_orbital=sp_orbital,
        sp_spin=sp_spin,
        isospin=isospin,
        sp_energy=sp_energy,
        sp_efluct1=sp_efluct1,
        sp_kinetic=sp_kinetic,
        sp_norm=sp_norm,
        sp_efluct2=sp_efluct2,
        sp_parity=sp_parity,
        wocc=wocc,
        wguv=wguv,
        pairwg=pairwg,
        wstates=wstates,
        deltaf=deltaf
    )
@jax.jit
def cdervx00(d, psin):
    iq, n, _, _ = psin.shape
    kfac = (jnp.pi + jnp.pi) / (d * n)
    half_n = n // 2

    d1psout = jnp.fft.fft(psin, axis=1, norm="backward")

    inds = jnp.arange(0, half_n)
    inds4d = inds[jnp.newaxis,:,jnp.newaxis,jnp.newaxis]

    d1psout = d1psout.at[:,0,:,:].set(0.0)
    d1psout = d1psout.at[:,inds,:,:].multiply((1j * inds4d) * kfac / n)
    d1psout = d1psout.at[:,n-inds,:,:].multiply(-((1j * inds4d) * kfac) / n)

    d1psout = d1psout.at[:,half_n,:,:].set(0.0)
    d1psout = jnp.fft.ifft(d1psout, axis=1, norm="forward")

    return d1psout

@jax.jit
def cdervy00(d, psin):
    iq, _, n, _ = psin.shape
    kfac = (jnp.pi + jnp.pi) / (d * n)
    half_n = n // 2

    d1psout = jnp.fft.fft(psin, axis=2, norm="backward")

    inds = jnp.arange(0, half_n)
    inds4d = inds[jnp.newaxis,jnp.newaxis,:,jnp.newaxis]

    d1psout = d1psout.at[:,:,0,:].set(0.0)
    d1psout = d1psout.at[:,:,inds,:].multiply((1j * inds4d) * kfac / n)
    d1psout = d1psout.at[:,:,n-inds,:].multiply(-((1j * inds4d) * kfac) / n)

    d1psout = d1psout.at[:,:,half_n,:].set(0.0)
    d1psout = jnp.fft.ifft(d1psout, axis=2, norm="forward")

    return d1psout

@jax.jit
def cdervz00(d, psin):
    iq, _, _, n = psin.shape
    kfac = (jnp.pi + jnp.pi) / (d * n)
    half_n = n // 2

    d1psout = jnp.fft.fft(psin, axis=3, norm="backward")

    inds = jnp.arange(0, half_n)
    inds4d = inds[jnp.newaxis,jnp.newaxis,jnp.newaxis,:]

    d1psout = d1psout.at[:,:,:,0].set(0.0)
    d1psout = d1psout.at[:,:,:,inds].multiply((1j * inds4d) * kfac / n)
    d1psout = d1psout.at[:,:,:,n-inds].multiply(-((1j * inds4d) * kfac) / n)

    d1psout = d1psout.at[:,:,:,half_n].set(0.0)
    d1psout = jnp.fft.ifft(d1psout, axis=3, norm="forward")

    return d1psout
@jax.jit
def cdervx01(d, psin):
    _, n, _, _ = psin.shape
    kfac = (jnp.pi + jnp.pi) / (d * n)
    half_n = n // 2

    d1psout = jnp.fft.fft(psin, axis=1, norm="backward")

    idx = jnp.arange(0, half_n)
    idx_4d = idx[jnp.newaxis,:,jnp.newaxis,jnp.newaxis]

    d2psout = jnp.copy(d1psout)
    d2psout = d2psout.at[:,idx,:,:].multiply(-(idx_4d * kfac) ** 2 / n)
    d2psout = d2psout.at[:,n-1-idx,:,:].multiply(-((idx_4d + 1) * kfac) ** 2 / n)

    d1psout = d1psout.at[:,0,:,:].set(0.0)
    d1psout = d1psout.at[:,idx,:,:].multiply((1j * idx_4d) * kfac / n)
    d1psout = d1psout.at[:,n-idx,:,:].multiply(-((1j * idx_4d) * kfac) / n)

    d1psout = d1psout.at[:,half_n,:,:].set(0.0)

    d1psout = d1psout.at[...].set(
        jnp.fft.ifft(d1psout, axis=1, norm="forward")
    )

    d2psout = d2psout.at[...].set(
        jnp.fft.ifft(d2psout, axis=1, norm="forward")
    )

    return d1psout, d2psout

@jax.jit
def cdervy01(d, psin):
    _, _, n, _ = psin.shape
    kfac = (jnp.pi + jnp.pi) / (d * n)
    half_n = n // 2

    d1psout = jnp.fft.fft(psin, axis=2, norm="backward")

    idx = jnp.arange(0, half_n)
    idx_4d = idx[jnp.newaxis, jnp.newaxis, :, jnp.newaxis]

    d2psout = jnp.copy(d1psout)
    d2psout = d2psout.at[:,:,idx,:].multiply(-(idx_4d * kfac) ** 2 / n)
    d2psout = d2psout.at[:,:,n-1-idx,:].multiply(-((idx_4d + 1) * kfac) ** 2 / n)

    d1psout = d1psout.at[:,:,0,:].set(0.0)
    d1psout = d1psout.at[:,:,idx,:].multiply((1j * idx_4d) * kfac / n)
    d1psout = d1psout.at[:,:,n-idx,:].multiply(-((1j * idx_4d) * kfac) / n)

    d1psout = d1psout.at[:,:,half_n,:].set(0.0)

    d1psout = d1psout.at[...].set(
        jnp.fft.ifft(d1psout, axis=2, norm="forward")
    )

    d2psout = d2psout.at[...].set(
        jnp.fft.ifft(d2psout, axis=2, norm="forward")
    )

    return d1psout, d2psout

@jax.jit
def cdervz01(d, psin):
    _, _, _, n = psin.shape
    kfac = (jnp.pi + jnp.pi) / (d * n)
    half_n = n // 2

    d1psout = jnp.fft.fft(psin, axis=3, norm="backward")

    idx = jnp.arange(0, half_n)
    idx_4d = idx[jnp.newaxis,jnp.newaxis,jnp.newaxis,:]

    d2psout = jnp.copy(d1psout)
    d2psout = d2psout.at[:,:,:,idx].multiply(-(idx_4d * kfac) ** 2 / n)
    d2psout = d2psout.at[:,:,:,n-1-idx].multiply(-((idx_4d + 1) * kfac) ** 2 / n)

    d1psout = d1psout.at[:,:,:,0].set(0.0)
    d1psout = d1psout.at[:,:,:,idx].multiply((1j * idx_4d) * kfac / n)
    d1psout = d1psout.at[:,:,:,n-idx].multiply(-((1j * idx_4d) * kfac) / n)

    d1psout = d1psout.at[:,:,:,half_n].set(0.0)

    d1psout = d1psout.at[...].set(
        jnp.fft.ifft(d1psout, axis=3, norm="forward")
    )

    d2psout = d2psout.at[...].set(
        jnp.fft.ifft(d2psout, axis=3, norm="forward")
    )

    return d1psout, d2psout

@jax.jit
def cdervx02(dx: float, psin: jnp.ndarray, pos_func: jnp.ndarray):
    iq, nx, ny, nz = psin.shape
    kfac = (jnp.pi + jnp.pi) / (dx * nx)

    d1psout = jnp.fft.fft(psin, axis=1, norm="backward")

    idx = jnp.arange(0, nx//2)
    idx_4d = idx[jnp.newaxis,:,jnp.newaxis,jnp.newaxis]

    d2psout = jnp.copy(d1psout)
    d2psout = d2psout.at[:,idx,:,:].multiply(-(idx_4d*kfac)**2/nx)
    d2psout = d2psout.at[:,nx-1-idx,:,:].multiply(-((idx_4d+1)*kfac)**2/nx)

    idx = jnp.arange(0, nx//2)
    idx_4d = idx[jnp.newaxis,:,jnp.newaxis,jnp.newaxis]

    d1psout_temp = jnp.copy(d1psout[:,nx//2,:,:])
    d1psout = d1psout.at[:,0,:,:].set(0.0)
    d1psout = d1psout.at[:,idx,:,:].multiply((1j * idx_4d) * kfac / nx)
    d1psout = d1psout.at[:,ny-idx,:,:].multiply(-((1j * idx_4d) * kfac) / nx)

    d1psout = d1psout.at[:,nx//2,:,:].set(0.0)

    d1psout = jnp.fft.ifft(d1psout, axis=1, norm="forward")

    d2psout_temp = jnp.multiply(d1psout, pos_func)
    d2psout = jnp.fft.fft(d2psout_temp, axis=1, norm="backward")
    d2psout = d2psout.at[:,idx,:,:].multiply(((1j * idx_4d)*kfac/nx))
    d2psout = d2psout.at[:,ny-idx,:,:].multiply((-((1j * idx_4d)*kfac)/nx))
    d2psout = d2psout.at[:,0,:,:].set(0.0)
    pos_func_ave = jnp.mean(pos_func, axis=0)
    d2psout = d2psout.at[:,nx//2,:,:].set(-(jnp.pi / dx) ** 2 * pos_func_ave * d1psout_temp / nx)

    d2psout = jnp.fft.ifft(d2psout, axis=1, norm="forward")

    return d1psout, d2psout

@jax.jit
def cdervy02(dy: float, psin: jnp.ndarray, pos_func: jnp.ndarray):
    iq, nx, ny, nz = psin.shape
    kfac = (jnp.pi + jnp.pi) / (dy * ny)

    d1psout = jnp.fft.fft(psin, axis=2, norm="backward")

    idy = jnp.arange(0, ny//2)
    idy_4d = idy[jnp.newaxis,jnp.newaxis,:,jnp.newaxis]

    d2psout = jnp.copy(d1psout)
    d2psout = d2psout.at[:,:,idy,:].multiply(-(idy_4d*kfac)**2/ny)
    d2psout = d2psout.at[:,:,ny-1-idy,:].multiply(-((idy_4d+1)*kfac)**2/ny)

    idy = jnp.arange(1, ny//2)
    idy_4d = idy[jnp.newaxis,jnp.newaxis,:,jnp.newaxis]

    d1psout_temp = jnp.copy(d1psout[:,:,ny//2,:])
    d1psout = d1psout.at[:,:,0,:].set(0.0)
    d1psout = d1psout.at[:,:,idy,:].multiply((1j * idy_4d) * kfac / ny)
    d1psout = d1psout.at[:,:,ny-idy,:].multiply(-((1j * idy_4d) * kfac) / ny)

    d1psout = d1psout.at[:,:,ny//2,:].set(0.0)

    d1psout = jnp.fft.ifft(d1psout, axis=2, norm="forward")

    d2psout_temp = jnp.multiply(d1psout, pos_func)
    d2psout = jnp.fft.fft(d2psout_temp, axis=2, norm="backward")
    d2psout = d2psout.at[:,:,idy,:].multiply(((1j * idy_4d)*kfac/ny))
    d2psout = d2psout.at[:,:,ny-idy,:].multiply((-((1j * idy_4d)*kfac)/ny))
    d2psout = d2psout.at[:,:,0,:].set(0.0)
    pos_func_ave = jnp.mean(pos_func, axis=1)
    d2psout = d2psout.at[:,:,ny//2,:].set(-(jnp.pi / dy) ** 2 * pos_func_ave * d1psout_temp / ny)

    d2psout = jnp.fft.ifft(d2psout, axis=2, norm="forward")

    return d1psout, d2psout

@jax.jit
def cdervz02(dz: float, psin: jnp.ndarray, pos_func: jnp.ndarray):
    iq, nx, ny, nz = psin.shape
    kfac = (jnp.pi + jnp.pi) / (dz * nz)

    d1psout = jnp.fft.fft(psin, axis=3, norm="backward")

    idz = jnp.arange(0, nz//2)
    idz_4d = idz[jnp.newaxis,jnp.newaxis,jnp.newaxis,:]

    d2psout = jnp.copy(d1psout)
    d2psout = d2psout.at[:,:,:,idz].multiply(-(idz_4d*kfac)**2/nz)
    d2psout = d2psout.at[:,:,:,nz-1-idz].multiply(-((idz_4d+1)*kfac)**2/nz)

    idz = jnp.arange(1, nz//2)
    idz_4d = idz[jnp.newaxis,jnp.newaxis,jnp.newaxis,:]

    d1psout_temp = jnp.copy(d1psout[:,:,:,nz//2])
    d1psout = d1psout.at[:,:,:,0].set(0.0)
    d1psout = d1psout.at[:,:,:,idz].multiply((1j * idz_4d) * kfac / nz)
    d1psout = d1psout.at[:,:,:,nz-idz].multiply(-((1j * idz_4d) * kfac) / nz)

    d1psout = d1psout.at[:,:,:,nz//2].set(0.0)

    d1psout = jnp.fft.ifft(d1psout, axis=3, norm="forward")

    d2psout_temp = jnp.multiply(d1psout, pos_func)
    d2psout = jnp.fft.fft(d2psout_temp, axis=3, norm="backward")
    d2psout = d2psout.at[:,:,:,idz].multiply(((1j * idz_4d)*kfac/nz))
    d2psout = d2psout.at[:,:,:,nz-idz].multiply((-((1j * idz_4d)*kfac)/nz))
    d2psout = d2psout.at[:,:,:,0].set(0.0)
    pos_func_ave = jnp.mean(pos_func, axis=2)
    d2psout = d2psout.at[:,:,:,nz//2].set(-(jnp.pi / dz) ** 2 * pos_func_ave * d1psout_temp / nz)

    d2psout = jnp.fft.ifft(d2psout, axis=3, norm="forward")

    return d1psout, d2psout

@jax.jit
def laplace(nst, iq, grids, forces, levels, wg, wguv, psin, v_pairmax, e0inv):
    weightmin = jnp.array([0.1])
    weight = jnp.maximum(wg, weightmin)
    weightuv = jnp.maximum(wguv, weightmin)

    kfacx = (jnp.pi + jnp.pi) / (grids.dx * grids.nx)
    kfacy = (jnp.pi + jnp.pi) / (grids.dy * grids.ny)
    kfacz = (jnp.pi + jnp.pi) / (grids.dz * grids.nz)

    k2facx = jnp.concatenate((
        -(jnp.arange(0, (grids.nx//2)) * kfacx) ** 2,
        -(jnp.arange((grids.nx//2), 0, -1) * kfacx) ** 2
    ))[:,jnp.newaxis,jnp.newaxis]

    k2facy = jnp.concatenate((
        -(jnp.arange(0, (grids.ny//2)) * kfacy) ** 2,
        -(jnp.arange((grids.ny//2), 0, -1) * kfacy) ** 2
    ))[jnp.newaxis,:,jnp.newaxis]

    k2facz = jnp.concatenate((
        -(jnp.arange(0, (grids.nz//2)) * kfacz) ** 2,
        -(jnp.arange((grids.nz//2), 0, -1) * kfacz) ** 2
    ))[jnp.newaxis,jnp.newaxis,:]

    psout = jnp.copy(psin)

    psout = psout.at[...].set(
        jnp.fft.fftn(psout, axes=(-3, -2, -1))
    )

    psout = psout.at[...].divide(
        ((weight * (e0inv - forces.h2ma * (k2facx + k2facy + k2facz))) + 0.5 * weightuv * v_pairmax) *
        (grids.nx * grids.ny * grids.nz)
    )

    psout = psout.at[...].set(
        jnp.fft.ifftn(psout, axes=(-3, -2, -1), norm='forward')
    )

    return psout