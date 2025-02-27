import jax
from jax import numpy as jnp
from dataclasses import dataclass, field

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
    default_kwargs = {
        'nneut': kwargs.get('nneut', 82),
        'nprot': kwargs.get('nprot', 50),
    }

    default_kwargs['nstmax'] = default_kwargs['nneut'] + default_kwargs['nprot']

    # get back here later
    default_kwargs['npsi'] = jnp.array(kwargs.get('npsi', [default_kwargs['nneut'], default_kwargs['nprot']]))
    default_kwargs['npmin'] = jnp.array(kwargs.get('npsi', [default_kwargs['nneut'], default_kwargs['nprot']]))

    default_kwargs['charge_number'] = default_kwargs['nprot']
    default_kwargs['mass_number'] = default_kwargs['nstmax']

    shape2d = (default_kwargs['nstmax'], 3)
    shape5d = (default_kwargs['nstmax'], 2, grids.nx, grids.ny, grids.nz)

    default_kwargs['psi'] = jnp.zeros(shape5d, dtype=jnp.complex128)
    default_kwargs['hampsi'] = jnp.zeros(shape5d, dtype=jnp.complex128)
    default_kwargs['lagrange'] = jnp.zeros(shape5d, dtype=jnp.complex128)
    default_kwargs['hmfpsi'] = jnp.zeros(shape5d, dtype=jnp.complex128)
    default_kwargs['delpsi'] = jnp.zeros(shape5d, dtype=jnp.complex128)

    default_kwargs['sp_orbital'] = jnp.zeros(shape2d, dtype=jnp.float64)
    default_kwargs['sp_spin'] = jnp.zeros(shape2d, dtype=jnp.float64)

    isospin = jnp.zeros(default_kwargs['nstmax'], dtype=jnp.int32)
    isospin = isospin.at[default_kwargs['nneut']:].set(1)
    default_kwargs['isospin'] = isospin

    default_kwargs['sp_energy'] = jnp.zeros(default_kwargs['nstmax'], dtype=jnp.float64)
    default_kwargs['sp_efluct1'] = jnp.zeros(default_kwargs['nstmax'], dtype=jnp.float64)
    default_kwargs['sp_kinetic'] = jnp.zeros(default_kwargs['nstmax'], dtype=jnp.float64)
    default_kwargs['sp_norm'] = jnp.zeros(default_kwargs['nstmax'], dtype=jnp.float64)
    default_kwargs['sp_efluct2'] = jnp.zeros(default_kwargs['nstmax'], dtype=jnp.float64)
    default_kwargs['sp_parity'] = jnp.zeros(default_kwargs['nstmax'], dtype=jnp.float64)
    default_kwargs['wocc'] = jnp.ones(default_kwargs['nstmax'], dtype=jnp.float64)
    default_kwargs['wguv'] = jnp.zeros(default_kwargs['nstmax'], dtype=jnp.float64)
    default_kwargs['pairwg'] = jnp.ones(default_kwargs['nstmax'], dtype=jnp.float64)
    default_kwargs['wstates'] = jnp.ones(default_kwargs['nstmax'], dtype=jnp.float64)
    default_kwargs['deltaf'] = jnp.zeros(default_kwargs['nstmax'], dtype=jnp.float64)

    return Levels(**default_kwargs)


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

    # 5.2245375141434353e-17
    # 1.36640756370672e-16
    return d1psout, d2psout


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

    # 3.966240216616929e-17
    # 1.170194769447655e-16
    return d1psout, d2psout

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

    # 4.105394205386439e-17
    # 1.258475307421716e-16
    return d1psout, d2psout


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



def laplace(grids, forces, wg, wguv, psin, v_pairmax, e0inv):
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
