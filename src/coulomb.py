import jax
from jax import numpy as jnp
from dataclasses import dataclass, field

@jax.tree_util.register_dataclass
@dataclass
class Coulomb:
    nx2: int = field(metadata=dict(static=True))
    ny2: int = field(metadata=dict(static=True))
    nz2: int = field(metadata=dict(static=True))
    wcoul: jax.Array
    q: jax.Array

def init_iq(n, d, periodic):
    idx = jnp.arange(n)
    idx = jnp.where(idx <= n//2, idx, idx - n)

    if periodic:
        return (2.0 * jnp.pi * idx / (n * d)) ** 2
    else:
        return (d * idx) ** 2

def init_coulomb(grids):
    default_kwargs = {
        'nx2': grids.nx if grids.periodic else 2 * grids.nx,
        'ny2': grids.ny if grids.periodic else 2 * grids.ny,
        'nz2': grids.nz if grids.periodic else 2 * grids.nz,
        'wcoul': jnp.zeros((grids.nx, grids.ny, grids.nz), dtype=jnp.float64)
    }

    iqx = init_iq(default_kwargs['nx2'], grids.dx, grids.periodic)
    iqy = init_iq(default_kwargs['ny2'], grids.dy, grids.periodic)
    iqz = init_iq(default_kwargs['nz2'], grids.dz, grids.periodic)

    i = iqx[:, jnp.newaxis, jnp.newaxis]
    j = iqy[jnp.newaxis, :, jnp.newaxis]
    k = iqz[jnp.newaxis, jnp.newaxis, :]
    q = (i + j + k).astype(jnp.complex128)

    q = q.at[0,0,0].set(1.0)

    if grids.periodic:
        q = q.at[...].set(1.0 / jnp.real(q))
        q = q.at[0,0,0].set(0.0)
    else:
        q = q.at[...].set(1.0 / jnp.sqrt(jnp.real(q)))
        q = q.at[0,0,0].set(2.84 / (grids.dx * grids.dy * grids.dz) ** (1.0/3.0))
        q = q.at[...].set(jnp.fft.fftn(q))

    default_kwargs['q'] = q

    return Coulomb(**default_kwargs)


def poisson(grids, params, coulomb, rho):
    rho2 = jnp.zeros(
        (coulomb.nx2, coulomb.ny2, coulomb.nz2),
        dtype=jnp.complex128
    )

    if not grids.periodic:
        rho2 = rho2.at[0,0,0].set(0.0)

    rho2 = rho2.at[:grids.nx, :grids.ny, :grids.nz].set(rho[1,...])

    rho2 = rho2.at[...].set(jnp.fft.fftn(rho2))

    if grids.periodic:
        rho2 = rho2.at[...].multiply(4.0 * jnp.pi * params.e2 * jnp.real(coulomb.q))
    else:
        rho2 = rho2.at[...].multiply(params.e2 * grids.wxyz * coulomb.q)

    wcoul = jnp.fft.ifftn(rho2)

    return jnp.real(wcoul[:grids.nx, :grids.ny, :grids.nz])
