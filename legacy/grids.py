import jax
import jax.numpy as jnp
from dataclasses import dataclass, field

@jax.tree_util.register_dataclass
@dataclass
class Grids:
    nx: int = field(metadata=dict(static=True))
    ny: int = field(metadata=dict(static=True))
    nz: int = field(metadata=dict(static=True))
    tabc_x: int
    tabc_y: int
    tabc_z: int
    bangx: float
    bangy: float
    bangz: float
    periodic: bool = field(metadata=dict(static=True))
    dx: float
    dy: float
    dz: float
    wxyz: float
    x: jax.Array
    y: jax.Array
    z: jax.Array
    der1x: jax.Array
    der2x: jax.Array
    cdmpx: jax.Array
    der1y: jax.Array
    der2y: jax.Array
    cdmpy: jax.Array
    der1z: jax.Array
    der2z: jax.Array
    cdmpz: jax.Array


def sder(nmax: int, d: float) -> jnp.ndarray:
    icn = (nmax + 1) // 2
    afac = jnp.pi / icn

    i = jnp.arange(1, nmax + 1, dtype=jnp.int64)
    j = jnp.arange(1, icn, dtype=jnp.int64)[:, jnp.newaxis, jnp.newaxis]
    grid = i[jnp.newaxis, :] - i[:, jnp.newaxis]

    der = -afac * ((jnp.sum(-j * jnp.sin(j * afac * (grid)), axis=0)) - (0.5 * icn * jnp.sin(icn * afac * grid))) / (icn * d)

    return der

sder_jit = jax.jit(sder, static_argnames=['nmax', 'd'])


def sder2(nmax: int, d: float) -> jnp.ndarray:
    icn = (nmax + 1) // 2
    afac = jnp.pi / icn

    i = jnp.arange(1, nmax + 1, dtype=jnp.int64)
    j = jnp.arange(1, icn, dtype=jnp.int64)[:, jnp.newaxis, jnp.newaxis]
    grid = i[jnp.newaxis, :] - i[:, jnp.newaxis]

    der = -(afac * afac) * ((jnp.sum((j ** 2) * jnp.cos(j * afac * (grid)), axis=0)) + (0.5 * (icn ** 2) * jnp.cos(icn * afac * grid))) / (icn * d * d)

    return der

sder2_jit = jax.jit(sder2, static_argnames=['nmax', 'd'])


def init_grids(params, **kwargs):
    default_kwargs = {
        'nx': 48,
        'ny': 48,
        'nz': 48,
        'dx': 0.8,
        'dy': 0.8,
        'dz': 0.8,
        'bangx': 0.0,
        'bangy': 0.0,
        'bangz': 0.0,
        'tabc_x': 0,
        'tabc_y': 0,
        'tabc_z': 0,
        'periodic': False
    }

    default_kwargs.update(kwargs)

    if not params.tfft and (abs(default_kwargs['bangx']) > 0.00001 or
                            abs(default_kwargs['bangy']) > 0.00001 or
                            abs(default_kwargs['bangz']) > 0.00001):
        raise ValueError('bloch boundaries cannot be used without tfft.')

    if params.tabc_nprocs > 1:
        # CALL tabc_init_blochboundary
        pass

    if params.tabc_nprocs == 1 and (default_kwargs['tabc_x'] != 0 or
                                    default_kwargs['tabc_y'] != 0 or
                                    default_kwargs['tabc_z'] != 0):
        raise ValueError('no tabc possible with tabc_nprocs=1.')

    default_kwargs['bangx'] *= params.pi
    default_kwargs['bangy'] *= params.pi
    default_kwargs['bangz'] *= params.pi

    if (default_kwargs['nx'] % 2 != 0 or
        default_kwargs['ny'] % 2 != 0 or
        default_kwargs['nz'] % 2 != 0):
        raise ValueError('grids: nx, ny, and nz must be even.')

    if default_kwargs['dx'] * default_kwargs['dy'] * default_kwargs['dz'] <= 0.0:
        if default_kwargs['dx'] <= 0.0:
            raise ValueError('grid spacing given as 0.')

        default_kwargs['dy'] = default_kwargs['dx']
        default_kwargs['dz'] = default_kwargs['dx']

    def init_coord_centered(n, d):
        indices = jnp.arange(n, dtype=jnp.float64)
        coords = (indices - (n-1)/2) * d
        return coords
    
    default_kwargs['x'] = init_coord_centered(default_kwargs['nx'], default_kwargs['dx'])
    default_kwargs['y'] = init_coord_centered(default_kwargs['ny'], default_kwargs['dy'])
    default_kwargs['z'] = init_coord_centered(default_kwargs['nz'], default_kwargs['dz'])

    default_kwargs['der1x'] = sder_jit(default_kwargs['nx'], default_kwargs['dx'])
    default_kwargs['der1y'] = sder_jit(default_kwargs['ny'], default_kwargs['dy'])
    default_kwargs['der1z'] = sder_jit(default_kwargs['nz'], default_kwargs['dz'])

    default_kwargs['der2x'] = sder2_jit(default_kwargs['nx'], default_kwargs['dx'])
    default_kwargs['der2y'] = sder2_jit(default_kwargs['ny'], default_kwargs['dy'])
    default_kwargs['der2z'] = sder2_jit(default_kwargs['nz'], default_kwargs['dz'])

    default_kwargs['cdmpx'] = jnp.zeros((default_kwargs['nx'], default_kwargs['nx']), dtype=jnp.float64)
    default_kwargs['cdmpy'] = jnp.zeros((default_kwargs['ny'], default_kwargs['ny']), dtype=jnp.float64)
    default_kwargs['cdmpz'] = jnp.zeros((default_kwargs['nz'], default_kwargs['nz']), dtype=jnp.float64)

    default_kwargs['wxyz'] = default_kwargs['dx'] * default_kwargs['dy'] * default_kwargs['dz']

    return Grids(**default_kwargs)

def print_grid_info(grids):
    print("\n=== GRID INFORMATION ===")
    print(f"Grid dimensions: {grids.nx}x{grids.ny}x{grids.nz}")
    print(f"Grid spacing: dx={grids.dx:.8f}, dy={grids.dy:.8f}, dz={grids.dz:.8f}")
    print(f"Integration measure wxyz = {grids.wxyz:.8f}")
    print(f"Expected value = {grids.dx * grids.dy * grids.dz:.8f}")
    
    # Calculate grid volume
    volume = grids.nx * grids.ny * grids.nz * grids.wxyz
    print(f"Total grid volume = {volume:.8f}")
    
    # Check that center is exactly at origin
    center_idx = grids.nx // 2
    print(f"Center index: {center_idx}")
    print(f"Center coordinates: x={grids.x[center_idx]:.10f}, y={grids.y[center_idx]:.10f}, z={grids.z[center_idx]:.10f}")
    print(f"Grid ranges: x=[{grids.x[0]:.3f}, {grids.x[-1]:.3f}], y=[{grids.y[0]:.3f}, {grids.y[-1]:.3f}], z=[{grids.z[0]:.3f}, {grids.z[-1]:.3f}]")