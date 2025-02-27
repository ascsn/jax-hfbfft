import jax
import jax.numpy as jnp
from reader import read_yaml

from dataclasses import dataclass, field

@jax.tree_util.register_dataclass
@dataclass
class Forces:
    name: str = field(metadata=dict(static=True))
    ex: int = field(metadata=dict(static=True))
    zpe: int
    h2m: jax.Array
    t0: float
    t1: float
    t2: float
    t3: float
    t4: float
    x0: float
    x1: float
    x2: float
    x3: float
    b4p: float
    power: float
    v0prot: float
    v0neut: float
    rho0pr: float
    ipair: int = field(metadata=dict(static=True))
    pair_reg: bool
    delta_fit: jax.Array
    pair_cutoff: jax.Array
    state_cutoff: jax.Array
    softcut_range: jax.Array
    tbcs: bool
    h2ma: float
    nucleon_mass: float
    b0: float
    b0p: float
    b1: float
    b1p: float
    b2: float
    b2p: float
    b3: float
    b3p: float
    b4: float
    b4p: float
    slate: float
    Crho0: float
    Crho1: float
    Crho0D: float
    Crho1D: float
    Cdrho0: float
    Cdrho1: float
    Ctau0: float
    Ctau1: float
    CdJ0: float
    CdJ1: float


def init_forces(params, **kwargs):
    force = read_yaml('_forces.yml').get(kwargs.get('name', 'SLy4'))

    if force is None:
        raise KeyError(f"Force '{force}' not found in the _forces.yml.")

    default_kwargs = {
        'name': kwargs.get('name', 'SLy4'),
        'ipair': kwargs.get('ipair', 0),
        'v0prot': 0.0,
        'v0neut': 0.0,
        'rho0pr': 0.16,
        'pair_reg': kwargs.get('pair_reg', False),
        'delta_fit': jnp.array(kwargs.get('delta_fit', [-1.0, -1.0])),
        'pair_cutoff': jnp.array(kwargs.get('pair_cutoff', [0.0, 0.0])),
        'state_cutoff': jnp.array(kwargs.get('state_cutoff', [0.0, 0.0])),
        'softcut_range': kwargs.get('softcut_range', 0.1),
        'tbcs': kwargs.get('tbcs', False)
    }

    default_kwargs['ex'] = force.get('ex')
    default_kwargs['zpe'] = force.get('zpe')
    default_kwargs['h2m'] = jnp.array(force.get('h2m'))
    default_kwargs['t0'] = force.get('t0')
    default_kwargs['t1'] = force.get('t1')
    default_kwargs['t2'] = force.get('t2')
    default_kwargs['t3'] = force.get('t3')
    default_kwargs['t4'] = force.get('t4')
    default_kwargs['x0'] = force.get('x0')
    default_kwargs['x1'] = force.get('x1')
    default_kwargs['x2'] = force.get('x2')
    default_kwargs['x3'] = force.get('x3')
    default_kwargs['b4p'] = force.get('b4p')
    default_kwargs['power'] = force.get('power')

    default_kwargs['b0'] = default_kwargs['t0'] * (1 + 0.5 * default_kwargs['x0'])
    default_kwargs['b0p'] = default_kwargs['t0'] * (0.5 + default_kwargs['x0'])
    default_kwargs['b1'] = (default_kwargs['t1'] + 0.5 * default_kwargs['x1'] *
                            default_kwargs['t1'] + default_kwargs['t2'] + 0.5 *
                            default_kwargs['x2'] * default_kwargs['t2']) / 4
    default_kwargs['b1p'] = (default_kwargs['t1'] * (0.5 + default_kwargs['x1']) -
                             default_kwargs['t2'] * (0.5 + default_kwargs['x2'])) / 4
    default_kwargs['b2'] = (3 * default_kwargs['t1'] *
                            (1 + 0.5 * default_kwargs['x1']) -
                            default_kwargs['t2'] * (1 + 0.5 * default_kwargs['x2'])) / 8
    default_kwargs['b2p'] = (3 * default_kwargs['t1'] *
                             (0.5 + default_kwargs['x1']) +
                             default_kwargs['t2'] * (0.5 + default_kwargs['x2'])) / 8
    default_kwargs['b3'] = default_kwargs['t3'] * (1 + 0.5 * default_kwargs['x3']) / 4
    default_kwargs['b3p'] = default_kwargs['t3'] * (0.5 + default_kwargs['x3']) / 4
    default_kwargs['b4'] = default_kwargs['t4'] / 2
    default_kwargs['slate'] = (3 / params.pi) ** (1/3) * params.e2

    default_kwargs['Crho0'] = 0.5 * default_kwargs['b0'] - 0.25 * default_kwargs['b0p']
    default_kwargs['Crho1'] = -0.25 * default_kwargs['b0p']
    default_kwargs['Crho0D'] = (1/3) * default_kwargs['b3'] - (1/6) * default_kwargs['b3p']
    default_kwargs['Crho1D'] = -(1/6) * default_kwargs['b3p']
    default_kwargs['Cdrho0'] = -0.5 * default_kwargs['b2'] + 0.25 * default_kwargs['b2p']
    default_kwargs['Cdrho1'] = 0.25 * default_kwargs['b2p']
    default_kwargs['Ctau0'] = default_kwargs['b1'] - 0.5 * default_kwargs['b1p']
    default_kwargs['Ctau1'] = -0.5 * default_kwargs['b1p']
    default_kwargs['CdJ0'] = -default_kwargs['b4'] - 0.5 * default_kwargs['b4p']
    default_kwargs['CdJ1'] = -0.5 * default_kwargs['b4p']

    if default_kwargs.get('ipair') == 5:
        default_kwargs['v0prot'] = force.get('vdi').get('v0prot')
        default_kwargs['v0neut'] = force.get('vdi').get('v0neut')
        default_kwargs['rho0pr'] = force.get('vdi').get('rho0pr')

    if default_kwargs.get('ipair') == 6:
        default_kwargs['v0prot'] = force.get('dddi').get('v0prot')
        default_kwargs['v0neut'] = force.get('dddi').get('v0neut')
        default_kwargs['rho0pr'] = force.get('dddi').get('rho0pr')

    default_kwargs['h2ma'] = float(0.5 * jnp.sum(default_kwargs['h2m']))
    default_kwargs['nucleon_mass'] = params.hbc ** 2 / (2.0 * default_kwargs['h2ma'])

    return Forces(**default_kwargs)
