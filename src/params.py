import jax
import jax.numpy as jnp
from dataclasses import dataclass, field

@jax.tree_util.register_dataclass
@dataclass
class Params:
    pi: float
    hbc: float
    e2: float
    cmplxone: complex
    cmplxhalf: complex
    cmplxzero: complex
    r0: float
    wffile: str = field(metadata=dict(static=True))
    converfile: str = field(metadata=dict(static=True))
    monopolesfile: str = field(metadata=dict(static=True))
    dipolesfile: str = field(metadata=dict(static=True))
    momentafile: str = field(metadata=dict(static=True))
    energiesfile: str = field(metadata=dict(static=True))
    diffenergiesfile: str = field(metadata=dict(static=True))
    quadrupolesfile: str = field(metadata=dict(static=True))
    spinfile: str = field(metadata=dict(static=True))
    extfieldfile: str = field(metadata=dict(static=True))
    tabcfile: str = field(metadata=dict(static=True))
    scratch: int
    scratch2: int
    tcoul: bool = field(metadata=dict(static=True))
    tstatic: bool
    tdynamic: bool
    tfft: bool
    trestart: bool
    mprint: int
    mplot: int
    mrest: int
    iteration: int
    time: float
    wflag: bool
    printnow: bool
    nselect: int
    writeselect: str = field(metadata=dict(static=True))
    write_isospin: bool
    mnof: int
    nof: int
    tabc_nprocs: int
    tabc_myid: int

def init_params(**kwargs):
    imode = kwargs.pop('imode')

    default_kwargs = {
        'pi': jnp.pi,
        'hbc': 197.328910,
        'e2': 1.4399784085965135,
        'cmplxone': 1.0 + 0.0j,
        'cmplxhalf': 0.5 + 0.0j,
        'cmplxzero': 0.0 + 0.0j,
        'r0': 1.2,
        'wffile': 'none',
        'converfile': 'conver',
        'monopolesfile': 'monopoles',
        'dipolesfile': 'dipoles',
        'momentafile': 'momenta',
        'energiesfile': 'energies',
        'diffenergiesfile': 'diffenergies',
        'quadrupolesfile': 'quadrupoles',
        'spinfile': 'spin',
        'extfieldfile': 'extfield',
        'tabcfile': 'tabc',
        'scratch': 11,
        'scratch2': 12,
        'tcoul': True,
        'tfft': True,
        'trestart': False,
        'mprint': 10,
        'mplot': 0,
        'mrest': 0,
        'iteration': 0,
        'time': 0.0,
        'wflag': False,
        'printnow': False,
        'nselect': 10,
        'writeselect': 'r',
        'write_isospin': False,
        'mnof': 4,
        'tabc_nprocs': 1,
        'tabc_myid': 0
    }

    default_kwargs.update(kwargs)

    default_kwargs['tstatic'] = True if imode == 1 else False
    default_kwargs['tdynamic'] = True if imode == 0 else False

    return Params(**default_kwargs)
