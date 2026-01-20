import jax
from dataclasses import dataclass, field

@jax.tree_util.register_dataclass
@dataclass
class Moment:
    cmtot: jax.Array


def init_moment(cmtot):
    return Moment(cmtot)
