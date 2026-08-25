import jax
from jax import numpy as jnp


@jax.jit
def rpsnorm(ps: jax.Array, wxyz: float) -> jax.Array:
    return wxyz * jnp.sum(jnp.real(jnp.conjugate(ps) * ps))


@jax.jit
def overlap(pl: jax.Array, pr: jax.Array, wxyz: float) -> jax.Array:
    return wxyz * jnp.sum(jnp.conjugate(pl) * pr)
