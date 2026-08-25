"""
Constraint support for modern HFB solver.

This module provides runtime constraint state and functions to build
constraint fields, update Lagrange multipliers, and compute the
constraint potential to be added to the mean-field.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import List, Tuple

import jax
import jax.numpy as jnp

from jax_hfbfft.core.grid import Grid
from jax_hfbfft.core.constraint import Constraint, multipole_name


@jax.tree_util.register_dataclass
@dataclass
class ConstraintState:
    """Runtime state for constraints during HFB iterations."""
    enabled: bool
    constr_field: jax.Array  # (nconstr, nx, ny, nz, 2)
    lambda_crank: jax.Array
    goal_crank: jax.Array
    actual_crank: jax.Array
    old_crank: jax.Array
    actual_crank2: jax.Array
    qcorr: jax.Array
    c0constr: float
    d0constr: float
    qepsconstr: float

    @classmethod
    def disabled(cls, grid: Grid) -> "ConstraintState":
        zeros_field = jnp.zeros((0, grid.nx, grid.ny, grid.nz, 2))
        zeros_vec = jnp.zeros((0,))
        return cls(
            enabled=False,
            constr_field=zeros_field,
            lambda_crank=zeros_vec,
            goal_crank=zeros_vec,
            actual_crank=zeros_vec,
            old_crank=zeros_vec,
            actual_crank2=zeros_vec,
            qcorr=zeros_vec,
            c0constr=0.0,
            d0constr=1e-4,
            qepsconstr=0.0,
        )


def _build_constraint_fields(
    grid: Grid,
    constraint: Constraint,
    mass_number: float,
) -> Tuple[jax.Array, jax.Array]:
    """Build constraint fields and goals for supported multipoles."""
    if not constraint.tconstraint:
        return jnp.zeros((0, grid.nx, grid.ny, grid.nz, 2)), jnp.zeros((0,))

    # Normalization: use raw multipole operators for Q20/Q22 (fm^2)
    prefacdxy = 0.5 * jnp.sqrt(jnp.pi / 5.0)  # Only used for principal-axis terms

    x_mesh = grid.x[:, None, None]
    y_mesh = grid.y[None, :, None]
    z_mesh = grid.z[None, None, :]

    r = jnp.sqrt(x_mesh**2 + y_mesh**2 + z_mesh**2)
    masking = 1.0 / (1.0 + jnp.exp((r - constraint.damprad) / constraint.dampgamma))

    constr_fields: List[jax.Array] = []
    goals: List[float] = []

    for (lam, mu), value in constraint.get_multipole_list():
        if (lam, mu) == (2, 0):
            q20_field = (2.0 * z_mesh**2 - x_mesh**2 - y_mesh**2) * masking
            field_both = jnp.stack([q20_field, q20_field], axis=-1)
            constr_fields.append(field_both)
            goals.append(float(value))
        elif (lam, mu) == (2, 2):
            q22_field = (x_mesh**2 - y_mesh**2) * masking
            field_both = jnp.stack([q22_field, q22_field], axis=-1)
            constr_fields.append(field_both)
            goals.append(float(value))
        else:
            name = multipole_name(lam, mu)
            raise ValueError(
                f"Constraint multipole {name} not supported in modern solver yet. "
                "Supported: Q20, Q22."
            )

    if constraint.tq_prin_axes:
        facdxy = prefacdxy / (constraint.r0rms**2 * mass_number**(5.0 / 3.0))

        xy_field = facdxy * x_mesh * y_mesh * masking
        constr_fields.append(jnp.stack([xy_field, xy_field], axis=-1))
        goals.append(0.0)

        xz_field = facdxy * x_mesh * z_mesh * masking
        constr_fields.append(jnp.stack([xz_field, xz_field], axis=-1))
        goals.append(0.0)

        yz_field = facdxy * y_mesh * z_mesh * masking
        constr_fields.append(jnp.stack([yz_field, yz_field], axis=-1))
        goals.append(0.0)

        x_field = facdxy * x_mesh * masking
        constr_fields.append(jnp.stack([x_field, x_field], axis=-1))
        goals.append(0.0)

        y_field = facdxy * y_mesh * masking
        constr_fields.append(jnp.stack([y_field, y_field], axis=-1))
        goals.append(0.0)

        z_field = facdxy * z_mesh * masking
        constr_fields.append(jnp.stack([z_field, z_field], axis=-1))
        goals.append(0.0)

    if not constr_fields:
        return jnp.zeros((0, grid.nx, grid.ny, grid.nz, 2)), jnp.zeros((0,))

    return jnp.stack(constr_fields, axis=0), jnp.array(goals)


def build_constraint_state(
    constraint: Constraint,
    grid: Grid,
    mass_number: float,
) -> ConstraintState:
    """Create a runtime ConstraintState from a Constraint config."""
    if constraint is None or not constraint.tconstraint:
        return ConstraintState.disabled(grid)

    constr_field, goal_crank = _build_constraint_fields(grid, constraint, mass_number)

    numconstraint = goal_crank.shape[0]
    if numconstraint == 0:
        return ConstraintState.disabled(grid)

    return ConstraintState(
        enabled=True,
        constr_field=constr_field,
        lambda_crank=jnp.zeros(numconstraint),
        goal_crank=goal_crank,
        actual_crank=jnp.zeros(numconstraint),
        old_crank=jnp.zeros(numconstraint),
        actual_crank2=jnp.zeros(numconstraint),
        qcorr=jnp.zeros(numconstraint),
        c0constr=constraint.c0constr,
        d0constr=constraint.d0constr,
        qepsconstr=constraint.qepsconstr,
    )


@jax.jit
def compute_constraint_expectations(constraint_state: ConstraintState, densities, grid: Grid) -> jax.Array:
    """Compute expectation values for all constraints."""
    if constraint_state.constr_field.shape[0] == 0:
        return jnp.zeros((0,))

    rho_transposed = jnp.transpose(densities.rho, (1, 2, 3, 0))
    expectations = grid.wxyz * jnp.sum(
        constraint_state.constr_field * rho_transposed[None, :, :, :, :],
        axis=(1, 2, 3, 4),
    )
    return expectations


@jax.jit
def compute_constraint_potential(
    constraint_state: ConstraintState,
    grid: Grid,
) -> jax.Array:
    """Compute constraint potential from Lagrange multipliers."""
    if constraint_state.constr_field.shape[0] == 0:
        return jnp.zeros((2, grid.nx, grid.ny, grid.nz))

    contribution = jnp.einsum(
        "i,ixyzs->sxyz",
        constraint_state.lambda_crank + constraint_state.qcorr,
        constraint_state.constr_field,
    )
    return -contribution


@jax.jit
def update_constraint_state(
    constraint_state: ConstraintState,
    densities,
    grid: Grid,
    e0act: float,
    x0act: float,
) -> ConstraintState:
    """Update Lagrange multipliers and expectations based on densities."""
    if constraint_state.constr_field.shape[0] == 0:
        return constraint_state

    rho_transposed = jnp.transpose(densities.rho, (1, 2, 3, 0))
    expectations = grid.wxyz * jnp.sum(
        constraint_state.constr_field * rho_transposed[None, :, :, :, :],
        axis=(1, 2, 3, 4),
    )

    total_density = jnp.sum(densities.rho)
    actual_numb = jnp.maximum(grid.wxyz * total_density, 1e-12)

    second_moments = grid.wxyz * jnp.sum(
        rho_transposed[None, :, :, :, :] * constraint_state.constr_field**2,
        axis=(1, 2, 3, 4),
    )

    variances = jnp.maximum(jnp.abs(second_moments - expectations**2 / actual_numb), 1e-12)

    denom = 2.0 * variances + constraint_state.d0constr
    denom = jnp.maximum(denom, 1e-12)

    e0act_safe = jnp.maximum(e0act, 1.0)
    x0act_safe = jnp.maximum(x0act, 1e-6)

    corrlambda = -constraint_state.qepsconstr * (e0act_safe / x0act_safe) * (
        expectations - constraint_state.goal_crank
    ) / denom

    # Clamp update to avoid instability (especially without explicit Q-correction)
    max_step = 0.2
    corrlambda = jnp.clip(corrlambda, -max_step, max_step)

    new_lambda = constraint_state.lambda_crank + corrlambda
    new_qcorr = constraint_state.c0constr * (expectations - constraint_state.goal_crank) / denom

    return dataclasses.replace(
        constraint_state,
        lambda_crank=new_lambda,
        qcorr=new_qcorr,
        old_crank=constraint_state.actual_crank,
        actual_crank=expectations,
        actual_crank2=variances,
    )
