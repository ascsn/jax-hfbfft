"""
Constraint support for modern HFB solver.

This module provides runtime constraint state and functions to build
constraint fields, update Lagrange multipliers, and compute the
constraint potential to be added to the mean-field.

The constrained problem is solved as an augmented Lagrangian,

    E - lambda (Q - Q0) + C (Q - Q0)^2,

where lambda (lambda_crank) is the multiplier and the quadratic penalty enters
the potential as an additive shift qcorr on it. Two ways of driving lambda are
supported:

  * in-loop (default): lambda is nudged every SCF iteration by
    update_constraint_state, with a trust region and a cap on the well depth;
  * outer/inner (solver.run_constrained_scan): lambda is frozen during each
    inner SCF solve (SolverConfig.freeze_constraint) and updated between
    solves by a secant step. This converges to tighter residuals.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp

from jax_hfbfft.core.grid import Grid
from jax_hfbfft.core.constraint import Constraint, multipole_name

# Set JAX_HFBFFT_CONSTR_DEBUG=1 to print the multiplier update every iteration.
_CONSTR_DEBUG = bool(int(os.environ.get('JAX_HFBFFT_CONSTR_DEBUG', '0')))


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
    penalty_denom: jax.Array  # frozen penalty response (see penalty_qcorr); 0 = recompute
    c0constr: jax.Array       # per-constraint penalty weight
    d0constr: float
    qepsconstr: float
    max_dv: float       # trust region: max change of the constraint potential per iteration (MeV)
    max_v: float        # cap on the depth of the constraint potential (MeV)
    relax_iters: float  # divides the in-loop gain (density relaxation time, in iterations)
    saturated: jax.Array  # per constraint: whether the depth cap was reached

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
            penalty_denom=zeros_vec,
            c0constr=zeros_vec,
            d0constr=1e-4,
            qepsconstr=0.0,
            max_dv=5.0,
            max_v=25.0,
            relax_iters=25.0,
            saturated=jnp.zeros((0,), dtype=bool),
        )


def _build_constraint_fields(
    grid: Grid,
    constraint: Constraint,
    mass_number: float,
) -> Tuple[jax.Array, jax.Array]:
    """Build constraint fields and goals for supported multipoles."""
    if not constraint.tconstraint:
        return jnp.zeros((0, grid.nx, grid.ny, grid.nz, 2)), jnp.zeros((0,))

    # Normalization: use raw multipole operators (fm^lambda)
    prefacdxy = 0.5 * jnp.sqrt(jnp.pi / 5.0)  # Only used for principal-axis terms

    x_mesh = grid.x[:, None, None]
    y_mesh = grid.y[None, :, None]
    z_mesh = grid.z[None, None, :]

    r = jnp.sqrt(x_mesh**2 + y_mesh**2 + z_mesh**2)
    masking = 1.0 / (1.0 + jnp.exp((r - constraint.damprad) / constraint.dampgamma))

    constr_fields: List[jax.Array] = []
    goals: List[float] = []

    # Operators in the convention Q_lam,mu = sqrt(16 pi / (2 lam + 1)) r^lam Y_lam,mu
    # (the one beta_gamma_to_multipoles uses) for Q20, Q30 and Q40:
    #   Q20 = 2z^2 - x^2 - y^2
    #   Q30 = 2z^3 - 3z(x^2 + y^2)
    #   Q40 = (35z^4 - 30z^2 r^2 + 3r^4) / 4
    # Q22 and Q32 are the raw forms x^2 - y^2 and z(x^2 - y^2); the strict
    # convention would put a factor sqrt(3/2) on Q22.
    r2 = x_mesh**2 + y_mesh**2 + z_mesh**2
    rho2 = x_mesh**2 + y_mesh**2

    for (lam, mu), value in constraint.get_multipole_list():
        if (lam, mu) == (1, 0):
            # <z>, the centre of mass. Q20 is not translation invariant (a shift
            # dz adds 2 A dz^2 at no energy cost), so a Q20 constraint alone can
            # be satisfied by moving the nucleus; constraining <z> = 0 removes
            # that direction.
            op = z_mesh
        elif (lam, mu) == (2, 0):
            op = 2.0 * z_mesh**2 - x_mesh**2 - y_mesh**2
        elif (lam, mu) == (2, 2):
            op = x_mesh**2 - y_mesh**2
        elif (lam, mu) == (3, 0):
            op = 2.0 * z_mesh**3 - 3.0 * z_mesh * rho2
        elif (lam, mu) == (3, 2):
            op = z_mesh * (x_mesh**2 - y_mesh**2)
        elif (lam, mu) == (4, 0):
            op = (35.0 * z_mesh**4 - 30.0 * z_mesh**2 * r2 + 3.0 * r2**2) / 4.0
        else:
            name = multipole_name(lam, mu)
            raise ValueError(
                f"Constraint multipole {name} not implemented. "
                "Supported: Q10, Q20, Q22, Q30, Q32, Q40."
            )
        # The multipole operators are damped beyond damprad because they grow
        # like r^lambda. The centre-of-mass operator is not: it is linear, and a
        # damped <z> would stop seeing displacements exactly where the damped
        # Q20 saturates.
        field = (op * jnp.ones_like(masking) if (lam, mu) == (1, 0)
                 else op * masking)
        constr_fields.append(jnp.stack([field, field], axis=-1))
        goals.append(float(value))

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

    # The centre-of-mass constraint gets its own penalty weight: its response
    # and residual scale differently from the multipoles'.
    multipoles = constraint.get_multipole_list()
    c0 = [constraint.c0_cm if (lam, mu) == (1, 0) else constraint.c0constr
          for (lam, mu), _ in multipoles]
    c0 += [constraint.c0constr] * (numconstraint - len(multipoles))

    return ConstraintState(
        enabled=True,
        constr_field=constr_field,
        lambda_crank=jnp.zeros(numconstraint),
        goal_crank=goal_crank,
        actual_crank=jnp.zeros(numconstraint),
        old_crank=jnp.zeros(numconstraint),
        actual_crank2=jnp.zeros(numconstraint),
        qcorr=jnp.zeros(numconstraint),
        penalty_denom=jnp.zeros(numconstraint),
        c0constr=jnp.array(c0),
        d0constr=constraint.d0constr,
        qepsconstr=constraint.qepsconstr,
        max_dv=5.0,
        max_v=25.0,
        relax_iters=25.0,
        saturated=jnp.zeros(numconstraint, dtype=bool),
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
def op_scale_guess(cs) -> jax.Array:
    """max|Q| per constraint; converts a potential depth in MeV into a bound on lambda."""
    return jnp.maximum(jnp.max(jnp.abs(cs.constr_field), axis=(1, 2, 3, 4)), 1e-12)


def linear_response(constraint_state, densities, grid: Grid,
                    mass_number: Optional[float] = None,
                    energy_weighted: bool = False) -> jax.Array:
    """
    Estimate the static response dQ/dlambda of each constraint.

    Without an energy scale this is 2 Var(Q) + d0constr, the non-energy-weighted
    sum rule 2 m_0, which overestimates dQ/dlambda by roughly a particle-hole
    energy (a factor ~20 for 16O). With `mass_number` given, or with
    energy_weighted=True (A then taken from the density), it is divided by the
    isoscalar GQR centroid 65 A^(-1/3) MeV, giving the static polarizability.
    """
    if constraint_state.constr_field.shape[0] == 0:
        return jnp.zeros((0,))
    rho_t = jnp.transpose(densities.rho, (1, 2, 3, 0))
    expectations = grid.wxyz * jnp.sum(
        constraint_state.constr_field * rho_t[None], axis=(1, 2, 3, 4))
    actual_numb = jnp.maximum(grid.wxyz * jnp.sum(densities.rho), 1e-12)
    second = grid.wxyz * jnp.sum(
        rho_t[None] * constraint_state.constr_field**2, axis=(1, 2, 3, 4))
    var = jnp.maximum(jnp.abs(second - expectations**2 / actual_numb), 1e-12)
    resp = jnp.maximum(2.0 * var + constraint_state.d0constr, 1e-12)
    if energy_weighted and mass_number is None:
        resp = resp / (65.0 / jnp.maximum(actual_numb, 1.0) ** (1.0 / 3.0))
    elif mass_number is not None:
        e_bar = 65.0 / max(float(mass_number), 1.0) ** (1.0 / 3.0)
        resp = resp / e_bar
    return jnp.maximum(resp, 1e-12)


def penalty_qcorr(constraint_state, densities, grid: Grid) -> jax.Array:
    """
    The augmented-Lagrangian penalty as a shift on the multiplier, computed
    from the current density at full strength.

    With lambda frozen, E - lambda Q alone is unbounded below (density can move
    to where the damped operator saturates); the quadratic penalty
    C (Q - Q0)^2, C = c0constr / response, makes the inner problem well posed.
    The shift vanishes at the target, so the converged solution is still set by
    lambda.

    C must stay fixed during an inner solve: -(lambda + qcorr) Q is the
    gradient of the penalized functional only for constant C. The response is
    therefore taken from penalty_denom (frozen once per solve by run_hfb) when
    set, and from the energy-weighted linear_response otherwise.
    """
    if constraint_state.constr_field.shape[0] == 0:
        return jnp.zeros((0,))
    rho_t = jnp.transpose(densities.rho, (1, 2, 3, 0))
    expectations = grid.wxyz * jnp.sum(
        constraint_state.constr_field * rho_t[None], axis=(1, 2, 3, 4))
    residual = expectations - constraint_state.goal_crank
    denom = jnp.where(
        constraint_state.penalty_denom > 0.0,
        constraint_state.penalty_denom,
        linear_response(constraint_state, densities, grid, energy_weighted=True),
    )
    qcorr = -2.0 * constraint_state.c0constr * residual / denom
    bound = constraint_state.max_v / op_scale_guess(constraint_state)
    return jnp.clip(qcorr, -bound, bound)


def update_constraint_state(
    constraint_state: ConstraintState,
    densities,
    grid: Grid,
    e0act: float,
    x0act: float,
) -> ConstraintState:
    """
    Update the multipliers from the current densities (in-loop driving).

    lambda <- lambda - qepsconstr * (Q - Q0) / (response * relax_iters), with:
      * a first-call warm start lambda = -(Q - Q0) / response;
      * a trust region: one step changes the potential by at most max_dv MeV;
      * a depth cap: |lambda| max|Q| <= max_v MeV. When it binds (the target is
        unreachable) `saturated` is set and the solver settles at the nearest
        reachable moment instead of tearing the nucleus apart.

    The gain is independent of the solver's step parameters: e0act and x0act
    are accepted for compatibility but not used. relax_iters = 25 was tuned on
    16O; the outer/inner scan (solver.run_constrained_scan) avoids that tuning.
    """
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

    residual = expectations - constraint_state.goal_crank
    op_scale = op_scale_guess(constraint_state)
    max_lambda = constraint_state.max_v / op_scale

    first_call = jnp.all(constraint_state.lambda_crank == 0.0)
    lam_guess = jnp.clip(-residual / denom, -max_lambda, max_lambda)
    lam_start = jnp.where(first_call, lam_guess, constraint_state.lambda_crank)

    stiffness = constraint_state.qepsconstr / denom
    dlambda = -stiffness * residual / constraint_state.relax_iters
    max_dlambda = constraint_state.max_dv / op_scale
    dlambda = jnp.clip(dlambda, -max_dlambda, max_dlambda)

    new_lambda = lam_start + dlambda
    if _CONSTR_DEBUG:
        jax.debug.print(
            "constraint goal={g} actual={a} lambda={l0}->{l1} step_capped={hd} depth_capped={hv}",
            g=constraint_state.goal_crank, a=expectations, l0=lam_start,
            l1=new_lambda, hd=jnp.abs(dlambda) >= 0.999 * max_dlambda,
            hv=jnp.abs(new_lambda) >= 0.999 * max_lambda)

    saturated = jnp.abs(new_lambda) > max_lambda
    new_lambda = jnp.clip(new_lambda, -max_lambda, max_lambda)

    # Quadratic penalty, applied on top of lambda in compute_constraint_potential.
    new_qcorr = (-2.0 * constraint_state.c0constr * residual / denom
                 / constraint_state.relax_iters)
    new_qcorr = jnp.clip(new_qcorr, -max_dlambda, max_dlambda)

    return dataclasses.replace(
        constraint_state,
        lambda_crank=new_lambda,
        qcorr=new_qcorr,
        saturated=saturated,
        old_crank=constraint_state.actual_crank,
        actual_crank=expectations,
        actual_crank2=variances,
    )
