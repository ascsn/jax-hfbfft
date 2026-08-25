import jax
import jax.numpy as jnp
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
import numpy as np

@jax.tree_util.register_dataclass
@dataclass
class Constraints:
    """Container for all constraint-related data and parameters"""
    
    # Control flags
    tconstraint: bool = False
    tq_prin_axes: bool = False  # Principal axes constraints (6 additional constraints)
    
    # Desired constraint values
    alpha20_wanted: float = -1e99  # Q20 quadrupole moment target
    alpha22_wanted: float = -1e99  # Q22 quadrupole moment target
    
    # Constraint algorithm parameters
    c0constr: float = 0.8          # Parameter for Q-corrective step
    d0constr: float = 1e-4         # Small parameter to avoid division by zero
    qepsconstr: float = 0.3        # Parameter for Lagrange multiplier update
    dampgamma: float = 1.0         # Damping parameter for masking function
    damprad: float = 6.0           # Damping radius for masking function
    
    # Physical constants
    r0rms: float = 0.93            # RMS radius parameter (corresponds to R0=1.2 fm)
    
    # Constraint arrays (allocated dynamically)
    numconstraint: int = 0
    constr_field: Optional[jax.Array] = None      # [iconstr, nx, ny, nz, isospin]
    lambda_crank: Optional[jax.Array] = None      # Lagrange multipliers
    dlambda_crank: Optional[jax.Array] = None     # Corrections to multipliers
    goal_crank: Optional[jax.Array] = None        # Target expectation values
    actual_crank: Optional[jax.Array] = None      # Actual expectation values
    old_crank: Optional[jax.Array] = None         # Previous expectation values
    actual_crank2: Optional[jax.Array] = None     # Variances
    qcorr: Optional[jax.Array] = None             # Q-correction factors

    def replace(self, **kwargs):
        """Create a new Constraints object with updated fields"""
        import dataclasses
        return dataclasses.replace(self, **kwargs)


def init_constraints(params: Dict[str, Any], mass_number: int, grids) -> Constraints:
    # Create constraints object with parameters from input
    constraints = Constraints()
    
    # Read constraint parameters
    if "alpha20_wanted" in params and params["alpha20_wanted"] > -1e90:
        constraints = constraints.replace(alpha20_wanted=params["alpha20_wanted"])
    if "alpha22_wanted" in params and params["alpha22_wanted"] > -1e90:
        constraints = constraints.replace(alpha22_wanted=params["alpha22_wanted"])
    if "tq_prin_axes" in params:
        constraints = constraints.replace(tq_prin_axes=params["tq_prin_axes"])
    
    # Update algorithm parameters if provided
    for param in ["c0constr", "d0constr", "qepsconstr", "dampgamma", "damprad", "r0rms"]:
        if param in params:
            constraints = constraints.replace(**{param: params[param]})
    
    # Count number of constraints
    numconstraint = 0
    if constraints.alpha20_wanted > -1e90:
        numconstraint += 1
    if constraints.alpha22_wanted > -1e90:
        numconstraint += 1
    if constraints.tq_prin_axes:
        numconstraint += 6  # xy, xz, yz, x, y, z
    
    if numconstraint == 0:
        print("No constraints specified - running spherical calculation")
        return constraints
    
    constraints = constraints.replace(numconstraint=numconstraint, tconstraint=True)
    
    # Initialize constraint arrays
    nx, ny, nz = grids.nx, grids.ny, grids.nz
    constraints = constraints.replace(
        constr_field=jnp.zeros((numconstraint, nx, ny, nz, 2)),
        lambda_crank=jnp.full(numconstraint, -0.2),  # Initial guess
        dlambda_crank=jnp.zeros(numconstraint),
        goal_crank=jnp.zeros(numconstraint),
        actual_crank=jnp.zeros(numconstraint),
        old_crank=jnp.zeros(numconstraint),
        actual_crank2=jnp.zeros(numconstraint),
        qcorr=jnp.zeros(numconstraint)
    )
    
    # Set up constraint fields
    constraints = _setup_constraint_fields(constraints, mass_number, grids)
    
    print(f"Constraints initialized with {numconstraint} constraint(s):")
    if constraints.alpha20_wanted > -1e90:
        print(f"  Q20 target: {constraints.alpha20_wanted:.3f}")
    if constraints.alpha22_wanted > -1e90:
        print(f"  Q22 target: {constraints.alpha22_wanted:.3f}")
    if constraints.tq_prin_axes:
        print(f"  Principal axes constraints enabled")
    
    return constraints


def _setup_constraint_fields(constraints: Constraints, mass_number: int, grids) -> Constraints:
    # Physical constants for normalization
    prefac20 = jnp.sqrt(jnp.pi / 5.0)           # Q20 normalization: sqrt(π/5)
    prefac22 = jnp.sqrt(1.2 * jnp.pi)           # Q22 normalization: sqrt(6π/5)  
    prefacdxy = 0.5 * jnp.sqrt(jnp.pi / 5.0)    # Off-diagonal normalization
    
    # Create coordinate meshes
    x_mesh = grids.x[:, None, None]               # [nx, 1, 1]
    y_mesh = grids.y[None, :, None]               # [1, ny, 1]
    z_mesh = grids.z[None, None, :]               # [1, 1, nz]
    
    # Create masking function for spatial localization
    # This ensures constraints are applied mainly in the nuclear interior
    r = jnp.sqrt(x_mesh**2 + y_mesh**2 + z_mesh**2)
    masking = 1.0 / (1.0 + jnp.exp((r - constraints.damprad) / constraints.dampgamma))
    
    iconstr = 0
    constr_fields = []
    goals = []
    
    # Q20 constraint: (2z² - x² - y²) - prolate/oblate deformation
    if constraints.alpha20_wanted > -1e90:
        # Normalization factor includes A^(5/3) dependence
        fac20 = prefac20 / (constraints.r0rms**2 * mass_number**(5.0/3.0))
        q20_field = fac20 * (2.0 * z_mesh**2 - x_mesh**2 - y_mesh**2) * masking
        
        # Apply to both isospins (isoscalar constraint)
        field_both_isospins = jnp.stack([q20_field, q20_field], axis=-1)  # [nx, ny, nz, 2]
        constr_fields.append(field_both_isospins)
        goals.append(constraints.alpha20_wanted)
        iconstr += 1
    
    # Q22 constraint: (x² - y²) - triaxial deformation
    if constraints.alpha22_wanted > -1e90:
        fac22 = prefac22 / (constraints.r0rms**2 * mass_number**(5.0/3.0))
        q22_field = fac22 * (x_mesh**2 - y_mesh**2) * masking
        
        # Apply to both isospins
        field_both_isospins = jnp.stack([q22_field, q22_field], axis=-1)
        constr_fields.append(field_both_isospins)
        goals.append(constraints.alpha22_wanted)
        iconstr += 1
    
    # Principal axes constraints (force alignment with coordinate axes)
    if constraints.tq_prin_axes:
        facdxy = prefacdxy / (constraints.r0rms**2 * mass_number**(5.0/3.0))
        
        # xy constraint (force Qxy = 0)
        xy_field = facdxy * x_mesh * y_mesh * masking
        field_both_isospins = jnp.stack([xy_field, xy_field], axis=-1)
        constr_fields.append(field_both_isospins)
        goals.append(0.0)
        
        # xz constraint (force Qxz = 0)
        xz_field = facdxy * x_mesh * z_mesh * masking
        field_both_isospins = jnp.stack([xz_field, xz_field], axis=-1)
        constr_fields.append(field_both_isospins)
        goals.append(0.0)
        
        # yz constraint (force Qyz = 0)
        yz_field = facdxy * y_mesh * z_mesh * masking
        field_both_isospins = jnp.stack([yz_field, yz_field], axis=-1)
        constr_fields.append(field_both_isospins)
        goals.append(0.0)
        
        # x constraint (force center of mass x = 0)
        x_field = facdxy * x_mesh * masking
        field_both_isospins = jnp.stack([x_field, x_field], axis=-1)
        constr_fields.append(field_both_isospins)
        goals.append(0.0)
        
        # y constraint (force center of mass y = 0)
        y_field = facdxy * y_mesh * masking
        field_both_isospins = jnp.stack([y_field, y_field], axis=-1)
        constr_fields.append(field_both_isospins)
        goals.append(0.0)
        
        # z constraint (force center of mass z = 0)
        z_field = facdxy * z_mesh * masking
        field_both_isospins = jnp.stack([z_field, z_field], axis=-1)
        constr_fields.append(field_both_isospins)
        goals.append(0.0)
    
    # Stack all fields and update constraints object
    if constr_fields:
        constraints = constraints.replace(
            constr_field=jnp.stack(constr_fields, axis=0),  # [numconstraint, nx, ny, nz, 2]
            goal_crank=jnp.array(goals)
        )
    
    return constraints


@jax.jit
def add_constraint_potential(potential: jax.Array, constraints: Constraints) -> jax.Array:
    if not constraints.tconstraint or constraints.constr_field is None:
        return potential
    
    # Add constraint contribution: V -= ∑ λ_i * field_i
    # Einstein summation over constraint index
    constraint_contribution = jnp.einsum('i,ixyzs->xyzs', 
                                       constraints.lambda_crank, 
                                       constraints.constr_field)
    
    return potential - constraint_contribution


@jax.jit  
def calculate_constraint_expectation_values(densities, constraints: Constraints, grids) -> Tuple[jax.Array, jax.Array]:
    if not constraints.tconstraint or constraints.constr_field is None:
        return jnp.array([]), jnp.array([])
    
    # Total particle number for normalization
    total_density = jnp.sum(densities.rho)  # Sum over isospins and space
    actual_numb = grids.wxyz * total_density
    
    # Rearrange density for easier computation: [nx, ny, nz, isospin]
    rho_transposed = jnp.transpose(densities.rho, (1, 2, 3, 0))
    
    # Calculate expectation values: <Q_i> = ∫ ρ(r) Q_i(r) dr
    expectations = grids.wxyz * jnp.sum(
        constraints.constr_field * rho_transposed[None, :, :, :, :], 
        axis=(1, 2, 3, 4)  # Sum over spatial dimensions and isospin
    )
    
    # Calculate second moments: <Q_i²> = ∫ ρ(r) Q_i(r)² dr
    second_moments = grids.wxyz * jnp.sum(
        rho_transposed[None, :, :, :, :] * constraints.constr_field**2,
        axis=(1, 2, 3, 4)
    )
    
    # Variances: Var[Q_i] = <Q_i²> - <Q_i>² / N
    # The division by N accounts for the fact that we're measuring fluctuations
    variances = jnp.abs(second_moments - expectations**2 / actual_numb)
    
    return expectations, variances


@jax.jit
def update_lagrange_multipliers(constraints: Constraints, 
                               expectations: jax.Array, 
                               variances: jax.Array,
                               e0act: float, 
                               x0act: float) -> Constraints:
    if not constraints.tconstraint:
        return constraints
    
    # Calculate corrections to Lagrange multipliers
    # This implements a stabilized gradient descent
    corrlambda = -constraints.qepsconstr * (jnp.maximum(e0act, 1.0) / x0act) * \
                 (expectations - constraints.old_crank) / \
                 (2.0 * variances + constraints.d0constr)
    
    # Update multipliers
    new_lambda_crank = constraints.lambda_crank + corrlambda
    
    # Calculate Q-correction factors for wavefunction correction
    new_qcorr = constraints.c0constr * (expectations - constraints.goal_crank) / \
                (2.0 * variances + constraints.d0constr)
    
    return constraints.replace(
        lambda_crank=new_lambda_crank,
        qcorr=new_qcorr,
        actual_crank=expectations,
        actual_crank2=variances
    )


@jax.jit
def apply_wavefunction_corrections(psi: jax.Array, 
                                 constraints: Constraints, 
                                 levels) -> jax.Array:
    if not constraints.tconstraint or constraints.constr_field is None:
        return psi
    
    # Calculate correction field: -∑ qcorr_i * field_i
    correction_field = -jnp.einsum('i,ixyzs->xyzs', 
                                  constraints.qcorr,
                                  constraints.constr_field)  # [nx, ny, nz, isospin]
    
    # Apply corrections state by state using vmap
    def apply_correction_to_state(state_idx, psi_state):
        """Apply correction to a single state"""
        isospin_idx = levels.isospin[state_idx] - 1  # Convert 1,2 -> 0,1
        weight = levels.wstates[state_idx] * levels.wocc[state_idx]
        
        # Get the correction for this isospin
        correction = correction_field[:, :, :, isospin_idx]  # [nx, ny, nz]
        
        # Apply exponential correction: ψ → exp(weight * correction) * ψ
        exp_correction = jnp.exp(weight * correction)  # [nx, ny, nz]
        
        # Apply to both spin components [nx, ny, nz, spin]
        corrected_state = psi_state * exp_correction[:, :, :, None]
        
        return corrected_state
    
    # Apply corrections to all states using vectorization
    corrected_psi = jax.vmap(apply_correction_to_state, in_axes=(0, 0))(
        jnp.arange(psi.shape[0]), psi
    )
    
    return corrected_psi


def before_constraint_step(constraints: Constraints, densities) -> Constraints:
    if not constraints.tconstraint:
        return constraints
    
    return constraints.replace(old_crank=constraints.actual_crank)


def tune_constraints(constraints: Constraints, 
                    densities, 
                    levels,
                    grids, 
                    e0act: float, 
                    x0act: float,
                    apply_corrections: bool = True) -> Tuple[Constraints, jax.Array]:
    if not constraints.tconstraint:
        return constraints, levels.psi
    
    # Calculate current expectation values and variances
    expectations, variances = calculate_constraint_expectation_values(densities, constraints, grids)
    
    # Update Lagrange multipliers
    updated_constraints = update_lagrange_multipliers(
        constraints, expectations, variances, e0act, x0act
    )
    
    # Apply wavefunction corrections if requested
    corrected_psi = levels.psi
    if apply_corrections:
        corrected_psi = apply_wavefunction_corrections(levels.psi, updated_constraints, levels)
    
    # Print progress information
    _print_constraint_progress(updated_constraints, expectations, variances)
    
    return updated_constraints, corrected_psi


def _print_constraint_progress(constraints: Constraints, 
                              expectations: jax.Array, 
                              variances: jax.Array):
    """Print constraint convergence information"""
    if not constraints.tconstraint:
        return
    
    print("  Constraint status:")
    
    iconstr = 0
    
    # Q20 constraint
    if constraints.alpha20_wanted > -1e90:
        goal = constraints.goal_crank[iconstr]
        actual = expectations[iconstr]
        diff = actual - goal
        lambda_val = constraints.lambda_crank[iconstr]
        status = "✓" if abs(diff) < 0.01 else "○"
        print(f"    {status} Q20: {actual:7.4f} → {goal:7.4f} (Δ={diff:+6.3f}, λ={lambda_val:7.3f})")
        iconstr += 1
    
    # Q22 constraint
    if constraints.alpha22_wanted > -1e90:
        goal = constraints.goal_crank[iconstr]
        actual = expectations[iconstr]
        diff = actual - goal
        lambda_val = constraints.lambda_crank[iconstr]
        status = "✓" if abs(diff) < 0.01 else "○"
        print(f"    {status} Q22: {actual:7.4f} → {goal:7.4f} (Δ={diff:+6.3f}, λ={lambda_val:7.3f})")
        iconstr += 1
    
    # Principal axes constraints (abbreviated output)
    if constraints.tq_prin_axes:
        max_diff = 0.0
        for i in range(6):  # xy, xz, yz, x, y, z
            diff = abs(expectations[iconstr + i] - constraints.goal_crank[iconstr + i])
            max_diff = max(max_diff, diff)
        status = "✓" if max_diff < 0.01 else "○"
        print(f"    {status} Principal axes: max deviation = {max_diff:.4f}")


def check_constraint_convergence(constraints: Constraints, tolerance: float = 0.01) -> bool:
    if not constraints.tconstraint:
        return True
    
    for i in range(constraints.numconstraint):
        diff = abs(constraints.actual_crank[i] - constraints.goal_crank[i])
        rel_tolerance = max(tolerance, abs(constraints.goal_crank[i]) * 0.01)
        if diff > rel_tolerance:
            return False
    
    return True


def get_constraint_summary(constraints: Constraints) -> Dict[str, Any]:
    if not constraints.tconstraint:
        return {"enabled": False}
    
    summary = {
        "enabled": True,
        "numconstraint": constraints.numconstraint,
        "converged": check_constraint_convergence(constraints)
    }
    
    iconstr = 0
    if constraints.alpha20_wanted > -1e90:
        summary["Q20"] = {
            "target": float(constraints.goal_crank[iconstr]),
            "actual": float(constraints.actual_crank[iconstr]),
            "lambda": float(constraints.lambda_crank[iconstr])
        }
        iconstr += 1
    
    if constraints.alpha22_wanted > -1e90:
        summary["Q22"] = {
            "target": float(constraints.goal_crank[iconstr]),
            "actual": float(constraints.actual_crank[iconstr]),
            "lambda": float(constraints.lambda_crank[iconstr])
        }
        iconstr += 1
    
    if constraints.tq_prin_axes:
        summary["principal_axes"] = True
    
    return summary