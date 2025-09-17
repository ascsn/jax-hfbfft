import jax
import jax.numpy as jnp
from functools import partial
from dataclasses import dataclass, field
from densities import add_density
from inout import sp_properties
from levels import laplace, cdervx01, cdervy01, cdervz01
from energies import integ_energy, sum_energy
from meanfield import skyrme, hpsi01
from trivial import rpsnorm, overlap
from grids import print_grid_info
from pairs import pair
import datetime
from output import FortranOutputWriter, complete_sinfo_with_fortran_output
from collections import defaultdict
import time
from scipy.optimize import brentq

@jax.tree_util.register_dataclass
@dataclass
class Static:
    tdiag: bool = field(metadata=dict(static=True))
    tlarge: bool
    tvaryx_0: bool
    ttime: bool
    tsort: bool
    maxiter: int
    iternat: int
    iternat_start: int
    iteranneal: int
    pairenhance: float
    inibcs: int
    inidiag: int
    delstepbas: float
    e0bas: float
    outerpot: int
    radinx: float
    radiny: float
    radinz: float
    serr: float
    delesum: float
    sumflu: float
    x0dmp: float
    e0dmp: float
    x0dmpmin: float
    outertype: str = field(metadata=dict(static=True))
    hmatrix: jax.Array
    gapmatrix: jax.Array
    symcond: jax.Array
    lambda_save: jax.Array


def init_static(forces, levels, **kwargs):
    nst = max(int(levels.npsi[0]), int(levels.npsi[1]))

    default_kwargs = {
        'tdiag': False,
        'tlarge': False,
        'tvaryx_0': False,
        'ttime': False,
        'tsort': False,
        'iternat': 100,
        'iternat_start': 40,
        'iteranneal': 0,
        'pairenhance': 0.0,
        'inibcs': 30,
        'inidiag': 30,
        'delstepbas': 2.0,
        'e0bas': 10.0,
        'delesum': 0.0,
        'sumflu': 0.0,
        'outerpot': 0,
        'x0dmp': 0.45,
        'e0dmp': 100,
        'x0dmpmin': 0.45,
        'outertype': 'N',
        'hmatrix': jnp.zeros((2, nst, nst), dtype=jnp.complex128),
        'gapmatrix': jnp.zeros((2, nst, nst), dtype=jnp.complex128),
        'symcond': jnp.zeros((2, nst, nst), dtype=jnp.complex128),
        'lambda_save': jnp.zeros((2, nst, nst), dtype=jnp.complex128),
    }

    default_kwargs.update(kwargs)

    default_kwargs['x0dmpmin'] =  default_kwargs['x0dmp']

    if forces.zpe == 0:
        forces.h2m = forces.h2m.at[...].multiply((levels.mass_number - 1.0) / levels.mass_number)

    return forces, Static(**default_kwargs)


@jax.jit
def e0dmp_gt_zero(args):
    nst, iq, psin, ps1, lagrange, forces, grids, levels, meanfield, params, static = args
    h_exp = jnp.real(overlap(psin, ps1, grids.wxyz))
    
    # Use jax.lax.cond for the first conditional: (params.iteration > 1) and (not forces.tbcs)
    ps1_orthogonal = jax.lax.cond(
        (params.iteration > 1) & (~forces.tbcs),
        lambda _: ps1 - lagrange,
        lambda _: ps1 - h_exp * psin,
        operand=None
    )
    
    # Use jax.lax.cond for the mprint conditional
    should_calculate_fluct = (params.mprint > 0) & (params.iteration % params.mprint == 0)
    
    def calculate_fluct(_):
        sp_efluct1 = jnp.sqrt(rpsnorm(ps1_orthogonal, grids.wxyz))
        sp_efluct2 = jnp.sqrt(rpsnorm(lagrange - h_exp * psin, grids.wxyz))
        return sp_efluct1, sp_efluct2
    
    def use_existing_fluct(_):
        return levels.sp_efluct1[nst], levels.sp_efluct2[nst]
    
    sp_efluct1, sp_efluct2 = jax.lax.cond(
        should_calculate_fluct,
        calculate_fluct,
        use_existing_fluct,
        operand=None
    )
    
    # Use jax.lax.cond for the tbcs conditional in laplace call
    def laplace_bcs(_):
        return laplace(nst, iq, grids, forces, levels, 1.0, 0.0, ps1_orthogonal, 0.0, static.e0dmp)
    
    def laplace_hfb(_):
        return laplace(
            nst, iq, grids, forces, levels,
            levels.wocc[nst],
            levels.wguv[nst] * levels.pairwg[nst],
            ps1_orthogonal,
            jnp.max(meanfield.v_pair[iq,...]),
            static.e0dmp
        )
    
    ps2 = jax.lax.cond(
        forces.tbcs,
        laplace_bcs,
        laplace_hfb,
        operand=None
    )
    
    psin_new = psin - static.x0dmp * ps2
    return psin_new, sp_efluct1, sp_efluct2
    
@jax.jit
def grstep_helper(nst, iq, spe_mf, psin, lagrange, forces, grids, levels, meanfield, params, static):
    
    # Step 1: Apply Hamiltonian using hpsi01 (ignore 3rd return value)
    def apply_hamiltonian_bcs(_):
        ps1, psi_mf, _ = hpsi01(grids, meanfield, iq, 1.0, 0.0, psin)
        return ps1, psi_mf
    
    def apply_hamiltonian_hfb(_):
        ps1, psi_mf, _ = hpsi01(
            grids, meanfield, iq,
            levels.wstates[nst] * levels.wocc[nst],
            levels.wstates[nst] * levels.wguv[nst] * levels.pairwg[nst],
            psin,
        )
        return ps1, psi_mf
    
    ps1, psi_mf = jax.lax.cond(
        forces.tbcs,
        apply_hamiltonian_bcs,
        apply_hamiltonian_hfb,
        operand=None
    )
    
    # Step 2: Calculate new single-particle energy
    spe_mf_new = jnp.real(overlap(psin, psi_mf, grids.wxyz))
    
    # Step 3: Apply the damped gradient step using jax.lax.cond
    def apply_e0dmp_correction(_):
        return e0dmp_gt_zero(
            (nst, iq, psin, ps1, lagrange, forces, grids, levels, meanfield, params, static)
        )
    
    def apply_simple_correction(_):
        psin_new = (1.0 + static.x0dmp * (spe_mf_new - spe_mf)) * psin - static.x0dmp * ps1
        sp_efluct1 = levels.sp_efluct1[nst]
        sp_efluct2 = levels.sp_efluct2[nst]
        return psin_new, sp_efluct1, sp_efluct2
    
    psin, sp_efluct1, sp_efluct2 = jax.lax.cond(
        static.e0dmp > 0.0,
        apply_e0dmp_correction,
        apply_simple_correction,
        operand=None
    )
    
    # Step 4: Recalculate H|psi> with the new wavefunction using hpsi01 for both branches
    def recalc_hamiltonian_bcs(_):
        return hpsi01(grids, meanfield, iq, 1.0, 0.0, psin)
    
    def recalc_hamiltonian_hfb(_):
        weight1 = levels.wstates[nst] * levels.wocc[nst]
        weight2 = levels.wstates[nst] * levels.wguv[nst] * levels.pairwg[nst]
        return hpsi01(grids, meanfield, iq, weight1, weight2, psin)
    
    ps1, hmfpsi, delpsi = jax.lax.cond(
        forces.tbcs,
        recalc_hamiltonian_bcs,
        recalc_hamiltonian_hfb,
        operand=None
    )
    
    # Final calculation
    denerg = (spe_mf - spe_mf_new) / jnp.abs(spe_mf_new)
    
    return psin, psi_mf, hmfpsi, delpsi, denerg, spe_mf_new, sp_efluct1, sp_efluct2

@jax.jit
def grstep(forces, grids, levels, meanfield, params, static):
    """
    Applies a damped gradient step to all wavefunctions in parallel using jax.vmap.
    """
    
    # 1. Prepare the input arrays that will be "mapped" over.
    #    These are the arguments to grstep_helper that are different for each state.
    nsts = jnp.arange(levels.nstmax)
    iqs = levels.isospin
    spe_mfs = levels.sp_energy
    psins = levels.psi
    lagranges = levels.lagrange

    # 2. Define the vectorized version of grstep_helper using jax.vmap.
    #    in_axes specifies how to handle each argument:
    #    - 0 means "map over the first axis of this array".
    #    - None means "this argument is static, broadcast it to every call".
    vmapped_grstep_helper = jax.vmap(
        grstep_helper,
        in_axes=(0, 0, 0, 0, 0, None, None, None, None, None, None)
    )

    # 3. Call the vectorized function once on all the data in parallel.
    #    This replaces the entire Python for-loop.
    psin, psi_mf, hmfpsi, delpsi, denerg, spe_mf_new, sp_efluct1, sp_efluct2 = vmapped_grstep_helper(
        nsts, iqs, spe_mfs, psins, lagranges,
        forces, grids, levels, meanfield, params, static
    )

    # 4. Update the levels object with the new batched results.
    #    The outputs from vmap are already stacked into arrays, ready for use.
    levels.psi = levels.psi.at[...].set(psin)
    levels.hampsi = levels.hampsi.at[...].set(psi_mf)
    levels.hmfpsi = levels.hmfpsi.at[...].set(hmfpsi)
    levels.delpsi = levels.delpsi.at[...].set(delpsi)
    levels.sp_energy = levels.sp_energy.at[...].set(spe_mf_new)
    levels.sp_efluct1 = levels.sp_efluct1.at[...].set(sp_efluct1)
    levels.sp_efluct2 = levels.sp_efluct2.at[...].set(sp_efluct2)

    # 5. Track convergence metrics (same as before).
    static.sumflu = static.sumflu + jnp.sum(levels.sp_efluct1)
    static.delesum = static.delesum + jnp.sum(levels.wocc * levels.wstates * denerg)

    return levels, static

#@partial(jax.jit, static_argnames=['diagonalize', 'construct'])
def diagstep(energies, forces, grids, levels, static, diagonalize=False, construct=True, npsi_neutron = None, npsi_proton = None):

    for iq in range(2):
        
        if iq == 0:  # Neutrons
            nst = npsi_neutron  # Use the full neutron basis size
            start, end = 0, nst
        else:  # Protons
            nst = npsi_proton  # Use the full proton basis size
            # Protons start after all neutron basis states
            start = npsi_neutron
            end = start + nst
        
        # Step 1: Reshape wave functions to 2D layout
        psi_2d = jnp.reshape(
            jnp.transpose(
                levels.psi[start:end,...],
                axes=(2, 3, 4, 1, 0)
            ),
            shape=(-1, nst),
            order='F'
        )
        
        hampsi_2d = jnp.reshape(
            jnp.transpose(
                levels.hampsi[start:end,...],
                axes=(2, 3, 4, 1, 0)
            ),
            shape=(-1, nst),
            order='F'
        )
               
        # Step 2 & 3: Perform Loewdin Orthonormalization
        unitary_rho, sp_norm_diag = loewdin_orthonormalize(psi_2d, grids.wxyz, nst)
        levels.sp_norm = levels.sp_norm.at[start:end].set(sp_norm_diag)

        # Calculate lambda matrix if needed
        if diagonalize:
            lambda_lin = jnp.dot(jnp.conjugate(psi_2d.T), hampsi_2d) * grids.wxyz
            _, unitary_lam = jnp.linalg.eigh(lambda_lin, symmetrize_input=False)
            unitary = jnp.dot(unitary_rho, unitary_lam)
        else:
            unitary = unitary_rho

        # Step 4: Handle lambda diagonalization if requested
        if diagonalize:
            _, unitary_lam = jnp.linalg.eigh(lambda_lin, symmetrize_input=False)
            unitary = jnp.dot(unitary_rho, unitary_lam)
            
        else:
            unitary = unitary_rho
        
        # Step 5: Apply transformation to wave functions
        transformed_psi = jnp.dot(psi_2d, unitary)

        # Reshape back to 5D
        levels.psi = levels.psi.at[start:end,...].set(
            jnp.transpose(
                jnp.reshape(
                    transformed_psi,
                    shape=(grids.nx, grids.ny, grids.nz, 2, nst),
                    order='F'
                ),
                axes=(4, 3, 0, 1, 2)
            )
        )
        # Step 6: Construct matrices or just update Lagrange multipliers
        if construct:
            energies, levels, static = construct_hfb_matrices(
                unitary, psi_2d, energies, forces, grids, levels, static, iq, start, end, nst
            )
        else:
            # Just update Lagrange multipliers
            psi_2d_new = jnp.reshape(
                jnp.transpose(levels.psi[start:end, ...], axes=(2, 3, 4, 1, 0)),
                shape=(-1, nst), order='F'
            )
            transformed_lagrange = jnp.dot(psi_2d_new, static.lambda_save[iq,:nst,:nst])
            levels.lagrange = levels.lagrange.at[start:end,...].set(
                jnp.transpose(
                    jnp.reshape(
                        transformed_lagrange,
                        shape=(grids.nx, grids.ny, grids.nz, 2, nst), order='F'
                    ),
                    axes=(4, 3, 0, 1, 2)
                )
            )

    return energies, levels, static

@jax.jit
def loewdin_orthonormalize(psi_2d, wxyz, nst):
    # 1. Calculate the overlap matrix S = psi_dagger * psi
    rhomatr_lin = jnp.dot(jnp.conjugate(psi_2d.T), psi_2d) * wxyz
    sp_norm_diag = jnp.real(jnp.diagonal(rhomatr_lin))

    # 2. Eigendecomposition: S * v = w * v
    w, v = jnp.linalg.eigh(rhomatr_lin, symmetrize_input=False)

    # 3. Regularize small eigenvalues to prevent division by zero
    machine_epsilon = jnp.finfo(jnp.float64).eps
    eigenvalue_threshold = jnp.maximum(1e-12, machine_epsilon * jnp.max(w) * nst)
    w_safe = jnp.maximum(w, eigenvalue_threshold)

    # 4. Construct the transformation matrix U = V * W^(-1/2) * V_dagger
    w_inv_sqrt = 1.0 / jnp.sqrt(w_safe)
    v_scaled = v * w_inv_sqrt[None, :]
    unitary_rho = jnp.dot(v, jnp.conjugate(v_scaled.T))

    return unitary_rho, sp_norm_diag

#@jax.jit
def construct_hfb_matrices(unitary, psi_2d, energies, forces, grids, levels, static, iq, start, end, nst):

    # Reshape H|psi_old> and Delta|psi_old>
    hmfpsi_2d = jnp.reshape(
        jnp.transpose(levels.hmfpsi[start:end, ...], axes=(2, 3, 4, 1, 0)),
        shape=(-1, nst), order='F'
    )

    # Calculate H in the OLD basis: <psi_old|H|psi_old>
    h_matrix_old = jnp.dot(jnp.conjugate(psi_2d.T), hmfpsi_2d) * grids.wxyz
    # Apply two-sided transformation to get H in the NEW basis: U_dagger * H_old * U
    h_matrix = jnp.dot(jnp.conjugate(unitary.T), jnp.dot(h_matrix_old, unitary))

    # Repeat the same logic for the gap matrix
    gap_matrix = jnp.zeros_like(h_matrix)
    if forces.ipair != 0:
        delpsi_2d = jnp.reshape(
            jnp.transpose(levels.delpsi[start:end, ...], axes=(2, 3, 4, 1, 0)),
            shape=(-1, nst), order='F'
        )
        # Calculate G in the OLD basis
        gap_matrix_old = jnp.dot(jnp.conjugate(psi_2d.T), delpsi_2d) * grids.wxyz
        # Transform G to the NEW basis
        gap_matrix = jnp.dot(jnp.conjugate(unitary.T), jnp.dot(gap_matrix_old, unitary))

    # Store matrices and update single-particle energies
    static.hmatrix = static.hmatrix.at[iq, :nst, :nst].set(h_matrix)
    static.gapmatrix = static.gapmatrix.at[iq, :nst, :nst].set(gap_matrix)
    levels.sp_energy = levels.sp_energy.at[start:end].set(
        jnp.real(jnp.diagonal(static.hmatrix[iq, :nst, :nst]))
    )

    if forces.ipair != 0:
        levels.deltaf = levels.deltaf.at[start:end].set(
            jnp.real(jnp.diagonal(static.gapmatrix[iq, :nst, :nst])) * levels.pairwg[start:end]
        )

    # Handle HF+BCS case by keeping only the diagonal of the gap matrix
    new_gapmatrix = jax.lax.cond(
        forces.tbcs,
        # Function to execute if forces.tbcs is True
        lambda gm: gm.at[iq, :nst, :nst].set(gm[iq, :nst, :nst] * jnp.eye(nst, dtype=gm.dtype)),
        # Function to execute if forces.tbcs is False
        lambda gm: gm,
        # The operand to pass to the chosen function
        static.gapmatrix
    )
    # Update the static object with the result from jax.lax.cond
    static.gapmatrix = new_gapmatrix

    # --- Calculate lambda, symcond, and Lagrange multipliers ---
    weight = levels.wocc[start:end] * levels.wstates[start:end]
    weightuv = levels.wguv[start:end] * levels.pairwg[start:end] * levels.wstates[start:end]
    lambda_temp = (
        weight[:, jnp.newaxis] * static.hmatrix[iq, :nst, :nst] -
        weightuv[:, jnp.newaxis] * static.gapmatrix[iq, :nst, :nst]
    )

    # Asymmetric part for fluctuation calculation
    lambda_asym = (0.5 + 0.5j) * (lambda_temp - jnp.conjugate(lambda_temp.T))
    energies.efluct1q = energies.efluct1q.at[iq].set(jnp.max(jnp.abs(lambda_asym)))
    energies.efluct2q = energies.efluct2q.at[iq].set(jnp.sqrt(jnp.sum(jnp.abs(lambda_asym)**2) / nst**2))
    static.symcond = static.symcond.at[iq, :nst, :nst].set(lambda_asym)

    # Symmetric part for Lagrange multipliers
    lambda_sym = (0.5 + 0.5j) * (lambda_temp + jnp.conjugate(lambda_temp.T))
    static.lambda_save = static.lambda_save.at[iq, :nst, :nst].set(lambda_sym)

    # Get the NEW wavefunctions to compute the Lagrange field
    psi_2d_new = jnp.reshape(
        jnp.transpose(levels.psi[start:end, ...], axes=(2, 3, 4, 1, 0)),
        shape=(-1, nst), order='F'
    )
    transformed_lagrange = jnp.dot(psi_2d_new, lambda_sym)
    levels.lagrange = levels.lagrange.at[start:end, ...].set(
        jnp.transpose(
            jnp.reshape(
                transformed_lagrange,
                shape=(grids.nx, grids.ny, grids.nz, 2, nst), order='F'
            ),
            axes=(4, 3, 0, 1, 2)
        )
    )

    # Update total convergence criteria
    energies.efluct1 = energies.efluct1.at[0].set(jnp.max(energies.efluct1q))
    energies.efluct2 = energies.efluct2.at[0].set(jnp.average(energies.efluct2q))

    return energies, levels, static


def harmosc(grids, levels, params, static):
    from trivial import rpsnorm

    # Initialize arrays
    psi = jnp.zeros_like(levels.psi)
    
    # Initialize occupation for the FULL basis
    wocc = jnp.zeros(levels.nstmax)
    # Only occupy the lowest energy states
    wocc = wocc.at[:levels.nneut].set(1.0)  # First nneut neutron states
    proton_start = int(levels.npsi[0])
    wocc = wocc.at[proton_start:proton_start + levels.nprot].set(1.0)  # First nprot proton states
    
    wguv = jnp.zeros_like(wocc)
    
    
    # Create base Gaussian - match FORTRAN
    x_mesh = grids.x[:, jnp.newaxis, jnp.newaxis]
    y_mesh = grids.y[jnp.newaxis, :, jnp.newaxis]  
    z_mesh = grids.z[jnp.newaxis, jnp.newaxis, :]
    
    xx2 = (x_mesh / static.radinx)**2
    y2 = (y_mesh / static.radiny)**2
    zz2 = (z_mesh / static.radinz)**2
    temp = xx2 + y2 + zz2
    gaussian = jnp.exp(-temp)
    

    # Normalize Gaussian using rpsnorm
    anorm = rpsnorm(jnp.stack([gaussian, jnp.zeros_like(gaussian)]), grids.wxyz)
    gaussian = gaussian / jnp.sqrt(anorm)
    
    # Generate shell structure
    # Pre-allocate quantum number array for the full basis
    nshell = jnp.zeros((3, levels.nstmax), dtype=jnp.int32)
        
    nst = 0  # Global state counter (0-based in Python)
    
    # Loop over isospins
    for iq in range(2):
        #Use full basis sizes, not just occupied particle numbers
        if iq == 0:
            nps = int(levels.npsi[0])  # neutron basis size
        else:
            nps = int(levels.npsi[1])  # proton basis size   
        
        # Track where this isospin starts in global array
        nst_start = nst
        
        # Generate quantum numbers exactly like FORTRAN
        done = False
        for ka in range(nps + 10):
            if done:
                break
            for k in range(ka + 1): 
                if done:
                    break
                for j in range(ka + 1):
                    if done:
                        break
                    for i in range(ka + 1):
                        if done:
                            break
                        if ka == i + j + k:
                            for is_spin in range(2):
                                #  Exit condition should be relative to isospin start
                                states_in_this_isospin = nst - nst_start + 1  # +1 because we're about to add one
                                if states_in_this_isospin > nps:  # FORTRAN: IF(nst>nps) EXIT ka_loop
                                    done = True
                                    break
                                    
                                if nst < levels.nstmax:  # bounds check
                                    nshell = nshell.at[0, nst].set(i)  # FORTRAN: nshell(1,nst)=i
                                    nshell = nshell.at[1, nst].set(j)  # FORTRAN: nshell(2,nst)=j  
                                    nshell = nshell.at[2, nst].set(k)  # FORTRAN: nshell(3,nst)=k
                                    nst += 1  # FORTRAN: nst=nst+1
                                else:
                                    done = True
                                    break
        
        states_generated = nst - nst_start
    
    
    # Now initialize ALL states in the basis
    for iq in range(2):
        if iq == 0:
            # Neutron states: indices 0 to npsi[0]-1
            nst_start = 0
            nst_end = int(levels.npsi[0])
        else:
            # Proton states: indices npsi[0] to nstmax-1  
            nst_start = int(levels.npsi[0])
            nst_end = levels.nstmax
        
        for nst in range(nst_start, min(nst_end, levels.nstmax)):
            if nst == nst_start:
                # Lowest state: pure Gaussian in first spin component
                psi = psi.at[nst, 0, :, :, :].set(gaussian)
                psi = psi.at[nst, 1, :, :, :].set(0.0)
                
                # Normalize
                anorm = rpsnorm(psi[nst], grids.wxyz)
                psi = psi.at[nst].set(psi[nst] / jnp.sqrt(anorm))
                
                
            else:
                # Higher states: Gaussian * polynomial
                # Spin component assignment exactly like FORTRAN
                # FORTRAN: is=MOD(nst-npmin(iq),2)+1
                # In 0-based indexing: is_component = (nst - nst_start) % 2
                is_component = (nst - nst_start) % 2
                
                # Get quantum numbers (nst is 0-based, nshell was filled correctly)
                i_qn = nshell[0, nst]
                j_qn = nshell[1, nst]  
                k_qn = nshell[2, nst]
                
                # Create polynomial factors exactly like FORTRAN
                if i_qn == 0:
                    xx = jnp.ones_like(grids.x)
                else:
                    xx = grids.x ** i_qn
                    
                if j_qn == 0:
                    yy = jnp.ones_like(grids.y)
                else:
                    yy = grids.y ** j_qn
                    
                if k_qn == 0:
                    zz = jnp.ones_like(grids.z)
                else:
                    zz = grids.z ** k_qn
                
                # Create 3D polynomial
                xx = xx[:, jnp.newaxis, jnp.newaxis]
                yy = yy[jnp.newaxis, :, jnp.newaxis]
                zz = zz[jnp.newaxis, jnp.newaxis, :]
                polynomial = xx * yy * zz
                
                # Create wavefunction: Gaussian * polynomial
                wave_func = gaussian * polynomial
                
                # Set in appropriate spin component
                psi = psi.at[nst, is_component, :, :, :].set(wave_func)
                psi = psi.at[nst, 1-is_component, :, :, :].set(0.0)
                
                # Normalize using rpsnorm like FORTRAN
                anorm = rpsnorm(psi[nst], grids.wxyz)
                if anorm > 1e-12:  # Avoid division by zero
                    psi = psi.at[nst].set(psi[nst] / jnp.sqrt(anorm))
                
    
    # Update levels with the corrected data
    levels.psi = psi
    levels.wocc = wocc
    levels.wguv = wguv
    
    
    return levels


def sinfo(coulomb, densities, energies, forces, grids, levels, meanfield, moment, params, static, pairs, printing=True):
    """
    Calculate data for informative output and write to appropriate files.
    """
    # Calculate static observables for printout
    from energies import integ_energy, sum_energy
    
    energies = integ_energy(coulomb, densities, energies, forces, grids, levels, params, pairs)
    energies = sum_energy(energies, levels, meanfield, pairs)
    
    if printing:
        
        # Print iteration information
        print(f"\n***** Iteration {params.iteration} *************************************")
        print(f"Total energy: {energies.ehf:.4f} MeV  Total kinetic energy: {energies.tke:.4f} MeV")
        print(f"c.m. energy:  {energies.e_zpe:.4f} MeV")
        print(f"de/e:      {static.delesum:.5e}      h**2  fluct.:    {energies.efluct1[0]:.5e} MeV, h*hfluc.:    {energies.efluct2[0]:.5e} MeV")
        print(f"Rearrangement E: {energies.e3corr:.5e} MeV. Pairing.Rearr.: {meanfield.ecorrp:.5e} MeV. Coul.Rearr.: {energies.ecorc:.5e} MeV")
        
        # Print energies integrated from density functional
        print("\nEnergies integrated from density functional:****************************************************")
        print(f"Total: {energies.ehfint:.6e} MeV. t0 part: {energies.ehf0:.6e} MeV. t1 part: {energies.ehf1:.6e} MeV. t2 part: {energies.ehf2:.6e} MeV.")
        print(f"t3 part: {energies.ehf3:.6e} MeV. t4 part: {energies.ehfls:.6e} MeV. Coulomb: {energies.ehfc:.6e} MeV.")
        print("*******************************************************************************************")
        print(f"Crho0:   {energies.ehfCrho0:.6e} MeV. Crho1:   {energies.ehfCrho1:.6e} MeV.")
        print(f"Cdrho0:  {energies.ehfCdrho0:.6e} MeV. Cdrho1:  {energies.ehfCdrho1:.6e} MeV.")
        print(f"Ctau0:   {energies.ehfCtau0:.6e} MeV. Ctau1:   {energies.ehfCtau1:.6e} MeV.")
        print(f"CdJ0:    {energies.ehfCdJ0:.6e} MeV. CdJ1:    {energies.ehfCdJ1:.6e} MeV.")
        print("*******************************************************************************************")
        
        # Print pairing information if relevant
        if forces.ipair != 0:
            print("\n--- Pairing Information ---")
            print('          e_ferm      e_pair     <uv delta>   <v2 delta>    aver_force ')
            for iq in range(2):
                isospin_label = "Neutrons" if iq == 0 else "Protons"
                print(f"{isospin_label:<9s} {pairs.eferm[iq]:11.4f} {pairs.epair[iq]:11.4f} "
                    f"{pairs.avdelt[iq]:11.4f} {pairs.avdeltv2[iq]:11.4f} {pairs.avg[iq]:11.4f}")

        
        # Print single-particle state details
        print("Neutron Single Particle States:")
        print("#  Par   v**2   var_h1   var_h2    Norm     Ekin    Energy    Lx      Ly      Lz     Sx     Sy     Sz    delta    E_bar_k    tke_np")
        
        tke_np = 0.0
        for i in range(int(levels.npsi[0])):
            tke_np += levels.wocc[i] * levels.sp_kinetic[i] * levels.wstates[i]
            print(f"{i:3d} {levels.sp_parity[i]:4.0f} {levels.wocc[i]:8.5f} {levels.sp_efluct1[i]:9.5f} {levels.sp_efluct2[i]:9.5f} {levels.sp_norm[i]:9.6f} {levels.sp_kinetic[i]:8.3f} {levels.sp_energy[i]:10.3f} {levels.sp_orbital[i, 0]:8.3f} {levels.sp_orbital[i, 1]:8.3f} {levels.sp_orbital[i, 2]:8.3f} {levels.sp_spin[i, 0]:7.3f} {levels.sp_spin[i, 1]:7.3f} {levels.sp_spin[i, 2]:7.3f} {levels.deltaf[i]:10.4f} {levels.wocc[i] * levels.sp_kinetic[i] * levels.wstates[i]:10.4f} {tke_np:10.4f}")
        
        print(f"Ebar_n: {tke_np:.4f}")
        
        print("Proton Single Particle States:")
        print("#  Par   v**2   var_h1   var_h2    Norm     Ekin    Energy    Lx      Ly      Lz     Sx     Sy     Sz    delta    E_bar_k    tke_np")
        
        tke_np = 0.0
        proton_start_index = int(levels.npsi[0])
        for i in range(proton_start_index, proton_start_index + int(levels.npsi[1])):
            tke_np += levels.wocc[i] * levels.sp_kinetic[i] * levels.wstates[i]
            print(f"{i:3d} {levels.sp_parity[i]:4.0f} {levels.wocc[i]:8.5f} {levels.sp_efluct1[i]:9.5f} {levels.sp_efluct2[i]:9.5f} {levels.sp_norm[i]:9.6f} {levels.sp_kinetic[i]:8.3f} {levels.sp_energy[i]:10.3f} {levels.sp_orbital[i, 0]:8.3f} {levels.sp_orbital[i, 1]:8.3f} {levels.sp_orbital[i, 2]:8.3f} {levels.sp_spin[i, 0]:7.3f} {levels.sp_spin[i, 1]:7.3f} {levels.sp_spin[i, 2]:7.3f} {levels.deltaf[i]:10.4f} {levels.wocc[i] * levels.sp_kinetic[i] * levels.wstates[i]:10.4f} {tke_np:10.4f}")
        
        print(f"Ebar_p: {tke_np:.4f}")
    
    return energies



def planewaves(grids, levels, params, static):
   
    # Initialize wave functions to zero
    psi = jnp.zeros_like(levels.psi, dtype=jnp.complex128)
    
    # Update occupation numbers
    wocc = jnp.zeros(levels.nstmax)
    wocc = wocc.at[:levels.nneut].set(1.0)
    wocc = wocc.at[levels.nneut:levels.nneut + levels.nprot].set(1.0)
    wguv = jnp.zeros_like(wocc)
    
    # Generate all possible k values for plane waves
    size = 7
    ki = jnp.zeros((3, 8 * size**3), dtype=jnp.int32)
    
    idx = 0
    for kx in range(-size//2, size//2 + 1):
        for ky in range(-size//2, size//2 + 1):
            for kz in range(-size//2, size//2 + 1):
                ki = ki.at[:, idx].set(jnp.array([kx, ky, kz]))
                idx += 1
    
    # Calculate energies for each k-value
    def calc_energy(k):
        dx, dy, dz = grids.dx, grids.dy, grids.dz
        return (params.hbc**2) / (2 * forces.nucleon_mass) * (
            ((2 * jnp.pi * k[0] + static.bangx) / (grids.nx * dx))**2 +
            ((2 * jnp.pi * k[1] + static.bangy) / (grids.ny * dy))**2 +
            ((2 * jnp.pi * k[2] + static.bangz) / (grids.nz * dz))**2
        )
    
    calc_energy_vmap = jax.vmap(calc_energy, in_axes=1)
    energies = calc_energy_vmap(ki[:, :idx])
    
    # Sort k-values by energy
    sorted_indices = jnp.argsort(energies)
    sorted_ki = ki[:, sorted_indices]
    
    # Initialize states with plane waves
    # For each isospin (neutrons and protons)
    for iq in range(2):
        if iq == 0:
            start, end = 0, levels.nneut
        else:
            start, end = levels.nneut, levels.nneut + levels.nprot
        
        # Initialize wave functions with plane waves
        state_idx = 0
        k_idx = 0
        
        while state_idx < (end - start):
            k = sorted_ki[:, k_idx]
            
            # Skip if k is already used (would need a more complex check in JAX)
            # For simplicity, we'll just use the next k
            
            # Initialize with plane wave for spin up
            if state_idx < (end - start):
                psi = initialize_plane_wave(psi, start + state_idx, k, 1, grids, static)
                state_idx += 1
            
            # Initialize with plane wave for spin down
            if state_idx < (end - start):
                psi = initialize_plane_wave(psi, start + state_idx, k, -1, grids, static)
                state_idx += 1
            
            k_idx += 1
    
    # Update levels
    levels = levels._replace(
        psi=psi,
        wocc=wocc,
        wguv=wguv
    )
    
    return levels

def initialize_plane_wave(psi, state_idx, k, spin, grids, static):
    nx, ny, nz = grids.nx, grids.ny, grids.nz
    
    # Generate mesh indices
    ix = jnp.arange(nx)[:, jnp.newaxis, jnp.newaxis]
    iy = jnp.arange(ny)[jnp.newaxis, :, jnp.newaxis]
    iz = jnp.arange(nz)[jnp.newaxis, jnp.newaxis, :]
    
    # Calculate phase factors
    facx = ix * ((2.0 * jnp.pi * k[0] + static.bangx) / nx)
    facy = iy * ((2.0 * jnp.pi * k[1] + static.bangy) / ny)
    facz = iz * ((2.0 * jnp.pi * k[2] + static.bangz) / nz)
    
    # Compute complex exponentials
    expx = jnp.cos(facx) + 1j * jnp.sin(facx)
    expy = jnp.cos(facy) + 1j * jnp.sin(facy)
    expz = jnp.cos(facz) + 1j * jnp.sin(facz)
    
    # Combine factors
    plane_wave = expx * expy * expz
    
    # Set spin components
    if spin > 0:
        psi = psi.at[state_idx, 0, :, :, :].set(plane_wave)
        psi = psi.at[state_idx, 1, :, :, :].set(0.0)
    else:
        psi = psi.at[state_idx, 1, :, :, :].set(plane_wave)
        psi = psi.at[state_idx, 0, :, :, :].set(0.0)
    
    # Normalize
    norm = jnp.sqrt(jnp.sum(jnp.abs(psi[state_idx])**2) * grids.wxyz)
    psi = psi.at[state_idx].divide(norm)
    
    
    return psi

@jax.jit
def hfb_natural(forces, grids, levels, meanfield, params, static, isospin, iternat):
    """
    Performs sub-iterations in configuration space to accelerate convergence.
    """
    if isospin == 0:
        start, end = 0, levels.nneut
        nst = levels.nneut
    else:
        start, end = levels.nneut, levels.nneut + levels.nprot
        nst = levels.nprot
    
    # Initialize configuration matrix cm as identity matrix
    cm = jnp.eye(nst, dtype=jnp.complex128)
    
    # Prepare damping coefficient
    if static.e0bas == 0.0:
        # Global damping coefficient
        energy_range = jnp.max(levels.sp_energy[start:end]) - jnp.min(levels.sp_energy[start:end])
        dampstep = jnp.full(nst, static.delstepbas / energy_range)
    else:
        # State-dependent damping coefficient
        min_energy = levels.sp_energy[start]
        dampstep = static.delstepbas / (levels.sp_energy[start:end] - min_energy + static.e0bas)
    
    # Extract relevant matrices for this isospin
    hmat = static.hmatrix[isospin, :nst, :nst]
    gapmat = static.gapmatrix[isospin, :nst, :nst]
    
    # Start sub-iterations
    for iter_sub in range(iternat):
        # Step 1: Gradient step using symcond matrix
        # cm[j,:] = cm[j,:] - dampstep[j] * symcond[j,:]
        for j in range(nst):
            cm = cm.at[j, :].set(
                cm[j, :] - dampstep[j] * static.symcond[isospin, j, :nst]
            )
        
        # Step 2: QR factorization for orthonormalization
        q, r = jnp.linalg.qr(cm)
        cm = q
        
        # Step 3: Calculate matrix transformations (calc_transform equivalent)
        # htran = cm† * hmat * cm
        htran = jnp.dot(jnp.dot(jnp.conjugate(cm.T), hmat), cm)
        gaptran = jnp.dot(jnp.dot(jnp.conjugate(cm.T), gapmat), cm)
        
        # Also calculate hcm = hmat * cm and gapcm = gapmat * cm for later use
        hcm = jnp.dot(hmat, cm)
        gapcm = jnp.dot(gapmat, cm)
        
        # Step 4: Update single-particle energies and pairing gaps
        should_update = (forces.ipair != 0) or (iter_sub == iternat - 1)
        
        if should_update:
            # Update sp_energy from diagonal of htran
            new_sp_energies = jnp.real(jnp.diag(htran))
            levels.sp_energy = levels.sp_energy.at[start:end].set(new_sp_energies)
            
            # Update deltaf from diagonal of gaptran if pairing is active
            if forces.ipair != 0:
                new_deltaf = jnp.real(jnp.diag(gaptran)) * levels.pairwg[start:end]
                levels.deltaf = levels.deltaf.at[start:end].set(new_deltaf)
            
            # Step 5: Redo pairing calculation if needed
            if forces.ipair != 0:
                if iter_sub != iternat - 1:
                    # Sub-iteration pairing (for single isospin)
                    levels = pair_subiter_single_isospin(levels, forces, isospin)
                else:
                    # Full pairing calculation at last iteration
                    levels, _ = pair(levels, meanfield, forces, params, grids, None)
        
        # Step 6: Calculate lambda matrix and update symcond for next iteration
        if iter_sub < iternat - 1:  # Not the last iteration
            # Calculate lambda matrix (calc_lambda equivalent)
            lambda_matrix = calculate_lambda_matrix(htran, gaptran, levels, start, end)
            
            # Calculate cmlambda = cm * lambda
            cmlambda = jnp.dot(cm, lambda_matrix)
            
            # Update symcond matrix for next gradient step
            # symcond = weight*hcm - weightuv*gapcm - cmlambda
            new_symcond = jnp.zeros((nst, nst), dtype=jnp.complex128)
            for j in range(nst):
                idx = start + j
                weight = levels.wocc[idx] * levels.wstates[idx]
                weightuv = levels.wguv[idx] * levels.pairwg[idx] * levels.wstates[idx]
                
                new_symcond = new_symcond.at[:, j].set(
                    weight * hcm[:, j] - weightuv * gapcm[:, j] - cmlambda[:, j]
                )
            
            # Update symcond in static
            static.symcond = static.symcond.at[isospin, :nst, :nst].set(new_symcond)
    
    # Step 7: Final updates after all sub-iterations
    # Update wave functions in coordinate space
    levels = update_wavefunctions_coordinate_space(levels, cm, htran, start, end, grids)
    
    # Update hmatrix and gapmatrix with final transformed matrices
    static.hmatrix = static.hmatrix.at[isospin, :nst, :nst].set(htran)
    static.gapmatrix = static.gapmatrix.at[isospin, :nst, :nst].set(gaptran)
    
    # Save lambda matrix for constraint calculations
    lambda_final = calculate_lambda_matrix(htran, gaptran, levels, start, end)
    static.lambda_save = static.lambda_save.at[isospin, :nst, :nst].set(lambda_final)
    
    return levels, static

def calculate_lambda_matrix(hmatrix, gapmatrix, levels, start, end):
    """Calculate lambda matrix following FORTRAN calc_lambda logic"""
    nst = end - start
    lambda_temp = jnp.zeros((nst, nst), dtype=jnp.complex128)
    
    for j in range(nst):
        idx = start + j
        weight = levels.wocc[idx] * levels.wstates[idx]
        weightuv = levels.wguv[idx] * levels.pairwg[idx] * levels.wstates[idx]
        lambda_temp = lambda_temp.at[:, j].set(
            weight * hmatrix[:, j] - weightuv * gapmatrix[:, j]
        )
    
    # Lambda = 0.5 * (lambda_temp + lambda_temp†)
    lambda_matrix = 0.5 * (lambda_temp + jnp.conjugate(lambda_temp.T))
    
    return lambda_matrix

def update_wavefunctions_coordinate_space(levels, cm, lambda_matrix, start, end, grids):
    """Update wave functions from configuration space transformation back to coordinate space"""
    nst = end - start
    
    # Transform wave functions: psi_new = psi_old * cm
    psi_old = levels.psi[start:end]
    psi_new = jnp.zeros_like(psi_old)
    
    for i in range(nst):
        for j in range(nst):
            psi_new = psi_new.at[i].add(psi_old[j] * cm[j, i])
    
    # Update lagrange: lagrange = psi_transformed * lambda
    lagrange_new = jnp.zeros_like(psi_old)
    for i in range(nst):
        for j in range(nst):
            lagrange_new = lagrange_new.at[i].add(psi_new[j] * lambda_matrix[j, i])
    
    # Update levels object
    levels.psi = levels.psi.at[start:end].set(psi_new)
    levels.lagrange = levels.lagrange.at[start:end].set(lagrange_new)
    
    return levels

def lastdiag(forces, grids, levels, params, static, pairs, isospin):
    # Determine range of states for this isospin
    if isospin == 0:
        start, end = 0, levels.nneut
        nst = levels.nneut
        npmin = 0
        npsi = levels.nneut
    else:
        start, end = levels.nneut, levels.nneut + levels.nprot
        nst = levels.nprot
        npmin = levels.nneut
        npsi = levels.nneut + levels.nprot
    
    # Use actual Fermi energy
    eferm = pairs.eferm[isospin]
    
    # Initialize matrices
    hfb_matrix = jnp.zeros((nst, nst), dtype=jnp.complex128)
    coeffmatrix = jnp.zeros((nst, nst), dtype=jnp.complex128)
    
    # Build HFB matri
    for i in range(nst):
        for j in range(nst):
            # Get global indices
            global_i = start + i
            global_j = start + j
            
            # Calculate xi and eta factors
            xi = (-jnp.sqrt(levels.wocc[global_i] * levels.wocc[global_j]) + 
                  jnp.sqrt((1.0 - levels.wocc[global_i]) * (1.0 - levels.wocc[global_j])))
            
            eta = (jnp.sqrt(levels.wocc[global_i] * (1.0 - levels.wocc[global_j])) + 
                   jnp.sqrt(levels.wocc[global_j] * (1.0 - levels.wocc[global_i])))
            
            # Get matrix elements from stored matrices
            h_term = static.hmatrix[isospin, i, j]
            delta_term = static.gapmatrix[isospin, i, j]
            
            # Apply Fermi energy shift to diagonal elements only
            if i == j:
                h_term = h_term - eferm
            
            # Build HFB matrix: H_ij = (h_mf - eferm * delta_ij) * xi_ij + Delta_ij * eta_ij
            hfb_matrix = hfb_matrix.at[i, j].set(h_term * xi + delta_term * eta)
    
    # Diagonalize the HFB matrix
    try:
        hfbegval, unitary_hfb = jnp.linalg.eigh(hfb_matrix)
        
        # Build coefficient matrix for lower component calculation
        noffset = npmin
        coeffmatrix = jnp.zeros((nst, nst), dtype=jnp.complex128)
        
        for i in range(nst):
            for j in range(nst):
                global_i = start + i  
                global_j = start + j
                
                if (global_i - global_j == 1) and (global_i % 2 == 0):
                    coeffmatrix = coeffmatrix.at[i, j].set(-jnp.sqrt(levels.wocc[global_i]))
                
                if (global_j - global_i == 1) and (global_i % 2 == 1):
                    coeffmatrix = coeffmatrix.at[i, j].set(jnp.sqrt(levels.wocc[global_i]))
        
        # Calculate lower component matrix
        lowmatrix = jnp.matmul(coeffmatrix, unitary_hfb)
        
        # Calculate quasi-particle norms
        qp_norm = jnp.zeros(nst, dtype=jnp.float64)
        
        for j in range(nst):  # Loop over eigenstates
            for i in range(nst):  # Loop over components
                # Square of lower component
                lowmatrix_squared = lowmatrix[i, j] * jnp.conj(lowmatrix[i, j])
                qp_norm = qp_norm.at[j].set(qp_norm[j] + jnp.real(lowmatrix_squared))
        
        qp_norm = jnp.sqrt(qp_norm)
        
        # Sort eigenvalues and norms together (eigenvalues should already be sorted from eigh)
        sorted_indices = jnp.argsort(hfbegval)
        hfbegval_sorted = hfbegval[sorted_indices]
        qp_norm_sorted = qp_norm[sorted_indices]
        
        return hfbegval_sorted, qp_norm_sorted
        
    except Exception as e:
        print(f"Error in lastdiag diagonalization: {e}")
        # Return fallback values
        return jnp.zeros(nst), jnp.zeros(nst)
        
    except Exception as e:
        print(f"ERROR: Failed to diagonalize HFB matrix: {e}")
        print(f"Matrix condition number: {jnp.linalg.cond(hfb_matrix)}")
        print(f"Matrix is finite: {jnp.all(jnp.isfinite(hfb_matrix))}")
        
        # Return dummy values on failure
        return jnp.ones(nst) * 1000.0, jnp.ones(nst)
    
def pair_subiter_single_isospin(levels, forces, pairs_data, isospin):
    """
    Solves the pairing problem for only one isospin, used in sub-iterations.
    Based on the FORTRAN pair_subiter subroutine.
    
    Args:
        levels: Levels object with current single-particle states
        forces: Forces object with pairing parameters
        pairs_data: Pairs object to store pairing results
        isospin: 0 for neutrons, 1 for protons
    
    Returns:
        Updated levels object
    """
    
    # Determine particle number based on isospin
    if isospin == 1:  # Protons
        particle_number = float(levels.nprot)
    else:  # Neutrons (isospin == 0)
        particle_number = float(levels.nneut)
    
    # Skip pairgap calculation - gaps are already calculated in the sub-iteration
    # This is the key difference from the full pair() function
    
    # Call pairdn equivalent for this specific isospin
    levels = _pairdn_single_isospin(isospin, particle_number, levels, forces, pairs_data)
    
    return levels

def _pairdn_single_isospin(iq, particle_number, levels, forces, pairs_data):
    """
    Equivalent to FORTRAN pairdn but for single isospin.
    Determines pairing solution using Brent's method to find correct Fermi energy.
    """
    
    # Get mask for this isospin
    mask = (levels.isospin == iq)
    indices = jnp.where(mask)[0]
    
    if len(indices) == 0:
        return levels
    
    # Extract data for this isospin only
    sp_energy_iso = levels.sp_energy[mask]
    deltaf_iso = levels.deltaf[mask]
    wstates_iso = levels.wstates[mask]
    pairwg_iso = levels.pairwg[mask]
    
    # Start with non-pairing value of Fermi energy
    # Following FORTRAN logic: it = npmin(iq) + NINT(particle_number) - 1
    n_filled = int(round(particle_number)) - 1
    if n_filled >= 0 and n_filled < len(sp_energy_iso) - 1:
        eferm_initial = 0.5 * (sp_energy_iso[n_filled] + sp_energy_iso[n_filled + 1])
    else:
        eferm_initial = jnp.mean(sp_energy_iso)
    
    # Use Brent's method to find correct Fermi energy
    def objective_func(eferm_trial):
        particle_num = _bcs_occupation_single_isospin(
            eferm_trial, sp_energy_iso, deltaf_iso, wstates_iso
        )
        return float(particle_num) - particle_number
    
    try:
        eferm_found = brentq(
            objective_func, 
            a=-100.0, 
            b=100.0, 
            xtol=1e-14, 
            rtol=1e-16
        )
    except ValueError:
        # If Brent's method fails, use initial estimate
        eferm_found = float(eferm_initial)
    
    # Calculate final BCS occupation numbers
    edif = sp_energy_iso - eferm_found
    deltaf_squared = deltaf_iso * pairwg_iso  # Include pairwg in deltaf
    equasi = jnp.sqrt(edif**2 + deltaf_squared**2)
    
    # Avoid division by zero
    equasi_safe = jnp.maximum(equasi, 1e-20)
    
    # BCS occupation probability v_k^2
    v2 = 0.5 * (1.0 - edif / equasi_safe)
    
    # Clip to avoid exactly 0 or 1 (following FORTRAN smallp parameter)
    smallp = 1e-6
    v2 = jnp.clip(v2, smallp, 1.0 - smallp)
    
    # Calculate u_k * v_k = sqrt(v_k^2 * (1 - v_k^2))
    wguv_new = jnp.sqrt(jnp.maximum(v2 * (1.0 - v2), smallp))
    
    # Apply soft cutoffs if configured
    wstates_new = wstates_iso
    pairwg_new = pairwg_iso
    
    if forces.pair_cutoff[iq] > 0.0:
        ecut_pair = eferm_found + forces.pair_cutoff[iq]
        cutwid_pair = forces.softcut_range * forces.pair_cutoff[iq]
        pairwg_cutoff = _soft_cutoff(sp_energy_iso, ecut_pair, cutwid_pair)
        pairwg_new = pairwg_new * pairwg_cutoff
    
    if forces.state_cutoff[iq] > 0.0:
        ecut_state = eferm_found + forces.state_cutoff[iq]
        cutwid_state = forces.softcut_range * forces.state_cutoff[iq]
        wstates_cutoff = _soft_cutoff(sp_energy_iso, ecut_state, cutwid_state)
        wstates_new = wstates_new * wstates_cutoff
    
    # Calculate pairing statistics
    xsmall = 1e-20
    vol = 0.5 * wguv_new * wstates_new  # No pairwg here as noted in FORTRAN
    
    sumuv = jnp.sum(vol)
    sumuv = jnp.maximum(sumuv, xsmall)
    
    sumduv = jnp.sum(vol * deltaf_iso)
    sumv2 = jnp.sum(v2 * wstates_new)
    sumdv2 = jnp.sum(deltaf_iso * v2 * wstates_new)
    
    # Update pairs_data for this isospin
    pairs_data.eferm = pairs_data.eferm.at[iq].set(eferm_found)
    pairs_data.avdelt = pairs_data.avdelt.at[iq].set(sumduv / sumuv)
    pairs_data.avdeltv2 = pairs_data.avdeltv2.at[iq].set(sumdv2 / sumv2)
    pairs_data.epair = pairs_data.epair.at[iq].set(sumduv)
    pairs_data.avg = pairs_data.avg.at[iq].set(sumduv / (sumuv * sumuv))
    
    # Update levels object for this isospin
    levels.wocc = levels.wocc.at[mask].set(v2)
    levels.wguv = levels.wguv.at[mask].set(wguv_new)
    
    # Update full arrays if cutoffs were applied
    if forces.pair_cutoff[iq] > 0.0:
        levels.pairwg = levels.pairwg.at[mask].set(pairwg_new)
    
    if forces.state_cutoff[iq] > 0.0:
        levels.wstates = levels.wstates.at[mask].set(wstates_new)
    
    return levels

def _bcs_occupation_single_isospin(efermi_trial, sp_energy, deltaf, wstates):
    """
    Calculates particle number for a given trial Fermi energy (single isospin).
    """
    edif = sp_energy - efermi_trial
    equasi = jnp.sqrt(edif**2 + deltaf**2)
    
    # Avoid division by zero
    equasi_safe = jnp.maximum(equasi, 1e-20)
    
    # BCS occupation probability
    wocc_trial = 0.5 * (1.0 - edif / equasi_safe)
    
    # Clip to avoid exactly 0 or 1
    smal = 1e-10
    wocc_trial = jnp.clip(wocc_trial, smal, 1.0 - smal)
    
    # Sum occupations weighted by wstates
    particle_number = jnp.sum(wocc_trial * wstates)
    
    return particle_number

def _soft_cutoff(sp_energy, ecut, cutwid):
    """
    Calculates soft cutoff as Fermi profile.
    """
    return 1.0 / (1.0 + jnp.exp((sp_energy - ecut) / cutwid))

@jax.jit
def sort_states(forces, grids, levels, isospin):
    """
    Sort single-particle states according to sp_energy.
    
    Args:
        forces: Forces parameters
        grids: Grid parameters
        levels: Levels parameters
        isospin: Isospin value (0 for neutrons, 1 for protons)
        
    Returns:
        levels: Updated levels with sorted states
    """
    if isospin == 0:
        start, end = 0, levels.nneut
    else:
        start, end = levels.nneut, levels.nneut + levels.nprot
        
    # Extract indices corresponding to this isospin
    state_indices = jnp.arange(start, end)
    
    # Sort indices based on single-particle energies
    sorted_indices = jnp.argsort(levels.sp_energy[start:end])
    sorted_global_indices = state_indices[sorted_indices]
    
    # Create sorted versions of arrays in levels
    sorted_psi = levels.psi.at[start:end].set(levels.psi[sorted_global_indices])
    sorted_lagrange = levels.lagrange.at[start:end].set(levels.lagrange[sorted_global_indices])
    sorted_sp_energy = levels.sp_energy.at[start:end].set(levels.sp_energy[sorted_global_indices])
    sorted_deltaf = levels.deltaf.at[start:end].set(levels.deltaf[sorted_global_indices])
    sorted_sp_norm = levels.sp_norm.at[start:end].set(levels.sp_norm[sorted_global_indices])
    sorted_sp_efluct1 = levels.sp_efluct1.at[start:end].set(levels.sp_efluct1[sorted_global_indices])
    sorted_sp_efluct2 = levels.sp_efluct2.at[start:end].set(levels.sp_efluct2[sorted_global_indices])
    sorted_wocc = levels.wocc.at[start:end].set(levels.wocc[sorted_global_indices])
    sorted_wguv = levels.wguv.at[start:end].set(levels.wguv[sorted_global_indices])
    sorted_wstates = levels.wstates.at[start:end].set(levels.wstates[sorted_global_indices])
    sorted_pairwg = levels.pairwg.at[start:end].set(levels.pairwg[sorted_global_indices])
    
    # Update levels with sorted arrays
    levels = levels._replace(
        psi=sorted_psi,
        lagrange=sorted_lagrange,
        sp_energy=sorted_sp_energy,
        deltaf=sorted_deltaf,
        sp_norm=sorted_sp_norm,
        sp_efluct1=sorted_sp_efluct1,
        sp_efluct2=sorted_sp_efluct2,
        wocc=sorted_wocc,
        wguv=sorted_wguv,
        wstates=sorted_wstates,
        pairwg=sorted_pairwg
    )
    
    return levels

def statichf(coulomb, densities, energies, forces, grids, levels, meanfield, moment, params, static, pairs, output_writer=None, output_interval=5):

    log_file_path = "hfb_convergence.log"

    # Create a clean log file at the start of a new calculation (not when restarting)
    if not params.trestart:
        with open(log_file_path, 'w') as f:
            f.write(f"# HFB Convergence Log - {datetime.datetime.now()}\n")
            f.write(f"# Force: {forces.name}, Grid: {grids.nx}x{grids.ny}x{grids.nz}, N={levels.nneut}, Z={levels.nprot}\n")
            f.write(f"# Convergence criterion: {static.serr:.6e}\n\n")

    """Main function for static iterations with added debugging."""

    npsi_neutron=int(levels.npsi[0])
    npsi_proton=int(levels.npsi[1])


    firstiter = 1
    addnew = 0.2
    addco = 1.0 - addnew
    taddnew = True

    if params.trestart:
        firstiter = params.iteration + 1
    else:
        params.iteration = 0
        energies, levels, static = diagstep(energies, forces, grids, levels, static, False, True,npsi_neutron, npsi_proton)


    densities = add_density(densities, grids, levels)

    meanfield, coulomb = skyrme(coulomb, densities, forces, grids, meanfield, params, static)

    levels, static = grstep(forces, grids, levels, meanfield, params, static)

    if forces.ipair != 0:
        levels, pairs = pair(levels, meanfield, forces, params, grids, pairs)


    energies, levels, static = diagstep(energies, forces, grids, levels, static, False, True,npsi_neutron, npsi_proton)


    levels = sp_properties(forces, grids, levels, moment)

    energies = integ_energy(coulomb, densities, energies, forces, grids, levels, params, pairs)
    energies = sum_energy(energies, levels, meanfield, pairs)

    if output_writer is not None:  # Pass this as parameter
        energies, observables = complete_sinfo_with_fortran_output(
            coulomb, densities, energies, forces, grids, levels, 
            meanfield, moment, params, static, pairs, output_writer
        )

    # Calculate and print information
    if output_writer is not None:
        energies, observables = complete_sinfo_with_fortran_output(
            coulomb, densities, energies, forces, grids, levels, 
            meanfield, moment, params, static, pairs, output_writer
        )
    else:
        if i % 10 == 0:
            energies = sinfo(coulomb, densities, energies, forces, grids, levels, meanfield, moment, params, static, pairs)
    write_convergence_log(log_file_path, params.iteration, energies, static, levels, densities, meanfield, grids, forces, pairs)
    
    # Set x0dmp to 3* its value to get faster convergence
    if static.tvaryx_0:
        static.x0dmp = static.x0dmp * 3.0

    # Save old pairing strengths for "annealing"
    v0protsav = forces.v0prot
    v0neutsav = forces.v0neut

    tbcssav = forces.tbcs

    for i in range(firstiter, static.maxiter + 1):
        print(i)

        params.iteration = i

        # Control pairing during iterations
        if i <= static.inibcs:
            forces.tbcs = True
        else:
            forces.tbcs = tbcssav

        if i > static.inidiag:
            static.tdiag = False
        else:
            static.tdiag = True

        # Annealing: enhance pairing strengths in first iteranneal iterations
        if static.iteranneal > 0:
            if i < static.iteranneal:
                forces.v0prot = v0protsav + v0protsav * static.pairenhance * (static.iteranneal - i) / (1.0 * static.iteranneal)
                forces.v0neut = v0neutsav + v0neutsav * static.pairenhance * (static.iteranneal - i) / (1.0 * static.iteranneal)
            else:
                forces.v0prot = v0protsav
                forces.v0neut = v0neutsav

        levels, static = grstep(forces, grids, levels, meanfield, params, static)


        if forces.tbcs:
            static.tdiag = True

        energies, levels, static = diagstep(energies, forces, grids, levels, static, static.tdiag, True,npsi_neutron, npsi_proton)


        #do pairing
        if forces.ipair != 0:
            levels, pairs = pair(levels, meanfield, forces, params, grids, pairs)

        # Sub-iterations in configuration space
        if forces.ipair != 0 and static.iternat > 0 and i > static.iternat_start and not forces.tbcs:
            for iq in range(2):
                levels, static = hfb_natural(forces, grids, levels, meanfield, params, static, iq, static.iternat)

        if taddnew:
            meanfield.upot = meanfield.upot.at[...].set(densities.rho)
            meanfield.bmass = meanfield.bmass.at[...].set(densities.tau)
            meanfield.v_pair = meanfield.v_pair.at[...].set(densities.chi)

        # Create temporary storage for old densities
        old_rho = jnp.copy(densities.rho)
        old_tau = jnp.copy(densities.tau)
        old_chi = jnp.copy(densities.chi)

        densities.rho = densities.rho.at[...].set(0.0)
        densities.chi = densities.chi.at[...].set(0.0)
        densities.tau = densities.tau.at[...].set(0.0)
        densities.current = densities.current.at[...].set(0.0)
        densities.sdens = densities.sdens.at[...].set(0.0)
        densities.sodens = densities.sodens.at[...].set(0.0)

        densities = add_density(densities, grids, levels)
        
        # Apply relaxation on densities
        if taddnew:
            densities.rho = densities.rho.at[...].set(addnew * densities.rho + addco * old_rho)
            densities.tau = densities.tau.at[...].set(addnew * densities.tau + addco * old_tau)
            densities.chi = densities.chi.at[...].set(addnew * densities.chi + addco * old_chi)

        # Construct potentials
        meanfield, coulomb = skyrme(coulomb, densities, forces, grids, meanfield, params, static)

        # Sort states if requested
        if static.tsort:
            for iq in range(2):
                levels = sort_states(forces, grids, levels, iq)

        levels = sp_properties(forces, grids, levels, moment)

        energies = integ_energy(coulomb, densities, energies, forces, grids, levels, params, pairs)
        energies = sum_energy(energies, levels, meanfield, pairs)

        if output_writer is not None and i % output_interval ==0:  # Pass this as parameter
            energies, observables = complete_sinfo_with_fortran_output(
                coulomb, densities, energies, forces, grids, levels, 
                meanfield, moment, params, static, pairs, output_writer
            )
        
        # Calculate and print information
        if i % 10 == 0:
            energies = sinfo(coulomb, densities, energies, forces, grids, levels, meanfield, moment, params, static, pairs)

        #write_convergence_log(log_file_path, params.iteration, energies, static, levels, densities, meanfield, grids, forces, pairs)

        # Check for convergence
        if energies.efluct1[0] < static.serr and i > 1:
            # Final .res output at convergence
            if output_writer is not None:
                energies, observables = complete_sinfo_with_fortran_output(
                    coulomb, densities, energies, forces, grids, levels, 
                    meanfield, moment, params, static, pairs, output_writer
                )
            print("Convergence achieved!")
            break

        # Adaptive step size if tvaryx_0 is true
        if static.tvaryx_0:
            if ((energies.ehf < energies.ehfprev and 
                 energies.efluct1[0] < (energies.efluct1prev * (1.0 - 1.0e-5))) or 
                (energies.efluct2[0] < (energies.efluct2prev * (1.0 - 1.0e-5)))):
                static.x0dmp = static.x0dmp * 1.005
            else:
                static.x0dmp = static.x0dmp * 0.8

            static.x0dmp = jnp.clip(static.x0dmp, static.x0dmpmin, static.x0dmpmin * 5.0)
            
            energies.efluct1prev = energies.efluct1[0]
            energies.efluct2prev = energies.efluct2[0]
            energies.ehfprev = energies.ehf

    # Final diagonalization to get quasiparticle states
    for iq in range(2):
        qp_energies, qp_norms = lastdiag(forces, grids, levels, params, static, iq)
        
        # Print quasiparticle energies
        if iq == 0:
            print("Neutron quasi-particle states (from low to high):")
        else:
            print("Proton quasi-particle states (from low to high):")
            
        print("#    qp energies   lower norm")
        for j in range(qp_energies.shape[0]):
            print(f"{j:3d}    {qp_energies[j]:9.5f}    {qp_norms[j]:9.5f}")

    return coulomb, densities, energies, forces, grids, levels, meanfield, moment, params, static


def write_convergence_log(file_path, iteration, energies, static, levels, densities, meanfield, grids, forces, pairs):
    # --- Calculate metrics for the summary log ---
    
    # Calculate neutron and proton occupation sums
    neutron_occ_sum = jnp.sum(levels.wocc[:int(levels.npsi[0])] * levels.wstates[:int(levels.npsi[0])])
    proton_occ_sum = jnp.sum(levels.wocc[int(levels.npsi[0]):] * levels.wstates[int(levels.npsi[0]):])
    
    # Format the change in energy from the previous step
    energy_change = "N/A"
    if hasattr(energies, 'ehfprev') and energies.ehfprev != 0.0 and iteration > 0:
        energy_change = f"{energies.ehf - energies.ehfprev:.6e}"
    
    # Calculate particle numbers from densities to check for conservation
    neutron_density_integral = jnp.sum(densities.rho[0,...]) * grids.wxyz
    proton_density_integral = jnp.sum(densities.rho[1,...]) * grids.wxyz
    
    # Check for loss of orthogonality between wavefunctions
    max_overlap = 0.0
    for iq in range(2):
        start = 0 if iq == 0 else int(levels.npsi[0])
        end = int(levels.npsi[0]) if iq == 0 else levels.nstmax
        for i in range(start, end):
            for j in range(i + 1, end):
                ovlp = abs(overlap(levels.psi[i], levels.psi[j], grids.wxyz))
                if ovlp > max_overlap:
                    max_overlap = ovlp

    # --- Write data to the log file ---
    with open(file_path, 'a') as f:
        # Write the header only at the beginning of a new calculation
        if iteration == 0:
            header = (
                f"{'Iter':>5s}  {'Energy(MeV)':>12s}  {'dE(MeV)':>12s}  {'TKE(MeV)':>12s}  {'Rearr(MeV)':>12s}  "
                f"{'Fluct1':>10s}  {'Fluct2':>10s}  {'N_Dens':>8s}  {'P_Dens':>8s}  {'Max_Ovlp':>10s}  {'x0dmp':>9s}\n"
            )
            f.write(header)
        
        # Write the main summary line for the current iteration
        data_line = (
            f"{iteration:5d}  {energies.ehf:12.6f}  {energy_change:12s}  {energies.tke:12.6f}  {energies.e3corr:12.6f}  "
            f"{energies.efluct1[0]:10.3e}  {energies.efluct2[0]:10.3e}  {neutron_density_integral:8.4f}  {proton_density_integral:8.4f}  "
            f"{max_overlap:10.3e}  {static.x0dmp:9.4f}\n"
        )
        f.write(data_line)
        
        # Optionally, write a detailed breakdown every 10 iterations
        if iteration % 10 == 0:
            f.write("#" + "-"*110 + "\n")
            f.write(f"# Detailed breakdown for iteration {iteration}:\n")
            
            # Energy components
            energy_components = {
                "t0 part": energies.ehf0, "t1 part": energies.ehf1, "t2 part": energies.ehf2,
                "t3 part": energies.ehf3, "Coulomb": energies.ehfc, "Spin-orbit": energies.ehfls,
                "Pairing (n)": pairs.epair[0], "Pairing (p)": pairs.epair[1]
            }
            f.write("# Energy Components:\n")
            for name, value in energy_components.items():
                f.write(f"#   {name:<12s}: {value:12.6f} MeV\n")
            
            f.write("#" + "-"*110 + "\n\n")

def statichf_with_benchmark(coulomb, densities, energies, forces, grids, levels, meanfield, moment, params, static, pairs, output_writer=None, output_interval=10):

    log_file_path = "hfb_convergence.log"
    npsi_neutron=int(levels.npsi[0])
    npsi_proton=int(levels.npsi[1])

    # Create a clean log file at the start of a new calculation (not when restarting)
    if not params.trestart:
        with open(log_file_path, 'w') as f:
            f.write(f"# HFB Convergence Log - {datetime.datetime.now()}\n")
            f.write(f"# Force: {forces.name}, Grid: {grids.nx}x{grids.ny}x{grids.nz}, N={levels.nneut}, Z={levels.nprot}\n")
            f.write(f"# Convergence criterion: {static.serr:.6e}\n\n")

    """Main function for static iterations with added debugging."""

    firstiter = 1
    addnew = 0.2
    addco = 1.0 - addnew
    taddnew = True

    Z = levels.nprot
    N = levels.nneut
    A = Z + N
    element_symbol = get_element_symbol(Z)
    benchmark_filename = f"{element_symbol}{A}_benchmark.txt"

    # Write a header to the benchmark file, overwriting any previous content
    with open(benchmark_filename, 'w') as f:
        f.write(f"# Benchmark Timing Report for {element_symbol}{A} (Z={Z}, N={N})\n")
        f.write(f"# Calculation started: {datetime.datetime.now()}\n")
        f.write("="*50 + "\n")
        f.write(f"{'Iteration':<12} {'Duration (s)':<20}\n")
        f.write("="*50 + "\n")

    if params.trestart:
        firstiter = params.iteration + 1
    else:
        params.iteration = 0
        energies, levels, static = diagstep(energies, forces, grids, levels, static, False, True, npsi_neutron, npsi_proton)


    densities = add_density(densities, grids, levels)

    meanfield, coulomb = skyrme(coulomb, densities, forces, grids, meanfield, params, static)

    levels, static = grstep(forces, grids, levels, meanfield, params, static)

    if forces.ipair != 0:
        levels, pairs = pair(levels, meanfield, forces, params, grids, pairs)


    energies, levels, static = diagstep(energies, forces, grids, levels, static, False, True,npsi_neutron, npsi_proton)


    levels = sp_properties(forces, grids, levels, moment)

    energies = integ_energy(coulomb, densities, energies, forces, grids, levels, params, pairs)
    energies = sum_energy(energies, levels, meanfield, pairs)

    if output_writer is not None:  # Pass this as parameter
        energies, observables = complete_sinfo_with_fortran_output(
            coulomb, densities, energies, forces, grids, levels, 
            meanfield, moment, params, static, pairs, output_writer
        )

    # Calculate and print information
    if output_writer is not None:
        energies, observables = complete_sinfo_with_fortran_output(
            coulomb, densities, energies, forces, grids, levels, 
            meanfield, moment, params, static, pairs, output_writer
        )
    else:
        if i % 10 == 0:
            energies = sinfo(coulomb, densities, energies, forces, grids, levels, meanfield, moment, params, static, pairs)
    #write_convergence_log(log_file_path, params.iteration, energies, static, levels, densities, meanfield, grids, forces, pairs)
    
    # Set x0dmp to 3* its value to get faster convergence
    if static.tvaryx_0:
        static.x0dmp = static.x0dmp * 3.0

    # Save old pairing strengths for "annealing"
    v0protsav = forces.v0prot
    v0neutsav = forces.v0neut

    tbcssav = forces.tbcs

    iteration_times = []


    for i in range(firstiter, static.maxiter + 1):
        print(i)

        start_time = time.perf_counter()

        params.iteration = i

        # Control pairing during iterations
        if i <= static.inibcs:
            forces.tbcs = True
        else:
            forces.tbcs = tbcssav

        if i > static.inidiag:
            static.tdiag = False
        else:
            static.tdiag = True

        # Annealing: enhance pairing strengths in first iteranneal iterations
        if static.iteranneal > 0:
            if i < static.iteranneal:
                forces.v0prot = v0protsav + v0protsav * static.pairenhance * (static.iteranneal - i) / (1.0 * static.iteranneal)
                forces.v0neut = v0neutsav + v0neutsav * static.pairenhance * (static.iteranneal - i) / (1.0 * static.iteranneal)
            else:
                forces.v0prot = v0protsav
                forces.v0neut = v0neutsav

        levels, static = grstep(forces, grids, levels, meanfield, params, static)


        if forces.tbcs:
            static.tdiag = True

        energies, levels, static = diagstep(energies, forces, grids, levels, static, static.tdiag, True,npsi_neutron, npsi_proton)


        #do pairing
        if forces.ipair != 0:
            levels, pairs = pair(levels, meanfield, forces, params, grids, pairs)

        # Sub-iterations in configuration space
        if forces.ipair != 0 and static.iternat > 0 and i > static.iternat_start and not forces.tbcs:
            for iq in range(2):
                levels, static = hfb_natural(forces, grids, levels, meanfield, params, static, iq, static.iternat)

        if taddnew:
            meanfield.upot = meanfield.upot.at[...].set(densities.rho)
            meanfield.bmass = meanfield.bmass.at[...].set(densities.tau)
            meanfield.v_pair = meanfield.v_pair.at[...].set(densities.chi)

        # Create temporary storage for old densities
        old_rho = jnp.copy(densities.rho)
        old_tau = jnp.copy(densities.tau)
        old_chi = jnp.copy(densities.chi)

        densities.rho = densities.rho.at[...].set(0.0)
        densities.chi = densities.chi.at[...].set(0.0)
        densities.tau = densities.tau.at[...].set(0.0)
        densities.current = densities.current.at[...].set(0.0)
        densities.sdens = densities.sdens.at[...].set(0.0)
        densities.sodens = densities.sodens.at[...].set(0.0)

        densities = add_density(densities, grids, levels)
        
        # Apply relaxation on densities
        if taddnew:
            densities.rho = densities.rho.at[...].set(addnew * densities.rho + addco * old_rho)
            densities.tau = densities.tau.at[...].set(addnew * densities.tau + addco * old_tau)
            densities.chi = densities.chi.at[...].set(addnew * densities.chi + addco * old_chi)

        # Construct potentials
        meanfield, coulomb = skyrme(coulomb, densities, forces, grids, meanfield, params, static)

        # Sort states if requested
        if static.tsort:
            for iq in range(2):
                levels = sort_states(forces, grids, levels, iq)

        levels = sp_properties(forces, grids, levels, moment)

        energies = integ_energy(coulomb, densities, energies, forces, grids, levels, params, pairs)
        energies = sum_energy(energies, levels, meanfield, pairs)

        if output_writer is not None and i % output_interval ==0:  # Pass this as parameter
            energies, observables = complete_sinfo_with_fortran_output(
                coulomb, densities, energies, forces, grids, levels, 
                meanfield, moment, params, static, pairs, output_writer
            )
        
        # Calculate and print information
        if i % 10 == 0:
            energies = sinfo(coulomb, densities, energies, forces, grids, levels, meanfield, moment, params, static, pairs)

        #write_convergence_log(log_file_path, params.iteration, energies, static, levels, densities, meanfield, grids, forces, pairs)

        # --- ADD THIS BLOCK ---
        end_time = time.perf_counter()
        iteration_duration = end_time - start_time
        iteration_times.append(iteration_duration)
        
        with open(benchmark_filename, 'a') as f:
            f.write(f"{i:<12} {iteration_duration:<20.6f}\n")

        # Check for convergence
        if energies.efluct1[0] < static.serr and i > 1:
            # Final .res output at convergence
            if output_writer is not None:
                energies, observables = complete_sinfo_with_fortran_output(
                    coulomb, densities, energies, forces, grids, levels, 
                    meanfield, moment, params, static, pairs, output_writer
                )
            print("Convergence achieved!")
            break

        # Adaptive step size if tvaryx_0 is true
        if static.tvaryx_0:
            if ((energies.ehf < energies.ehfprev and 
                 energies.efluct1[0] < (energies.efluct1prev * (1.0 - 1.0e-5))) or 
                (energies.efluct2[0] < (energies.efluct2prev * (1.0 - 1.0e-5)))):
                static.x0dmp = static.x0dmp * 1.005
            else:
                static.x0dmp = static.x0dmp * 0.8

            static.x0dmp = jnp.clip(static.x0dmp, static.x0dmpmin, static.x0dmpmin * 5.0)
            
            energies.efluct1prev = energies.efluct1[0]
            energies.efluct2prev = energies.efluct2[0]
            energies.ehfprev = energies.ehf

    # --- ADD THIS ENTIRE BLOCK AFTER THE LOOP ---
    if iteration_times:
        total_time = sum(iteration_times)
        num_iterations = len(iteration_times)
        average_time = total_time / num_iterations
        min_time = min(iteration_times)
        max_time = max(iteration_times)

        summary_string = (
            f"\n\n" + "="*50 + "\n" +
            f"           BENCHMARK TIMING SUMMARY\n" +
            "="*50 + "\n" +
            f"Total iterations run: {num_iterations}\n" +
            f"Total computation time: {total_time:.4f} seconds\n" +
            "-" * 50 + "\n" +
            f"Average time per iteration: {average_time:.4f} seconds\n" +
            f"Fastest iteration:          {min_time:.4f} seconds\n" +
            f"Slowest iteration:          {max_time:.4f} seconds\n" +
            "="*50 + "\n"
        )
        
        # Print to console
        print(summary_string)

        # Append summary to the benchmark file
        with open(benchmark_filename, 'a') as f:
            f.write(summary_string)
    # --- END MODIFIED ---
    
    # Final diagonalization to get quasiparticle states
    for iq in range(2):
        qp_energies, qp_norms = lastdiag(forces, grids, levels, params, static, pairs,iq)
        
        # Print quasiparticle energies
        if iq == 0:
            print("Neutron quasi-particle states (from low to high):")
        else:
            print("Proton quasi-particle states (from low to high):")
            
        print("#    qp energies   lower norm")
        for j in range(qp_energies.shape[0]):
            print(f"{j:3d}    {qp_energies[j]:9.5f}    {qp_norms[j]:9.5f}")

    return coulomb, densities, energies, forces, grids, levels, meanfield, moment, params, static

def get_element_symbol(Z):
    """Returns the chemical symbol for a given proton number Z."""
    symbols = [
        "n", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", 
        "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", 
        "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", 
        "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", 
        "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", 
        "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", 
        "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", 
        "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", 
        "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", 
        "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
    ]
    if 0 <= Z < len(symbols):
        return symbols[Z]
    return "X" # Placeholder for unknown elements

def print_detailed_timing_report(iteration_times, detailed_times=None):
    """Print comprehensive timing analysis."""
    
    total_time = sum(iteration_times)
    num_iterations = len(iteration_times)
    average_time = total_time / num_iterations
    min_time = min(iteration_times)
    max_time = max(iteration_times)

    print("\n" + "="*70)
    print("           DETAILED BENCHMARK TIMING REPORT")
    print("="*70)
    print(f"Total iterations: {num_iterations}")
    print(f"Total time: {total_time:.4f} seconds")
    print(f"Average per iteration: {average_time:.4f} seconds")
    print(f"Range: {min_time:.4f} - {max_time:.4f} seconds")
    
    if detailed_times:
        print("\nStep-by-step breakdown (average times):")
        print("-" * 50)
        
        step_totals = {}
        for step_name, times in detailed_times.items():
            if times:  # Check if list is not empty
                avg_time = sum(times) / len(times)
                step_totals[step_name] = avg_time
                percentage = (avg_time / average_time) * 100
                print(f"{step_name:<15s}: {avg_time:8.4f}s ({percentage:5.1f}%)")
        
        # Find bottlenecks
        if step_totals:
            print("\nBottleneck analysis:")
            print("-" * 50)
            sorted_steps = sorted(step_totals.items(), key=lambda x: x[1], reverse=True)
            print("Slowest steps:")
            for i, (step, time_val) in enumerate(sorted_steps[:3]):
                print(f"  {i+1}. {step}: {time_val:.4f}s")
    
    print("="*70 + "\n")


def statichf_with_detailed_benchmark(coulomb, densities, energies, forces, grids, levels, meanfield, moment, params, static, pairs, output_writer=None, output_interval=5, enable_detailed_timing=True):
   
    npsi_neutron=int(levels.npsi[0])
    npsi_proton=int(levels.npsi[1])

    # Generate unique filename based on nucleus properties
    def generate_unique_filename(base_name, levels, forces, grids):
        """Generate unique filename based on nucleus and calculation parameters."""
        nucleus_id = f"Z{levels.nprot:02d}N{levels.nneut:03d}"  # e.g., Z50N082 for Sn-132
        force_id = forces.name.replace(' ', '_')  # Remove spaces from force name
        grid_id = f"{grids.nx}x{grids.ny}x{grids.nz}"
        
        # Add pairing info
        if forces.ipair == 0:
            pair_id = "nopair"
        elif forces.ipair == 5:
            pair_id = "VDI"
        elif forces.ipair == 6:
            pair_id = "DDDI"
        else:
            pair_id = f"pair{forces.ipair}"
            
        return f"{base_name}_{nucleus_id}_{force_id}_{grid_id}_{pair_id}.log"
    
    # Create unique log file paths
    log_file_path = generate_unique_filename("hfb_convergence", levels, forces, grids)
    timing_log_path = generate_unique_filename("hfb_detailed_timing", levels, forces, grids)
    
    # Initialize timing log
    if enable_detailed_timing and not params.trestart:
        with open(timing_log_path, 'w') as f:
            f.write(f"# Detailed HFB Timing Log - {datetime.datetime.now()}\n")
            f.write(f"# Nucleus: Z={levels.nprot}, N={levels.nneut}, A={levels.mass_number}\n")
            f.write(f"# Force: {forces.name}, Grid: {grids.nx}x{grids.ny}x{grids.nz}, dx={grids.dx:.3f} fm\n")
            f.write(f"# Pairing: ipair={forces.ipair}")
            if forces.ipair != 0:
                f.write(f", v0neut={forces.v0neut:.3f}, v0prot={forces.v0prot:.3f}")
            f.write(f"\n# All times in seconds\n\n")
            header = (f"{'Iter':>5s} {'Total':>8s} {'GrStep':>8s} {'Diag':>8s} {'Pair':>8s} "
                     f"{'Density':>8s} {'Skyrme':>8s} {'SPProp':>8s} {'Energy':>8s} {'Info':>8s}\n")
            f.write(header)

    # Regular initialization...
    if not params.trestart:
        with open(log_file_path, 'w') as f:
            f.write(f"# HFB Convergence Log - {datetime.datetime.now()}\n")
            f.write(f"# Nucleus: Z={levels.nprot}, N={levels.nneut}, A={levels.mass_number}\n")
            f.write(f"# Force: {forces.name}, Grid: {grids.nx}x{grids.ny}x{grids.nz}, dx={grids.dx:.3f} fm\n")
            f.write(f"# Pairing: ipair={forces.ipair}")
            if forces.ipair != 0:
                f.write(f", v0neut={forces.v0neut:.3f}, v0prot={forces.v0prot:.3f}")
            f.write(f"\n# Convergence criterion: {static.serr:.6e}\n\n")

    firstiter = 1
    addnew = 0.2
    addco = 1.0 - addnew
    taddnew = True

    if params.trestart:
        firstiter = params.iteration + 1
    else:
        params.iteration = 0
        energies, levels, static = diagstep(energies, forces, grids, levels, static, False, True,npsi_neutron, npsi_proton)

    # Initial setup steps (not timed in detail since they're one-time)
    densities = add_density(densities, grids, levels)
    meanfield, coulomb = skyrme(coulomb, densities, forces, grids, meanfield, params, static)
    levels, static = grstep(forces, grids, levels, meanfield, params, static)

    if forces.ipair != 0:
        levels, pairs = pair(levels, meanfield, forces, params, grids, pairs)

    energies, levels, static = diagstep(energies, forces, grids, levels, static, False, True,npsi_neutron, npsi_proton)
    levels = sp_properties(forces, grids, levels, moment)
    energies = integ_energy(coulomb, densities, energies, forces, grids, levels, params, pairs)
    energies = sum_energy(energies, levels, meanfield, pairs)

    if output_writer is not None:
        energies, observables = complete_sinfo_with_fortran_output(
            coulomb, densities, energies, forces, grids, levels, 
            meanfield, moment, params, static, pairs, output_writer
        )

    energies = sinfo(coulomb, densities, energies, forces, grids, levels, meanfield, moment, params, static, pairs)
    write_convergence_log(log_file_path, params.iteration, energies, static, levels, densities, meanfield, grids, forces, pairs)

    if static.tvaryx_0:
        static.x0dmp = static.x0dmp * 3.0

    v0protsav = forces.v0prot
    v0neutsav = forces.v0neut
    tbcssav = forces.tbcs

    # Storage for timing data
    iteration_times = []
    detailed_times = defaultdict(list) if enable_detailed_timing else None

    for i in range(firstiter, static.maxiter + 1):
        print(f"Iteration {i}")
        
        iter_start_time = time.perf_counter()
        step_times = {} if enable_detailed_timing else None

        params.iteration = i

        # Control pairing during iterations
        if i <= static.inibcs:
            forces.tbcs = True
        else:
            forces.tbcs = tbcssav

        if i > static.inidiag:
            static.tdiag = False
        else:
            static.tdiag = True

        # Annealing logic...
        if static.iteranneal > 0:
            if i < static.iteranneal:
                forces.v0prot = v0protsav + v0protsav * static.pairenhance * (static.iteranneal - i) / (1.0 * static.iteranneal)
                forces.v0neut = v0neutsav + v0neutsav * static.pairenhance * (static.iteranneal - i) / (1.0 * static.iteranneal)
            else:
                forces.v0prot = v0protsav
                forces.v0neut = v0neutsav

        # STEP 1: Gradient Step
        if enable_detailed_timing:
            step_start = time.perf_counter()
        
        levels, static = grstep(forces, grids, levels, meanfield, params, static)
        
        if enable_detailed_timing:
            jax.block_until_ready([levels.psi, static.delesum])  # Ensure computation is complete
            step_times['grstep'] = time.perf_counter() - step_start

        # STEP 2: Diagonalization
        if enable_detailed_timing:
            step_start = time.perf_counter()

        if forces.tbcs:
            static.tdiag = True
        energies, levels, static = diagstep(energies, forces, grids, levels, static, static.tdiag, True,npsi_neutron, npsi_proton)

        if enable_detailed_timing:
            jax.block_until_ready([levels.psi, energies.efluct1])
            step_times['diagstep'] = time.perf_counter() - step_start

        # STEP 3: Pairing
        if enable_detailed_timing:
            step_start = time.perf_counter()

        if forces.ipair != 0:
            levels, pairs = pair(levels, meanfield, forces, params, grids, pairs)

        if enable_detailed_timing:
            jax.block_until_ready([levels.wocc, pairs.eferm])
            step_times['pairing'] = time.perf_counter() - step_start

        # Sub-iterations in configuration space
        if forces.ipair != 0 and static.iternat > 0 and i > static.iternat_start and not forces.tbcs:
            for iq in range(2):
                levels, static = hfb_natural(forces, grids, levels, meanfield, params, static, iq, static.iternat)

        # Density mixing setup
        if taddnew:
            meanfield.upot = meanfield.upot.at[...].set(densities.rho)
            meanfield.bmass = meanfield.bmass.at[...].set(densities.tau)
            meanfield.v_pair = meanfield.v_pair.at[...].set(densities.chi)

        old_rho = jnp.copy(densities.rho)
        old_tau = jnp.copy(densities.tau)
        old_chi = jnp.copy(densities.chi)

        # Reset densities
        densities.rho = densities.rho.at[...].set(0.0)
        densities.chi = densities.chi.at[...].set(0.0)
        densities.tau = densities.tau.at[...].set(0.0)
        densities.current = densities.current.at[...].set(0.0)
        densities.sdens = densities.sdens.at[...].set(0.0)
        densities.sodens = densities.sodens.at[...].set(0.0)

        # STEP 4: Density Calculation
        if enable_detailed_timing:
            step_start = time.perf_counter()

        densities = add_density(densities, grids, levels)

        if enable_detailed_timing:
            jax.block_until_ready([densities.rho, densities.tau])
            step_times['add_density'] = time.perf_counter() - step_start

        # Apply relaxation
        if taddnew:
            densities.rho = densities.rho.at[...].set(addnew * densities.rho + addco * old_rho)
            densities.tau = densities.tau.at[...].set(addnew * densities.tau + addco * old_tau)
            densities.chi = densities.chi.at[...].set(addnew * densities.chi + addco * old_chi)

        # STEP 5: Skyrme Functional
        if enable_detailed_timing:
            step_start = time.perf_counter()

        meanfield, coulomb = skyrme(coulomb, densities, forces, grids, meanfield, params, static)

        if enable_detailed_timing:
            jax.block_until_ready([meanfield.upot, coulomb.wcoul])
            step_times['skyrme'] = time.perf_counter() - step_start

        # Sort states if requested
        if static.tsort:
            for iq in range(2):
                levels = sort_states(forces, grids, levels, iq)

        # STEP 6: Single-particle properties
        if enable_detailed_timing:
            step_start = time.perf_counter()

        levels = sp_properties(forces, grids, levels, moment)

        if enable_detailed_timing:
            jax.block_until_ready([levels.sp_orbital, levels.sp_spin])
            step_times['sp_properties'] = time.perf_counter() - step_start

        # STEP 7: Energy calculations
        if enable_detailed_timing:
            step_start = time.perf_counter()

        energies = integ_energy(coulomb, densities, energies, forces, grids, levels, params, pairs)
        energies = sum_energy(energies, levels, meanfield, pairs)

        if enable_detailed_timing:
            jax.block_until_ready([energies.ehf, energies.tke])
            step_times['energy_calc'] = time.perf_counter() - step_start

        # Output handling
        if output_writer is not None and i % output_interval == 0:
            energies, observables = complete_sinfo_with_fortran_output(
                coulomb, densities, energies, forces, grids, levels, 
                meanfield, moment, params, static, pairs, output_writer
            )

        # STEP 8: Information printing
        if enable_detailed_timing:
            step_start = time.perf_counter()

        energies = sinfo(coulomb, densities, energies, forces, grids, levels, meanfield, moment, params, static, pairs)
        #write_convergence_log(log_file_path, params.iteration, energies, static, levels, densities, meanfield, grids, forces, pairs)

        if enable_detailed_timing:
            step_times['sinfo'] = time.perf_counter() - step_start

        # Calculate total iteration time
        iter_end_time = time.perf_counter()
        iteration_duration = iter_end_time - iter_start_time
        iteration_times.append(iteration_duration)

        # Log detailed timing
        if enable_detailed_timing:
            # Store timing data
            for step_name, step_time in step_times.items():
                detailed_times[step_name].append(step_time)

            # Write to detailed timing log
            with open(timing_log_path, 'a') as f:
                line = f"{i:5d} {iteration_duration:8.4f} "
                line += f"{step_times.get('grstep', 0):8.4f} {step_times.get('diagstep', 0):8.4f} {step_times.get('pairing', 0):8.4f} "
                line += f"{step_times.get('add_density', 0):8.4f} {step_times.get('skyrme', 0):8.4f} "
                line += f"{step_times.get('sp_properties', 0):8.4f} {step_times.get('energy_calc', 0):8.4f} {step_times.get('sinfo', 0):8.4f}\n"
                f.write(line)

        print(f"--- Iteration {i} completed in {iteration_duration:.4f} seconds ---")

        # Check for convergence
        if energies.efluct1[0] < static.serr and i > 1:
            if output_writer is not None:
                energies, observables = complete_sinfo_with_fortran_output(
                    coulomb, densities, energies, forces, grids, levels, 
                    meanfield, moment, params, static, pairs, output_writer
                )
            print("Convergence achieved!")
            break

        # Adaptive step size logic...
        if static.tvaryx_0:
            if ((energies.ehf < energies.ehfprev and 
                 energies.efluct1[0] < (energies.efluct1prev * (1.0 - 1.0e-5))) or 
                (energies.efluct2[0] < (energies.efluct2prev * (1.0 - 1.0e-5)))):
                static.x0dmp = static.x0dmp * 1.005
            else:
                static.x0dmp = static.x0dmp * 0.8

            static.x0dmp = jnp.clip(static.x0dmp, static.x0dmpmin, static.x0dmpmin * 5.0)
            
            energies.efluct1prev = energies.efluct1[0]
            energies.efluct2prev = energies.efluct2[0]
            energies.ehfprev = energies.ehf

    # Print comprehensive timing report
    if iteration_times:
        summary_file_path = generate_unique_filename("hfb_timing_summary", levels, forces, grids)
        print_detailed_timing_report(iteration_times, detailed_times if enable_detailed_timing else None, 
                                   summary_file_path, levels, forces, grids, static)

    # Final diagonalization...
    for iq in range(2):
        qp_energies, qp_norms = lastdiag(forces, grids, levels, params, static, pairs, iq)
        
        if iq == 0:
            print("Neutron quasi-particle states (from low to high):")
        else:
            print("Proton quasi-particle states (from low to high):")
            
        print("#    qp energies   lower norm")
        for j in range(min(10, qp_energies.shape[0])):  # Limit output
            print(f"{j:3d}    {qp_energies[j]:9.5f}    {qp_norms[j]:9.5f}")

    return coulomb, densities, energies, forces, grids, levels, meanfield, moment, params, static


def print_detailed_timing_report(iteration_times, detailed_times=None, summary_file_path=None, levels=None, forces=None, grids=None, static=None):
    """Print comprehensive timing analysis and optionally write to file."""
    
    total_time = sum(iteration_times)
    num_iterations = len(iteration_times)
    average_time = total_time / num_iterations
    min_time = min(iteration_times)
    max_time = max(iteration_times)
    std_time = (sum((t - average_time)**2 for t in iteration_times) / num_iterations)**0.5

    # Prepare report content
    report_lines = []
    report_lines.append("="*70)
    report_lines.append("           DETAILED BENCHMARK TIMING REPORT")
    report_lines.append("="*70)
    
    if levels and forces and grids:
        report_lines.append(f"Nucleus: Z={levels.nprot}, N={levels.nneut}, A={levels.mass_number}")
        report_lines.append(f"Force: {forces.name}")
        report_lines.append(f"Grid: {grids.nx}x{grids.ny}x{grids.nz}, dx={grids.dx:.3f} fm")
        if forces.ipair != 0:
            report_lines.append(f"Pairing: ipair={forces.ipair}, v0neut={forces.v0neut:.3f}, v0prot={forces.v0prot:.3f}")
        else:
            report_lines.append("Pairing: disabled")
        if static:
            report_lines.append(f"Convergence criterion: {static.serr:.2e}")
        report_lines.append("-" * 70)
    
    report_lines.append(f"Total iterations: {num_iterations}")
    report_lines.append(f"Total computation time: {total_time:.4f} seconds ({total_time/60:.2f} minutes)")
    report_lines.append(f"Average per iteration: {average_time:.4f} seconds")
    report_lines.append(f"Standard deviation: {std_time:.4f} seconds")
    report_lines.append(f"Range: {min_time:.4f} - {max_time:.4f} seconds")
    
    if detailed_times:
        report_lines.append("")
        report_lines.append("Step-by-step breakdown (average times):")
        report_lines.append("-" * 50)
        
        step_totals = {}
        for step_name, times in detailed_times.items():
            if times:  # Check if list is not empty
                avg_time = sum(times) / len(times)
                std_time_step = (sum((t - avg_time)**2 for t in times) / len(times))**0.5
                step_totals[step_name] = avg_time
                percentage = (avg_time / average_time) * 100
                report_lines.append(f"{step_name:<15s}: {avg_time:8.4f}s ± {std_time_step:6.4f}s ({percentage:5.1f}%)")
        
        # Find bottlenecks
        if step_totals:
            report_lines.append("")
            report_lines.append("Bottleneck analysis:")
            report_lines.append("-" * 50)
            sorted_steps = sorted(step_totals.items(), key=lambda x: x[1], reverse=True)
            report_lines.append("Slowest steps:")
            for i, (step, time_val) in enumerate(sorted_steps[:3]):
                percentage = (time_val / average_time) * 100
                report_lines.append(f"  {i+1}. {step}: {time_val:.4f}s ({percentage:.1f}% of iteration)")
            
            # Performance insights
            report_lines.append("")
            report_lines.append("Performance insights:")
            report_lines.append("-" * 50)
            
            # Identify potential optimization opportunities
            if 'grstep' in step_totals and step_totals['grstep'] / average_time > 0.4:
                report_lines.append("• Gradient step takes >40% of time - consider optimizing wavefunction updates")
            
            if 'diagstep' in step_totals and step_totals['diagstep'] / average_time > 0.3:
                report_lines.append("• Diagonalization is expensive - check basis size or matrix construction")
            
            if 'skyrme' in step_totals and step_totals['skyrme'] / average_time > 0.2:
                report_lines.append("• Skyrme functional evaluation is costly - potential for grid optimization")
            
            if 'add_density' in step_totals and step_totals['add_density'] / average_time > 0.15:
                report_lines.append("• Density construction is slow - consider vectorization improvements")
    
    report_lines.append("="*70)
    
    # Print to console
    print("\n" + "\n".join(report_lines) + "\n")
    
    # Write to file if path provided
    if summary_file_path:
        with open(summary_file_path, 'w') as f:
            f.write(f"# HFB Timing Summary - {datetime.datetime.now()}\n")
            f.write("# " + "="*68 + "\n")
            for line in report_lines:
                f.write(line + "\n")
            
            # Add raw timing data for further analysis
            if detailed_times:
                f.write("\n\n# Raw timing data for analysis:\n")
                f.write("# Iteration, Total, " + ", ".join(detailed_times.keys()) + "\n")
                for i, total_time in enumerate(iteration_times):
                    data_line = f"{i+1}, {total_time:.6f}"
                    for step_name in detailed_times.keys():
                        if i < len(detailed_times[step_name]):
                            data_line += f", {detailed_times[step_name][i]:.6f}"
                        else:
                            data_line += ", 0.000000"
                    f.write(data_line + "\n")
        
        print(f"Detailed timing report saved to: {summary_file_path}")


