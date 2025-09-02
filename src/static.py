import jax
import jax.numpy as jnp
from functools import partial
from dataclasses import dataclass, field
from densities import add_density
from inout import sp_properties
from levels import laplace
from meanfield import skyrme, hpsi00, hpsi01
from trivial import rpsnorm, overlap

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
    nst = max(levels.nneut, levels.nprot)

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
        'x0dmp': 0.2,
        'e0dmp': 100,
        'x0dmpmin': 0.2,
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


def e0dmp_gt_zero(args):
    nst, iq, psin, ps1, lagrange, forces, grids, levels, meanfield, params, static  = args
    h_exp = jnp.real(overlap(psin, ps1, grids.wxyz))

    ps1 = ps1.at[...].set(
        jax.lax.cond(
            (params.iteration > 1) & (~forces.tbcs),
            lambda _: ps1 - lagrange,
            lambda _: ps1 - h_exp * psin,
            operand=None,
        )
    )

    sp_efluct1, sp_efluct2 = jax.lax.cond(
        (params.mprint > 0) & (params.iteration % params.mprint == 0),
        lambda _: (
            jnp.sqrt(rpsnorm(ps1, grids.wxyz)),
            jnp.sqrt(rpsnorm(lagrange - h_exp * psin, grids.wxyz))
        ),
        lambda _: (
            levels.sp_efluct1[nst],
            levels.sp_efluct2[nst]
        ),
        operand=None,
    )

    ps2 = jax.lax.cond(
        forces.tbcs,
        lambda _: laplace(grids, forces, 1.0, 0.0, ps1, 0, static.e0dmp),
        lambda _: laplace(
            grids,
            forces,
            levels.wocc[nst],
            levels.wguv[nst] * levels.pairwg[nst],
            ps1,
            jnp.max(meanfield.v_pair[iq,...]),
            static.e0dmp
        ),
        operand=None,
    )

    psin = psin.at[...].add(-(static.x0dmp * ps2))

    return psin, sp_efluct1, sp_efluct2

def grstep_helper(nst, iq, spe_mf, psin, lagrange, forces, grids, levels, meanfield, params, static):
    # Step 1
    ps1, psi_mf = jax.lax.cond(
        forces.tbcs,
        lambda _: hpsi00(grids, meanfield, iq, 1.0, 0.0, psin),
        lambda _: hpsi00(
            grids,
            meanfield,
            iq,
            levels.wstates[nst] * levels.wocc[nst],
            levels.wstates[nst] * levels.wguv[nst] * levels.pairwg[nst],
            psin,
        ),
        operand=None
    )

    # Step 2
    spe_mf_new = jnp.real(overlap(psin, psi_mf, grids.wxyz))

    psin, sp_efluct1, sp_efluct2 = jax.lax.cond(
        static.e0dmp > 0.0,
        e0dmp_gt_zero,
        lambda _: (
            (1.0 + static.x0dmp * (spe_mf_new - spe_mf)) * psin - static.x0dmp * ps1,
            levels.sp_efluct1[nst],
            levels.sp_efluct2[nst],
        ),
        operand=(nst, iq, psin, ps1, lagrange, forces, grids, levels, meanfield, params, static)
    )

    ps1, hmfpsi, delpsi = hpsi01(grids, meanfield, iq, 1.0, 0.0, psin)

    denerg = (spe_mf - spe_mf_new) / jnp.abs(spe_mf_new)

    return psin, psi_mf, hmfpsi, delpsi, denerg, spe_mf_new, sp_efluct1, sp_efluct2


grstep_helper_jit = jax.jit(grstep_helper)
grstep_helper_vmap = jax.vmap(
    grstep_helper,
    in_axes=(0, 0, 0, 0, 0, None, None, None, None, None, None)
)

@jax.jit
def grstep(forces, grids, levels, meanfield, params, static):
    nst = jnp.arange(levels.nstmax)
    isospin = levels.isospin
    sp_energy = levels.sp_energy
    psi = levels.psi
    lagrange = levels.lagrange

    psin, psi_mf, hmfpsi, delpsi, denerg, spe_mf_new, sp_efluct1, sp_efluct2 = grstep_helper_vmap(
        nst, isospin, sp_energy, psi, lagrange, forces,
        grids, levels, meanfield, params, static
    )

    levels.psi = levels.psi.at[...].set(psin)
    levels.hampsi = levels.hampsi.at[...].set(psi_mf)
    levels.hmfpsi = levels.hmfpsi.at[...].set(hmfpsi)
    levels.delpsi = levels.delpsi.at[...].set(delpsi)
    levels.sp_energy = levels.sp_energy.at[...].set(spe_mf_new)

    static.sumflu = static.sumflu + jnp.sum(levels.sp_efluct1)
    static.delesum = static.delesum + jnp.sum(levels.wocc * levels.wstates * denerg)

    levels.sp_efluct1 =  levels.sp_efluct1.at[...].set(sp_efluct1)
    levels.sp_efluct2 =  levels.sp_efluct2.at[...].set(sp_efluct2)

    return levels, static


@partial(jax.jit, static_argnames=['diagonalize', 'construct'])
def diagstep(energies, forces, grids, levels, static, diagonalize=False, construct=True):
    for iq in range(2):
        start, end = (0, levels.nneut) if iq == 0 else (levels.nneut, levels.nneut + levels.nprot)
        nst = end - start

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

        rhomatr_lin = jnp.dot(
            jnp.conjugate(psi_2d.T),
            psi_2d
        ) * grids.wxyz

        if diagonalize:
            lambda_lin = jnp.dot(
                jnp.conjugate(psi_2d.T),
                hampsi_2d
            ) * grids.wxyz

        energies.efluct1q = energies.efluct1q.at[iq].set(
            jax.lax.cond(
                forces.tbcs,
                lambda _: (
                    jnp.sqrt(
                        jnp.sum(
                            levels.wocc[start:end] *
                            levels.wstates[start:end] *
                            levels.sp_efluct1[start:end] ** 2
                        ) /
                        jnp.sum(
                            levels.wocc[start:end] *
                            levels.wstates[start:end]
                        )
                    )
                ),
                lambda _: energies.efluct1q[iq],
                operand=None
            )
        )

        levels.sp_norm = levels.sp_norm.at[start:end].set(
            jnp.real(
                jnp.diagonal(rhomatr_lin)
            )
        )

        if diagonalize:
            _, unitary_lam = jnp.linalg.eigh(lambda_lin, symmetrize_input=False)

        w, v = jnp.linalg.eigh(rhomatr_lin, symmetrize_input=False)
        unitary_rho = jnp.dot(
            v,
            jnp.dot(
                jnp.diag(jnp.sqrt(1.0 / w)),
                jnp.conjugate(v.T)
            )
        )

        if diagonalize:
            unitary = jnp.dot(
                unitary_rho,
                unitary_lam
            )
        else:
            unitary = unitary_rho

        levels.psi = levels.psi.at[start:end,...].set(
            jnp.transpose(
                jnp.reshape(
                    jnp.dot(psi_2d, unitary),
                    shape=(grids.nx, grids.ny, grids.nz, 2, nst),
                    order='F'
                ),
                axes=(4, 3, 0, 1, 2)
            )
        )

        psi_2d = jnp.reshape(
            jnp.transpose(
                levels.psi[start:end,...],
                axes=(2, 3, 4, 1, 0)
            ),
            shape=(-1, nst),
            order='F'
        )

        if construct:
            hmfpsi_2d = jnp.reshape(
                jnp.transpose(
                    levels.hmfpsi[start:end,...],
                    axes=(2, 3, 4, 1, 0)
                ),
                shape=(-1, nst),
                order='F'
            )

            lambda_lin = jnp.dot(
                jnp.conjugate(psi_2d.T), hmfpsi_2d
            ) * grids.wxyz

            static.hmatrix = static.hmatrix.at[iq,:nst,:nst].set(
                jnp.dot(lambda_lin, unitary)
            )

            # if forces.ipair != 0:
            #     delpsi_2d = jnp.reshape(
            #         jnp.transpose(
            #             levels.delpsi[start:end,...],
            #             axes=(2, 3, 4, 1, 0)
            #         ),
            #         shape=(-1, nst),
            #         order='F'
            #     )

            #     lambda_lin = jnp.dot(
            #         jnp.conjugate(psi_2d.T), delpsi_2d
            #     ) * grids.wxyz

            #     static.gapmatrix = static.gapmatrix.at[iq,:nst,:nst].set(
            #         jnp.dot(lambda_lin, unitary)
            #     )

            levels.sp_energy = levels.sp_energy.at[start:end].set(
                jnp.real(
                    jnp.diagonal(static.hmatrix[iq,:nst,:nst])
                )
            )

            # if forces.ipair != 0:
            #     levels.deltaf = levels.deltaf.at[start:end].set(
            #         jnp.diagonal(static.gapmatrix[iq,:nst,:nst]) * levels.pairwg[start:end]
            #     )

            static.gapmatrix = static.gapmatrix.at[iq, :nst, :nst].set(
                jnp.where(
                    forces.tbcs,
                    static.gapmatrix[iq, :nst, :nst] * jnp.eye(nst),
                    static.gapmatrix[iq, :nst, :nst]
                )
            )

            weight = levels.wocc[start:end] * levels.wstates[start:end]
            weightuv = levels.wguv[start:end] * levels.pairwg[start:end] * levels.wstates[start:end]
            lambda_temp = (
                weight[:,jnp.newaxis] *
                static.hmatrix[iq,:nst,:nst] -
                weightuv[:,jnp.newaxis] *
                static.gapmatrix[iq,:nst,:nst]
            )

            lambda_lin = (0.5 + 0.5j) * (lambda_temp - jnp.conjugate(lambda_temp.T))
            energies.efluct1q = energies.efluct1q.at[iq].set(
                jnp.max(jnp.abs(lambda_lin))
            )
            energies.efluct2q = energies.efluct1q.at[iq].set(
                jnp.sqrt(
                    jnp.sum(
                        jnp.real(lambda_lin)**2 +
                        jnp.imag(lambda_lin)**2
                    ) / nst**2
                )
            )

            static.symcond = static.symcond.at[iq,:nst,:nst].set(lambda_lin)

            lambda_lin = (0.5 + 0.5j) * (lambda_temp + jnp.conjugate(lambda_temp.T))

            levels.lagrange = levels.lagrange.at[start:end,...].set(
                jnp.transpose(
                    jnp.reshape(
                        jnp.dot(psi_2d, lambda_lin),
                        shape=(grids.nx, grids.ny, grids.nz, 2, nst),
                        order='F'
                    ),
                    axes=(4, 3, 0, 1, 2)
                )
            )

            static.lambda_save = static.lambda_save.at[iq,:nst,:nst].set(lambda_lin)

            energies.efluct1 = energies.efluct1.at[0].set(jnp.max(energies.efluct1q))
            energies.efluct2 = energies.efluct2.at[0].set(jnp.average(energies.efluct2q))
        else:
            levels.lagrange = levels.lagrange.at[start:end,...].set(
                jnp.transpose(
                    jnp.reshape(
                        jnp.dot(psi_2d, static.lambda_save[iq,:nst,:nst]),
                        shape=(grids.nx, grids.ny, grids.nz, 2, nst),
                        order='F'
                    ),
                    axes=(4, 3, 0, 1, 2)
                )
            )

    return energies, levels, static


def statichf(coulomb, densities, energies, forces, grids, levels, meanfield, moment, params, static):
    firstiter = 1
    addnew = 0.2
    addco = 1.0 - addnew
    taddnew = True

    if params.trestart:
        firstiter = params.iteration + 1
    else:
        params.iteration = 0
        energies, levels, static = diagstep(energies, forces, grids, levels, static, False, True)

    densities = add_density(densities, grids, levels)

    meanfield, coulomb = skyrme(coulomb, densities, forces, grids, meanfield, params, static)

    levels, static = grstep(forces, grids, levels, meanfield, params, static)

    energies, levels, static = diagstep(energies, forces, grids, levels, static, False, True)

    levels = sp_properties(forces, grids, levels, moment)

    # CALL sinfo(wflag)

    # set x0dmp to 3* its value to get faster convergence
    if static.tvaryx_0:
        static.x0dmp = static.x0dmp * 3.0

    # save old pairing strengths for "annealing"
    v0protsav = forces.v0prot
    v0neutsav = forces.v0neut

    tbcssav = forces.tbcs

    for i in range(firstiter, static.maxiter + 1):
        params.iteration = i

        if i <= static.inibcs:
            forces.tbcs = True
        else:
            forces.tbcs = tbcssav

        if i > static.inidiag:
            static.tdiag = False
        else:
            static.tdiag = True

        if static.iteranneal > 0:
            if i < static.iteranneal:
                forces.v0prot = v0protsav + v0protsav * static.pairenhance * (static.iteranneal - i) / (1.0 * static.iteranneal)
                forces.v0prot = v0neutsav + v0neutsav * static.pairenhance * (static.iteranneal - i) / (1.0 * static.iteranneal)
            else:
                forces.v0prot = v0protsav
                forces.v0prot = v0neutsav

        levels, static = grstep(forces, grids, levels, meanfield, params, static)

        if forces.tbcs:
            static.tdiag = True

        energies, levels, static = diagstep(energies, forces, grids, levels, static, static.tdiag, True)

        if taddnew:
            meanfield.upot = meanfield.upot.at[...].set(densities.rho)
            meanfield.bmass = meanfield.bmass.at[...].set(densities.tau)
            meanfield.v_pair = meanfield.v_pair.at[...].set(densities.chi)

        densities.rho = densities.rho.at[...].set(0.0)
        densities.chi = densities.chi.at[...].set(0.0)
        densities.tau = densities.tau.at[...].set(0.0)
        densities.current = densities.current.at[...].set(0.0)
        densities.sdens = densities.sdens.at[...].set(0.0)
        densities.sodens = densities.sodens.at[...].set(0.0)

        densities = add_density(densities, grids, levels)

        # Step 8a: optional constraint step
        # skipped

        if taddnew:
            densities.rho = densities.rho.at[...].set(
                addnew * densities.rho + addco * meanfield.upot
            )
            densities.tau = densities.tau.at[...].set(
                addnew * densities.tau + addco * meanfield.bmass
            )
            densities.chi = densities.chi.at[...].set(
                addnew * densities.chi + addco * meanfield.v_pair
            )

        # Step 8b: construct potentials
        meanfield, coulomb = skyrme(coulomb, densities, forces, grids, meanfield, params, static)

        # calculate and print information
        # skipped

        levels = sp_properties(forces, grids, levels, moment)

        if params.mprint > 0:
            # CALL sinfo(MOD(iter,mprint)==0.AND.wflag)
            pass

        if energies.efluct1 < static.serr and i > 1:
            # CALL write_wavefunctions
            print("break")
            break
        if params.mrest > 0:
            if i % params.mrest == 0:
                # CALL write_wavefunctions
                pass

        # if static.tvaryx_0:
        #     if (energies.ehf < energies.ehfprev and energies.efluct1 < (energies.efluct1prev * (1.0 - 1.0e-5))) or (energies.efluct2 < (energies.efluct2prev * (1.0 - 1.0e-5))):
        #         static.x0dmp = static.x0dmp * 1.005
        #         print("\n\n1111\n\n")
        #     else:
        #         static.x0dmp = static.x0dmp * 0.8
        #         print("\n\n2222\n\n")

        #     if static.x0dmp < static.x0dmpmin:
        #         static.x0dmp = static.x0dmpmin

        #     if static.x0dmp > static.x0dmpmin * 5.0:
        #         static.x0dmp = static.x0dmpmin * 5.0

        #     energies.efluct1prev = energies.efluct1
        #     energies.efluct2prev = energies.efluct2
        #     energies.ehfprev = energies.ehf

    return coulomb, densities, energies, forces, grids, levels, meanfield, moment, params, static
