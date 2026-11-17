"""
Local nuclear energy density functionals by automatic differentiation.

A functional is one Python function, its energy density

    e(densities, force, grid, use_coulomb) -> (e_total (nx, ny, nz), e_pair (2, nx, ny, nz))

of the local densities (rho, tau, J, j, s, chi and, for J^2 forces, J^S and T), including the kinetic energy, Coulomb
exchange and pairing (e_pair is the pairing part per isospin, already contained in e_total). The direct Coulomb energy
is added outside, 1/2 int rho_p V_C with field V_C, as in the hand-coded path. Every mean field the solver uses is a
functional derivative of E = int e, all obtained in one reverse-mode pass:

    upot   = dE/drho  (+ V_C on protons, + constraint)     bmass = dE/dtau         wlspot = dE/dJ
    aq     = dE/dj                                         spot  = dE/ds           v_pair = -1/2 dE/dchi
    wtens  = dE/dJ^S (off-diagonal halved: they appear twice in sum_mu,nu)         fspin  = dE/dT

so fields and energy are consistent by construction (variational on the grid). A new functional is an energy density
registered with @register('name') and selected with Force.functional; the solver, TDHF/TDHFB and the response and
derivative machinery (physics/response.py, physics/second_order.py) need no change.

Registered: 'skyrme_ad' (Skyrme in coupling-constant form, Bender, Heenen & Reinhard RMP 75, 121) and 'fayans'
(Fayans, JETP Lett. 68, 169; Reinhard & Nazarewicz, PRC 95, 064328; ported from SkyAx's fyfunc).
"""
import dataclasses
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from jax_hfbfft.physics.meanfield import (Meanfield, compute_laplacian, compute_gradient, compute_divergence,
                                          compute_curl)

FLOOR = 1.0e-25          # keeps rho^alpha differentiable where rho = 0 (as meanfield.py / energies.py)
FUNCTIONALS = {}


def register(name):
    """Register an energy density e(densities, force, grid, use_coulomb) -> (e_total, e_pair) under `name`."""
    def deco(fn):
        FUNCTIONALS[name] = fn
        return fn
    return deco


# -- building blocks (the same derivative matrices as the hand-coded path) ---------------------------------------
def _dot(a, b):
    return jnp.sum(a * b, axis=0)


def _vlap(v, grid):
    return jnp.stack([compute_laplacian(v[k], grid) for k in range(3)], axis=0)


def _pos(x, floor):
    """max(x, floor), except for complex (gauge-rotated) densities, which take the principal branch."""
    return x if jnp.iscomplexobj(x) else jnp.maximum(x, floor)


def kinetic(d, f):
    return f.h2m[0] * d.tau[0] + f.h2m[1] * d.tau[1]


def skyrme_volume(d, f, grid):
    """
    Skyrme EDF in coupling-constant form, t = 0 (n + p) and t = 1 (p - n), with the Galilean-invariant time-odd
    partners (C^j = -C^tau, C^nabla j = C^nabla J):
        sum_t (C^rho_t + C^rhoD_t rho0^alpha) rho_t^2 + C^tau_t (rho_t tau_t - j_t^2) + C^Drho_t rho_t Lap rho_t
              + C^nablaJ_t (rho_t div J_t + s_t . curl j_t)
    """
    rho0, rho1 = d.rho[0] + d.rho[1], d.rho[1] - d.rho[0]
    tau0, tau1 = d.tau[0] + d.tau[1], d.tau[1] - d.tau[0]
    j0, j1 = d.current[0] + d.current[1], d.current[1] - d.current[0]
    J0, J1 = d.sodens[0] + d.sodens[1], d.sodens[1] - d.sodens[0]
    s0, s1 = d.sdens[0] + d.sdens[1], d.sdens[1] - d.sdens[0]
    ra = _pos(rho0, FLOOR) ** f.power
    e = (f.Crho0 + f.Crho0D * ra) * rho0 ** 2 + (f.Crho1 + f.Crho1D * ra) * rho1 ** 2
    e = e + f.Ctau0 * (rho0 * tau0 - _dot(j0, j0)) + f.Ctau1 * (rho1 * tau1 - _dot(j1, j1))
    e = e + f.Cdrho0 * rho0 * compute_laplacian(rho0, grid) + f.Cdrho1 * rho1 * compute_laplacian(rho1, grid)
    e = e + f.CdJ0 * (rho0 * compute_divergence(J0, grid) + _dot(s0, compute_curl(j0, grid)))
    e = e + f.CdJ1 * (rho1 * compute_divergence(J1, grid) + _dot(s1, compute_curl(j1, grid)))
    return e


def skyrme_tensor(d, f):
    """J^2 terms: 1/2 C^J_t J_t^2 (vector part), C^J_t (J^S_t)^2 and the time-odd partner -C^J_t s_t . T_t."""
    J0, J1 = d.sodens[0] + d.sodens[1], d.sodens[1] - d.sodens[0]
    e = 0.5 * (f.CJ0 * _dot(J0, J0) + f.CJ1 * _dot(J1, J1))
    if d.jsym is not None:
        sq = lambda j: jnp.sum(j[:3] ** 2, axis=0) + 2.0 * jnp.sum(j[3:] ** 2, axis=0)
        e = e + f.CJ0 * sq(d.jsym[0] + d.jsym[1]) + f.CJ1 * sq(d.jsym[1] - d.jsym[0])
    if d.tspin is not None:
        s0, s1 = d.sdens[0] + d.sdens[1], d.sdens[1] - d.sdens[0]
        T0, T1 = d.tspin[0] + d.tspin[1], d.tspin[1] - d.tspin[0]
        e = e - f.CJ0 * _dot(s0, T0) - f.CJ1 * _dot(s1, T1)
    return e


def spin_channel(d, f, grid):
    """Independent time-odd s-channel terms: (c_s + c_ds rho0^alpha) s_t^2 + c_sDs s_t . Lap s_t."""
    ra = _pos(d.rho[0] + d.rho[1], FLOOR) ** f.power
    s0, s1 = d.sdens[0] + d.sdens[1], d.sdens[0] - d.sdens[1]
    return ((f.c_s0 + f.c_ds0 * ra) * _dot(s0, s0) + (f.c_s1 + f.c_ds1 * ra) * _dot(s1, s1)
            + f.c_sDs0 * _dot(s0, _vlap(s0, grid)) + f.c_sDs1 * _dot(s1, _vlap(s1, grid)))


def coulomb_exchange(d, f, use_coulomb):
    """Slater approximation, -3/4 (3/pi)^(1/3) e^2 rho_p^(4/3)."""
    if f.ex == 0:
        return 0.0
    return jnp.where(use_coulomb, -0.75 * f.slate * _pos(d.rho[1], 0.0) ** (4.0 / 3.0), 0.0)


def regularized_coupling(v, rho, f):
    """
    Bulgac-Yu regularised coupling v / (1 + v K) per isospin (Force.pair_reg_ecut), v (2, ...) the bare one;
    rho (2, ...) the neutron and proton densities. Identity when pair_reg_ecut = 0.
    """
    if not f.pair_reg_ecut:
        return v
    r0, r1 = rho[0] + rho[1], rho[1] - rho[0]
    B = jnp.stack([f.h2m[0] + f.Ctau0 * r0 - f.Ctau1 * r1, f.h2m[1] + f.Ctau0 * r0 + f.Ctau1 * r1])
    kf = (3.0 * jnp.pi ** 2 * _pos(rho, FLOOR)) ** (1.0 / 3.0)
    kc = jnp.sqrt(kf ** 2 + f.pair_reg_ecut / B)
    x = kf / kc
    K = kc / (4.0 * jnp.pi ** 2 * B) * (1.0 - 0.5 * x * jnp.log((1.0 + x) / (1.0 - x)))
    return v / (1.0 + v * K)


def contact_pairing(d, f):
    """Contact pairing per isospin: -v_q chi_q^2, v_q = V0_q g(rho), g = 1 (ipair 5) or 1 - rho/rho0pr (ipair 6),
    regularised (Force.pair_reg_ecut) when set."""
    if f.ipair not in (5, 6):
        return jnp.zeros_like(d.chi)
    g = 1.0 if f.ipair == 5 else 1.0 - (d.rho[0] + d.rho[1]) / f.rho0pr
    v0 = jnp.stack([jnp.asarray(f.v0neut, float), jnp.asarray(f.v0prot, float)])[:, None, None, None]
    return -regularized_coupling(v0 * g * jnp.ones_like(d.chi), d.rho, f) * d.chi ** 2


@register('skyrme_ad')
def skyrme(d, f, grid, use_coulomb):
    """The Skyrme functional of the hand-coded path, written once as an energy density."""
    pair = contact_pairing(d, f)
    e = (kinetic(d, f) + skyrme_volume(d, f, grid) + skyrme_tensor(d, f) + spin_channel(d, f, grid)
         + coulomb_exchange(d, f, use_coulomb) + jnp.sum(pair, axis=0))
    return e, pair


# -- Fayans ---------------------------------------------------------------------------------------------------------
FY_KEYS = ('avp', 'h1vp', 'h2vp', 'avm', 'h1vm', 'h2vm', 'asp', 'hsp', 'hgradp', 'asm', 'hsm', 'hgradm',
           'fxi_n', 'h1xi_n', 'hgradxi_n', 'fxi_p', 'h1xi_p', 'hgradxi_p', 'c0nabj', 'c1nabj', 'rho0fy',
           'sigmavol', 'rho0xi', 'gammaxi', 'pairstab')
_FY = {k: i for i, k in enumerate(FY_KEYS)}
STAB_CAP = 50.0


def _fy_scales(fy, h2m):
    """SkyAx ini_fyfunc auxiliaries: r0, eferm, efac, pfac, c0fac."""
    h2mav = 0.5 * (h2m[0] + h2m[1])
    rho0fy = fy[_FY['rho0fy']]
    r0 = (0.375 / (jnp.pi * rho0fy)) ** (1.0 / 3.0)
    eferm = (1.125 * jnp.pi) ** (2.0 / 3.0) * h2mav / r0 ** 2
    return dict(r0=r0, efac=eferm * rho0fy * 2.0 / 3.0, pfac=2.0 * eferm / (3.0 * rho0fy),
                c0fac=2.0 * eferm / (3.0 * rho0fy) * r0 ** 2)


def fayans_parameters(block, h2m):
    """(fy array, t4, b4p) from a force file's 'fayans:' block; spin-orbit -clsp rho div J - clsm rho1 div J1."""
    b = dict(block)
    for k, v in (('rho0fy', 0.08), ('sigmavol', 1.0 / 3.0), ('rho0xi', 0.16), ('gammaxi', 2.0 / 3.0),
                 ('hsp', 0.0), ('asm', 0.0), ('hsm', 0.0), ('hgradm', 0.0), ('pairstab', -0.3)):
        b.setdefault(k, v)
    dfx = b.get('delfxi', 0.0)                     # isovector pairing, SkyAx isovectormodel 2: f_n,p = f -+ delfxi
    for q, sgn in (('n', -1.0), ('p', 1.0)):
        b[f'fxi_{q}'], b[f'h1xi_{q}'], b[f'hgradxi_{q}'] = b['fxi'] + sgn * dfx, b['h1xi'], b['hgradxi']
    fy = jnp.array([float(b[k]) for k in FY_KEYS])
    c0fac = float(_fy_scales(fy, np.asarray(h2m, float))['c0fac'])
    clsp, clsm = c0fac * b['c0nabj'], c0fac * b['c1nabj']
    return fy, 2.0 * (clsp - clsm), 2.0 * clsm


def _sq_grad(field, grid):
    gx, gy, gz = compute_gradient(field, grid)
    return gx * gx + gy * gy + gz * gz


def fayans_ph(d, f, grid):
    """Fayans volume and surface energy density."""
    fy, s = f.fy, _fy_scales(f.fy, f.h2m)
    p = lambda k: fy[_FY[k]]
    rs, rm = d.rho[0] + d.rho[1], d.rho[1] - d.rho[0]
    inv = 0.5 / p('rho0fy')
    xp, xm = _pos(rs, FLOOR) * inv, rm * inv
    xps = xp ** p('sigmavol')
    d2 = _sq_grad(rs, grid) * s['r0'] ** 2 * inv ** 2
    d2m = _sq_grad(rm, grid) * s['r0'] ** 2 * inv ** 2
    e = (p('avp') * xp * xp * (1.0 - p('h1vp') * xps) / (1.0 + p('h2vp') * xps)
         + p('avm') * xm * xm * (1.0 - p('h1vm') * xp) / (1.0 + p('h2vm') * xp)
         + p('asp') * d2 / (1.0 + p('hsp') * xps + p('hgradp') * d2)
         + p('asm') * d2m / (1.0 + p('hsm') * xp + p('hgradm') * d2m))
    return s['efac'] * e


def fayans_pairing(d, f, grid):
    """Fayans pairing per isospin, rhfac_q(rho, grad rho) chi_q^2 (rhfac < 0 attracts); zero for ipair 0."""
    if f.ipair == 0:
        return jnp.zeros_like(d.chi)
    fy, s = f.fy, _fy_scales(f.fy, f.h2m)
    p = lambda k: fy[_FY[k]]
    rs = d.rho[0] + d.rho[1]
    xg = (_pos(rs, FLOOR) / p('rho0xi')) ** p('gammaxi')
    d2 = _sq_grad(rs, grid) * (s['r0'] / p('rho0xi')) ** 2
    rh = jnp.stack([s['pfac'] * (p(f'fxi_{q}') + p(f'h1xi_{q}') * xg + p(f'hgradxi_{q}') * d2) for q in 'np'])
    return rh * d.chi ** 2


@register('fayans')
def fayans(d, f, grid, use_coulomb):
    """Fayans: volume/surface and pairing functionals, Skyrme-form spin-orbit (t4/b4p), no effective mass."""
    pair = fayans_pairing(d, f, grid)
    e = (kinetic(d, f) + fayans_ph(d, f, grid) + skyrme_volume(d, f, grid) + spin_channel(d, f, grid)
         + coulomb_exchange(d, f, use_coulomb) + jnp.sum(pair, axis=0))
    return e, pair


# -- fields and energy from any registered functional ---------------------------------------------------------------
def _energy(d, f, grid, use_coulomb):
    return grid.wxyz * jnp.sum(FUNCTIONALS[f.functional](d, f, grid, use_coulomb)[0])


def _pair_energy(rho, d, f, grid, use_coulomb):
    return grid.wxyz * jnp.sum(FUNCTIONALS[f.functional](dataclasses.replace(d, rho=rho), f, grid, use_coulomb)[1])


@jax.jit
def meanfield(densities, coulomb_potential, constraint_potential, force, grid, use_coulomb):
    """The Meanfield of force.functional: every field a derivative of the energy (one reverse pass)."""
    w = grid.wxyz
    g = jax.grad(_energy)(densities, force, grid, use_coulomb)
    upot = g.rho / w + constraint_potential
    upot = upot.at[1].add(jnp.where(use_coulomb, coulomb_potential, 0.0))
    bmass = g.tau / w
    aq = g.current / w
    v_pair = -0.5 * g.chi / w
    if force.functional == 'fayans' and force.ipair != 0:
        # SkyAx pair stabilisation: the gap field (not the energy) of isospin q is scaled by 1 + (pairstab/E_q)^2
        e_q = w * jnp.sum(v_pair * densities.chi, axis=(1, 2, 3))
        f_q = 1.0 + (force.fy[_FY['pairstab']] / jnp.maximum(e_q, 1.0e-6)) ** 2
        v_pair = v_pair * jnp.minimum(f_q, STAB_CAP)[:, None, None, None]
    # pairing rearrangement correction for the Koopman sum, -1/2 int rho dE_pair/drho (meanfield.py's ecorrp)
    gp = jax.grad(_pair_energy)(densities.rho, densities, force, grid, use_coulomb) / w
    ecorrp = -0.5 * w * jnp.sum(gp * densities.rho)
    wtens = None
    if densities.jsym is not None:
        wtens = (g.jsym / w).at[:, 3:].multiply(0.5)
    fspin = g.tspin / w if densities.tspin is not None else None
    divaq = jnp.stack([compute_divergence(aq[q], grid) for q in (0, 1)])
    dbmass = jnp.stack([jnp.stack(compute_gradient(bmass[q], grid)) for q in (0, 1)])
    return Meanfield(upot=upot, bmass=bmass, divaq=divaq, v_pair=v_pair, aq=aq, spot=g.sdens / w,
                     wlspot=g.sodens / w, dbmass=dbmass, ecorrp=jnp.asarray(ecorrp), wtens=wtens, fspin=fspin)


def integrated_energy(densities, force, grid, coulomb_potential=None, mass_number=1, use_coulomb=True,
                      wocc=None, wstates=None, sp_kinetic=None, sp_energy=None, e_ln=None):
    """
    Energies of force.functional. ehfint = int e + 1/2 int rho_p V_C - E_zpe (+ Lipkin-Nogami); the Koopman sum uses
    the general rearrangement R = E - 1/2 (T + sum_d int (dE/dd) d) over all densities but chi, which reproduces the
    hand-coded e3corr + ecorc + ecorrp - E_pair for Skyrme.
    """
    from jax_hfbfft.physics.energies import Energies
    w = grid.wxyz
    e_tot, e_pair = FUNCTIONALS[force.functional](densities, force, grid, use_coulomb)
    E = w * jnp.sum(e_tot)
    epair_q = -w * jnp.sum(e_pair, axis=(1, 2, 3))           # positive for attraction, as the hand path reports
    ehft = w * jnp.sum(kinetic(densities, force))
    exch = w * jnp.sum(coulomb_exchange(densities, force, use_coulomb))
    ehfc_dir = 0.0
    if coulomb_potential is not None:
        ehfc_dir = jnp.where(use_coulomb, 0.5 * w * jnp.sum(densities.rho[1] * coulomb_potential), 0.0)
    e_zpe = jnp.where(((force.zpe == 1) | (force.zpe == 3)) & (mass_number > 1),
                      17.3 / jnp.maximum(1.0, mass_number) ** 0.2, 0.0)
    eln = jnp.sum(e_ln) if e_ln is not None else 0.0
    ehfint = E + ehfc_dir - e_zpe + eln
    g = jax.grad(_energy)(densities, force, grid, use_coulomb)
    leaves = lambda t: [x for x in jax.tree_util.tree_leaves(dataclasses.replace(t, chi=None))]
    contraction = w * sum(jnp.sum(a * b) for a, b in zip(leaves(g), leaves(densities)))
    R = E - 0.5 * (ehft + contraction)
    if wocc is not None and wstates is not None and sp_kinetic is not None and sp_energy is not None:
        tke = jnp.sum(wocc * wstates * sp_kinetic)
        ehf = jnp.sum(wocc * wstates * (sp_kinetic + sp_energy)) / 2.0 + R - e_zpe + eln
    else:
        tke, ehf = ehft, 0.0
    z = jnp.zeros(())
    return Energies(
        ehft=ehft, ehf0=E - ehft - exch + jnp.sum(epair_q), ehf1=z, ehf2=z, ehf3=z, ehfls=z, ehflsodd=z, ehf_odd=z,
        ehfc=ehfc_dir + exch, ecorc=z, ehfint=ehfint, ehf=ehf, tke=tke, e3corr=R, e_zpe=e_zpe,
        efluct1=jnp.zeros(1), efluct1q=jnp.zeros(2), efluct2=jnp.zeros(1), efluct2q=jnp.zeros(2),
        orbital=jnp.zeros(3), spin=jnp.zeros(3), total_angmom=jnp.zeros(3), epair=epair_q,
        ehfCrho0=z, ehfCrho1=z, ehfCdrho0=z, ehfCdrho1=z, ehfCtau0=z, ehfCtau1=z, ehfCdJ0=z, ehfCdJ1=z,
        ehfCj0=z, ehfCj1=z)
