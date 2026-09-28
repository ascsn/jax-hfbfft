#!/usr/bin/env python3
"""
TDHF heavy-ion collision, with animation.

    python3 tdhf_collide.py                        # 16O+16O fusion
    python3 tdhf_collide.py --b 5.0                # grazing, re-separates
    python3 tdhf_collide.py --e-cm 120             # too fast to fuse
    python3 tdhf_collide.py --render out/coll_O16_E34_b0.npz

Environment is handled by tdhf_env.py, imported at the top -- no wrapper needed.
The one exception is a shell where `python3` itself will not start ("error while
loading shared libraries: libpython3.11.so.1.0"); that is settled by the dynamic
linker before any Python runs, so use ./tdhf_env.sh for it, or export
LD_LIBRARY_PATH from your profile.  Run `python3 tdhf_env.py` to check.

Converge ONE nucleus, make two copies of it, put them at opposite ends of an
elongated box, boost them toward each other, and propagate.  They approach,
touch, form a neck, and then -- depending on energy and impact parameter --
either fuse into a single hot blob or tear apart again.

WHAT SETS THE OUTCOME
---------------------
  --e-cm   Total kinetic energy in the centre-of-mass frame, MeV.  For 16O+16O
           the Coulomb barrier is ~10 MeV.  Below it they bounce off without
           touching.  From there up to ~50 MeV they fuse.  Push past ~100 MeV
           and the system punches through and comes out the other side.
  --b      Impact parameter, fm.  b=0 is head-on.  Increasing b spins the
           composite up; past the critical value fusion fails and you get a
           neck that stretches and snaps -- visually the best run of the set,
           and the closest thing this code can currently show to fission.

READING THE OUTPUT
------------------
Same headline number as every other TDHF run here: dE/E, plotted underneath
the movie.  Energy is exactly conserved by the continuum equations, so drift
measures everything that is wrong at once.  Under 1e-6 is healthy.  A sudden
jump means density reached the box wall -- the run is over at that point and
frames after it are artefact.  The script prints the boundary density at the
end so you can check.

THE TWO THINGS THAT MAKE THIS DIFFERENT FROM A ONE-NUCLEUS RUN
--------------------------------------------------------------
1. The centre-of-mass correction.  The composite of mass A_tot needs
   h2m * (A_tot-1)/A_tot, not the fragment's (A-1)/A.  run_hfb applies no c.m.
   correction itself, so each fragment is relaxed directly in the composite's
   Hamiltonian; see matched_forces().  The stationarity residual printed during
   setup is the check that it worked.

2. Only occupied orbitals are propagated.  The static solver carries empty
   states (20 neutron states for 8 neutrons) and propagating them costs full
   time per step while contributing exactly nothing to any density.  Dropping
   them is exact, not an approximation, and it is worth a factor of ~2.5.

NO PAIRING
----------
TDHF needs a Slater determinant, so this runs at ipair=0.  Pairing matters for
the fusion barrier and matters a great deal for anything resembling fission;
that needs TDHFB, which this code does not have.
"""

# Must run before jax or matplotlib are imported: it puts ./src on sys.path,
# repairs matplotlib's missing pyparsing, and sets JAX_ENABLE_X64 / XLA_FLAGS,
# neither of which JAX re-reads after import.
import tdhf_env
tdhf_env.setup()

import sys
import time
import argparse
import dataclasses
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from jax_hfbfft.core.grid import Grid
from jax_hfbfft.core.force import Force
from jax_hfbfft.physics.solver import SolverConfig, run_hfb, apply_cm_correction
from jax_hfbfft.physics.densities import compute_densities
from jax_hfbfft.physics.dynamics import (
    TDConfig, TDState, tdhf_step, build_fields, compute_moments, total_energy,
    apply_boost, stationarity_residual, orthonormality_error,
)

# name: (Z, N, npsi_n).  Every entry must have a decent closed-shell HF ground
# state, because TDHF runs at ipair=0 -- see the module docstring.
NUCLEI = {
    'O16':  (8, 8, 20),
    'C12':  (6, 6, 16),
    'Mg24': (12, 12, 26),
    'Ca40': (20, 20, 40),
    'Ca48': (20, 28, 34),
    'Ni56': (28, 28, 40),
}

# Static iterations that are past convergence but BEFORE the solver collapses.
#
# run_hfb at ipair=0 eventually leaks the least-bound occupied orbitals into
# unbound box states: the energy drops well below the true ground state, the
# peak density climbs past saturation, and a flat density pedestal reaches the
# box walls.  It is not a convergence failure that more iterations would cure --
# it is a divergence, and the collapsed state has a LOWER energy and a LOWER
# efluct than the correct one, so neither of the usual criteria notices.
#
# Measured with diag_static.py on a 26^2 x 52 box.  The onset tracks how weakly
# bound the highest occupied orbital is: 16O (deeply bound) survived 300
# iterations, 40Ca went at ~190, and 48Ca -- whose valence f7/2 neutrons sit
# near -9.9 MeV, closest to the continuum -- went at ~110.  Re-measure with
# diag_static.py before trusting a nucleus that is not listed here, and expect
# the window to shrink for anything more weakly bound.
SAFE_STATIC_ITER = {
    'O16': 300, 'C12': 300, 'Mg24': 200,
    'Ca40': 150, 'Ca48': 80, 'Ni56': 100,
}


# ── Force bookkeeping ────────────────────────────────────────────────────────

def matched_forces(force: Force, a_frag: int, a_tot: int):
    """
    Return (force_for_run_hfb, force_for_propagation) such that a fragment of
    mass a_frag is relaxed in exactly the Hamiltonian the composite of mass
    a_tot is propagated with.

    run_hfb does NOT apply the (A-1)/A correction itself, so the fragment can be
    relaxed directly in the composite's Hamiltonian: both returns are
    apply_cm_correction(force, a_tot).  (a_frag is kept in the signature so
    callers do not change.)  For zpe != 0 apply_cm_correction is a no-op.
    """
    if getattr(force, 'zpe', 0) != 0 or a_frag <= 1:
        return force, force
    force_td = apply_cm_correction(force, a_tot)
    return force_td, force_td


# ── Building the two-nucleus initial state ───────────────────────────────────

def occupied_only(psi, wocc, isospin, tol=1e-6):
    """
    Keep only the occupied orbitals.

    Raises on fractional occupation for the same reason prepare_tdhf_state does:
    TDHF is an evolution of a Slater determinant, and propagating BCS weights
    with frozen v^2 violates the continuity equation.
    """
    w = np.asarray(wocc)
    frac = np.minimum(np.abs(w), np.abs(1.0 - w))
    if frac.max() > tol:
        raise SystemExit(
            f"Static state has {(frac > tol).sum()} fractionally occupied states "
            f"(worst {frac.max():.3e}).  TDHF needs sharp occupations -- run with "
            f"ipair=0."
        )
    keep = np.where(np.round(w) > 0.5)[0]
    return psi[keep], jnp.ones(len(keep)), isospin[keep], keep


def translate(psi, grid, dx_fm=0.0, dz_fm=0.0):
    """
    Rigid translation by whole grid points.

    Snapping to the grid makes this exact -- a roll, not an interpolation -- at
    the cost of quantising the requested offset to dx.  The wrap-around is
    harmless because the fragment density is ~1e-8 at the box edge; the caller
    prints the boundary density so that assumption stays checkable.
    """
    nx_shift = int(round(dx_fm / grid.dx))
    nz_shift = int(round(dz_fm / grid.dz))
    out = jnp.roll(psi, nx_shift, axis=2)
    out = jnp.roll(out, nz_shift, axis=4)
    return out, nx_shift * grid.dx, nz_shift * grid.dz


def lowdin_orthonormalize(psi, isospin, wxyz):
    """
    Symmetric (Lowdin) orthonormalisation within each isospin block.

    Two copies of the same nucleus 16 fm apart are very nearly orthogonal, but
    "very nearly" is not "exactly", and the Slater determinant the propagator
    assumes requires exactly.  Lowdin is the right choice over Gram-Schmidt
    here because it is symmetric: it does not privilege one fragment over the
    other, so it cannot bias the initial condition toward either side.

    Neutron-proton overlaps are deliberately NOT included.  They are different
    species and are not required to be orthogonal.

    Returns (psi, max|S - I| before orthonormalisation).
    """
    psi = np.asarray(psi)
    iso = np.asarray(isospin)
    worst = 0.0
    out = np.array(psi)
    for iq in (0, 1):
        idx = np.where(iso == iq)[0]
        if len(idx) == 0:
            continue
        flat = psi[idx].reshape(len(idx), -1)
        S = (flat.conj() @ flat.T) * wxyz
        worst = max(worst, float(np.abs(S - np.eye(len(idx))).max()))
        ev, U = np.linalg.eigh(S)
        if ev.min() <= 0:
            raise SystemExit(
                f"Overlap matrix for isospin {iq} is singular (min eigenvalue "
                f"{ev.min():.2e}).  The fragments are on top of each other -- "
                f"increase --sep."
            )
        s_inv_half = (U * ev**-0.5) @ U.conj().T
        # phi_i = sum_j psi_j (S^-1/2)_{ji}: with states as ROWS that is the
        # TRANSPOSE.  Without it the result is orthonormal only when S is real;
        # boosted fragments make S complex, and the old form left overlaps of
        # ~2e-2 (worse than before), which the first time step then silently
        # repaired -- the constant ~4e-3 'drift' of Int rho in every paired run.
        out[idx] = (s_inv_half.T @ flat).reshape(psi[idx].shape)
    return jnp.asarray(out), worst


def make_td_state(psi, wocc, isospin, coulomb_solver, grid, force, use_coulomb=True):
    """Assemble a TDState from bare arrays (the composite is not a SolverState)."""
    d, mf, w = build_fields(psi, wocc, isospin, grid, force,
                            coulomb_solver, use_coulomb)
    return TDState(psi=psi, wocc=wocc, isospin=isospin, densities=d,
                   meanfield=mf, coulomb_solver=coulomb_solver, wcoul=w,
                   time=0.0, step=0)


# ── Diagnostics ──────────────────────────────────────────────────────────────

def fragment_separation(densities, grid):
    """
    Distance between the density centroids of the z<0 and z>0 half-spaces.

    A crude measure and deliberately so: it is defined at every instant, needs
    no fragment-finding, and does the one job the movie needs -- collapsing to a
    small value when the system becomes one object and growing again if it comes
    apart.  It is measured along the BEAM axis, so for a non-central collision,
    once the composite has rotated appreciably it reports the projection rather
    than the true fragment distance.
    """
    rho = densities.rho[0] + densities.rho[1]
    _, _, Z = grid.get_meshgrid()
    w = grid.wxyz
    hi = Z > 0
    lo = ~hi
    m_hi = jnp.sum(rho * hi) * w
    m_lo = jnp.sum(rho * lo) * w
    z_hi = jnp.sum(rho * Z * hi) * w / m_hi
    z_lo = jnp.sum(rho * Z * lo) * w / m_lo
    return float(z_hi - z_lo)


def boundary_density(densities, grid):
    """Max density on the six faces of the box -- the box-is-too-small alarm."""
    rho = np.asarray(densities.rho[0] + densities.rho[1])
    faces = [rho[0], rho[-1], rho[:, 0], rho[:, -1], rho[:, :, 0], rho[:, :, -1]]
    return max(float(np.abs(f).max()) for f in faces)


def check_bound(psi, wocc, isospin, grid, mass_number, label):
    """
    Verify a static fragment is actually BOUND before it is used.

    The static solver can converge on a state that is lower in energy than the
    true ground state and completely unphysical: an over-dense core plus a
    fraction of a nucleon spread flat across the box in an unbound orbital.  For
    40Ca on a 26^2 x 52 box it happens around iteration 200 and costs 2.6
    nucleons, and the density plateau reaches the walls.

    efluct does NOT catch it.  efluct measures how stationary the state is, and
    the collapsed state is perfectly stationary -- it recovers to 6e-4, lower
    than it was beforehand, so an efluct threshold accepts it.  The energy does
    not catch it either, because the bogus state is 30 MeV LOWER.  What does
    catch it is asking how much of the nucleus is not in the nucleus.

    Returns (a_out, face).  Prints a loud warning; deliberately does not raise,
    because a small a_out is normal and the caller may legitimately want to
    proceed with a marginal state.
    """
    zeros = jnp.zeros_like(wocc)
    dens = compute_densities(psi, wocc, zeros, jnp.ones_like(wocc),
                             jnp.ones_like(wocc), isospin, grid)
    rho = np.asarray(dens.rho[0] + dens.rho[1])
    X, Y, Z = (np.asarray(a) for a in grid.get_meshgrid())
    r_nuc = 1.2 * mass_number ** (1.0 / 3.0)
    far = np.sqrt(X**2 + Y**2 + Z**2) > 2.5 * r_nuc
    a_out = float((rho * far).sum() * grid.wxyz)
    face = max(float(rho[0].max()), float(rho[-1].max()),
               float(rho[:, 0].max()), float(rho[:, -1].max()),
               float(rho[:, :, 0].max()), float(rho[:, :, -1].max()))
    print(f"  boundness: peak rho = {rho.max():.4f} fm^-3, "
          f"A outside {2.5*r_nuc:.1f} fm = {a_out:.4f}, box face = {face:.2e}",
          flush=True)
    if a_out > 0.05 or face > 1e-4 or rho.max() > 0.20:
        print(f"  !! {label} IS NOT PROPERLY BOUND.  A flat density pedestal "
              f"reaching the box walls and a peak above ~0.19 fm^-3 mean the\n"
              f"     static solver has dumped density into unbound box states. "
              f"This is NOT fixed by running longer -- it gets worse.\n"
              f"     Reduce --static-iter (150 is past convergence and before "
              f"the collapse for 40Ca) and re-check this line.", flush=True)
    return a_out, face


# ── The run ──────────────────────────────────────────────────────────────────

def run_case(args):
    nuc2 = args.nucleus2 or args.nucleus
    z1, n1, npsi1 = NUCLEI[args.nucleus]
    z2, n2, npsi2 = NUCLEI[nuc2]
    a1, a2 = z1 + n1, z2 + n2
    a_tot = a1 + a2

    grid = Grid.create(nx=args.nx, ny=args.nx, nz=args.nz,
                       dx=args.dx, dy=args.dx, dz=args.dx)
    print(f"box {args.nx}x{args.nx}x{args.nz} @ {args.dx} fm  "
          f"= {args.nx*args.dx:.0f} x {args.nx*args.dx:.0f} x {args.nz*args.dx:.0f} fm",
          flush=True)
    print(f"reaction {args.nucleus} + {nuc2}   A = {a1} + {a2} = {a_tot}",
          flush=True)

    force_bare = Force.from_name("SLy4", ipair=0)
    # Each fragment needs its OWN pre-scaling but they share the composite force.
    force_st1, force_td = matched_forces(force_bare, a1, a_tot)
    force_st2, _ = matched_forces(force_bare, a2, a_tot)
    if args.time_odd == 'skyrme':
        force_td = force_td.with_time_odd_from_skyrme()

    # Both fragments share one solver config, so the safe count is the smaller
    # of the two: 40Ca + 48Ca must stop at 48Ca's limit, not 40Ca's.
    n_iter = args.static_iter
    if n_iter is None:
        n_iter = min(SAFE_STATIC_ITER.get(args.nucleus, 150),
                     SAFE_STATIC_ITER.get(nuc2, 150))
        print(f"  static iterations {n_iter} (pre-collapse limit for "
              f"{args.nucleus}/{nuc2}; see SAFE_STATIC_ITER)", flush=True)

    cfg = SolverConfig(max_iterations=n_iter, verbose=False,
                       bcs_start=30, diag_start=30, output_interval=9999,
                       tvaryx_0=True)

    def static(name, zz, nn, npsi, fst):
        t0 = time.time()
        s = run_hfb(grid, fst, nucleus_z=zz, nucleus_n=nn, npsi_n=npsi,
                    config=cfg)
        print(f"static {name}: E={float(s.energies.ehfint):.4f} MeV  "
              f"iters={s.iteration}  efluct={float(s.efluct):.2e}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        p, w, iso, keep = occupied_only(s.psi, s.wocc, s.isospin)
        print(f"  {len(keep)} occupied orbitals of {s.psi.shape[0]} "
              f"({int((iso == 0).sum())} n + {int((iso == 1).sum())} p)",
              flush=True)
        check_bound(p, w, iso, grid, zz + nn, name)
        one = make_td_state(p, w, iso, s.coulomb_solver, grid, force_td)
        resid = float(stationarity_residual(one, grid, force_td))
        print(f"  stationarity residual ||(h-<h>)psi|| = {resid:.3e} MeV "
              f"(composite Hamiltonian)", flush=True)
        if resid > 1.0:
            print("  !! large -- this fragment is NOT stationary in the "
                  "propagation Hamiltonian; expect spurious breathing",
                  flush=True)
        return s, p, w, iso

    s1, psi1, wocc1, iso1 = static(args.nucleus, z1, n1, npsi1, force_st1)
    if nuc2 == args.nucleus:
        s2, psi2, wocc2, iso2 = s1, psi1, wocc1, iso1
    else:
        s2, psi2, wocc2, iso2 = static(nuc2, z2, n2, npsi2, force_st2)

    # ── place, boost, combine ────────────────────────────────────────────────
    # Offsets are weighted by the OTHER fragment's mass so the centre of mass
    # sits at the origin: an asymmetric pair placed symmetrically would drift
    # bodily through the box and drag the whole collision off centre.
    zl = -args.sep * a2 / a_tot
    zh = +args.sep * a1 / a_tot
    bl = -args.b * a2 / a_tot
    bh = +args.b * a1 / a_tot
    psi_lo, bx_l, bz_l = translate(psi1, grid, dx_fm=bl, dz_fm=zl)
    psi_hi, bx_h, bz_h = translate(psi2, grid, dx_fm=bh, dz_fm=zh)
    print(f"  {args.nucleus} at z={bz_l:+.1f} x={bx_l:+.1f} fm,  "
          f"{nuc2} at z={bz_h:+.1f} x={bx_h:+.1f} fm  "
          f"(sep {args.sep}, b {args.b}, snapped to grid)", flush=True)

    # Each nucleon of fragment i carries momentum hbar*k_i, so fragment momentum
    # is A_i*hbar*k_i.  Zero total momentum -- the definition of the CM frame --
    # requires a1*k1 = a2*k2, so the lighter fragment moves faster.  Then
    #     E_cm = k1^2*S1 + k2^2*S2,   S_i = N_i*h2m_n + Z_i*h2m_p
    # with k2 = (a1/a2)*k1, which inverts for k1.  h2m is the COMPOSITE one,
    # matching the propagator.
    h2m = np.asarray(force_td.h2m)
    S1 = n1 * h2m[0] + z1 * h2m[1]
    S2 = n2 * h2m[0] + z2 * h2m[1]
    ratio = a1 / a2
    k1 = float(np.sqrt(args.e_cm / (S1 + ratio**2 * S2)))
    k2 = ratio * k1
    v1 = 197.3269804 * k1 / 939.0
    v2 = 197.3269804 * k2 / 939.0
    print(f"  boost k = {k1:.4f} / {k2:.4f} /fm  ->  v = {v1:.4f} / {v2:.4f} c, "
          f"closing speed {v1+v2:.4f} c", flush=True)

    psi_lo = apply_boost(psi_lo, grid, (0.0, 0.0, +k1))  # lower z, moving up
    psi_hi = apply_boost(psi_hi, grid, (0.0, 0.0, -k2))  # upper z, moving down

    psi = jnp.concatenate([psi_lo, psi_hi], axis=0)
    wocc = jnp.concatenate([wocc1, wocc2])
    isospin = jnp.concatenate([iso1, iso2])

    psi, overlap = lowdin_orthonormalize(psi, isospin, grid.wxyz)
    print(f"  max |<a|b> - delta| between the two fragments before "
          f"orthonormalisation: {overlap:.2e}", flush=True)

    td = make_td_state(psi, wocc, isospin, s1.coulomb_solver, grid, force_td)

    # Energy check: the same configuration at rest, minus the boosted one, must
    # come out to E_cm.  This validates the boost normalisation the way test_t4
    # validates the single-nucleus one.
    psi_rest = jnp.concatenate([
        translate(psi1, grid, dx_fm=bl, dz_fm=zl)[0],
        translate(psi2, grid, dx_fm=bh, dz_fm=zh)[0]], axis=0)
    psi_rest, _ = lowdin_orthonormalize(psi_rest, isospin, grid.wxyz)
    td_rest = make_td_state(psi_rest, wocc, isospin, s1.coulomb_solver, grid,
                            force_td)
    e_rest = total_energy(td_rest, grid, force_td)
    E0 = total_energy(td, grid, force_td)
    e_frag = float(s1.energies.ehfint) + float(s2.energies.ehfint)
    print(f"  E(at rest) = {e_rest:.4f} MeV   E(boosted) = {E0:.4f} MeV", flush=True)
    print(f"  kinetic energy injected = {E0-e_rest:.4f} MeV   "
          f"requested E_cm = {args.e_cm:.4f} MeV   "
          f"error {abs(E0-e_rest-args.e_cm)/args.e_cm:.2e}", flush=True)
    print(f"  sum of fragment energies = {e_frag:.4f} MeV  "
          f"(the composite starts {e_rest-e_frag:+.3f} MeV above that: "
          f"Coulomb repulsion between the fragments)", flush=True)

    # ── propagate ────────────────────────────────────────────────────────────
    tdcfg = TDConfig(dt=args.dt, n_steps=args.steps, use_coulomb=True,
                     verbose=False)
    mid = args.nx // 2
    frames, times, seps, drifts, q20s = [], [], [], [], []
    vols_n, vols_p = [], []
    every = max(1, args.steps // args.frames)

    def snap(st):
        rn = np.asarray(st.densities.rho[0])
        rp = np.asarray(st.densities.rho[1])
        frames.append((rn + rp)[:, mid, :].T)           # x-z plane through y=0
        # Full 3D densities, float32, neutron and proton SEPARATELY.  At the
        # default size this is ~34 MB for the whole run, which buys every
        # downstream view -- the three orthogonal planes, the isosurface, a
        # transverse-integrated spacetime diagram, and anything that needs the
        # local proton fraction (neutron skin, charge equilibration) -- without
        # ever re-running.  Storing only the summed slice, as the first version
        # did, made every new way of looking at the data cost a fresh
        # propagation, and no amount of post-processing can recover n from p.
        vols_n.append(rn.astype(np.float32))
        vols_p.append(rp.astype(np.float32))
        times.append(st.time)
        seps.append(fragment_separation(st.densities, grid))
        q20s.append(compute_moments(st.densities, grid)['Q20'])
        drifts.append((total_energy(st, grid, force_td) - E0) / abs(E0))

    snap(td)
    print(f"\n  propagating {args.steps} steps x {args.dt} fm/c "
          f"= {args.steps*args.dt:.0f} fm/c", flush=True)
    t0 = time.time()
    for i in range(args.steps):
        td = tdhf_step(td, grid, force_td, tdcfg)
        if (i + 1) % every == 0:
            snap(td)
            print(f"  t={td.time:7.1f} fm/c   sep={seps[-1]:6.2f} fm   "
                  f"Q20={q20s[-1]:9.1f}   dE/E={drifts[-1]:+.2e}", flush=True)
    dt_wall = time.time() - t0
    print(f"  {args.steps} steps in {dt_wall:.0f}s "
          f"({dt_wall/max(args.steps,1):.3f} s/step)", flush=True)

    ortho = float(orthonormality_error(td.psi, td.isospin, grid.wxyz))
    bmax = boundary_density(td.densities, grid)
    print(f"\n  final orthonormality error {ortho:.2e}")
    print(f"  final energy drift         {drifts[-1]:+.3e}")
    print(f"  max density on box faces   {bmax:.2e} fm^-3", flush=True)
    if bmax > 1e-4:
        print("  !! density has reached the box wall -- frames from the point "
              "where dE/E jumps are artefact.  Enlarge --nz.", flush=True)

    Path('out').mkdir(exist_ok=True)
    pair = args.nucleus if nuc2 == args.nucleus else f"{args.nucleus}-{nuc2}"
    tag = f"{pair}_E{args.e_cm:g}_b{args.b:g}"
    path = f"out/coll_{tag}.npz"
    np.savez_compressed(
        path, frames=np.array(frames), times=np.array(times),
        seps=np.array(seps), drifts=np.array(drifts), q20s=np.array(q20s),
        rho3d_n=np.array(vols_n, dtype=np.float32),
        rho3d_p=np.array(vols_p, dtype=np.float32),
        extent=np.array([float(grid.x[0]), float(grid.x[-1]),
                         float(grid.z[0]), float(grid.z[-1])]),
        yextent=np.array([float(grid.y[0]), float(grid.y[-1])]),
        spacing=np.array([grid.dx, grid.dy, grid.dz]),
        nucleus=args.nucleus, nucleus2=nuc2, e_cm=args.e_cm, b=args.b)
    print(f"  saved {path}", flush=True)
    return path


# ── Rendering ────────────────────────────────────────────────────────────────

def render(path, fps=15):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    d = np.load(path, allow_pickle=True)
    frames, times = d['frames'], d['times']
    seps, drifts, q20s, extent = d['seps'], d['drifts'], d['q20s'], d['extent']
    nuc, e_cm, b = str(d['nucleus']), float(d['e_cm']), float(d['b'])
    nuc2 = str(d['nucleus2']) if 'nucleus2' in d.files else nuc

    bg, fg, dim = '#0e0e12', '#e8e8ee', '#8a8f98'
    fig = plt.figure(figsize=(12, 5.4))
    fig.patch.set_facecolor(bg)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1], hspace=0.42, wspace=0.24,
                          left=0.06, right=0.97, top=0.90, bottom=0.12)
    ax = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1])
    for a in (ax, ax1, ax2):
        a.set_facecolor(bg)
        for sp in a.spines.values():
            sp.set_color('#3a3a44')
        a.tick_params(colors=dim, labelsize=8)

    # Density is plotted with z (the beam axis) horizontal, so the fragments
    # come in from left and right the way a beam-line picture is usually drawn.
    vmax = float(np.percentile(frames, 99.9))
    im = ax.imshow(frames[0].T, origin='lower',
                   extent=[extent[2], extent[3], extent[0], extent[1]],
                   cmap='inferno', vmin=0.0, vmax=vmax,
                   interpolation='bilinear', aspect='equal')
    ax.set_xlabel('z  (fm)   — beam axis', color=dim, fontsize=9)
    ax.set_ylabel('x  (fm)', color=dim, fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
    cb.set_label(r'$\rho$  (fm$^{-3}$)', color=dim, fontsize=9)
    cb.ax.tick_params(colors=dim, labelsize=8)
    cb.outline.set_edgecolor('#3a3a44')
    ttl = ax.set_title('', color=fg, fontsize=12, family='monospace', pad=10)

    fig.text(0.06, 0.965, f"{nuc} + {nuc2}   TDHF   "
             f"$E_{{cm}}$ = {e_cm:g} MeV   b = {b:g} fm",
             color=fg, fontsize=12, family='monospace')

    ax1.plot(times, seps, color='#5ac8fa', lw=1.5)
    mk1, = ax1.plot([times[0]], [seps[0]], 'o', color='#ffd60a', ms=6)
    ax1.set_ylabel('fragment sep.  (fm)', color=dim, fontsize=9)
    ax1.grid(alpha=0.13, color=dim)
    ax1.tick_params(labelbottom=False)

    ax2.plot(times, drifts, color='#ff9f0a', lw=1.5)
    mk2, = ax2.plot([times[0]], [drifts[0]], 'o', color='#ffd60a', ms=6)
    ax2.set_ylabel(r'$\Delta E/E$', color=dim, fontsize=9)
    ax2.set_xlabel('t  (fm/c)', color=dim, fontsize=9)
    ax2.grid(alpha=0.13, color=dim)
    ax2.axhline(0.0, color=dim, lw=0.7, alpha=0.5)

    def update(i):
        im.set_data(frames[i].T)
        mk1.set_data([times[i]], [seps[i]])
        mk2.set_data([times[i]], [drifts[i]])
        ttl.set_text(f"t = {times[i]:6.1f} fm/c")
        return im, mk1, mk2, ttl

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 // fps,
                         blit=False)
    out = path.replace('.npz', '.gif')
    anim.save(out, writer=PillowWriter(fps=fps), dpi=95,
              savefig_kwargs={'facecolor': bg})
    print(f"  wrote {out}  ({len(frames)} frames)")
    return out


def main():
    p = argparse.ArgumentParser(
        description="TDHF heavy-ion collision + animation",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument('--nucleus', default='O16', choices=list(NUCLEI),
                   help='projectile (and target too, unless --nucleus2)')
    p.add_argument('--nucleus2', default=None, choices=list(NUCLEI),
                   help='target, for an asymmetric reaction; defaults to '
                        '--nucleus.  Try Ca40 + Ca48 for charge equilibration.')
    p.add_argument('--e-cm', type=float, default=34.0,
                   help='centre-of-mass kinetic energy, MeV')
    p.add_argument('--b', type=float, default=0.0, help='impact parameter, fm')
    p.add_argument('--sep', type=float, default=16.0,
                   help='initial centre-to-centre separation, fm')
    p.add_argument('--nx', type=int, default=24, help='transverse grid points')
    p.add_argument('--nz', type=int, default=48, help='beam-axis grid points')
    p.add_argument('--dx', type=float, default=1.0, help='grid spacing, fm')
    p.add_argument('--dt', type=float, default=0.2, help='time step, fm/c')
    p.add_argument('--steps', type=int, default=2400,
                   help='2400 x 0.2 = 480 fm/c: enough for the fragments to '
                        'close a 10 fm gap and for the composite to settle')
    p.add_argument('--frames', type=int, default=150)
    p.add_argument('--static-iter', type=int, default=None,
                   help='static iterations; default is the pre-collapse limit '
                        'from SAFE_STATIC_ITER.  MORE IS NOT BETTER -- past the '
                        'limit the solver leaks density into unbound box '
                        'states.  Check the boundness line if you override it.')
    p.add_argument('--time-odd', choices=['sky3d', 'skyrme'], default='sky3d',
                   help="sky3d (default) matches the reference code; skyrme adds "
                        "the s-channel couplings, unstable for SLy4")
    p.add_argument('--fps', type=int, default=15)
    p.add_argument('--no-render', action='store_true',
                   help='run the physics only, leave the npz for later')
    p.add_argument('--render', metavar='NPZ', help='skip the run, just re-render')
    args = p.parse_args()

    if args.render:
        render(args.render, args.fps)
        return 0
    path = run_case(args)
    if not args.no_render:
        render(path, args.fps)
    return 0


if __name__ == '__main__':
    sys.exit(main())
