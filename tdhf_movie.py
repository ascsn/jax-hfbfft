#!/usr/bin/env python3
"""
TDHF showcase runs, with animation.

    python tdhf_movie.py --mode release   --nucleus Mg24     # <-- start here
    python tdhf_movie.py --mode gdr       --nucleus O16
    python tdhf_movie.py --mode quad      --nucleus O16
    python tdhf_movie.py --mode monopole  --nucleus O16
    python tdhf_movie.py --render out/release_Mg24.npz       # re-render only

MODES
-----
release   THE ONE TO RUN FIRST.  Converge the nucleus under a strong Q20
          constraint so it is squeezed into a cigar, then switch the constraint
          off and let go.  It swings through spherical, overshoots into an
          oblate pancake, and back.  This is genuine large-amplitude collective
          motion -- not linear response -- so it exercises the time-odd terms
          properly, and it is by far the most fun to watch.  It is also a real
          test: the oscillation should be periodic and the energy flat.

gdr       Isovector giant dipole.  Neutrons and protons are boosted in
          OPPOSITE directions and then slosh against each other -- the mode
          that dominates nuclear photoabsorption.  Animates rho_n - rho_p, so
          you see the charge separation directly.  The boost uses the
          CM-corrected operator (protons +N/A, neutrons -Z/A) so it does not
          excite the spurious translational mode.

quad      Small quadrupole kick: the giant quadrupole resonance in linear
          response.  Same shape oscillation as `release` but tiny amplitude.

monopole  Breathing mode.  The nucleus pulses radially.  Least dramatic
          visually, but the cleanest single frequency.

OUTPUT
------
Writes  out/<mode>_<nucleus>.npz  (density slices + moment history) and
        out/<mode>_<nucleus>.gif

TIME-ODD TERMS
--------------
By default this matches Sky3D.  Two families of time-odd term exist and they
are NOT on the same footing:

  ALWAYS ON, and not optional -- the j^2 current coupling and the time-odd
  spin-orbit partners.  Their coefficients are fixed by Galilean invariance
  (C^j = -C^tau) and by the same b4 that sets the spin-orbit term, so they are
  determined by the time-even fit and carry no freedom.  A moving nucleus
  needs them.

  OFF BY DEFAULT -- the independent s-channel couplings (s^2, s.Lap(s), s.T).
  Nothing in a time-even fit constrains these, and for SLy4 the derived
  isoscalar coupling C^s_0(rho) turns NEGATIVE below rho ~ 0.0058 fm^-3.
  Below that density s = 0 is an energy MAXIMUM, so spin density in the
  surface region grows exponentially: measured here as max|s| rising a factor
  ~5 every 5 fm/c and dragging the energy with it.  Sky3D omits these terms
  entirely (sdens enters its meanfield only through the spin-orbit curl);
  HFODD takes them as user inputs for the same reason.

Pass --time-odd skyrme to switch them on anyway.  You will get a warning and,
for SLy4, an unstable run.

Always watch the dE/E trace printed in the corner of the animation: it is the
single number that tells you whether the frames are physics or artefact.
"""

import os
os.environ['JAX_ENABLE_X64'] = 'True'

import sys
import time
import argparse
import dataclasses
from pathlib import Path

import numpy as np
import jax.numpy as jnp

from jax_hfbfft.core.grid import Grid
from jax_hfbfft.core.force import Force
from jax_hfbfft.core.constraint import Constraint
from jax_hfbfft.physics.solver import SolverConfig, run_hfb
from jax_hfbfft.physics.dynamics import (
    TDConfig, prepare_tdhf_state, tdhf_step, build_fields,
    compute_moments, total_energy, apply_multipole_boost,
)

NUCLEI = {'O16': (8, 8, 20), 'Mg24': (12, 12, 26), 'Ca40': (20, 20, 40)}


def isovector_dipole_boost(psi, isospin, grid, eta, Z, N):
    """
    Boost protons and neutrons in opposite directions along z.

    The naive operator (just z, opposite signs) drags the centre of mass and so
    excites the spurious translational mode on top of the physics.  Weighting
    by N/A and Z/A makes the operator orthogonal to translation, so what you
    see afterwards is the dipole resonance and nothing else.
    """
    A = Z + N
    _, _, ZC = grid.get_meshgrid()
    w = jnp.where(isospin == 1, N / A, -Z / A)          # protons +, neutrons -
    phase = jnp.exp(-1j * eta * w[:, None, None, None, None] * ZC[None, None])
    return psi * phase


def run_case(args):
    z, n, npsi = NUCLEI[args.nucleus]
    grid = Grid.create(nx=args.nx, ny=args.nx, nz=args.nx,
                       dx=args.dx, dy=args.dx, dz=args.dx)
    force = Force.from_name("SLy4")

    cfg = SolverConfig(max_iterations=args.static_iter, verbose=False,
                       bcs_start=30, diag_start=30, output_interval=9999,
                       tvaryx_0=True)

    constraint = None
    if args.mode == 'release':
        # Squeeze into a prolate shape.  Q20 in fm^2; a few hundred is a
        # strongly deformed light nucleus.
        # The damping radius must sit WELL INSIDE the box, not at the wall.
        # V = -lambda(2z^2 - x^2 - y^2) is an inverted parabola in z: unbounded
        # below, so it drags density outward without limit.  The mask is the
        # only thing stopping that.  Setting damprad to the box half-width --
        # as an earlier version did -- leaves the operator effectively undamped
        # everywhere inside the box, and the constraint pulls the nucleus apart
        # onto the walls instead of deforming it.
        #
        # Use ~2x the nuclear radius, and keep it at least 3 fm clear of the
        # boundary so the mask has room to fall off.
        R = 1.2 * (z + n) ** (1.0 / 3.0)
        damprad = min(2.2 * R, args.nx * args.dx / 2.0 - 3.0)
        print(f"  damping radius {damprad:.1f} fm  (nuclear R ~ {R:.1f} fm, "
              f"box half-width {args.nx*args.dx/2.0:.1f} fm)", flush=True)
        constraint = Constraint.from_multipoles({'Q20': args.q20},
                                                damprad=damprad,
                                                dampgamma=0.8)
        print(f"constraining Q20 -> {args.q20} fm^2 ...", flush=True)
        print("  (a light nucleus reaches only a few tens of fm^2; ask for far "
              "more and the\n   constraint saturates and you get a squashed, "
              "hollow state, not a deformed one)", flush=True)

    t0 = time.time()
    s = run_hfb(grid, force, nucleus_z=z, nucleus_n=n, npsi_n=npsi,
                config=cfg, constraint=constraint)
    rho0 = np.asarray(s.densities.rho[0] + s.densities.rho[1])
    XX, YY, ZZ = grid.get_meshgrid()
    q_got = float(jnp.sum((s.densities.rho[0] + s.densities.rho[1])
                          * (2 * ZZ**2 - XX**2 - YY**2)) * grid.wxyz)
    ctr = rho0[args.nx // 2, args.nx // 2, args.nx // 2]
    print(f"static {args.nucleus}: E={float(s.energies.ehfint):.4f} MeV  "
          f"iters={s.iteration}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"  peak rho = {rho0.max():.5f} fm^-3   rho(centre) = {ctr:.5f}   "
          f"Q20 achieved = {q_got:.1f} fm^2", flush=True)
    if constraint is not None and abs(q_got - args.q20) > 0.25 * abs(args.q20):
        print(f"  !! the constraint did NOT reach its target "
              f"({q_got:.1f} vs {args.q20:.1f} fm^2)", flush=True)
    if rho0.max() < 0.10 or ctr < 0.05:
        raise SystemExit(
            f"\nRefusing to animate: the static state is not a nucleus.\n"
            f"  peak density {rho0.max():.5f} fm^-3 (saturation is 0.16)\n"
            f"  central density {ctr:.5f} fm^-3\n"
            f"A hollow, half-density blob means the constraint overwhelmed the\n"
            f"mean field.  Lower --q20 (try 40-80 for a light nucleus) or drop\n"
            f"the constraint with --mode quad."
        )

    # Default matches Sky3D: the j^2 and spin-orbit time-odd terms are always
    # on (Galilean-fixed), the independent s-channel couplings are off.  They
    # are dynamically unstable for SLy4 -- see --time-odd skyrme.
    f_td = force.with_time_odd_from_skyrme() if args.time_odd == 'skyrme' else force
    td = prepare_tdhf_state(s, grid, f_td)
    fu = f_td

    X, Y, ZC = grid.get_meshgrid()
    if args.mode == 'quad':
        td = dataclasses.replace(td, psi=apply_multipole_boost(
            td.psi, grid, 2 * ZC**2 - X**2 - Y**2, args.eta))
    elif args.mode == 'monopole':
        td = dataclasses.replace(td, psi=apply_multipole_boost(
            td.psi, grid, X**2 + Y**2 + ZC**2, args.eta))
    elif args.mode == 'gdr':
        td = dataclasses.replace(td, psi=isovector_dipole_boost(
            td.psi, td.isospin, grid, args.eta_dip, z, n))
    # 'release' needs no kick -- switching the constraint off IS the kick.

    d, mf, w = build_fields(td.psi, td.wocc, td.isospin, grid, fu,
                            td.coulomb_solver, True)
    td = dataclasses.replace(td, densities=d, meanfield=mf, wcoul=w)

    E0 = total_energy(td, grid, fu)
    tdcfg = TDConfig(dt=args.dt, n_steps=args.steps, use_coulomb=True, verbose=False)

    mid = args.nx // 2
    frames, times, moms, drifts = [], [], [], []
    every = max(1, args.steps // args.frames)

    def snap(st):
        rho = np.asarray(st.densities.rho[0] + st.densities.rho[1])
        if args.mode == 'gdr':
            field = np.asarray(st.densities.rho[0] - st.densities.rho[1])
        else:
            field = rho
        frames.append(field[:, mid, :].T)              # x-z plane
        times.append(st.time)
        m = compute_moments(st.densities, grid)
        moms.append(m['Q20'] if args.mode != 'monopole' else m['r2'])
        drifts.append((total_energy(st, grid, fu) - E0) / abs(E0))

    snap(td)
    t0 = time.time()
    for i in range(args.steps):
        td = tdhf_step(td, grid, fu, tdcfg)
        if (i + 1) % every == 0:
            snap(td)
            print(f"  t={td.time:7.1f} fm/c   moment={moms[-1]:10.3f}   "
                  f"dE/E={drifts[-1]:+.2e}", flush=True)
    print(f"  {args.steps} steps in {time.time()-t0:.0f}s", flush=True)

    Path('out').mkdir(exist_ok=True)
    path = f"out/{args.mode}_{args.nucleus}.npz"
    np.savez(path, frames=np.array(frames), times=np.array(times),
             moments=np.array(moms), drifts=np.array(drifts),
             extent=np.array([grid.x[0], grid.x[-1], grid.z[0], grid.z[-1]]),
             mode=args.mode, nucleus=args.nucleus)
    print(f"  saved {path}", flush=True)
    return path


def render(path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    d = np.load(path, allow_pickle=True)
    frames, times = d['frames'], d['times']
    moms, drifts, extent = d['moments'], d['drifts'], d['extent']
    mode, nuc = str(d['mode']), str(d['nucleus'])

    diverging = (mode == 'gdr')
    vmax = float(np.max(np.abs(frames)))
    cmap = 'RdBu_r' if diverging else 'magma'
    vmin = -vmax if diverging else 0.0

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(11, 4.6), gridspec_kw={'width_ratios': [1.15, 1]})
    fig.patch.set_facecolor('#101014')
    for a in (ax, ax2):
        a.set_facecolor('#101014')
        for sp in a.spines.values():
            sp.set_color('#555')
        a.tick_params(colors='#aaa', labelsize=8)

    im = ax.imshow(frames[0], origin='lower', extent=extent, cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation='bilinear')
    ax.set_xlabel('x (fm)', color='#aaa', fontsize=9)
    ax.set_ylabel('z (fm)', color='#aaa', fontsize=9)
    label = r'$\rho_n-\rho_p$' if diverging else r'$\rho$'
    cb = fig.colorbar(im, ax=ax, fraction=0.046)
    cb.set_label(label + r'  (fm$^{-3}$)', color='#aaa', fontsize=9)
    cb.ax.tick_params(colors='#aaa', labelsize=8)
    ttl = ax.set_title('', color='w', fontsize=11, family='monospace')

    ax2.plot(times, moms, color='#5ac8fa', lw=1.4)
    mk, = ax2.plot([times[0]], [moms[0]], 'o', color='#ffd60a', ms=7)
    ax2.set_xlabel('t (fm/c)', color='#aaa', fontsize=9)
    ax2.set_ylabel(r'$\langle r^2\rangle$' if mode == 'monopole' else r'$Q_{20}$',
                   color='#aaa', fontsize=9)
    ax2.grid(alpha=0.15)
    dtxt = ax2.text(0.03, 0.06, '', transform=ax2.transAxes,
                    color='#8a8f98', fontsize=8, family='monospace')

    def update(i):
        im.set_data(frames[i])
        mk.set_data([times[i]], [moms[i]])
        ttl.set_text(f"{nuc}  {mode}   t = {times[i]:6.1f} fm/c")
        dtxt.set_text(f"dE/E = {drifts[i]:+.2e}")
        return im, mk, ttl, dtxt

    anim = FuncAnimation(fig, update, frames=len(frames), interval=70, blit=False)
    out = path.replace('.npz', '.gif')
    anim.save(out, writer=PillowWriter(fps=14), dpi=95)
    print(f"  wrote {out}  ({len(frames)} frames)")


def main():
    p = argparse.ArgumentParser(
        description="TDHF showcase + animation",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument('--mode', choices=['release', 'gdr', 'quad', 'monopole'],
                   default='release')
    p.add_argument('--nucleus', default='Mg24', choices=list(NUCLEI))
    p.add_argument('--nx', type=int, default=24)
    p.add_argument('--dx', type=float, default=1.0)
    p.add_argument('--dt', type=float, default=0.2)
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--frames', type=int, default=120)
    p.add_argument('--eta', type=float, default=0.004)
    p.add_argument('--eta-dip', type=float, default=0.03)
    p.add_argument('--q20', type=float, default=60.0,
                   help='target Q20 in fm^2; light nuclei reach only tens')
    p.add_argument('--static-iter', type=int, default=200)
    p.add_argument('--time-odd', choices=['sky3d', 'skyrme'], default='sky3d',
                   help="sky3d (default) omits the s-channel couplings, matching "
                        "the reference code; skyrme derives them from (t,x) and "
                        "is unstable for SLy4")
    p.add_argument('--render', metavar='NPZ', help="skip the run, just re-render")
    args = p.parse_args()

    if args.render:
        render(args.render)
    else:
        render(run_case(args))
    return 0


if __name__ == '__main__':
    sys.exit(main())
