#!/usr/bin/env python3
"""
Collective oscillations of a single nucleus in TDHF, with animation.

    python oscillations.py --mode release  --nucleus Mg24
    python oscillations.py --mode gdr      --nucleus O16
    python oscillations.py --mode quad     --nucleus O16
    python oscillations.py --mode monopole --nucleus O16
    python oscillations.py --render out/release_Mg24.npz    # re-render only

Modes:
    release   Converge under a Q20 constraint (a prolate shape), then release
              the constraint and propagate: large-amplitude shape oscillation.
    gdr       Isovector giant dipole: protons and neutrons are kicked in
              opposite directions (weights +N/A and -Z/A, so the centre of
              mass is not excited). Animates rho_n - rho_p.
    quad      Small isoscalar quadrupole kick (linear response).
    monopole  Small monopole (breathing) kick; tracks <r^2>.

Output: out/<mode>_<nucleus>.npz (x-z slices, the tracked moment and the
energy drift per frame) and an animated GIF.

Time-odd terms: the current (j^2) and spin-orbit time-odd terms are always
included. The s-channel couplings (s^2, s.Lap(s)) are off by default, as in
Sky3D; --time-odd skyrme derives them from the Skyrme parameters. For SLy4 the
isoscalar s^2 coupling is negative at low density, which makes the spin
density unstable in the nuclear surface.
"""

import argparse
import math
import sys
import time
from pathlib import Path

import jax_hfbfft  # noqa: F401  (enables 64-bit JAX before jax is imported)
import numpy as np

from jax_hfbfft import TDHF, TDConfig
from jax_hfbfft.core.constraint import Constraint
from jax_hfbfft.core.force import Force
from jax_hfbfft.core.grid import Grid
from jax_hfbfft.physics.solver import SolverConfig, apply_cm_correction, run_hfb

NUCLEI = {'O16': (8, 8, 20), 'Mg24': (12, 12, 26), 'Ca40': (20, 20, 40)}


def static_state(args, grid, force):
    z, n, npsi = NUCLEI[args.nucleus]
    constraint = None
    if args.mode == 'release':
        # The constraining operator 2z^2 - x^2 - y^2 is unbounded below along z;
        # the damping mask confines it to about twice the nuclear radius and at
        # least 3 fm inside the box.
        R = 1.2 * (z + n) ** (1.0 / 3.0)
        damprad = min(2.2 * R, args.nx * args.dx / 2.0 - 3.0)
        constraint = Constraint.from_multipoles({'Q20': args.q20},
                                                damprad=damprad, dampgamma=0.8)
        print(f"constraining Q20 to {args.q20} fm^2 (damping radius {damprad:.1f} fm)")

    cfg = SolverConfig(max_iterations=args.static_iter, verbose=False, bcs_start=30,
                       diag_start=30, output_interval=10**9, sinfo_interval=10**9,
                       tvaryx_0=True)
    t0 = time.time()
    s = run_hfb(grid, force, nucleus_z=z, nucleus_n=n, npsi_n=npsi,
                config=cfg, constraint=constraint)
    rho = np.asarray(s.densities.rho[0] + s.densities.rho[1])
    X, Y, Z = (np.asarray(a) for a in grid.get_meshgrid())
    q20 = float((rho * (2 * Z**2 - X**2 - Y**2)).sum() * grid.wxyz)
    centre = rho[args.nx // 2, args.nx // 2, args.nx // 2]
    print(f"static {args.nucleus}: E = {float(s.energies.ehfint):.4f} MeV, "
          f"{s.iteration} iterations ({time.time() - t0:.0f} s)")
    print(f"  peak rho = {rho.max():.5f} fm^-3, central rho = {centre:.5f}, Q20 = {q20:.1f} fm^2")
    if constraint is not None and abs(q20 - args.q20) > 0.25 * abs(args.q20):
        print(f"  WARNING: the constraint did not reach its target "
              f"({q20:.1f} vs {args.q20:.1f} fm^2)")
    if rho.max() < 0.10 or centre < 0.05:
        raise SystemExit(
            f"The static state is not a bound nucleus (peak density {rho.max():.4f} "
            f"fm^-3). The constraint is too strong: lower --q20 (light nuclei "
            f"reach a few tens of fm^2) or use --mode quad."
        )
    return s


def run(args):
    z, n, _ = NUCLEI[args.nucleus]
    grid = Grid.create(nx=args.nx, ny=args.nx, nz=args.nx,
                       dx=args.dx, dy=args.dx, dz=args.dx)
    force = apply_cm_correction(Force.from_name("SLy4", ipair=0), z + n)
    s = static_state(args, grid, force)

    force_td = force.with_time_odd_from_skyrme() if args.time_odd == 'skyrme' else force
    every = max(1, args.steps // args.frames)
    td = TDHF.from_static(
        s, grid, force=force_td,
        config=TDConfig(dt=args.dt, diag_interval=every),
        # Releasing the constraint makes the starting state non-stationary on purpose.
        stationarity_tol=math.inf if args.mode == 'release' else 1.0,
    )
    if args.mode == 'quad':
        td.kick_quadrupole(args.eta)
    elif args.mode == 'monopole':
        td.kick_monopole(args.eta)
    elif args.mode == 'gdr':
        td.kick_isovector_dipole(args.eta_dip)

    mid = args.nx // 2
    frames, times, moments, drifts = [], [], [], []
    t0 = time.time()
    for rec in td.iterate(args.steps):
        rho = np.asarray(td.state.densities.rho)
        field = rho[0] - rho[1] if args.mode == 'gdr' else rho[0] + rho[1]
        frames.append(field[:, mid, :].T)                   # x-z plane
        times.append(rec['time'])
        moments.append(rec['r2'] if args.mode == 'monopole' else rec['Q20'])
        drifts.append(rec['dE_rel'])
        print(f"  t = {rec['time']:7.1f} fm/c   moment = {moments[-1]:10.3f}   "
              f"dE/E = {rec['dE_rel']:+.2e}", flush=True)
    print(f"{args.steps} steps in {time.time() - t0:.0f} s")

    Path('out').mkdir(exist_ok=True)
    path = f"out/{args.mode}_{args.nucleus}.npz"
    np.savez(path, frames=np.array(frames), times=np.array(times),
             moments=np.array(moments), drifts=np.array(drifts),
             extent=np.array([grid.x[0], grid.x[-1], grid.z[0], grid.z[-1]]),
             mode=args.mode, nucleus=args.nucleus)
    print(f"saved {path}")
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    p.add_argument('--mode', choices=['release', 'gdr', 'quad', 'monopole'],
                   default='release')
    p.add_argument('--nucleus', default='Mg24', choices=list(NUCLEI))
    p.add_argument('--nx', type=int, default=24, help='grid points per direction')
    p.add_argument('--dx', type=float, default=1.0, help='grid spacing, fm')
    p.add_argument('--dt', type=float, default=0.2, help='time step, fm/c')
    p.add_argument('--steps', type=int, default=1200, help='number of time steps')
    p.add_argument('--frames', type=int, default=120, help='frames to record')
    p.add_argument('--eta', type=float, default=0.004,
                   help='kick strength for quad and monopole (fm^-2)')
    p.add_argument('--eta-dip', type=float, default=0.03,
                   help='kick strength for gdr (fm^-1)')
    p.add_argument('--q20', type=float, default=60.0,
                   help='target Q20 for release, fm^2')
    p.add_argument('--static-iter', type=int, default=200)
    p.add_argument('--time-odd', choices=['sky3d', 'skyrme'], default='sky3d',
                   help="'sky3d' (default) omits the s-channel time-odd couplings; "
                        "'skyrme' derives them from the Skyrme parameters")
    p.add_argument('--fps', type=int, default=14)
    p.add_argument('--no-render', action='store_true', help='skip the animation')
    p.add_argument('--render', metavar='NPZ', help='only render an existing run')
    args = p.parse_args()

    path = args.render or run(args)
    if args.render or not args.no_render:
        from jax_hfbfft.viz.tdhf import render_oscillation
        print(f"wrote {render_oscillation(path, args.fps)}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
