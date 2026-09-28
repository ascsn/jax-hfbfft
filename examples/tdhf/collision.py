#!/usr/bin/env python3
"""
TDHF heavy-ion collision with animation.

    python collision.py                          # 16O + 16O, head-on, E_cm = 34 MeV
    python collision.py --b 5.0                  # non-central
    python collision.py --e-cm 120               # well above the fusion window
    python collision.py --nucleus Ca40 --nucleus2 Ca48
    python collision.py --render out/coll_O16_E34_b0.npz   # re-render only

Each nucleus is converged with the static solver (ipair=0), the two are placed
on the z axis `--sep` fm apart, boosted toward each other with centre-of-mass
kinetic energy `--e-cm`, and propagated as one Slater determinant.

Parameters that set the outcome:
    --e-cm   Centre-of-mass kinetic energy (MeV), excluding Coulomb. For
             16O + 16O the Coulomb barrier is about 10 MeV.
    --b      Impact parameter (fm). Larger b brings more angular momentum;
             beyond a critical value the fragments re-separate.

Output: out/coll_<pair>_E<e_cm>_b<b>.npz (density slices, full neutron and
proton densities as float32, separation, Q20 and energy drift per frame) and,
unless --no-render, an animated GIF. See render_views.py for other views.

Accuracy: the relative energy drift dE/E is printed per frame and should stay
small (about 1e-6 here). A jump usually means density has reached the box
boundary; the script reports the density on the box faces at the end.
Pairing is not included (TDHF); the static states use ipair=0.
"""

import argparse
import sys
import time
from pathlib import Path

import jax_hfbfft  # noqa: F401  (enables 64-bit JAX before jax is imported)
import numpy as np

from jax_hfbfft.core.force import Force
from jax_hfbfft.core.grid import Grid
from jax_hfbfft.physics import collisions as C
from jax_hfbfft.physics import dynamics as D
from jax_hfbfft.physics.coulomb import CoulombSolver
from jax_hfbfft.physics.solver import SolverConfig, apply_cm_correction, run_hfb

# name: (Z, N, number of neutron basis states). Even-even nuclei with a
# well-defined ipair=0 ground state: closed shells or subshells (C12, O16, Ca40,
# Ca48, Ni56), plus Mg24, whose no-pairing ground state is prolate.
NUCLEI = {
    'O16':  (8, 8, 20),
    'C12':  (6, 6, 16),
    'Mg24': (12, 12, 26),
    'Ca40': (20, 20, 40),
    'Ca48': (20, 28, 34),
    'Ni56': (28, 28, 40),
}

# Default static iteration counts. They converge each nucleus while staying
# below the point where the ipair=0 static solver starts moving weakly bound
# density into unbound box states on a 24^2 x 48 fm box. That failure mode
# lowers the energy while looking converged, so the boundness check below is
# the thing to watch when changing these.
STATIC_ITER = {'O16': 300, 'C12': 300, 'Mg24': 200, 'Ca40': 150, 'Ca48': 80, 'Ni56': 100}


def relax(name, grid, force, n_iter):
    """Converge one nucleus in `force` and return it as a Fragment."""
    z, n, npsi = NUCLEI[name]
    cfg = SolverConfig(max_iterations=n_iter, verbose=False, bcs_start=30,
                       diag_start=30, output_interval=10**9, sinfo_interval=10**9,
                       tvaryx_0=True)
    t0 = time.time()
    s = run_hfb(grid, force, nucleus_z=z, nucleus_n=n, npsi_n=npsi, config=cfg)
    frag = C.Fragment.from_static(s)
    b = C.boundness(frag.psi, frag.isospin, grid, frag.A)
    print(f"static {name}: E = {float(s.energies.ehfint):.4f} MeV, "
          f"{s.iteration} iterations, efluct = {float(s.efluct):.2e} "
          f"({time.time() - t0:.0f} s)")
    print(f"  peak rho = {b['peak_rho']:.4f} fm^-3, nucleons beyond "
          f"{b['radius']:.1f} fm = {b['a_outside']:.4f}, box-face density = {b['face']:.1e}")
    if not b['bound']:
        print(f"  WARNING: {name} is not properly bound (density in unbound box "
              f"states). Reduce --static-iter and check this line again.")
    resid = D.stationarity_residual(D.prepare_tdhf_state(s, grid, force), grid)
    print(f"  stationarity residual in the propagation Hamiltonian: {resid:.2e} MeV")
    return frag, float(s.energies.ehfint)


def run(args):
    name2 = args.nucleus2 or args.nucleus
    grid = Grid.create(nx=args.nx, ny=args.nx, nz=args.nz,
                       dx=args.dx, dy=args.dx, dz=args.dx)
    a_tot = sum(NUCLEI[n][0] + NUCLEI[n][1] for n in (args.nucleus, name2))
    print(f"box {args.nx}x{args.nx}x{args.nz} at {args.dx} fm; "
          f"{args.nucleus} + {name2}, A = {a_tot}")

    # One Hamiltonian for both the fragments and the composite (see collisions.py).
    force = apply_cm_correction(Force.from_name("SLy4", ipair=0), a_tot)
    n_iter = args.static_iter or min(STATIC_ITER.get(args.nucleus, 150),
                                     STATIC_ITER.get(name2, 150))

    frag1, e1 = relax(args.nucleus, grid, force, n_iter)
    frag2, e2 = (frag1, e1) if name2 == args.nucleus else relax(name2, grid, force, n_iter)

    force_td = force.with_time_odd_from_skyrme() if args.time_odd == 'skyrme' else force
    csolver = CoulombSolver.create(grid)
    setup = dict(grid=grid, force=force_td, e_cm=args.e_cm, separation=args.sep,
                 impact_parameter=args.b, coulomb_solver=csolver)
    state, info = C.make_collision_state(frag1, frag2, **setup)
    rest, _ = C.make_collision_state(frag1, frag2, boost=False, **setup)

    E0 = D.total_energy(state, grid, force_td)
    e_rest = D.total_energy(rest, grid, force_td)
    print(f"  {args.nucleus} at z = {info['z1']:+.1f}, x = {info['x1']:+.1f} fm; "
          f"{name2} at z = {info['z2']:+.1f}, x = {info['x2']:+.1f} fm")
    print(f"  k = {info['k1']:.4f} / {info['k2']:.4f} fm^-1; injected kinetic energy "
          f"{E0 - e_rest:.4f} MeV (requested {args.e_cm:.4f})")
    print(f"  initial energy above the separated fragments: {e_rest - e1 - e2:+.3f} MeV "
          f"(mostly the Coulomb repulsion between them)")

    cfg = D.TDConfig(dt=args.dt)
    every = max(1, args.steps // args.frames)
    mid = args.nx // 2
    rec = {k: [] for k in ('frames', 'times', 'seps', 'drifts', 'q20s', 'rho_n', 'rho_p')}

    def snapshot(st):
        rn = np.asarray(st.densities.rho[0])
        rp = np.asarray(st.densities.rho[1])
        obs = D.observables(st, grid, force_td)
        rec['frames'].append((rn + rp)[:, mid, :].T)       # x-z plane through y = 0
        rec['rho_n'].append(rn.astype(np.float32))
        rec['rho_p'].append(rp.astype(np.float32))
        rec['times'].append(obs['time'])
        rec['seps'].append(C.fragment_separation(st.densities, grid))
        rec['q20s'].append(obs['Q20'])
        rec['drifts'].append((obs['E'] - E0) / abs(E0))

    snapshot(state)
    print(f"\npropagating {args.steps} steps of {args.dt} fm/c")
    t0 = time.time()
    done = 0
    while done < args.steps:
        n = min(every, args.steps - done)
        state = D.advance(state, grid, force_td, cfg, n)
        done += n
        snapshot(state)
        print(f"  t = {rec['times'][-1]:7.1f} fm/c   sep = {rec['seps'][-1]:6.2f} fm   "
              f"Q20 = {rec['q20s'][-1]:9.1f}   dE/E = {rec['drifts'][-1]:+.2e}", flush=True)
    wall = time.time() - t0
    print(f"{args.steps} steps in {wall:.0f} s ({wall / max(args.steps, 1):.3f} s/step)")

    face = C.boundary_density(state.densities, grid)
    print(f"final orthonormality error {D.orthonormality_error(state.psi, state.isospin, grid.wxyz):.2e}, "
          f"energy drift {rec['drifts'][-1]:+.2e}, box-face density {face:.1e} fm^-3")
    if face > 1e-4:
        print("WARNING: density has reached the box boundary; frames after the "
              "energy drift jumps are unreliable. Increase --nz.")

    Path('out').mkdir(exist_ok=True)
    pair = args.nucleus if name2 == args.nucleus else f"{args.nucleus}-{name2}"
    path = f"out/coll_{pair}_E{args.e_cm:g}_b{args.b:g}.npz"
    np.savez_compressed(
        path, frames=np.array(rec['frames']), times=np.array(rec['times']),
        seps=np.array(rec['seps']), drifts=np.array(rec['drifts']),
        q20s=np.array(rec['q20s']),
        rho3d_n=np.array(rec['rho_n']), rho3d_p=np.array(rec['rho_p']),
        extent=np.array([float(grid.x[0]), float(grid.x[-1]),
                         float(grid.z[0]), float(grid.z[-1])]),
        yextent=np.array([float(grid.y[0]), float(grid.y[-1])]),
        spacing=np.array([grid.dx, grid.dy, grid.dz]),
        nucleus=args.nucleus, nucleus2=name2, e_cm=args.e_cm, b=args.b)
    print(f"saved {path}")
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    p.add_argument('--nucleus', default='O16', choices=list(NUCLEI),
                   help='projectile (also the target unless --nucleus2 is given)')
    p.add_argument('--nucleus2', default=None, choices=list(NUCLEI), help='target')
    p.add_argument('--e-cm', type=float, default=34.0,
                   help='centre-of-mass kinetic energy, MeV')
    p.add_argument('--b', type=float, default=0.0, help='impact parameter, fm')
    p.add_argument('--sep', type=float, default=16.0, help='initial separation, fm')
    p.add_argument('--nx', type=int, default=24, help='transverse grid points')
    p.add_argument('--nz', type=int, default=48, help='grid points along the beam')
    p.add_argument('--dx', type=float, default=1.0, help='grid spacing, fm')
    p.add_argument('--dt', type=float, default=0.2, help='time step, fm/c')
    p.add_argument('--steps', type=int, default=2400, help='number of time steps')
    p.add_argument('--frames', type=int, default=150, help='frames to record')
    p.add_argument('--static-iter', type=int, default=None,
                   help='static iterations (default: per-nucleus value in STATIC_ITER)')
    p.add_argument('--time-odd', choices=['sky3d', 'skyrme'], default='sky3d',
                   help="'sky3d' (default) omits the s-channel time-odd couplings, "
                        "as Sky3D does; 'skyrme' derives them from the Skyrme "
                        "parameters (unstable for SLy4)")
    p.add_argument('--fps', type=int, default=15)
    p.add_argument('--no-render', action='store_true', help='skip the animation')
    p.add_argument('--render', metavar='NPZ', help='only render an existing run')
    args = p.parse_args()

    path = args.render or run(args)
    if args.render or not args.no_render:
        from jax_hfbfft.viz.tdhf import render_collision
        print(f"wrote {render_collision(path, args.fps)}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
