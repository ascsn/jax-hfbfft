#!/usr/bin/env python3
"""
Paired static ground state for TDHFB.

    python static.py --nucleus O20 --nx 13 --dx 1.2 --v0 300 --iters 3000
    python static.py --nucleus Sn120 --nx 33 --dx 0.8 --v0 362 --iters 20000
    python static.py --nucleus Sn120 --nx 33 --iters 10000 --restart out/Sn120_33.npz

Solves HFB with DDDI pairing (v_pair = v0 (1 - rho/0.32)) and writes an .npz
that run.py and collide.py read (see jax_hfbfft.core.tdhfb.save_static).

What TDHFB needs from the static:
  * an odd grid (--nx/--ny/--nz odd): the kinetic operator is then exactly the
    gradient of the energy, which time-dependent runs need;
  * no soft state cutoff: every state weight is 1 and the orbital space itself
    is the pairing cutoff, so choose npsi with that in mind;
  * convergence: efluct small and the energy no longer moving. Statics are
    cheap next to the dynamics, so run long. Also check the density on the box
    faces (printed at the end).

A deformed nucleus has to be restarted from a state of the right shape
(--restart); from a harmonic-oscillator start U238 relaxes to the oblate
minimum.
"""

import argparse
import os
import time

import jax_hfbfft  # noqa: F401  (enables 64-bit JAX before jax is imported)
import jax.numpy as jnp
import numpy as np

from jax_hfbfft import HFBFFT, Force, Grid, Nucleus
from jax_hfbfft.core.tdhfb import save_static

# name: (Z, N, (neutron states, proton states), HO start radius in fm)
NUCLEI = {
    'O20':   (8, 12, (40, 20), 2.7),
    'Ca44':  (20, 24, (48, 40), 3.1),
    'Sn118': (50, 68, (136, 100), 3.1),
    'Sn120': (50, 70, (140, 100), 3.1),
    'Sn122': (50, 72, (144, 100), 3.1),
    'Sn132': (50, 82, (164, 100), 3.1),
    'U238':  (92, 146, (292, 184), 3.1),
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--nucleus', default='O20', help=f"one of {sorted(NUCLEI)} (or give --z/--n/--npsi)")
    p.add_argument('--z', type=int, default=None)
    p.add_argument('--n', type=int, default=None)
    p.add_argument('--npsi', type=int, nargs=2, default=None, metavar=('NEUTRON', 'PROTON'))
    p.add_argument('--nx', type=int, default=13)
    p.add_argument('--ny', type=int, default=None, help='defaults to --nx')
    p.add_argument('--nz', type=int, default=None, help='defaults to --nx')
    p.add_argument('--dx', type=float, default=1.2)
    p.add_argument('--functional', default='SLy4')
    p.add_argument('--v0', type=float, default=362.0, help='DDDI pairing strength (MeV fm^3)')
    p.add_argument('--v0p', type=float, default=None, help='proton pairing strength (default --v0)')
    p.add_argument('--iters', type=int, default=3000)
    p.add_argument('--restart', default=None, help='start from this static .npz (same grid and basis)')
    p.add_argument('--out', default=None)
    a = p.parse_args()

    Z, N, npsi, radin = NUCLEI.get(a.nucleus, (None, None, None, 3.1))
    Z, N = a.z or Z, a.n or N
    npsi = tuple(a.npsi) if a.npsi else npsi
    if Z is None or N is None or npsi is None:
        raise SystemExit(f"unknown nucleus {a.nucleus}: give --z, --n and --npsi")
    ny, nz = a.ny or a.nx, a.nz or a.nx
    if not (a.nx % 2 and ny % 2 and nz % 2):
        print("  WARNING: even grid dimension. TDHFB energy conservation needs odd grids.")
    out = a.out or f"out/{a.nucleus}_{a.nx}x{ny}x{nz}_dx{a.dx:g}_v{a.v0:g}.npz"
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)

    v0p = a.v0 if a.v0p is None else a.v0p
    force = Force.from_name(a.functional).with_pairing(
        ipair=6, v0neut=a.v0, v0prot=v0p, rho0pr=0.32, tbcs=False,
        pair_cutoff=(0.0, 0.0), state_cutoff=(0.0, 0.0))
    grid = Grid.create(nx=a.nx, ny=ny, nz=nz, dx=a.dx, dy=a.dx, dz=a.dx)
    calc = HFBFFT(nucleus=Nucleus(protons=Z, neutrons=N), force=force, grid=grid, npsi=npsi)
    calc.diag_start, calc.bcs_start, calc.tvaryx_0 = 30, 30, True
    calc.initialize_wavefunctions(method="harmonic_oscillator", radinx=radin, radiny=radin, radinz=radin)
    if a.restart:
        r = np.load(a.restart)
        for k in ('psi', 'wocc', 'wguv', 'pairwg', 'wstates'):
            setattr(calc.state, k, jnp.asarray(r[k]))
        print(f"  restart from {a.restart} (E {float(r['E']):.4f}, efluct {float(r['efluct']):.2e})")
    print(f"  {a.nucleus}: Z={Z} N={N} npsi={npsi}  box {a.nx}x{ny}x{nz} @ {a.dx} fm  v0={a.v0:g}/{v0p:g}")

    t0 = time.time()
    calc.run(max_iterations=a.iters, convergence_threshold=1e-20, print_interval=500, sinfo_interval=10**9)
    st = calc.solver_state
    rho = np.abs(np.asarray(st.densities.rho[0] + st.densities.rho[1]))
    face = max(rho[[0, -1]].max(), rho[:, [0, -1]].max(), rho[:, :, [0, -1]].max())
    print(f"\n  E {float(st.energies.ehfint):.6f} MeV   efluct {float(st.efluct):.3e}   "
          f"box-face density {face:.1e}   ({time.time() - t0:.0f} s)")
    uv, iso, gap = np.asarray(st.wguv), np.asarray(st.isospin), np.asarray(st.deltaf)
    for q, name in ((0, 'neutrons'), (1, 'protons')):
        m = iso == q
        print(f"  mean gap {name:<8} {float((uv[m] * gap[m]).sum() / max(uv[m].sum(), 1e-12)):.3f} MeV")
    save_static(out, st, grid, functional=a.functional, v0=a.v0, v0p=v0p, ipair=6, rho0pr=0.32,
                nucleus=a.nucleus)
    print(f"  wrote {out}")


if __name__ == '__main__':
    main()
