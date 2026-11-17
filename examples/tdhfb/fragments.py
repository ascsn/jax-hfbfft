#!/usr/bin/env python3
"""
Fragment observables from saved 3D frames (frames.py, --frames3d-every).

    python fragments.py out/fisQ30k               # fission
    python fragments.py out/ca48u238 --every 10   # collision, every 10th frame

For each frame: the neck (minimum of the line density rho(z) = Int rho dx dy
within one rms length of the centre; the neck plane counts half to each side),
the N, Z and centre of mass of each side, the side velocities (central
differences of the centres), and the Coulomb interaction energy of the two
sides computed exactly from their proton densities, E_C = Int rho_p,L V_R with
V_R = e^2 Int rho_p,R / |r - r'| (isolated boundary conditions). After
separation TKE = 1/2 M_L v_L^2 + 1/2 M_R v_R^2 + E_C, with the nuclear
interaction negligible once the gap between the fragments exceeds a few fm.
Writes STEM_fragments.npz and prints a table; --selftest checks E_C against
two well-separated point-like Gaussian charges.
"""

import argparse
import glob
import os

import jax_hfbfft  # noqa: F401  (enables 64-bit JAX before jax is imported)
import jax.numpy as jnp
import numpy as np

from jax_hfbfft import Grid
from jax_hfbfft.physics.coulomb import CoulombSolver, solve_poisson

MN = 938.9  # nucleon mass (MeV)


def neck_split(rho, z):
    """Index of the neck plane and the side weights (n_z,) for left and right."""
    lz = rho.sum((0, 1))
    zc = (lz * z).sum() / lz.sum()
    sz = np.sqrt((lz * (z - zc) ** 2).sum() / lz.sum())
    iz = np.where(np.abs(z - zc) < sz)[0]
    i = iz[np.argmin(lz[iz])] if len(iz) else int(np.argmin(np.abs(z - zc)))
    wl = (z < z[i]).astype(float) + 0.5 * (np.arange(len(z)) == i)
    return i, wl, 1.0 - wl


def analyse(stem, every=1):
    files = sorted(glob.glob(stem + '_3d/t*.npz'), key=lambda f: float(os.path.basename(f)[1:-4]))[::every]
    f0 = np.load(files[0])
    x, y, z = f0['x'], f0['y'], f0['z']
    d = float(x[1] - x[0])
    grid = Grid.create(nx=len(x), ny=len(y), nz=len(z), dx=d, dy=d, dz=d)
    cs = CoulombSolver.create(grid)
    w = d ** 3
    rows = []
    for fn in files:
        f = np.load(fn)
        rn, rp = f['rho_n'].astype(np.float64), f['rho_p'].astype(np.float64)
        rho = rn + rp
        i, wl, wr = neck_split(rho, z)
        side = {}
        for nm, m in (('L', wl), ('R', wr)):
            N, Zp = w * (rn * m).sum(), w * (rp * m).sum()
            lz = (rho * m).sum((0, 1))
            side[nm] = (N, Zp, (lz * z).sum() / lz.sum())
        VR = np.asarray(solve_poisson(jnp.asarray(rp * wr[None, None, :]), cs, grid))
        ec = w * float((rp * wl[None, None, :] * VR).sum())
        rows.append((float(f['t']), float(z[i]), float(rho[:, :, i].max()), *side['L'], *side['R'], ec))
    a = np.array(rows)
    t, zL, zR = a[:, 0], a[:, 5], a[:, 8]
    vL, vR = np.gradient(zL, t), np.gradient(zR, t)          # fm per fm/c = units of c
    AL, AR = a[:, 3] + a[:, 4], a[:, 6] + a[:, 7]
    ek = 0.5 * MN * (AL * vL ** 2 + AR * vR ** 2)
    out = dict(t=t, z_neck=a[:, 1], neck_peak=a[:, 2], N_L=a[:, 3], Z_L=a[:, 4], z_L=zL, N_R=a[:, 6], Z_R=a[:, 7],
               z_R=zR, v_L=vL, v_R=vR, E_kin=ek, E_C=a[:, 9], TKE=ek + a[:, 9])
    np.savez(stem + '_fragments.npz', **out)
    return out


def selftest():
    """Two Gaussian charges of Z = 50 and 40, 30 fm apart: E_C -> 1.44 * 50 * 40 / 30."""
    n, d = 41, 1.0
    grid = Grid.create(nx=n, ny=n, nz=81, dx=d, dy=d, dz=d)
    cs = CoulombSolver.create(grid)
    X, Y, Z = np.meshgrid(*(np.asarray(v) for v in (grid.x, grid.y, grid.z)), indexing='ij')
    g = lambda z0, q: q * np.exp(-((X ** 2 + Y ** 2 + (Z - z0) ** 2) / (2 * 2.0 ** 2))) / ((2 * np.pi * 4.0) ** 1.5)
    a, b = g(-15.0, 50), g(15.0, 40)
    V = np.asarray(solve_poisson(jnp.asarray(b), cs, grid))
    ec = d ** 3 * float((a * V).sum())
    print(f"selftest: E_C {ec:.3f} MeV, point charges {1.4399784 * 50 * 40 / 30:.3f} MeV")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stem', nargs='?')
    p.add_argument('--every', type=int, default=1, help='use every n-th 3D frame')
    p.add_argument('--selftest', action='store_true')
    a = p.parse_args()
    if a.selftest:
        selftest()
        return
    o = analyse(a.stem, a.every)
    print("     t   neck  rho_neck    A_L     Z_L    A_R     Z_R   gap(cm) v_rel(c)   E_kin    E_C     TKE (MeV)")
    for k in range(0, len(o['t']), max(1, len(o['t']) // 25)):
        print(f"{o['t'][k]:7.0f} {o['z_neck'][k]:+5.0f} {o['neck_peak'][k]:8.4f} {o['N_L'][k] + o['Z_L'][k]:7.2f} "
              f"{o['Z_L'][k]:6.2f} {o['N_R'][k] + o['Z_R'][k]:7.2f} {o['Z_R'][k]:6.2f} {o['z_R'][k] - o['z_L'][k]:7.2f} "
              f"{o['v_R'][k] - o['v_L'][k]:8.4f} {o['E_kin'][k]:7.2f} {o['E_C'][k]:7.2f} {o['TKE'][k]:7.2f}")
    print(f"wrote {a.stem}_fragments.npz")


if __name__ == '__main__':
    main()
