#!/usr/bin/env python3
"""
TDHFB fission from a constrained starting point.

    python fission.py --static out/U238_Q16000.npz --nz 101 --T 3000 --ckpt ck.npz --resume --out out/U238_fis.npz

The starting point is a relaxed constrained static beyond the outer saddle
(relaxed under the constraint, so it is a stationary state of the same
functional apart from the constraint). The constraint is removed at t = 0 and
the nucleus evolves freely; --kick-q20 adds a small push along the
elongation. --nz embeds the state in a longer box so the fragments have room.

Records every --every fm/c: energy, N, Z, pairing energies, Q20, Q30, the neck
(position of the minimum of the line density rho(z) = Int rho dx dy between the
fragments, that minimum, and the peak density in the neck plane), the
fragments on either side of the neck (N, Z, centres, separation) and the
density on the box faces. Scission: the peak density in the neck plane falls
below --scission-rho; the run continues --after fm/c past it and stops.
Each record also stores animation frames (frames.py): y = 0 slices in
OUT_frames.npz, the canonical levels in OUT_levels.npz (--no-levels to skip),
and with --frames3d-every T the full 3D density and pair field every T fm/c in
OUT_3d/ (~4 MB per frame for U238 in a 37x37x121 box). viz/build_3d.py turns
them into the raymarched viewer.

Fission takes thousands of fm/c: use --ckpt FILE --resume and chain jobs.
"""

import argparse
import json
import os
import time

import jax_hfbfft  # noqa: F401  (enables 64-bit JAX before jax is imported)
import numpy as np

from jax_hfbfft import TDHFB
from frames import FrameRecorder


def shape(td, flds):
    """Neck, fragments, Q30 and box-face density of the current state."""
    rho_q = np.asarray(flds[0].rho)                       # (2, nx, ny, nz)
    rho = rho_q[0] + rho_q[1]
    g = td.grid
    x, y, z = (np.asarray(v) for v in (g.x, g.y, g.z))
    w = float(g.wxyz)
    lz = rho.sum((0, 1)) * g.dx * g.dy                   # line density along z
    zc = (lz * z).sum() / lz.sum()
    sz = np.sqrt((lz * (z - zc) ** 2).sum() / lz.sum())
    # the neck: the minimum of rho(z) within one rms length of the centre (a compact shape has none: the minimum
    # then sits at the window edge, where the density is high, and no scission is signalled)
    iz = np.where(np.abs(z - zc) < sz)[0]
    i_neck = iz[np.argmin(lz[iz])] if len(iz) else int(np.argmin(np.abs(z - zc)))
    left = z < z[i_neck]
    plane = 0.5 * np.eye(len(z))[i_neck]                 # the neck plane counts half to each side
    sides = {}
    for nm, m in (('L', left.astype(float) + plane), ('R', (~left).astype(float) - plane + 0.0)):
        m = np.clip(m, 0, 1)
        N = w * (rho_q[0] * m).sum(); Z = w * (rho_q[1] * m).sum()
        zf = (lz * m * z).sum() / max((lz * m).sum(), 1e-12)
        sides[nm] = (N, Z, zf)
    X, Y, Zg = np.meshgrid(x, y, z, indexing='ij')
    q30 = w * (rho * (2 * Zg ** 3 - 3 * Zg * (X ** 2 + Y ** 2))).sum()
    face = max(rho[[0, -1]].max(), rho[:, [0, -1]].max(), rho[:, :, [0, -1]].max())
    return dict(z_neck=float(z[i_neck]), neck_line=float(lz[i_neck]), neck_peak=float(rho[:, :, i_neck].max()),
                N_L=float(sides['L'][0]), Z_L=float(sides['L'][1]), N_R=float(sides['R'][0]), Z_R=float(sides['R'][1]),
                sep=float(sides['R'][2] - sides['L'][2]), Q30=float(q30), face=float(face))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--static', required=True, help='relaxed constrained starting point')
    p.add_argument('--nz', type=int, default=None, help='embed in a box this long in z (odd)')
    p.add_argument('--kick-q20', type=float, default=0.0, help='push along the elongation, exp(i eta Q20)')
    p.add_argument('--dt', type=float, default=0.2,
                   help='time step (fm/c); for U238 on a 1 fm grid dt 0.2 agrees with dt 0.025 to 6e-7 in the '
                        'density and 2 keV in E_pair after 10 fm/c')
    p.add_argument('--n-comp', type=int, default=1)
    p.add_argument('--taylor', type=int, default=4)
    p.add_argument('--strang', action=argparse.BooleanOptionalAction, default=True,
                   help='symmetric pair/mean-field splitting (with --midpoint: second order)')
    p.add_argument('--midpoint', action=argparse.BooleanOptionalAction, default=True,
                   help='exponential-midpoint complement rotations (--no-midpoint: forward Euler)')
    p.add_argument('--T', type=float, default=3000.0)
    p.add_argument('--every', type=float, default=2.0)
    p.add_argument('--scission-rho', type=float, default=0.04, help='neck-plane peak density at scission (fm^-3)')
    p.add_argument('--after', type=float, default=100.0, help='fm/c to continue after scission')
    p.add_argument('--check-at', type=float, nargs='*', default=[], help='continuity checks (fm/c)')
    p.add_argument('--save-every', type=float, default=20.0)
    p.add_argument('--ckpt', default=None)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--out', required=True)
    p.add_argument('--absorb-width', type=float, default=0.0,
                   help='absorbing boundary layer at the box faces (fm); 0 = reflecting box')
    p.add_argument('--absorb-strength', type=float, default=10.0, help='absorbing potential at the faces (MeV)')
    p.add_argument('--frames3d-every', type=float, default=0.0, help='full 3D density/pair-field frames (fm/c)')
    p.add_argument('--levels', action=argparse.BooleanOptionalAction, default=True,
                   help='record the canonical occupations and single-particle energies')
    a = p.parse_args()
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)

    td = TDHFB.from_static(a.static, nz=a.nz, max_defect=1e-2, dt=a.dt, n_comp=a.n_comp, taylor_order=a.taylor,
                           strang=a.strang, midpoint=a.midpoint,
                           absorb_width=a.absorb_width, absorb_strength=a.absorb_strength)
    if a.kick_q20:
        td.kick_quadrupole(a.kick_q20)
    checks, t_scission = [], None
    frames = FrameRecorder(a.out, a.frames3d_every, a.levels)
    if a.resume and a.ckpt and os.path.exists(a.ckpt):
        td.load(a.ckpt)
        if os.path.exists(a.out):
            o = np.load(a.out)
            keys = json.loads(str(o['keys']))
            td.history = [dict(zip(keys, row)) for row in o['history'] if row[0] <= td.time + 1e-9]
            checks = [c for c in json.loads(str(o['checks'])) if c['time'] <= td.time + 1e-9]
            ts = float(o['t_scission'])
            t_scission = ts if ts >= 0 else None
        frames.resume(td.time)
        print(f"  resumed from {a.ckpt} at t = {td.time:.2f} fm/c", flush=True)
    g = td.grid
    print(f"  {td.state.nstates} orbitals, box {g.nx}x{g.ny}x{g.nz} @ {g.dx} fm, dt {a.dt}", flush=True)

    def save():
        h = td.history_arrays()
        keys = list(h)
        np.savez(a.out + '.tmp.npz', history=np.column_stack([h[k] for k in keys]), keys=json.dumps(keys),
                 checks=json.dumps(checks), args=json.dumps(vars(a)),
                 t_scission=-1.0 if t_scission is None else t_scission)
        os.replace(a.out + '.tmp.npz', a.out)
        frames.save(td.grid)
        if a.ckpt:
            td.save(a.ckpt)

    every = max(1, int(round(a.every / a.dt)))
    n_steps = int(round((a.T - td.time) / a.dt))
    pending = sorted(t for t in a.check_at if t >= td.time - 1e-9)
    next_save = td.time + a.save_every
    t0 = time.time()
    def shape_and_frame(td, flds):
        frames(td, flds)
        return shape(td, flds)

    for k, rec in enumerate(td.iterate(n_steps, record_every=every, extra=shape_and_frame)):
        t, E0 = rec['time'], td.history[0]['E']
        if k % 5 == 0:
            print(f"  t {t:7.1f}  dE {rec['E'] - E0:+.3e}  N {rec['N']:.6f}  Q20 {rec['Q20']:9.1f}  Q30 {rec['Q30']:9.1f}"
                  f"  neck {rec['neck_peak']:.4f} at z {rec['z_neck']:+.1f}  A_L {rec['N_L'] + rec['Z_L']:6.2f}"
                  f"  Z_L {rec['Z_L']:5.2f}  sep {rec['sep']:5.2f}  Epair {rec['Epair_n']:.2f} {rec['Epair_p']:.2f}"
                  f"  face {rec['face']:.1e}  ({time.time() - t0:.0f} s)", flush=True)
        if pending and t >= pending[0] - 1e-9:
            pending = [x for x in pending if x > t + 1e-9]
            checks.append(dict(time=t, **td.continuity()))
            print(f"    continuity t = {t:.1f}: {checks[-1]}", flush=True)
        if t_scission is None and rec['neck_peak'] < a.scission_rho:
            t_scission = t
            print(f"  SCISSION at t = {t:.1f} fm/c: A_L {rec['N_L'] + rec['Z_L']:.2f} (Z {rec['Z_L']:.2f}), "
                  f"A_R {rec['N_R'] + rec['Z_R']:.2f} (Z {rec['Z_R']:.2f})", flush=True)
        if t >= next_save - 1e-9:
            save()
            next_save += a.save_every
        if t_scission is not None and t >= t_scission + a.after:
            break
    save()
    print(f"  wrote {a.out}" + (f"; scission at {t_scission:.1f} fm/c" if t_scission is not None else ""))


if __name__ == '__main__':
    main()
