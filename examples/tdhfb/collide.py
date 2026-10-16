#!/usr/bin/env python3
"""
Head-on TDHFB collision of two paired nuclei.

    python collide.py --frag out/Sn120_33.npz --nz 69 --sep 16 --e-asym 277 --phase 3.1416 --T 240
    python collide.py --frag S.npz --phase 1.5708 --scheme ebata           # comparison arm
    python collide.py --frag Sn118.npz --frag2 Sn122.npz --e-asym 277      # asymmetric

Each fragment (a static from static.py, embedded in a box of length --nz by
zero-padding in z) is placed at -/+ sep/2 and boosted toward the other so that
the asymptotic c.m. energy is --e-asym (MeV, Coulomb included: the kinetic
energy at the starting separation is e_asym - 1.44 Z1 Z2 / sep). Fragment B
gets the relative pairing phase --phase. For Sn+Sn the barrier is ~267 MeV.

Records every --every fm/c: energy, N, Z, pairing energies, Q20, the
separation of the two density halves and N_left - N_right (on an odd nz the
centre plane counts half to each side). --snap-every saves y=0 slices of the
density and of the neutron pairing density xi_n. --check-at measures local
continuity against its floor.

Long runs: --ckpt FILE --resume restarts from the last save (every
--save-every fm/c). Beyond ~300 fm/c, when the pairing has collapsed, check
the energy against a run with dt/2.
"""

import argparse
import json
import os
import time

import jax_hfbfft  # noqa: F401  (enables 64-bit JAX before jax is imported)
import numpy as np

from jax_hfbfft import TDHFB
from frames import FrameRecorder


def halves(td, flds):
    """sep (fm) and N_left - N_right of the total density; odd nz: centre plane split half/half."""
    rho = np.asarray(flds[0].rho[0] + flds[0].rho[1])
    z = np.asarray(td.grid.z)
    c0 = np.isclose(z, 0.0, atol=1e-9 * td.grid.dz)
    wv = float(td.grid.wxyz)
    rz = rho.sum((0, 1))
    NL = wv * (rz[z < 0].sum() + 0.5 * rz[c0].sum())
    NR = wv * (rz[(z > 0) & ~c0].sum() + 0.5 * rz[c0].sum())
    zl = (rz[z < 0] * z[z < 0]).sum() / rz[z < 0].sum()
    zr = (rz[z >= 0] * z[z >= 0]).sum() / rz[z >= 0].sum()
    return dict(sep=float(zr - zl), dN=float(NL - NR))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--frag', required=True, help='fragment A static .npz (static.py)')
    p.add_argument('--frag2', default=None, help='fragment B if different from A')
    p.add_argument('--nz', type=int, default=69, help='collision box length in grid points (odd)')
    p.add_argument('--sep', type=float, default=16.0, help='initial distance of the centres (fm)')
    p.add_argument('--e-asym', type=float, default=277.0, help='asymptotic c.m. energy (MeV)')
    p.add_argument('--phase', type=float, default=0.0, help='relative pairing phase of fragment B')
    p.add_argument('--boost-band', type=float, default=6.0)
    p.add_argument('--scheme', choices=['global', 'ebata'], default='global')
    p.add_argument('--dt', type=float, default=0.1)
    p.add_argument('--T', type=float, default=240.0)
    p.add_argument('--every', type=float, default=2.0, help='record interval (fm/c)')
    p.add_argument('--check-at', type=float, nargs='*', default=[0.0], help='continuity checks (fm/c)')
    p.add_argument('--eps', type=float, default=1e-7)
    p.add_argument('--n-comp', type=int, default=1)
    p.add_argument('--strang', action=argparse.BooleanOptionalAction, default=True,
                   help='symmetric pair/mean-field splitting (with --midpoint: second order)')
    p.add_argument('--midpoint', action=argparse.BooleanOptionalAction, default=True,
                   help='exponential-midpoint complement rotations (--no-midpoint: forward Euler)')
    p.add_argument('--pred-n-comp', type=int, default=1, help='predictor sub-rotations (0 = n-comp, exact)')
    p.add_argument('--drop-below', type=float, default=0.0, help='remove orbitals whose occupation falls below this')
    p.add_argument('--snap-every', type=float, default=0.0, help='save y=0 slices every this many fm/c')
    p.add_argument('--save-every', type=float, default=20.0)
    p.add_argument('--ckpt', default=None)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--out', required=True)
    p.add_argument('--absorb-width', type=float, default=0.0,
                   help='absorbing boundary layer at the box faces (fm); 0 = reflecting box')
    p.add_argument('--absorb-strength', type=float, default=10.0, help='absorbing potential at the faces (MeV)')
    p.add_argument('--frames3d-every', type=float, default=0.0,
                   help='full 3D density/pair-field frames every this many fm/c (frames.py, for viz/build_3d.py)')
    p.add_argument('--levels', action=argparse.BooleanOptionalAction, default=True,
                   help='record the canonical occupations and single-particle energies')
    a = p.parse_args()
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)

    td = TDHFB.collision(a.frag, a.frag2, nz=a.nz, sep=a.sep, e_asym=a.e_asym, phase=a.phase,
                         scheme=a.scheme, band=a.boost_band, dt=a.dt, eps=a.eps, n_comp=a.n_comp, strang=a.strang, midpoint=a.midpoint, absorb_width=a.absorb_width, absorb_strength=a.absorb_strength,
                         pred_n_comp=a.pred_n_comp or None)
    i0 = td.info
    print(f"  {td.scheme} scheme, {td.state.nstates} orbitals, box {td.grid.nx}x{td.grid.ny}x{td.grid.nz}; "
          f"fragments at z = {i0['zA']:+.2f}, {i0['zB']:+.2f} fm, k = {i0['kA']:.4f} fm^-1, "
          f"kinetic {i0['e_kin']:.1f} + Coulomb {i0['e_coul']:.1f} MeV, phase {a.phase:.4f}", flush=True)
    snaps = {k: [] for k in ('t', 'rho_xz', 'xi_n_xz')}
    frames = FrameRecorder(a.out, a.frames3d_every, a.levels)
    checks = []
    if a.resume and a.ckpt and os.path.exists(a.ckpt):
        td.load(a.ckpt)
        if os.path.exists(a.out):
            o = np.load(a.out)
            keys = json.loads(str(o['keys']))
            td.history = [dict(zip(keys, row)) for row in o['history'] if row[0] <= td.time + 1e-9]
            checks = [c for c in json.loads(str(o['checks'])) if c['time'] <= td.time + 1e-9]
            if 'snap_t' in o.files:
                keep = o['snap_t'] <= td.time + 1e-9
                snaps = {k: list(o['snap_' + k][keep]) for k in snaps}
        frames.resume(td.time)
        print(f"  resumed from {a.ckpt} at t = {td.time:.2f} fm/c, {td.state.nstates} orbitals")

    def save():
        h = td.history_arrays()
        keys = list(h)
        np.savez_compressed(a.out + '.tmp.npz', history=np.column_stack([h[k] for k in keys]),
                            keys=json.dumps(keys), checks=json.dumps(checks), args=json.dumps(vars(a)),
                            z=np.asarray(td.grid.z), x=np.asarray(td.grid.x),
                            **{'snap_' + k: np.asarray(v) for k, v in snaps.items() if len(v)})
        os.replace(a.out + '.tmp.npz', a.out)
        frames.save(td.grid)
        if a.ckpt:
            td.save(a.ckpt)

    def extra(td_, flds):
        frames(td_, flds)
        rec = halves(td_, flds)
        if a.snap_every > 0 and (not snaps['t'] or td_.time >= snaps['t'][-1] + a.snap_every - 1e-9):
            j = td_.grid.ny // 2
            snaps['t'].append(td_.time)
            snaps['rho_xz'].append(np.asarray(flds[0].rho[0, :, j] + flds[0].rho[1, :, j], np.float32))
            snaps['xi_n_xz'].append(np.asarray(flds[4][0, :, j], np.complex64))
        return rec

    every = max(1, int(round(a.every / a.dt)))
    n_steps = int(round((a.T - td.time) / a.dt))
    pending = sorted(t for t in a.check_at if t >= td.time - 1e-9 and not any(abs(c['time'] - t) < 1e-9 for c in checks))
    next_save = td.time + a.save_every
    t0 = time.time()
    for k, rec in enumerate(td.iterate(n_steps, record_every=every, extra=extra)):
        t, E0 = rec['time'], td.history[0]['E']
        if a.drop_below > 0 and td.scheme == 'global':
            n, lost = td.drop_empty(a.drop_below)
            if n:
                print(f"  t = {t:7.2f}  removed {n} orbitals ({lost:.2e} particles), {td.state.nstates} left", flush=True)
        if k % 5 == 0:
            print(f"  t = {t:7.2f}  E-E0 {rec['E'] - E0:+.3e}  N {rec['N']:.8f}  NL-NR {rec['dN']:+.5f}"
                  f"  sep {rec['sep']:6.2f} fm  Epair_n {rec['Epair_n']:.3f}  ({time.time() - t0:.0f} s)", flush=True)
        if pending and t >= pending[0] - 1e-9 and td.scheme == 'global':
            pending = [x for x in pending if x > t + 1e-9]
            c = dict(time=t, **td.continuity())
            checks.append(c)
            print(f"    continuity t = {t:.1f}: n {c['n']:.3e} (floor {c['floor_n']:.3e})", flush=True)
        if t >= next_save - 1e-9:
            save()
            next_save += a.save_every
    save()
    print(f"  wrote {a.out}")


if __name__ == '__main__':
    main()
