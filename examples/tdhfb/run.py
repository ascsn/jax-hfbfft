#!/usr/bin/env python3
"""
TDHFB evolution of one nucleus, at rest or after a quadrupole kick.

    python run.py --static out/O20_13x13x13_dx1.2_v300.npz --kick 0.005 --T 60
    python run.py --static S.npz --kick 0 --T 500 --check-at 0 100 250
    python run.py --static S.npz --kick 0.005 --T 60 --scheme ebata     # comparison arm

Records every --every fm/c: energy, particle numbers, pairing energies, Q20 and
(global scheme) the purity of the generalized density. At the --check-at times
the local continuity defect is measured against its floor (see
TDHFB.continuity); it is meaningful for moving nuclei, not at rest.

Accuracy checks: N and Z drift by ~5e-9 and the impurity stays below 1e-12 over
60 fm/c; the energy drift for a kicked 20O on 13^3 is ~9e-3 MeV over
60 fm/c at dt = 0.05. A kick of 0.01 on Sn120 injects ~40 MeV and breaks the
pairing, which is physics; use ~0.002 for pairing studies.

Output: --out (.npz) with the history, the continuity checks and the arguments,
rewritten every --save-every fm/c. --ckpt FILE --resume continues a run.
"""

import argparse
import json
import os
import time

import jax_hfbfft  # noqa: F401  (enables 64-bit JAX before jax is imported)
import numpy as np

from jax_hfbfft import TDHFB


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--static', required=True, help='static .npz from static.py')
    p.add_argument('--scheme', choices=['global', 'ebata'], default='global')
    p.add_argument('--kick', type=float, default=0.005, help='quadrupole kick exp(i eta Q20), eta in fm^-2')
    p.add_argument('--dt', type=float, default=0.05)
    p.add_argument('--T', type=float, default=60.0, help='total time (fm/c)')
    p.add_argument('--every', type=float, default=1.0, help='record interval (fm/c)')
    p.add_argument('--check-at', type=float, nargs='*', default=[0.0], help='continuity checks (fm/c)')
    p.add_argument('--eps', type=float, default=1e-7, help='regulator of rho~^-1')
    p.add_argument('--n-comp', type=int, default=4)
    p.add_argument('--pred-n-comp', type=int, default=1, help='predictor sub-rotations (0 = n-comp, exact)')
    p.add_argument('--drop-below', type=float, default=0.0, help='remove orbitals below this occupation at t=0')
    p.add_argument('--save-every', type=float, default=50.0)
    p.add_argument('--ckpt', default=None)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--out', default=None)
    a = p.parse_args()
    out = a.out or os.path.join('out', os.path.basename(a.static).replace('.npz', f'_{a.scheme}_k{a.kick:g}.npz'))
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)

    td = TDHFB.from_static(a.static, scheme=a.scheme, dt=a.dt, eps=a.eps, n_comp=a.n_comp,
                           pred_n_comp=a.pred_n_comp or None)
    if a.drop_below > 0:
        n, lost = td.drop_empty(a.drop_below)
        print(f"  removed {n} orbitals below occupation {a.drop_below:g} ({lost:.2e} particles)")
    td.kick_quadrupole(a.kick)
    checks = []
    if a.resume and a.ckpt and os.path.exists(a.ckpt):
        td.load(a.ckpt)
        if os.path.exists(out):
            o = np.load(out)
            keys = json.loads(str(o['keys']))
            td.history = [dict(zip(keys, row)) for row in o['history'] if row[0] <= td.time + 1e-9]
            checks = [c for c in json.loads(str(o['checks'])) if c['time'] <= td.time + 1e-9]
        print(f"  resumed from {a.ckpt} at t = {td.time:.2f} fm/c")
    print(f"  {td.scheme} scheme, {td.state.nstates} orbitals, kick {a.kick}, dt {a.dt}")

    def save():
        h = td.history_arrays()
        keys = list(h)
        np.savez(out + '.tmp.npz', history=np.column_stack([h[k] for k in keys]), keys=json.dumps(keys),
                 checks=json.dumps(checks), args=json.dumps(vars(a)))
        os.replace(out + '.tmp.npz', out)
        if a.ckpt:
            td.save(a.ckpt)

    every = max(1, int(round(a.every / a.dt)))
    n_steps = int(round((a.T - td.time) / a.dt))
    pending = sorted(t for t in a.check_at if t >= td.time - 1e-9 and not any(abs(c['time'] - t) < 1e-9 for c in checks))
    next_save = td.time + a.save_every
    t0 = time.time()
    for k, rec in enumerate(td.iterate(n_steps, record_every=every)):
        t, E0 = rec['time'], td.history[0]['E']
        if k % 10 == 0:
            print(f"  t = {t:7.2f}  E-E0 {rec['E'] - E0:+.3e}  N {rec['N']:.10f}  Z {rec['Z']:.10f}"
                  f"  Epair_n {rec['Epair_n']:.4f}  Q20 {rec['Q20']:.3f}  ({time.time() - t0:.0f} s)", flush=True)
        if pending and t >= pending[0] - 1e-9 and td.scheme == 'global':
            pending = [x for x in pending if x > t + 1e-9]
            c = dict(time=t, **td.continuity())
            checks.append(c)
            print(f"    continuity t = {t:.1f}: n {c['n']:.3e} (floor {c['floor_n']:.3e})"
                  f"  p {c['p']:.3e} (floor {c['floor_p']:.3e})", flush=True)
        if t >= next_save - 1e-9:
            save()
            next_save += a.save_every
    save()
    print(f"  wrote {out}")


if __name__ == '__main__':
    main()
