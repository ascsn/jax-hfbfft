#!/usr/bin/env python3
"""
Relax a static into the minimum of the TDHFB functional in its own orbital space.

    python relax.py --static out/U238.npz --iters 5000 --out out/U238_relaxed.npz
    python relax.py --static out/U238_relaxed.npz --iters 3000      # continue

The canonical static solver (static.py) leaves nearly empty orbitals
unconverged and keeps only the diagonal gaps, so its result is not stationary
under TDHFB: at rest the dynamics relaxes it, which shows up as a slow growth
of the pairing energy. physics.tdhfb.relax solves HFB in the representation of
the dynamics (full HFB matrix in the span of the orbitals, preconditioned
descent out of it); started from its result TDHFB stays still.

The output keeps the static's metadata and adds the relaxed matrices (rho,
kap); run.py and collide.py use them directly. The pairing grows by a few
percent at fixed v0, so fit v0 on relaxed states (the mean gaps are printed).
Run the Ebata comparison arm from the unrelaxed static: his equations assume a
BCS-type state in a Kramers basis.
"""

import argparse
import os
import time

import jax_hfbfft  # noqa: F401  (enables 64-bit JAX before jax is imported)
import jax.numpy as jnp
import numpy as np

from jax_hfbfft import TDHFB
from jax_hfbfft.core.tdhfb import load_static
from jax_hfbfft.physics import tdhfb as T


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--static', required=True)
    p.add_argument('--iters', type=int, default=3000)
    p.add_argument('--step', type=float, default=1.0)
    p.add_argument('--mix', type=float, default=0.5)
    p.add_argument('--every', type=int, default=200, help='print and save every this many iterations')
    p.add_argument('--v0', type=float, default=None, help='relax with this neutron pairing strength instead')
    p.add_argument('--v0p', type=float, default=None, help='relax with this proton pairing strength instead')
    p.add_argument('--pair-kcut', type=float, default=None,
                   help='momentum cutoff of the pair channel (fm^-1; 0 = bare contact pairing). Heavy nuclei need one: '
                        'with bare pairing the relaxed gap either vanishes or grows into grid-scale states')
    p.add_argument('--pair-kwidth', type=float, default=None)
    p.add_argument('--out', default=None)
    a = p.parse_args()
    out = a.out or (a.static if a.static.endswith('_relaxed.npz') else a.static.replace('.npz', '_relaxed.npz'))

    d, grid, force = load_static(a.static)
    if a.pair_kcut is not None or a.pair_kwidth is not None:
        d['pair_kcut'] = a.pair_kcut if a.pair_kcut is not None else float(d.get('pair_kcut', 0.0))
        d['pair_kwidth'] = a.pair_kwidth if a.pair_kwidth is not None else float(d.get('pair_kwidth', 0.1))
        force = force.with_pairing(pair_kcut=d['pair_kcut'], pair_kwidth=d['pair_kwidth'])
        print(f"  pair channel momentum cutoff {d['pair_kcut']:g} fm^-1 (width {d['pair_kwidth']:g})", flush=True)
    if a.v0 is not None or a.v0p is not None:      # pairing-strength scan: same start, new force
        d['v0'] = a.v0 if a.v0 is not None else float(d['v0'])
        d['v0p'] = a.v0p if a.v0p is not None else float(d.get('v0p', d['v0']))
        force = force.with_pairing(v0neut=d['v0'], v0prot=d['v0p'])
        print(f"  pairing strengths v0 n/p = {d['v0']:g}/{d['v0p']:g}", flush=True)
    td = TDHFB.from_static(a.static, force=force, max_defect=1e-2,   # relax restores the symmetry
                           kramers_tol=1e-3)
    S = td.state
    print(f"  {S.nstates} orbitals; start E {T.energy(S, grid, force, td.cs)[0]:.6f} MeV, "
          f"mean gaps {T.mean_gaps(S, grid, force, td.cs)}", flush=True)
    t0 = time.time()

    def checkpoint(it, S):
        save(out, d, S, grid, force, td.cs)
        print(f"  {it + 1} iterations ({time.time() - t0:.0f} s), mean gaps n/p "
              f"{' '.join(f'{g:.4f}' for g in T.mean_gaps(S, grid, force, td.cs))} MeV, saved {out}", flush=True)

    T.relax(S, grid, force, td.cs, n_iter=a.iters, step=a.step, mix=a.mix, verbose=a.every, callback=checkpoint)


def save(path, d, S, grid, force, cs):
    psi, rho, kap = T._recanon(S.psi, S.rho, S.kap, S.isospin, T._canon_perm(S.isospin))
    occ = np.clip(np.real(np.asarray(jnp.diagonal(rho))), 0.0, 1.0)
    E, _ = T.energy(S, grid, force, cs)
    meta = {k: v for k, v in d.items() if k not in ('psi', 'rho', 'kap', 'wocc', 'wguv', 'E', 'relaxed')}
    tmp = path + '.tmp.npz'                       # atomic: a kill mid-write keeps the old checkpoint
    np.savez_compressed(tmp, **meta, psi=np.asarray(psi), rho=np.asarray(rho), kap=np.asarray(kap),
                        wocc=occ, wguv=np.sqrt(occ * (1 - occ)), E=E, relaxed=True)
    os.replace(tmp, path)


if __name__ == '__main__':
    main()
