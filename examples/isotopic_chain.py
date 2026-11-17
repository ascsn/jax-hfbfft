"""
Even-even isotopic chain with two-neutron separation energies.

    python isotopic_chain.py --z 20 --n 20 30 --force SLy4 --v0 362 --out ca_chain.csv
"""

import argparse
import csv

import jax_hfbfft  # noqa: F401  (64-bit JAX)
from jax_hfbfft import Force, Grid, run_chain, separation_energies


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--z', type=int, required=True)
    p.add_argument('--n', type=int, nargs=2, required=True, metavar=('FIRST', 'LAST'), help='even N range')
    p.add_argument('--force', default='SLy4')
    p.add_argument('--v0', type=float, default=362.0, help='DDDI pairing strength (0 = no pairing)')
    p.add_argument('--nx', type=int, default=32)
    p.add_argument('--dx', type=float, default=0.8)
    p.add_argument('--iters', type=int, default=1500)
    p.add_argument('--ln', action='store_true', help='Lipkin-Nogami')
    p.add_argument('--checkpoints', default=None, help='directory for per-nucleus .npz files')
    p.add_argument('--out', default=None, help='CSV file')
    a = p.parse_args()

    force = Force.from_name(a.force)
    force = (force.with_pairing(ipair=6, v0neut=a.v0, v0prot=a.v0, rho0pr=0.32) if a.v0 > 0
             else force.with_pairing(ipair=0))
    nuclei = [(a.z, n) for n in range(a.n[0], a.n[1] + 1, 2)]
    points = run_chain(nuclei, force, Grid.create_cubic(n=a.nx, d=a.dx), max_iterations=a.iters,
                       checkpoint_dir=a.checkpoints,
                       configure=(lambda c: setattr(c, 'use_ln', True)) if a.ln else None)

    rows = [dict(r, gap_n=p.gap_n, gap_p=p.gap_p, r_ch=p.r_ch, beta2=p.beta2, converged=p.converged)
            for r, p in zip(separation_energies(points), points)]
    print(f"{'N':>4} {'B (MeV)':>11} {'S2n':>8} {'gap n':>7} {'r_ch':>7} {'beta2':>7}")
    for r in rows:
        print(f"{r['N']:4d} {r['B']:11.3f} {r['S2n']:8.3f} {r['gap_n']:7.3f} {r['r_ch']:7.4f} {r['beta2']:7.3f}"
              + ("" if r['converged'] else "  (not converged)"))
    if a.out:
        with open(a.out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == '__main__':
    main()
