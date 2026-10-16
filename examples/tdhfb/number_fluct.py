#!/usr/bin/env python3
"""
Fragment particle-number widths from a TDHFB checkpoint (exact second moments).

    python number_fluct.py --static U238_Q30000.npz --nz 121 --ckpt fis_ck.npz

Splits the box at the neck (as fission.py and fragments.py do) and, with
--outside D, adds the region farther than D fm beyond both fragments (emitted
nucleons). For neutrons and protons: the mean and width sigma of the number in
each region (physics.tdhfb.region_number_moments), their correlation, and the
fragment width conditioned on the total number (a Gaussian estimate of the
width after projecting the total N or Z: HFB pairing makes the total number
fluctuate, which a projected calculation would remove). The mass-number width
of a fragment is sqrt(var_N + var_Z) (neutrons and protons are independent).
"""

import argparse

import jax_hfbfft  # noqa: F401
import numpy as np

from jax_hfbfft import TDHFB
from jax_hfbfft.physics import tdhfb as T


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--static', required=True)
    p.add_argument('--nz', type=int, default=None)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--frag2', default=None, help='second fragment static (collisions): use TDHFB.collision geometry')
    p.add_argument('--outside', type=float, default=0.0, help='also count beyond R + this many fm from both sides')
    a = p.parse_args()
    td = TDHFB.from_static(a.static, nz=a.nz, max_defect=1e-2)
    td.load(a.ckpt)
    S, g = td.state, td.grid
    dens = T.fields(S, g, td.force, td.cs)[0]
    rho = np.asarray(dens.rho[0] + dens.rho[1])
    x, y, z = (np.asarray(v) for v in (g.x, g.y, g.z))
    lz = rho.sum((0, 1)); zc = (lz * z).sum() / lz.sum(); sz = np.sqrt((lz * (z - zc) ** 2).sum() / lz.sum())
    iz = np.where(np.abs(z - zc) < sz)[0]; i = iz[np.argmin(lz[iz])]
    wl = (z < z[i]).astype(float) + 0.5 * (np.arange(len(z)) == i)
    L = np.broadcast_to(wl[None, None, :], rho.shape).copy(); R = 1.0 - L
    regions, names = [L, R], ['L', 'R']
    if a.outside > 0:
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        far = np.ones(rho.shape, bool)
        for side in (L > 0.5, R > 0.5):
            A = float(g.wxyz) * rho[side].sum(); c = (Z[side] * rho[side]).sum() / rho[side].sum()
            s_z = np.sqrt((rho[side] * (Z[side] - c) ** 2).sum() / rho[side].sum())
            s_x = np.sqrt((rho[side] * X[side] ** 2).sum() / rho[side].sum())
            r = np.sqrt((X / (np.sqrt(5) * s_x)) ** 2 + (Y / (np.sqrt(5) * s_x)) ** 2 + ((Z - c) / (np.sqrt(5) * s_z)) ** 2)
            far &= r > 1.0 + a.outside / (1.2 * A ** (1 / 3))
        far = far.astype(float)
        regions = [L * (1 - far), R * (1 - far), far]
        names = ['L', 'R', 'out']
    mom = T.region_number_moments(S, regions, g)
    print(f"{a.ckpt}: t = {td.time:.1f} fm/c, neck plane z = {z[i]:+.1f} fm")
    var_A = {nm: 0.0 for nm in names}
    for q, lab in ((0, 'N'), (1, 'Z')):
        m, C = mom[q]['mean'], mom[q]['cov']
        tot_var = C.sum()
        iq = np.asarray(S.isospin) == q
        kq = np.asarray(S.kap)[np.ix_(iq, iq)]
        check = 2 * float(np.real(np.trace(kq @ kq.conj().T)))
        print(f"  {lab}: total {m.sum():.4f}, sigma(total) {np.sqrt(max(tot_var, 0)):.3f} (HFB number fluctuation; "
              f"check 2 Tr kap kap^+ -> sigma {np.sqrt(check):.3f})")
        for k, nm in enumerate(names):
            cond = C[k, k] - C[k].sum() ** 2 / tot_var if tot_var > 0 else C[k, k]
            var_A[nm] += C[k, k]
            print(f"     {nm:3s}: <{lab}> {m[k]:9.4f}   sigma {np.sqrt(max(C[k, k], 0)):.3f}   "
                  f"sigma at fixed total {np.sqrt(max(cond, 0)):.3f}")
        if len(names) >= 2:
            print(f"     corr(L, R) {C[0, 1] / np.sqrt(C[0, 0] * C[1, 1]):+.3f}")
    print("  mass-number widths: " + ", ".join(f"{nm} sigma_A {np.sqrt(v):.3f}" for nm, v in var_A.items()))


if __name__ == '__main__':
    main()
