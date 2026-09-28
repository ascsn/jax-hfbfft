#!/usr/bin/env python3
"""
Additional views of a finished collision (post-processing only).

    python render_views.py out/coll_O16_E34_b0.npz              # every view
    python render_views.py out/coll_O16_E34_b0.npz --spacetime
    python render_views.py out/coll_Ca40-Ca48_E60_b0.npz --isospin
    python render_views.py out/coll_O16_E34_b0.npz --iso --iso-mode plain

Views (all written next to the input file):
    --spacetime  Density along the beam axis versus time, as a single image.
    --threeview  The x-z, y-z and x-y planes through the centre, animated,
                 as filled density bands.
    --iso        Ray-cast 3D view from three cameras. --iso-mode selects nested
                 translucent density shells (default), one surface coloured by
                 proton fraction ('frac'), or one lit surface ('plain').
    --isospin    rho_n - rho_p, the proton fraction, and the neutron and proton
                 transfer across the neck (informative when N != Z).
"""

import argparse
import sys
from pathlib import Path

from jax_hfbfft.viz import tdhf as V


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    p.add_argument('npz')
    p.add_argument('--spacetime', action='store_true')
    p.add_argument('--threeview', action='store_true')
    p.add_argument('--iso', action='store_true')
    p.add_argument('--isospin', action='store_true')
    p.add_argument('--iso-level', type=float, default=0.08,
                   help='isosurface density for frac/plain modes, fm^-3')
    p.add_argument('--iso-mode', choices=['shells', 'frac', 'plain'], default='shells')
    p.add_argument('--iso-upsample', type=int, default=3,
                   help='resampling factor before ray casting (cost grows as its cube)')
    p.add_argument('--flat-slices', action='store_true',
                   help='continuous colour scale instead of density bands in --threeview')
    p.add_argument('--fps', type=int, default=15)
    args = p.parse_args()

    selected = args.spacetime or args.threeview or args.iso or args.isospin
    if not selected:
        for path in V.render_all_views(args.npz, args.fps, args.iso_level, args.iso_mode,
                                       args.iso_upsample, bands=not args.flat_slices):
            print(f"wrote {path}")
        return 0

    d = V.load_collision(args.npz)
    stem = str(Path(args.npz).with_suffix(''))
    written = []
    if args.spacetime:
        written.append(V.spacetime(d, stem + '_spacetime.png'))
    if args.threeview:
        written.append(V.threeview(d, stem + '_3view.gif', args.fps,
                                   bands=not args.flat_slices))
    if args.iso:
        written.append(V.isosurface(d, stem + '_iso.gif', args.iso_level, args.fps,
                                    upsample=args.iso_upsample, mode=args.iso_mode))
    if args.isospin:
        written.append(V.isospin_transfer(d, stem + '_isospin.gif', args.fps))
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
