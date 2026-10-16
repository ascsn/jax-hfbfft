#!/usr/bin/env python3
"""
Raymarched 3D viewer for TDHFB runs: one self-contained HTML page (WebGL2).

    python viz/build_3d.py OUT.html --run q30 "Release from 300 b" out/fisQ30k \\
                                    --run rest "U238 at rest" out/rest --title "U238 in 3D"

Each --run ID LABEL STEM reads what run.py, collide.py or fission.py wrote with
--frames3d-every (see frames.py):
  STEM_3d/t*.npz      full 3D rho_q and Delta_q        (required)
  STEM.npz            the observable history           (required)
  STEM_levels.npz     canonical occupations / energies (optional: levels panel)

The page shows the density as an opaque core surface and a translucent cloud
surface, and the pair density xi_q = Delta_q / (v0_q (1 - rho/rho0pr)) as a
glow coloured by its phase, with orbit, time and threshold controls. Every run
is cropped to the region its frames ever touch, quantized to 8 bits per channel
and gzipped (the page inflates it with DecompressionStream); a run with more
than --max-frames frames is thinned evenly. Pages above ~16 MB will not
publish as artifacts: lower --max-frames or split runs over pages.
"""

import argparse
import base64
import glob
import gzip
import json
import os

import numpy as np

RHO_TOP = 0.2                  # fm^-3, the 8-bit density scale
HERE = os.path.dirname(os.path.abspath(__file__))


def frame_files(stem, max_frames, t_min=None, t_max=None):
    tof = lambda f: float(os.path.basename(f)[1:-4])
    files = sorted(glob.glob(stem + '_3d/t*.npz'), key=tof)
    files = [f for f in files if (t_min is None or tof(f) >= t_min - 1e-9) and (t_max is None or tof(f) <= t_max + 1e-9)]
    if len(files) > max_frames:
        files = [files[i] for i in np.unique(np.linspace(0, len(files) - 1, max_frames).round().astype(int))]
    return files


def pairing_constants(frame, hist):
    """v0, v0p, rho0pr: stored in the frames by frames.py, else read from the static the run started from."""
    if 'v0' in frame.files:
        return float(frame['v0']), float(frame['v0p']), float(frame['rho0pr'])
    args = json.loads(str(hist['args']))
    st = np.load(args.get('static') or args.get('frag'))
    return float(st['v0']), float(st['v0p']), float(st['rho0pr'])


def observables(hist, times):
    keys = json.loads(str(hist['keys']))
    h = hist['history']
    col = lambda k: h[:, keys.index(k)]
    ii = [int(np.argmin(np.abs(col('time') - t))) for t in times]
    pick = lambda k, f=1.0, nd=3: [round(float(col(k)[i] * f), nd) for i in ii]
    obs = dict(dE=[round(float(col('E')[i] - col('E')[0]), 6) for i in ii], En=pick('Epair_n'), Ep=pick('Epair_p'),
               Q20=pick('Q20', 0.01, 1))
    if 'neck_peak' in keys:
        obs.update(neck=pick('neck_peak', nd=4), ZL=pick('Z_L', nd=1), ZR=pick('Z_R', nd=1),
                   AL=[round(float(col('N_L')[i] + col('Z_L')[i]), 1) for i in ii],
                   AR=[round(float(col('N_R')[i] + col('Z_R')[i]), 1) for i in ii])
    if 'sep' in keys:
        obs['sep'] = pick('sep', nd=2)
    if 'dN' in keys:
        obs['dN'] = pick('dN', nd=3)
    ts = float(hist['t_scission']) if 't_scission' in hist.files else -1.0
    return obs, (ts if ts >= 0 else None)


def levels(stem, times):
    """Canonical levels at the frame times: occupation as uint8, energy in 0.1 MeV as int16, isospin."""
    path = stem + '_levels.npz'
    if not os.path.exists(path):
        return None
    lv = np.load(path)
    ii = [int(np.argmin(np.abs(lv['t'] - t))) for t in times]
    near = np.abs(lv['t'][ii] - np.asarray(times)) < 1.01           # frames with no level record nearby show none
    occ, e, iso = lv['occ'][ii], lv['e'][ii], lv['iso'][ii]
    ok = np.isfinite(occ) & np.isfinite(e) & (iso >= 0) & near[:, None]
    q_occ = np.where(ok, np.clip(np.rint(occ * 254), 0, 254), 255).astype(np.uint8)       # 255 = no orbital
    q_e = np.clip(np.rint(np.where(ok, e, 0) * 10), -32000, 32000).astype(np.int16)
    q_iso = np.where(ok, iso, 0).astype(np.uint8)
    blob = np.concatenate([q_occ.ravel(), q_iso.ravel(), q_e.view(np.uint8).ravel()]).tobytes()
    return dict(n=int(occ.shape[1]), data=base64.b64encode(gzip.compress(blob, 9)).decode())


def dataset(key, label, stem, max_frames, t_min=None, t_max=None):
    files = frame_files(stem, max_frames, t_min, t_max)
    hist = np.load(stem + '.npz')
    fr = [np.load(f) for f in files]
    v0, v0p, rho0 = pairing_constants(fr[0], hist)
    rho = [f['rho_n'] + f['rho_p'] for f in fr]
    xin = [f['delta_n'] / (v0 * (1 - r / rho0)) for f, r in zip(fr, rho)]
    xip = [f['delta_p'] / (v0p * (1 - r / rho0)) for f, r in zip(fr, rho)]
    an = max(float(np.abs(x).max()) for x in xin) or 1.0
    ap = max(float(np.abs(x).max()) for x in xip) or 1.0
    dmax = [max(float(np.abs(f['delta_n']).max()) for f in fr), max(float(np.abs(f['delta_p']).max()) for f in fr)]
    mask = np.zeros(rho[0].shape, bool)
    for r, a, b in zip(rho, xin, xip):
        mask |= (r > 2e-4) | (np.abs(a) > 0.01 * an) | (np.abs(b) > 0.01 * ap)
    sl = []
    for ax in range(3):
        idx = np.where(mask.any(axis=tuple(j for j in range(3) if j != ax)))[0]
        sl.append(slice(max(int(idx[0]) - 2, 0), min(int(idx[-1]) + 3, mask.shape[ax])))
    sl = tuple(sl)
    q = lambda a, lo, hi: np.clip(np.rint((a - lo) / (hi - lo) * 255), 0, 255).astype(np.uint8)
    # channel-last voxels in (x, y, z) C order, z fastest: the texture width is nz
    blobs = [np.stack([q(f['rho_n'][sl], 0, RHO_TOP), q(f['rho_p'][sl], 0, RHO_TOP),
                       q(a[sl].real, -an, an), q(a[sl].imag, -an, an), q(b[sl].real, -ap, ap), q(b[sl].imag, -ap, ap)],
                      -1).tobytes() for f, a, b in zip(fr, xin, xip)]
    times = [float(f['t']) for f in fr]
    frac_n = float(fr[0]['rho_n'].sum() / (fr[0]['rho_n'].sum() + fr[0]['rho_p'].sum()))   # N / A
    obs, ts = observables(hist, times)
    shape = [int(s.stop - s.start) for s in sl]
    return dict(id=key, label=label, nx=shape[0], ny=shape[1], nz=shape[2], dx=float(fr[0]['x'][1] - fr[0]['x'][0]),
                amax=[an, ap], dmax=dmax, rho_top=RHO_TOP, frac_n=frac_n, times=times, obs=obs, t_scission=ts, levels=levels(stem, times),
                data=base64.b64encode(gzip.compress(b''.join(blobs), 9)).decode(), raw_mb=sum(map(len, blobs)) / 1e6)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('out')
    p.add_argument('--run', nargs=3, action='append', required=True, metavar=('ID', 'LABEL', 'STEM'))
    p.add_argument('--max-frames', type=int, default=400)
    p.add_argument('--t-min', type=float, default=None, help='use frames from this time on (fm/c)')
    p.add_argument('--t-max', type=float, default=None, help='use frames up to this time (fm/c); also keeps the crop box small')
    p.add_argument('--title', default='TDHFB in 3D', help='page title (a short name)')
    p.add_argument('--heading', default=None, help='page heading (HTML allowed; default: the title)')
    p.add_argument('--subtitle', default='TDHFB with dynamical pairing, raymarched from the full 3D density and the '
                                         'complex pair density, 1 fm voxels.')
    a = p.parse_args()
    data = [dataset(k, l, s, a.max_frames, a.t_min, a.t_max) for k, l, s in a.run]
    tpl = open(os.path.join(HERE, 'raymarch_template.html')).read()
    html = (tpl.replace('__TITLE__', a.title).replace('__HEADING__', a.heading or a.title)
            .replace('__SUBTITLE__', a.subtitle).replace('/*__DATA__*/null', json.dumps(data, separators=(',', ':'))))
    with open(a.out, 'w') as f:
        f.write(html)
    print(f'wrote {a.out}: {len(html) / 1e6:.2f} MB;',
          '; '.join(f"{d['id']} {len(d['times'])} frames to t={d['times'][-1]:.0f}, box {d['nx']}x{d['ny']}x{d['nz']}"
                    f"{', levels' if d['levels'] else ''}" for d in data))


if __name__ == '__main__':
    main()
