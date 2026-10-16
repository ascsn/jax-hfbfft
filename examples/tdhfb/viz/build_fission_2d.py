"""
Build the 2D U238 fission-study viewer (one self-contained HTML page) from saved data.
Study-specific: the paths below are those of the 2026-10-06 U238 study in
/mnt/scratch/knigh137/fission; run.py/collide.py output goes to build_3d.py instead.


  stage3/anim_path.npz                 Stage 2 constrained path (extract_slices.py)
  stage3/anim_old_Q{20,30}k.npz        old-step runs: t = 0 and the latest checkpoint
  stage3/{fisQ*.npz, oldstep/...}      old-step observable histories
  stage3/fisQ*_dt02{.npz,_frames.npz}  relaunched runs (frames every record), when present

usage: python .../viz/build_fission_2d.py OUT.html   (run in the study directory)
"""
import base64, json, os, sys
import numpy as np

MAX_FRAMES = 160
KEYS = ['time', 'E', 'Epair_n', 'Epair_p', 'Q20', 'Q30', 'neck_peak', 'N_L', 'Z_L', 'N_R', 'Z_R', 'sep']


def pack(rho_n, rho_p, d_n, d_p):
    """Frames (F, nx, nz) -> uint8 [F][6][nx][nz]: rho_n, rho_p, |dn|, arg dn, |dp|, arg dp, plus the scales."""
    rmax = float(max(rho_n.max(), rho_p.max()))
    amax = [float(np.abs(d).max()) or 1.0 for d in (d_n, d_p)]
    q = lambda a, m: np.clip(np.rint(255 * a / m), 0, 255).astype(np.uint8)
    ph = lambda d: np.rint((np.angle(d) + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
    arr = np.stack([q(rho_n, rmax), q(rho_p, rmax), q(np.abs(d_n), amax[0]), ph(d_n),
                    q(np.abs(d_p), amax[1]), ph(d_p)], axis=1)
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode(), rmax, amax


def grid_info(x, z, nframes):
    return dict(nx=len(x), nz=len(z), x0=float(x[0]), dx=float(x[1] - x[0]), z0=float(z[0]),
                dz=float(z[1] - z[0]), nframes=nframes)


def history(path):
    o = np.load(path)
    keys = json.loads(str(o['keys']))
    h = o['history']
    cols = {k: h[:, keys.index(k)] for k in KEYS if k in keys}
    ts = float(o['t_scission'])
    return cols, (ts if ts >= 0 else None)


def series(cols):
    E0 = cols['E'][0]
    return dict(t=cols['time'].round(2).tolist(), dE=(cols['E'] - E0).round(5).tolist(),
                Q20=(cols['Q20'] / 100).round(2).tolist(), neck=cols['neck_peak'].round(5).tolist(),
                En=cols['Epair_n'].round(4).tolist(), Ep=cols['Epair_p'].round(4).tolist(),
                AL=(cols['N_L'] + cols['Z_L']).round(2).tolist(), ZL=cols['Z_L'].round(2).tolist(),
                AR=(cols['N_R'] + cols['Z_R']).round(2).tolist(), ZR=cols['Z_R'].round(2).tolist())


def path_dataset():
    p = dict(np.load('stage3/anim_path.npz'))
    keep = p['Q20'] < 33000              # Q20 >= 34000 b: box too short, those statics are not on the path
    p = {k: (v[keep] if v.ndim and len(v) == len(keep) else v) for k, v in p.items()}
    b64, rmax, amax = pack(p['rho_n'], p['rho_p'], p['delta_n'], p['delta_p'])
    E = p['E']
    return dict(id='path', label='Fission path', kind='path',
                note='Static, relaxed under a Q20 constraint (k_c 3.0). Each frame is one point on the path; '
                     'the curve is the energy relative to the lowest point (Q20 >= 340 b left out: box too short).',
                frames=b64, rmax=rmax, amax=amax, **grid_info(p['x'], p['z'], len(E)),
                param=(p['Q20'] / 100).round(1).tolist(),
                series=dict(Q20=(p['Q20'] / 100).round(2).tolist(), E=(E - E.min()).round(4).tolist(),
                            En=p['Epair_n'].round(4).tolist(), Ep=p['Epair_p'].round(4).tolist(),
                            Q30=(p['Q30'] / 1000).round(3).tolist()))


def new_dataset(name, start_q20, new_out, old_hist):
    """Relaunched run (second-order step, dt 0.2): frames so far, with the old run's curves for comparison."""
    f = np.load(new_out[:-4] + '_frames.npz')
    t = f['t']
    idx = np.unique(np.linspace(0, len(t) - 1, min(len(t), MAX_FRAMES)).round().astype(int))
    b64, rmax, amax = pack(f['rho_n'][idx], f['rho_p'][idx], f['delta_n'][idx], f['delta_p'][idx])
    cols, ts = history(new_out)
    old, _ = history(old_hist)
    return dict(id=name, label=f'Release from {start_q20} b', kind='time', step='new',
                note=f'Second-order step (Strang + midpoint), dt 0.2 fm/c; running, {len(t)} frames up to '
                     f't = {t[-1]:.0f} fm/c (every {t[1] - t[0]:.0f} fm/c, {len(idx)} shown). Dashed: the old '
                     'first-order run over the same times.',
                frames=b64, rmax=rmax, amax=amax, **grid_info(f['x'], f['z'], len(idx)),
                param=t[idx].round(2).tolist(), series=series(cols), old=series(old), t_scission=ts)


def old_dataset(name, start_q20, old_hist, old_anim):
    a = np.load(old_anim)
    b64, rmax, amax = pack(a['rho_n'], a['rho_p'], a['delta_n'], a['delta_p'])
    cols, ts = history(old_hist)
    return dict(id=name, label=f'{start_q20} b, old step', kind='time', step='old',
                note='First-order step (dt 0.1): energy drifts ~2 keV per fm/c. Densities exist only at t = 0 '
                     'and at the latest checkpoint; the dt 0.2 relaunch records a frame every record.',
                frames=b64, rmax=rmax, amax=amax, **grid_info(a['x'], a['z'], len(a['t'])),
                param=a['t'].round(2).tolist(), series=series(cols), t_scission=ts)


def releases_vs_q20():
    """The dynamic releases (second-order step) as Epair vs Q20, to overlay on the static path."""
    out = []
    for name, f in (('release from 300 b', 'stage3/fisQ30k_dt02.npz'), ('release from 200 b', 'stage3/fisQ20k_dt02.npz')):
        if not os.path.exists(f):
            continue
        cols, ts = history(f)
        k = slice(None, None, max(1, len(cols['time']) // 300))
        out.append(dict(name=name, Q20=(cols['Q20'][k] / 100).round(1).tolist(), En=cols['Epair_n'][k].round(3).tolist(),
                        Ep=cols['Epair_p'][k].round(3).tolist(), t_end=float(cols['time'][-1]),
                        q20_scission=None if ts is None else float(cols['Q20'][np.argmin(np.abs(cols['time'] - ts))] / 100)))
    return out


def main(out):
    path = path_dataset()
    path['releases'] = releases_vs_q20()
    data = [path,
            new_dataset('q30', 300, 'stage3/fisQ30k_dt02.npz', 'stage3/fisQ30k.npz'),
            new_dataset('q20', 200, 'stage3/fisQ20k_dt02.npz', 'stage3/oldstep/fisQ20k.npz'),
            old_dataset('q30old', 300, 'stage3/fisQ30k.npz', 'stage3/anim_old_Q30k.npz'),
            old_dataset('q20old', 200, 'stage3/oldstep/fisQ20k.npz', 'stage3/anim_old_Q20k.npz')]
    tpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'viewer_template.html')).read()
    html = tpl.replace('/*__DATA__*/null', json.dumps(data, separators=(',', ':')))
    open(out, 'w').write(html)
    print(f'wrote {out}: {len(html) / 1e6:.2f} MB;', ', '.join(f"{d['id']} {d['nframes']} frames ({d.get('step', 'static')})"
                                                          for d in data))


if __name__ == '__main__':
    main(sys.argv[1])
