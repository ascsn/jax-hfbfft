"""
Frames for the viewers in viz/, recorded by run.py, collide.py and fission.py.

At every record (FrameRecorder is the `extra` hook of TDHFB.iterate):
  OUT_frames.npz   y = 0 slices: rho_n, rho_p (float32) and the complex pair
                   density xi_q and pair field Delta_q (complex64);
  OUT_levels.npz   the canonical levels: occupation v2 and <h> (MeV) of each
                   eigen-orbital of rho~, with its isospin (--levels);
and every `every3d` fm/c:
  OUT_3d/t<time>.npz  full 3D rho_q (float32) and Delta_q (complex64), plus
                   v0, v0p, rho0pr, so xi_q = Delta_q / (v0_q (1 - rho/rho0pr)).
Files are written atomically; `resume(t)` keeps what was recorded up to t.
"""

import os

import jax
import numpy as np

from jax_hfbfft.physics import tdhfb as T


def stem(out: str) -> str:
    return out[:-4] if out.endswith('.npz') else out


class FrameRecorder:
    KEYS = ('t', 'rho_n', 'rho_p', 'xi_n', 'xi_p', 'delta_n', 'delta_p')

    def __init__(self, out: str, every3d: float = 0.0, levels: bool = True):
        s = stem(out)
        self.path2d, self.dir3d, self.path_lv = s + '_frames.npz', s + '_3d', s + '_levels.npz'
        self.every3d, self.levels = every3d, levels
        self.f2 = {k: [] for k in self.KEYS}
        self.lv = dict(t=[], iso=[], occ=[], e=[])
        self._lev = None
        if every3d > 0:
            os.makedirs(self.dir3d, exist_ok=True)

    def resume(self, t: float):
        if os.path.exists(self.path2d):
            fr = np.load(self.path2d)
            keep = fr['t'] <= t + 1e-9
            self.f2 = {k: list(fr[k][keep]) for k in self.KEYS}
        if self.levels and os.path.exists(self.path_lv):
            lv = np.load(self.path_lv)
            keep = lv['t'] <= t + 1e-9
            self.lv = {k: list(lv[k][keep]) for k in self.lv}

    def __call__(self, td, flds):
        t = td.time
        if self.f2['t'] and t <= self.f2['t'][-1] + 1e-9:
            return
        g, iy = td.grid, td.grid.ny // 2
        rho = np.asarray(flds[0].rho)
        delta, xi = np.asarray(flds[3]), np.asarray(flds[4])
        self.f2['t'].append(t)
        for q, nm in ((0, 'n'), (1, 'p')):
            self.f2['rho_' + nm].append(rho[q, :, iy, :].astype(np.float32))
            self.f2['xi_' + nm].append(xi[q, :, iy, :].astype(np.complex64))
            self.f2['delta_' + nm].append(delta[q, :, iy, :].astype(np.complex64))
        if self.every3d > 0:
            k3 = t / self.every3d
            f3 = os.path.join(self.dir3d, f't{t:07.1f}.npz')
            if abs(k3 - round(k3)) < 1e-6 and not os.path.exists(f3):
                f = td.force
                np.savez(f3 + '.tmp.npz', t=t, rho_n=rho[0].astype(np.float32), rho_p=rho[1].astype(np.float32),
                         delta_n=delta[0].astype(np.complex64), delta_p=delta[1].astype(np.complex64),
                         v0=float(f.v0neut), v0p=float(f.v0prot), rho0pr=float(f.rho0pr),
                         x=np.asarray(g.x), y=np.asarray(g.y), z=np.asarray(g.z))
                os.replace(f3 + '.tmp.npz', f3)
        if self.levels and td.scheme == 'global':
            if self._lev is None:
                self._lev = jax.jit(lambda psi, r, iso, mf: T.canonical_levels(psi, r, iso, mf, g))
            S = td.state
            iso, occ, e = (np.asarray(v) for v in self._lev(S.psi, S.rho, S.isospin, flds[1]))
            self.lv['t'].append(t)
            self.lv['iso'].append(iso.astype(np.int8))
            self.lv['occ'].append(occ.astype(np.float32))
            self.lv['e'].append(e.astype(np.float32))

    def save(self, grid):
        if self.f2['t']:
            np.savez(self.path2d + '.tmp.npz', x=np.asarray(grid.x), z=np.asarray(grid.z),
                     **{k: np.asarray(v) for k, v in self.f2.items()})
            os.replace(self.path2d + '.tmp.npz', self.path2d)
        if self.lv['t']:
            n = max(len(v) for v in self.lv['occ'])            # orbitals can be dropped: pad with NaN / -1
            pad = lambda vs, fill, dt: np.stack([np.concatenate([v, np.full(n - len(v), fill, dt)]) for v in vs])
            np.savez(self.path_lv + '.tmp.npz', t=np.asarray(self.lv['t']), iso=pad(self.lv['iso'], -1, np.int8),
                     occ=pad(self.lv['occ'], np.nan, np.float32), e=pad(self.lv['e'], np.nan, np.float32))
            os.replace(self.path_lv + '.tmp.npz', self.path_lv)
