"""
Rendering of TDHF runs saved as .npz files (see examples/tdhf).

Requires matplotlib and scipy (pip install jax-hfbfft[viz]). Figures are drawn
on a private Agg canvas, so importing this module does not change the
caller's matplotlib backend.

Collision files are written by examples/tdhf/collision.py and hold, per frame,
the x-z density slice and (in current files) the full neutron and proton
densities rho3d_n / rho3d_p. Slice-only files support render_collision and
spacetime; the other views need the 3D densities.
"""

from pathlib import Path

import numpy as np
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.animation import FuncAnimation, PillowWriter

BG, FG, DIM, EDGE = '#0e0e12', '#e8e8ee', '#8a8f98', '#3a3a44'
RHO_SAT = 0.16   # fm^-3


def _figure(figsize):
    fig = Figure(figsize=figsize)
    FigureCanvasAgg(fig)
    fig.patch.set_facecolor(BG)
    return fig


def _style(ax):
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_color(EDGE)
    ax.tick_params(colors=DIM, labelsize=8)
    return ax


def _colorbar(fig, im, ax, label=None, fraction=0.030, labelsize=8):
    cb = fig.colorbar(im, ax=ax, fraction=fraction, pad=0.02)
    if label:
        cb.set_label(label, color=DIM, fontsize=9)
    cb.ax.tick_params(colors=DIM, labelsize=labelsize)
    cb.outline.set_edgecolor(EDGE)
    return cb


def _save_animation(fig, update, n_frames, out, fps, dpi=90):
    FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps,
                  blit=False).save(out, writer=PillowWriter(fps=fps), dpi=dpi,
                                   savefig_kwargs={'facecolor': BG})
    return str(out)


# ── Loading ──────────────────────────────────────────────────────────────────

def load_collision(path) -> dict:
    """
    Read a collision .npz. Adds 'hasnp' (separate neutron/proton densities
    present) and 'has3d' (a 3D density present), and 'rho3d' (total density)
    when available.
    """
    d = np.load(path, allow_pickle=True)
    out = {k: d[k] for k in d.files}
    out['nucleus'] = str(out['nucleus'])
    out['nucleus2'] = str(out['nucleus2']) if 'nucleus2' in d.files else out['nucleus']
    out['e_cm'] = float(out['e_cm'])
    out['b'] = float(out['b'])
    out['hasnp'] = 'rho3d_n' in d.files and 'rho3d_p' in d.files
    out['has3d'] = out['hasnp'] or 'rho3d' in d.files
    if out['hasnp']:
        out['rho3d'] = out['rho3d_n'] + out['rho3d_p']
    return out


def _title(d):
    return (f"{d['nucleus']} + {d['nucleus2']}   "
            f"$E_{{cm}}$ = {d['e_cm']:g} MeV   b = {d['b']:g} fm")


def _require_3d(d, view):
    if not d['has3d']:
        raise ValueError(f"{view} needs the 3D density, which this file does not contain.")


def contour_mask(field, levels, width=0.8, grad_min=None):
    """
    Pixels within `width` pixels of any of `levels`.

    The offset |field - level| is divided by the local gradient to convert it
    into a distance in pixels, so lines have an even width whether the field
    changes quickly or slowly. Where the gradient is below `grad_min` (flat
    regions, e.g. the constant proton fraction in the interior) no line is
    drawn. The default is 0.15 x the level spacing per pixel.
    """
    gy, gx = np.gradient(field)
    grad = np.hypot(gx, gy)
    if grad_min is None:
        spacing = min(np.diff(sorted(levels))) if len(levels) > 1 else 1.0
        grad_min = 0.15 * spacing
    ok = grad > grad_min
    m = np.zeros(field.shape, dtype=bool)
    for L in levels:
        m |= ok & ((np.abs(field - L) / (grad + 1e-12)) < width)
    return m


# ── Animations written during the run ───────────────────────────────────────

def render_collision(path, fps: int = 15) -> str:
    """
    Animate a collision: the x-z density slice with the fragment separation
    and energy drift underneath. Writes <path>.gif and returns its path.
    """
    d = np.load(path, allow_pickle=True)
    frames, times = d['frames'], d['times']
    seps, drifts, extent = d['seps'], d['drifts'], d['extent']
    nuc, e_cm, b = str(d['nucleus']), float(d['e_cm']), float(d['b'])
    nuc2 = str(d['nucleus2']) if 'nucleus2' in d.files else nuc

    fig = _figure((12, 5.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1], hspace=0.42, wspace=0.24,
                          left=0.06, right=0.97, top=0.90, bottom=0.12)
    ax = _style(fig.add_subplot(gs[:, 0]))
    ax1 = _style(fig.add_subplot(gs[0, 1]))
    ax2 = _style(fig.add_subplot(gs[1, 1]))

    # z (the beam axis) horizontal, so the fragments approach from left and right.
    vmax = float(np.percentile(frames, 99.9))
    im = ax.imshow(frames[0].T, origin='lower',
                   extent=[extent[2], extent[3], extent[0], extent[1]],
                   cmap='inferno', vmin=0.0, vmax=vmax,
                   interpolation='bilinear', aspect='equal')
    ax.set_xlabel('z  (fm)   — beam axis', color=DIM, fontsize=9)
    ax.set_ylabel('x  (fm)', color=DIM, fontsize=9)
    _colorbar(fig, im, ax, r'$\rho$  (fm$^{-3}$)')
    ttl = ax.set_title('', color=FG, fontsize=12, family='monospace', pad=10)
    fig.text(0.06, 0.965, f"{nuc} + {nuc2}   TDHF   "
             f"$E_{{cm}}$ = {e_cm:g} MeV   b = {b:g} fm",
             color=FG, fontsize=12, family='monospace')

    ax1.plot(times, seps, color='#5ac8fa', lw=1.5)
    mk1, = ax1.plot([times[0]], [seps[0]], 'o', color='#ffd60a', ms=6)
    ax1.set_ylabel('fragment sep.  (fm)', color=DIM, fontsize=9)
    ax1.grid(alpha=0.13, color=DIM)
    ax1.tick_params(labelbottom=False)

    ax2.plot(times, drifts, color='#ff9f0a', lw=1.5)
    mk2, = ax2.plot([times[0]], [drifts[0]], 'o', color='#ffd60a', ms=6)
    ax2.set_ylabel(r'$\Delta E/E$', color=DIM, fontsize=9)
    ax2.set_xlabel('t  (fm/c)', color=DIM, fontsize=9)
    ax2.grid(alpha=0.13, color=DIM)
    ax2.axhline(0.0, color=DIM, lw=0.7, alpha=0.5)

    def update(i):
        im.set_data(frames[i].T)
        mk1.set_data([times[i]], [seps[i]])
        mk2.set_data([times[i]], [drifts[i]])
        ttl.set_text(f"t = {times[i]:6.1f} fm/c")
        return im, mk1, mk2, ttl

    return _save_animation(fig, update, len(frames), str(path).replace('.npz', '.gif'), fps,
                           dpi=95)


def render_oscillation(path, fps: int = 14) -> str:
    """
    Animate a single-nucleus run (examples/tdhf/oscillations.py): the x-z slice
    of rho (or rho_n - rho_p for the dipole mode) next to the tracked moment.
    Writes <path>.gif and returns its path.
    """
    d = np.load(path, allow_pickle=True)
    frames, times = d['frames'], d['times']
    moms, drifts, extent = d['moments'], d['drifts'], d['extent']
    mode, nuc = str(d['mode']), str(d['nucleus'])

    diverging = (mode == 'gdr')
    vmax = float(np.max(np.abs(frames)))
    cmap = 'RdBu_r' if diverging else 'magma'
    vmin = -vmax if diverging else 0.0

    fig = _figure((11, 4.6))
    ax, ax2 = fig.subplots(1, 2, gridspec_kw={'width_ratios': [1.15, 1]})
    _style(ax), _style(ax2)

    im = ax.imshow(frames[0], origin='lower', extent=extent, cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation='bilinear')
    ax.set_xlabel('x (fm)', color=DIM, fontsize=9)
    ax.set_ylabel('z (fm)', color=DIM, fontsize=9)
    label = r'$\rho_n-\rho_p$' if diverging else r'$\rho$'
    _colorbar(fig, im, ax, label + r'  (fm$^{-3}$)', fraction=0.046)
    ttl = ax.set_title('', color=FG, fontsize=11, family='monospace')

    ax2.plot(times, moms, color='#5ac8fa', lw=1.4)
    mk, = ax2.plot([times[0]], [moms[0]], 'o', color='#ffd60a', ms=7)
    ax2.set_xlabel('t (fm/c)', color=DIM, fontsize=9)
    ax2.set_ylabel(r'$\langle r^2\rangle$' if mode == 'monopole' else r'$Q_{20}$',
                   color=DIM, fontsize=9)
    ax2.grid(alpha=0.15)
    dtxt = ax2.text(0.03, 0.06, '', transform=ax2.transAxes,
                    color=DIM, fontsize=8, family='monospace')

    def update(i):
        im.set_data(frames[i])
        mk.set_data([times[i]], [moms[i]])
        ttl.set_text(f"{nuc}  {mode}   t = {times[i]:6.1f} fm/c")
        dtxt.set_text(f"dE/E = {drifts[i]:+.2e}")
        return im, mk, ttl, dtxt

    return _save_animation(fig, update, len(frames), str(path).replace('.npz', '.gif'), fps,
                           dpi=95)


# ── Spacetime ────────────────────────────────────────────────────────────────

def spacetime(d, out) -> str:
    """
    Density along the beam axis versus time, as one static image.

    With the 3D density this is the transverse integral int rho dx dy, whose
    integral over z is A at every time. Slice-only files fall back to the
    y = 0 slice summed over x, and the figure says so.
    """
    t = d['times']
    ext = d['extent']
    if d['has3d']:
        dx, dy, _ = d['spacing']
        prof = d['rho3d'].sum(axis=(1, 2)) * float(dx) * float(dy)
        lab = r'$\int \rho\, dx\, dy$   (fm$^{-1}$)'
        note = 'transverse-integrated'
    else:
        # frames are stored as rho[:, mid, :].T, i.e. (nt, nz, nx): sum over x.
        prof = d['frames'].sum(axis=2)
        lab = r'$\sum_x \rho(x, y{=}0, z)$   (arb.)'
        note = 'from the y=0 slice only'

    fig = _figure((8.5, 6.5))
    ax = _style(fig.add_subplot())
    im = ax.imshow(prof, origin='lower', aspect='auto', cmap='inferno',
                   extent=[ext[2], ext[3], t[0], t[-1]], interpolation='bilinear')
    ax.set_xlabel('z  (fm)   — beam axis', color=DIM, fontsize=10)
    ax.set_ylabel('t  (fm/c)', color=DIM, fontsize=10)
    ax.set_title(_title(d) + '\nspacetime', color=FG, fontsize=11,
                 family='monospace', pad=10)
    _colorbar(fig, im, ax, lab, fraction=0.045)
    ax.text(0.985, 0.015, note, transform=ax.transAxes, ha='right',
            color=DIM, fontsize=7.5, family='monospace')
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=BG)
    return str(out)


# ── Three orthogonal planes ──────────────────────────────────────────────────

def threeview(d, out, fps: int = 15, upsample: int = 5, bands: bool = True) -> str:
    """
    Animate the x-z, y-z and x-y planes through the box centre.

    With bands=True the density is drawn as filled bands at 1/8, 1/4, 1/2,
    3/4 and 1 x saturation, so the area above a given density reads directly.
    Slices are upsampled before colouring so band edges are smooth rather
    than one grid cell wide.
    """
    _require_3d(d, "threeview")
    from scipy.ndimage import zoom
    vol, t = d['rho3d'], d['times']
    ext, yext = d['extent'], d['yextent']
    nt, nx, ny, nz = vol.shape
    ix, iy, iz = nx // 2, ny // 2, nz // 2
    vmax = float(np.percentile(vol, 99.95))

    # Each slice is (vertical, horizontal) to match its extent; only the
    # beam's-eye panel (x horizontal) needs a transpose.
    panels = [
        (lambda v: v[:, iy, :], [ext[2], ext[3], ext[0], ext[1]],
         'z  (fm)', 'x  (fm)', 'x–z   side'),
        (lambda v: v[ix, :, :], [ext[2], ext[3], yext[0], yext[1]],
         'z  (fm)', 'y  (fm)', 'y–z   side, quarter turn'),
        (lambda v: v[:, :, iz].T, [ext[0], ext[1], yext[0], yext[1]],
         'x  (fm)', 'y  (fm)', 'x–y   down the beam'),
    ]

    levels = [0.02, 0.04, 0.08, 0.12, 0.16]
    cmap = matplotlib.colormaps['inferno']
    if bands:
        norm = matplotlib.colors.BoundaryNorm([0.0] + levels + [max(vmax, 0.2)], cmap.N)
    else:
        norm = matplotlib.colors.Normalize(0.0, vmax)

    def rgb(field):
        # Cubic upsampling can overshoot slightly below zero at the vacuum edge.
        f = np.clip(zoom(field, upsample, order=3), 0.0, None)
        img = cmap(norm(f))[..., :3]
        if bands:
            img[contour_mask(f, levels, width=0.9, grad_min=0.02 / upsample)] *= 0.55
        return img

    fig = _figure((15, 4.6))
    axes = fig.subplots(1, 3)
    ims = []
    for ax, (fn, e, xl, yl, name) in zip(axes, panels):
        _style(ax)
        im = ax.imshow(rgb(fn(vol[0])), origin='lower', extent=e,
                       interpolation='bilinear', aspect='equal')
        ax.set_xlabel(xl, color=DIM, fontsize=9)
        ax.set_ylabel(yl, color=DIM, fontsize=9)
        ax.set_title(name, color=FG, fontsize=10, family='monospace')
        ims.append((im, fn))
    sup = fig.suptitle('', color=FG, fontsize=12, family='monospace')
    fig.text(0.5, 0.015,
             ('density bands at ρ = 0.02, 0.04, 0.08, 0.12, 0.16 fm⁻³'
              if bands else 'ρ, continuous scale'),
             color=DIM, fontsize=8, family='monospace', ha='center')
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])

    def update(i):
        for im, fn in ims:
            im.set_data(rgb(fn(vol[i])))
        sup.set_text(f"{_title(d)}    t = {t[i]:6.1f} fm/c")
        return [im for im, _ in ims] + [sup]

    return _save_animation(fig, update, nt, out, fps)


# ── Isospin: relative motion of neutrons and protons ─────────────────────────

def neck_indices(tot, w_xy, dz):
    """
    Per frame, the z index of the plane dividing the two fragments: the
    minimum of the linear density between the two half-box centroids. This
    follows the neck, and for an asymmetric pair it differs from z = 0.
    """
    nt, _, _, nz = tot.shape
    prof = tot.sum(axis=(1, 2)) * w_xy          # (nt, nz), linear density
    out = np.empty(nt, dtype=int)
    zc = np.arange(nz)
    half = nz // 2
    for i in range(nt):
        p = prof[i]
        lo = int(round((p[:half] * zc[:half]).sum() / max(p[:half].sum(), 1e-12)))
        hi = int(round((p[half:] * zc[half:]).sum() / max(p[half:].sum(), 1e-12)))
        lo, hi = max(min(lo, nz - 2), 0), min(max(hi, lo + 1), nz - 1)
        out[i] = lo + int(np.argmin(p[lo:hi + 1]))
    return out


def isospin_transfer(d, out, fps: int = 15) -> str:
    """
    Animate rho_n - rho_p and the proton fraction Z/A in the y = 0 plane, with
    the net neutron and proton transfer across the neck and the Z/A of each
    side. Informative for N != Z reactions (e.g. 40Ca + 48Ca).
    """
    if not d['hasnp']:
        raise ValueError("isospin_transfer needs separate neutron and proton "
                         "densities (rho3d_n, rho3d_p).")
    from scipy.ndimage import zoom
    rn, rp, tot, t = d['rho3d_n'], d['rho3d_p'], d['rho3d'], d['times']
    ext = d['extent']
    dx, dy, dz = (float(v) for v in d['spacing'])
    nt, nx, ny, nz = tot.shape
    iy = ny // 2
    up = 5
    vlim = float(np.percentile(np.abs((rn - rp)[:, :, iy, :]), 99.5)) or 1e-3

    def slices(i):
        # Upsample the two densities and form the ratio afterwards: Z/A has a
        # step at the vacuum edge that cubic interpolation would ring across.
        # Vacuum (rho < 0.02 fm^-3) is masked out.
        n_ = zoom(rn[i, :, iy, :], up, order=3)
        p_ = zoom(rp[i, :, iy, :], up, order=3)
        tt = np.clip(n_ + p_, 0.0, None)
        keep = tt > 0.02
        return (np.where(keep, n_ - p_, np.nan),
                np.where(keep, p_ / np.maximum(tt, 1e-9), np.nan))

    dif0, frac0 = slices(0)

    ineck = neck_indices(tot, dx * dy, dz)
    wv = dx * dy * dz
    NL = np.array([rn[i, :, :, :ineck[i]].sum() * wv for i in range(nt)])
    ZL = np.array([rp[i, :, :, :ineck[i]].sum() * wv for i in range(nt)])
    NR = np.array([rn[i, :, :, ineck[i]:].sum() * wv for i in range(nt)])
    ZR = np.array([rp[i, :, :, ineck[i]:].sum() * wv for i in range(nt)])
    dN, dZ = NL - NL[0], ZL - ZL[0]
    fL = ZL / np.maximum(NL + ZL, 1e-9)
    fR = ZR / np.maximum(NR + ZR, 1e-9)

    fig = _figure((14.5, 6.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0], hspace=0.42,
                          wspace=0.22, left=0.06, right=0.96, top=0.88, bottom=0.10)
    axd = _style(fig.add_subplot(gs[0, 0]))
    axf = _style(fig.add_subplot(gs[1, 0]))
    axt = _style(fig.add_subplot(gs[0, 1]))
    axq = _style(fig.add_subplot(gs[1, 1]))
    zext = [ext[2], ext[3], ext[0], ext[1]]

    dcm = matplotlib.colormaps['RdBu_r'].copy()
    dcm.set_bad(BG)
    imd = axd.imshow(dif0, origin='lower', extent=zext, cmap=dcm,
                     vmin=-vlim, vmax=vlim, interpolation='bilinear', aspect='equal')
    axd.set_ylabel('x  (fm)', color=DIM, fontsize=9)
    axd.set_title(r'$\rho_n-\rho_p$   (red neutron-rich)', color=FG,
                  fontsize=10, family='monospace')
    _colorbar(fig, imd, axd, labelsize=7)

    fcm = matplotlib.colormaps['coolwarm'].copy()
    fcm.set_bad(BG)
    imf = axf.imshow(frac0, origin='lower', extent=zext, cmap=fcm,
                     vmin=0.35, vmax=0.55, interpolation='bilinear', aspect='equal')
    axf.set_xlabel('z  (fm)   — beam axis', color=DIM, fontsize=9)
    axf.set_ylabel('x  (fm)', color=DIM, fontsize=9)
    axf.set_title('proton fraction Z/A   (masked below ρ = 0.02)', color=FG,
                  fontsize=10, family='monospace')
    _colorbar(fig, imf, axf, labelsize=7)

    axt.plot(t, dN, color='#5ac8fa', lw=1.6, label='ΔN')
    axt.plot(t, dZ, color='#ff9f0a', lw=1.6, label='ΔZ')
    axt.axhline(0, color=DIM, lw=0.7, alpha=0.5)
    mkt1, = axt.plot([t[0]], [dN[0]], 'o', color='#ffd60a', ms=5)
    mkt2, = axt.plot([t[0]], [dZ[0]], 'o', color='#ffd60a', ms=5)
    axt.set_ylabel('net transfer\nonto left side', color=DIM, fontsize=9)
    # At least +/- half a nucleon, so numerical noise does not fill the panel.
    lim = max(float(np.abs(np.concatenate([dN, dZ])).max()) * 1.25, 0.5)
    axt.set_ylim(-lim, lim)
    axt.grid(alpha=0.13, color=DIM)
    axt.legend(facecolor=BG, edgecolor=EDGE, labelcolor=FG, fontsize=8)

    axq.plot(t, fL, color='#5ac8fa', lw=1.6, label='left')
    axq.plot(t, fR, color='#ff453a', lw=1.6, label='right')
    mkq1, = axq.plot([t[0]], [fL[0]], 'o', color='#ffd60a', ms=5)
    mkq2, = axq.plot([t[0]], [fR[0]], 'o', color='#ffd60a', ms=5)
    axq.set_xlabel('t  (fm/c)', color=DIM, fontsize=9)
    axq.set_ylabel('Z/A of each side', color=DIM, fontsize=9)
    axq.grid(alpha=0.13, color=DIM)
    axq.legend(facecolor=BG, edgecolor=EDGE, labelcolor=FG, fontsize=8)

    sup = fig.suptitle('', color=FG, fontsize=12, family='monospace')
    nline = axd.axvline(0.0, color=DIM, lw=0.9, ls='--', alpha=0.8)
    zgrid = np.linspace(ext[2], ext[3], nz)

    def update(i):
        dd, ff = slices(i)
        imd.set_data(dd)
        imf.set_data(ff)
        nline.set_xdata([zgrid[ineck[i]], zgrid[ineck[i]]])
        mkt1.set_data([t[i]], [dN[i]])
        mkt2.set_data([t[i]], [dZ[i]])
        mkq1.set_data([t[i]], [fL[i]])
        mkq2.set_data([t[i]], [fR[i]])
        sup.set_text(f"{_title(d)}    t = {t[i]:6.1f} fm/c")
        return imd, imf, nline, mkt1, mkt2, mkq1, mkq2, sup

    return _save_animation(fig, update, nt, out, fps)


# ── Isosurface ───────────────────────────────────────────────────────────────

def raycast(vol, level, light=(-0.55, 0.40, 0.73), paint=None, inset=3):
    """
    Shade the first crossing of `level` along axis 0.

    Returns (shade, alpha, painted), each of shape vol.shape[1:]. The crossing
    depth is refined to sub-voxel precision by linear interpolation, and the
    normal is -grad(rho) sampled at the hit voxel. Alpha fades in over a thin
    density band around the level so the silhouette anti-aliases. If `paint`
    (a co-registered scalar volume) is given, it is sampled `inset` voxels
    behind the surface, away from the noisy low-density edge.
    """
    mask = vol >= level
    hit = mask.any(axis=0)
    first = mask.argmax(axis=0)

    g0, g1, g2 = np.gradient(vol)
    idx = first[None, ...]
    n0 = -np.take_along_axis(g0, idx, 0)[0]
    n1 = -np.take_along_axis(g1, idx, 0)[0]
    n2 = -np.take_along_axis(g2, idx, 0)[0]
    norm = np.sqrt(n0**2 + n1**2 + n2**2) + 1e-12
    n0, n1, n2 = n0 / norm, n1 / norm, n2 / norm

    L = np.array(light, dtype=float)
    L /= np.linalg.norm(L)
    diff = np.clip(n0 * L[0] + n1 * L[1] + n2 * L[2], 0.0, 1.0)

    kp = np.maximum(first - 1, 0)
    v_hi = np.take_along_axis(vol, first[None], 0)[0]
    v_lo = np.take_along_axis(vol, kp[None], 0)[0]
    frac = np.where(v_hi > v_lo, (level - v_lo) / (v_hi - v_lo + 1e-12), 0.0)
    depth = (kp + np.clip(frac, 0.0, 1.0)) / max(vol.shape[0] - 1, 1)

    shade = np.clip(0.18 + 0.76 * diff * (1.0 - 0.42 * depth), 0, 1)
    peak = vol.max(axis=0)
    alpha = np.clip((peak - level) / (0.30 * level), 0.0, 1.0) * hit

    if paint is None:
        return shade, alpha, None
    deep = np.clip(first + inset, 0, vol.shape[0] - 1)
    painted = np.take_along_axis(paint, deep[None], 0)[0]
    return shade, alpha, painted


def _camera(vol, rot_z, tilt, upsample, smooth):
    """
    Rotate and resample the volume for one camera. The result has the view
    direction on axis 0, x on axis 1 (vertical) and z on axis 2 (horizontal).
    Cubic upsampling is needed: a light nucleus is only a few voxels across.
    """
    from scipy.ndimage import rotate as ndrotate, zoom, gaussian_filter
    v = vol
    if rot_z:
        v = ndrotate(v, rot_z, axes=(0, 1), reshape=False, order=1)   # about z
    if tilt:
        v = ndrotate(v, tilt, axes=(1, 2), reshape=False, order=1)    # about x
    if upsample > 1:
        v = zoom(v, upsample, order=3)
    if smooth:
        v = gaussian_filter(v, smooth)
    return np.moveaxis(v, 1, 0)          # look down y -> (ny, nx, nz)


def _camera_np(rn, rp, rot_z, tilt, upsample, smooth, fill):
    """
    Camera transform for a neutron/proton pair, returning (total, proton
    fraction). The fraction is formed after resampling, because it has a step
    at the vacuum edge that interpolation would ring across.
    """
    a = _camera(rn, rot_z, tilt, upsample, smooth)
    b = _camera(rp, rot_z, tilt, upsample, smooth)
    tot = a + b
    return tot, np.where(tot > 1e-3, b / np.maximum(tot, 1e-3), fill)


def isosurface(d, out, level=0.08, fps=15, upsample=3, smooth=1.0,
               mode='shells') -> str:
    """
    Animate a ray-cast 3D view from three cameras (oblique large, two
    axis-aligned small).

    Modes:
        'shells': nested translucent surfaces at several densities, composited
            front to back, so the diffuse surfaces are seen touching before
            the cores merge.
        'frac': one surface at `level`, coloured by proton fraction relative to
            the composite's equilibrium value (needs rho3d_n / rho3d_p).
        'plain': one lit surface at `level`.
    """
    _require_3d(d, "isosurface")
    if mode not in ('shells', 'frac', 'plain'):
        raise ValueError(f"Unknown mode {mode!r}; use 'shells', 'frac' or 'plain'.")
    if mode == 'frac' and not d['hasnp']:
        raise ValueError("mode='frac' needs separate neutron and proton densities.")
    vol, t = d['rho3d'], d['times']
    nt = vol.shape[0]
    peak = float(vol.max())
    if level >= peak:
        raise ValueError(f"level {level} is above the peak density {peak:.4f} fm^-3.")

    # (label, rotation about z, tilt about x)
    cams = [('down y', 0.0, 0.0), ('down x', 90.0, 0.0), ('oblique', 35.0, 24.0)]

    # One crop for the whole run, from the union of all frames, so the object
    # does not jump between frames. ever has shape (nx, ny, nz).
    ever = vol.max(axis=0) >= level
    xs = np.where(ever.any(axis=(1, 2)))[0]
    zs = np.where(ever.any(axis=(0, 1)))[0]
    pad = 4
    x0, x1 = max(xs[0] - pad, 0), min(xs[-1] + pad, vol.shape[1] - 1)
    z0, z1 = max(zs[0] - pad, 0), min(zs[-1] + pad, vol.shape[3] - 1)
    sl = (slice(x0 * upsample, (x1 + 1) * upsample),
          slice(z0 * upsample, (z1 + 1) * upsample))

    bg_rgb = np.array(matplotlib.colors.to_rgb(BG))
    plain = matplotlib.colormaps['copper']

    if mode == 'frac':
        pn, pp = d['rho3d_n'], d['rho3d_p']
        z_tot = float(pp[0].sum())
        f_eq = z_tot / (float(pn[0].sum()) + z_tot)
        fnorm = matplotlib.colors.Normalize(f_eq - 0.10, f_eq + 0.10)
        flevels = [f_eq + s for s in (-0.08, -0.06, -0.04, -0.02, 0.0,
                                      0.02, 0.04, 0.06, 0.08)]
        iso_cmap = matplotlib.colormaps['coolwarm']

    # (density, opacity), outermost first. Outer shells are nearly transparent
    # so that the dense core still dominates the composited pixel.
    shells = [(0.02, 0.10), (0.06, 0.18), (0.10, 0.30), (0.145, 0.97)]
    shell_cmap = matplotlib.colormaps['inferno']
    shell_cols = [np.array(shell_cmap(0.32 + 0.60 * j / max(len(shells) - 1, 1))[:3])
                  for j in range(len(shells))]

    def frame_shells(i, rot_z, tilt):
        cam = _camera(vol[i], rot_z, tilt, upsample, smooth)
        acc = np.zeros(cam.shape[1:] + (3,))
        cov = np.zeros(cam.shape[1:])
        for (lev, op), col in zip(shells, shell_cols):
            if lev >= float(vol[i].max()):
                continue
            sh, al, _ = raycast(cam, lev)
            a = al * op
            lit = col * (0.45 + 0.75 * sh)[..., None]
            acc += (1.0 - cov)[..., None] * a[..., None] * lit
            cov += (1.0 - cov) * a
        acc, cov = acc[sl], cov[sl][..., None]
        return np.clip(acc, 0, 1) + bg_rgb * (1.0 - cov)

    def frame(i, rot_z, tilt):
        if mode == 'shells':
            return frame_shells(i, rot_z, tilt)
        if mode == 'frac':
            cv, cf = _camera_np(pn[i], pp[i], rot_z, tilt, upsample, smooth, f_eq)
            # Sample ~1 fm inside the surface: past the noisy outer edge, but
            # shallow enough to be the composition of the surface itself.
            sh, al, painted = raycast(cv, level, paint=cf, inset=upsample)
            sh, al, painted = sh[sl], al[sl][..., None], painted[sl]
            img = iso_cmap(fnorm(painted))[..., :3] * (0.25 + 0.92 * sh[..., None])
            # No contours near the silhouette, where the sampled fraction is unreliable.
            img[contour_mask(painted, flevels) & (al[..., 0] > 0.75)] *= 0.45
            return np.clip(img, 0, 1) * al + bg_rgb * (1.0 - al)
        sh, al, _ = raycast(_camera(vol[i], rot_z, tilt, upsample, smooth), level)
        sh, al = sh[sl], al[sl][..., None]
        return plain(sh)[..., :3] * al + bg_rgb * (1.0 - al)

    fig = _figure((14.5, 5.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.05, 1.0], hspace=0.16,
                          wspace=0.06, left=0.02, right=0.98, top=0.88, bottom=0.09)
    order = [(cams[2], gs[:, 0]), (cams[0], gs[0, 1]), (cams[1], gs[1, 1])]
    ims = []
    for (name, rz, tl), cell in order:
        ax = _style(fig.add_subplot(cell))
        ax.set_xticks([])
        ax.set_yticks([])
        im = ax.imshow(frame(0, rz, tl), origin='lower',
                       interpolation='bilinear', aspect='equal')
        ax.set_title(name, color=FG, fontsize=10, family='monospace', pad=4)
        ims.append((im, rz, tl))
    sup = fig.suptitle('', color=FG, fontsize=12, family='monospace')

    if mode == 'shells':
        caption = ('nested translucent shells at ρ = '
                   + ', '.join(f'{lv:g}' for lv, _ in shells)
                   + ' fm⁻³  (outer diffuse → inner dense core)')
    elif mode == 'frac':
        caption = (f'surface ρ = {level:g} fm⁻³, coloured by proton fraction Z/A '
                   f'(blue neutron-rich → red proton-rich, equilibrium {f_eq:.3f}); '
                   f'contours every 0.02')
    else:
        caption = f'surface ρ = {level:g} fm⁻³'
    fig.text(0.5, 0.022, caption, color=DIM, fontsize=8, family='monospace', ha='center')

    def update(i):
        for im, rz, tl in ims:
            im.set_data(frame(i, rz, tl))
        sup.set_text(f"{_title(d)}    "
                     f"$\\rho$ = {level:g} fm$^{{-3}}$    t = {t[i]:6.1f} fm/c")
        return [im for im, _, _ in ims] + [sup]

    return _save_animation(fig, update, nt, out, fps)


def render_all_views(path, fps: int = 15, iso_level: float = 0.08,
                     iso_mode: str = 'shells', iso_upsample: int = 3,
                     bands: bool = True) -> list:
    """Write every view the file supports next to it; returns the paths."""
    d = load_collision(path)
    stem = str(Path(path).with_suffix(''))
    out = [spacetime(d, stem + '_spacetime.png')]
    if d['has3d']:
        out.append(threeview(d, stem + '_3view.gif', fps, bands=bands))
        out.append(isosurface(d, stem + '_iso.gif', iso_level, fps,
                              upsample=iso_upsample, mode=iso_mode))
    if d['hasnp']:
        out.append(isospin_transfer(d, stem + '_isospin.gif', fps))
    return out
