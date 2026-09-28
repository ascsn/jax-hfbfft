#!/usr/bin/env python3
"""
Alternative views of a finished TDHF collision.  Pure post-processing -- reads
an npz written by tdhf_collide.py and re-renders.  No propagation, no GPU.

    python3 tdhf_views.py out/coll_O16_E34_b0.npz --all
    python3 tdhf_views.py out/coll_O16_E34_b0.npz --spacetime
    python3 tdhf_views.py out/coll_O16_E34_b0.npz --iso --iso-level 0.08

WHAT EACH ONE IS FOR
--------------------
--spacetime   z horizontally, time vertically, density as colour.  The entire
              collision as ONE static image: approach, merger, and every
              subsequent oscillation readable at a glance, with the oscillation
              period measurable straight off the axis.  This is the view an
              animation cannot give you and the only one of the three that
              survives being printed in a paper.

--threeview   The three orthogonal planes through the box, animated together:
              x-z (side), y-z (side, rotated a quarter turn about the beam),
              and x-y (looking down the beam).  Drawn as filled density BANDS,
              a topographic map rather than a heat map, so "how much of the
              nucleus is above half density" reads as an area.  Each slice is
              upsampled before colouring, without which every band edge is one
              grid cell -- about sixteen screen pixels -- of staircase.
              --flat-slices restores a continuous colour scale.

              These are SLICES, not projections, so they show internal
              structure: the beam's-eye panel stays empty until the neck
              actually reaches the midplane, which is informative rather than a
              bug.  For a head-on collision the two side panels are identical by
              symmetry; they diverge as soon as b is non-zero.

--isospin     How the two species move relative to each other, which the matter
              views cannot show at all because they only ever see rho_n + rho_p.
              Four panels: rho_n - rho_p (where the imbalance sits), Z/A (what
              each region is made of), the net neutrons and protons transferred
              across the neck, and Z/A of each side converging as the system
              equilibrates.  The dividing plane is the density minimum between
              the fragments, not z=0 -- an asymmetric pair is not centred
              between its fragments, and once a neck forms the meaningful
              boundary is where the matter is thinnest.

              Says nothing on an N = Z + N = Z reaction: Z/A is 0.5 everywhere
              and there is no imbalance to equilibrate.  40Ca + 48Ca is the case
              it exists for.

--iso         3D view, three cameras, oblique one large.  --iso-mode picks what
              is drawn:

                shells (default)  Nested translucent surfaces at several
                    densities, composited front to back.  Better than a single
                    opaque surface for a collision: the diffuse outer surfaces
                    touch and bridge well before the dense cores do, and one
                    opaque surface hides that entirely.
                frac  A single surface at --iso-level coloured by local proton
                    fraction, with contours of it.  Shows the neutron skin and,
                    during a collision, charge equilibration across the neck.
                    Needs the neutron/proton split, and says nothing at all on
                    an N = Z system, where the fraction is flat at 0.5.
                plain  One lit surface, no colouring.

ISOSURFACE METHOD
-----------------
No marching cubes: scikit-image is not installed here.  Instead each pixel casts
a ray along the view axis, takes the first crossing of the level (refined to
sub-voxel by linear interpolation), and shades it with the local density
gradient as the surface normal.  That is a genuine isosurface, it is pure numpy,
and it is a good deal faster than meshing 27k voxels 150 times over -- which
matters because this runs per frame per camera.

Oblique cameras are obtained by rotating the VOLUME (scipy.ndimage.rotate) and
casting along a fixed axis, rather than by rotating rays.  Same picture, far
less code.

OLDER FILES
-----------
npz files written before rho3d was added carry only the x-z slice.  --spacetime
still works from those (it falls back to summing the slice, and says so on the
plot); --threeview and --iso need the 3D density and will tell you to re-run.
"""

import sys
import argparse
from pathlib import Path

import tdhf_env
tdhf_env.setup()

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

BG, FG, DIM = '#0e0e12', '#e8e8ee', '#8a8f98'


def _style(ax):
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_color('#3a3a44')
    ax.tick_params(colors=DIM, labelsize=8)
    return ax


def load(path):
    """
    Read an npz and normalise across the three formats this has had.

      oldest   frames only (a single y=0 slice)
      middle   + rho3d, the summed 3D density
      current  + rho3d_n / rho3d_p, neutrons and protons separately

    `has3d` gates the 3D views, `hasnp` gates anything needing the proton
    fraction.  n cannot be recovered from a summed density, so files written
    before the split simply cannot support the isospin views.
    """
    d = np.load(path, allow_pickle=True)
    out = {k: d[k] for k in d.files}
    out['nucleus'] = str(out['nucleus'])
    out['nucleus2'] = str(out['nucleus2']) if 'nucleus2' in d.files \
        else out['nucleus']
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


def contour_mask(field, levels, width=0.8, grad_min=None):
    """
    Pixels lying within `width` PIXELS of any of `levels`.

    Dividing the value offset by the local gradient converts it into a distance
    measured in pixels, so the lines come out an even width whether the field is
    changing fast or slowly.  Thresholding |field - level| directly instead
    gives lines that balloon across the flat interior and vanish on the steep
    surface -- the opposite of what a contour map is for.

    `grad_min` suppresses lines where the field is flat.  Without it the ratio
    is 0/0 across the nuclear interior, where the proton fraction is constant to
    within rounding, and every level within noise of the plateau value paints
    the whole interior with speckle.  A flat region genuinely has no contours in
    it; the guard says so explicitly.  Defaults to a twentieth of the level
    spacing per pixel, i.e. lines are dropped once they would exceed ~20 px.
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


# ── spacetime ────────────────────────────────────────────────────────────────

def spacetime(d, out):
    """
    Density along the beam axis versus time.

    With the 3D density this is a true transverse integral, rho_lin(z,t) =
    int rho dx dy, which is the physically meaningful quantity: its integral
    over z is exactly A at every time, so vertical stripes of constant total
    brightness mean particle number is conserved and you can see that by eye.
    Falling back to a slice loses that property, so the plot says which it used.
    """
    t = d['times']
    ext = d['extent']
    if d['has3d']:
        vol = d['rho3d']                        # (nt, nx, ny, nz)
        dx, dy, _ = d['spacing']
        prof = vol.sum(axis=(1, 2)) * float(dx) * float(dy)
        lab = r'$\int \rho\, dx\, dy$   (fm$^{-1}$)'
        note = 'transverse-integrated'
    else:
        # frames[i] is (nz, nx) -- stored as rho[:, mid, :].T -- so the stacked
        # array is (nt, nz, nx) and the transverse axis to collapse is 2, not 1.
        # Summing axis=1 instead collapses z and yields a transverse profile
        # that plots without error against the z extent and looks like a blob
        # sitting at the origin for the whole run.
        prof = d['frames'].sum(axis=2)
        lab = r'$\sum_x \rho(x, y{=}0, z)$   (arb.)'
        note = 'from the y=0 slice only — re-run to get the true integral'

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    fig.patch.set_facecolor(BG)
    _style(ax)
    im = ax.imshow(prof, origin='lower', aspect='auto', cmap='inferno',
                   extent=[ext[2], ext[3], t[0], t[-1]],
                   interpolation='bilinear')
    ax.set_xlabel('z  (fm)   — beam axis', color=DIM, fontsize=10)
    ax.set_ylabel('t  (fm/c)', color=DIM, fontsize=10)
    ax.set_title(_title(d) + '\nspacetime', color=FG, fontsize=11,
                 family='monospace', pad=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label(lab, color=DIM, fontsize=9)
    cb.ax.tick_params(colors=DIM, labelsize=8)
    cb.outline.set_edgecolor('#3a3a44')
    ax.text(0.985, 0.015, note, transform=ax.transAxes, ha='right',
            color=DIM, fontsize=7.5, family='monospace')
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=BG)
    plt.close(fig)
    print(f"  wrote {out}")


# ── three orthogonal planes ──────────────────────────────────────────────────

def threeview(d, out, fps=15, upsample=5, bands=True):
    if not d['has3d']:
        raise SystemExit("--threeview needs the 3D density; this npz predates "
                         "rho3d.  Re-run tdhf_collide.py to get it.")
    from scipy.ndimage import zoom
    vol, t = d['rho3d'], d['times']
    ext, yext = d['extent'], d['yextent']
    nt, nx, ny, nz = vol.shape
    ix, iy, iz = nx // 2, ny // 2, nz // 2
    vmax = float(np.percentile(vol, 99.95))

    # imshow reads rows as vertical and columns as horizontal, so each slice
    # must already be (vertical, horizontal) to match its extent.  v[:, iy, :]
    # is (nx, nz) = (x, z), which is correct as-is for an extent whose
    # horizontal axis is z; transposing it there plots the fragments separated
    # across the beam instead of along it.  Only the beam's-eye panel, whose
    # extent puts x horizontal, needs the transpose.
    panels = [
        (lambda v: v[:, iy, :], [ext[2], ext[3], ext[0], ext[1]],
         'z  (fm)', 'x  (fm)', 'x–z   side'),
        (lambda v: v[ix, :, :], [ext[2], ext[3], yext[0], yext[1]],
         'z  (fm)', 'y  (fm)', 'y–z   side, quarter turn'),
        (lambda v: v[:, :, iz].T, [ext[0], ext[1], yext[0], yext[1]],
         'x  (fm)', 'y  (fm)', 'x–y   down the beam'),
    ]

    # Density bands, in units of saturation (0.16 fm^-3): 1/8, 1/4, 1/2, 3/4 and
    # saturation.  The 0.08 boundary is the same surface the isosurface view
    # renders, so the two pictures can be read against each other.
    #
    # Filled bands rather than lines.  A thin contour line is one grid cell wide
    # and there is no way to draw it that is not either aliased or blurred; a
    # FILLED region has an edge that upsampling genuinely sharpens, and it also
    # carries the reading "how much of the nucleus is above half density" as an
    # area rather than as a distance between two lines.
    levels = [0.02, 0.04, 0.08, 0.12, 0.16]
    cmap = matplotlib.colormaps['inferno']
    if bands:
        norm = matplotlib.colors.BoundaryNorm([0.0] + levels + [max(vmax, 0.2)],
                                              cmap.N)
    else:
        norm = matplotlib.colors.Normalize(0.0, vmax)

    def rgb(field):
        # Upsample BEFORE colouring.  Colouring at grid resolution and letting
        # imshow interpolate afterwards is what made this look pixelated: at
        # 1 fm spacing one cell is ~16 screen pixels, so every band edge was a
        # staircase of 16-pixel steps that no amount of interpolation could
        # hide.  Cubic zoom overshoots slightly at the vacuum edge, hence the
        # clip -- a small negative density would otherwise wrap to the top of
        # the colour scale.
        f = np.clip(zoom(field, upsample, order=3), 0.0, None)
        img = cmap(norm(f))[..., :3]
        if bands:
            # Darken the band boundaries just enough to read as contour edges.
            img[contour_mask(f, levels, width=0.9,
                             grad_min=0.02 * upsample**-1)] *= 0.55
        return img

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    fig.patch.set_facecolor(BG)
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

    FuncAnimation(fig, update, frames=nt, interval=1000 // fps, blit=False).save(
        out, writer=PillowWriter(fps=fps), dpi=90,
        savefig_kwargs={'facecolor': BG})
    plt.close(fig)
    print(f"  wrote {out}  ({nt} frames)")


# ── isospin: how neutrons and protons move relative to each other ────────────

def neck_indices(tot, w_xy, dz):
    """
    Per frame, the z index of the dividing plane between the two fragments.

    Splitting at z=0 instead would be wrong twice over: an asymmetric pair is
    not centred between its fragments (40Ca+48Ca starts at -8 and +7 fm), and
    once a neck forms the physically meaningful boundary is where the density is
    thinnest, not where the box happens to be zero.  So: locate each side's
    centroid, then take the minimum of the linear density between them.

    After the two have fully merged this degenerates to "the thinnest plane near
    the middle", which is still the right split for asking how much material has
    crossed from one side to the other.
    """
    nt, _, _, nz = tot.shape
    prof = tot.sum(axis=(1, 2)) * w_xy          # (nt, nz), linear density
    out = np.empty(nt, dtype=int)
    zc = np.arange(nz)
    for i in range(nt):
        p = prof[i]
        half = nz // 2
        lo = int(round((p[:half] * zc[:half]).sum() / max(p[:half].sum(), 1e-12)))
        hi = int(round((p[half:] * zc[half:]).sum() / max(p[half:].sum(), 1e-12)))
        lo, hi = max(min(lo, nz - 2), 0), min(max(hi, lo + 1), nz - 1)
        out[i] = lo + int(np.argmin(p[lo:hi + 1]))
    return out


def isospin(d, out, fps=15):
    """
    Isovector density, local composition, and the resulting nucleon transfer.

    This is the view for how the two species act on each other rather than how
    the matter moves as a whole: rho_n - rho_p says where the imbalance sits,
    Z/A says what each region is made of, and the traces integrate that into how
    many neutrons and protons have actually crossed the neck.
    """
    if not d['hasnp']:
        raise SystemExit("--isospin needs rho3d_n / rho3d_p; this npz predates "
                         "the neutron/proton split.  Re-run tdhf_collide.py.")
    rn, rp, tot, t = d['rho3d_n'], d['rho3d_p'], d['rho3d'], d['times']
    ext, yext = d['extent'], d['yextent']
    dx, dy, dz = (float(v) for v in d['spacing'])
    nt, nx, ny, nz = tot.shape
    iy = ny // 2

    from scipy.ndimage import zoom
    US = 5
    vlim = float(np.percentile(np.abs((rn - rp)[:, :, iy, :]), 99.5)) or 1e-3

    def slices(i):
        """
        Upsampled (rho_n - rho_p, Z/A) in the y=0 plane, vacuum masked out.

        The two DENSITIES are upsampled and the ratio formed afterwards, for the
        same reason as in the isosurface: Z/A steps at the vacuum edge and cubic
        interpolation rings across a step, painting false composition into the
        interior.  Masking below 0.02 fm^-3 also keeps the panels from being
        dominated by a large block of vacuum, where an isovector density and a
        proton fraction are both meaningless.
        """
        n_ = zoom(rn[i, :, iy, :], US, order=3)
        p_ = zoom(rp[i, :, iy, :], US, order=3)
        tt = np.clip(n_ + p_, 0.0, None)
        keep = tt > 0.02
        dd = np.where(keep, n_ - p_, np.nan)
        ff = np.where(keep, p_ / np.maximum(tt, 1e-9), np.nan)
        return dd, ff

    dif0, frac0 = slices(0)

    # Transfer across the neck.
    ineck = neck_indices(tot, dx * dy, dz)
    wv = dx * dy * dz
    NL = np.array([rn[i, :, :, :ineck[i]].sum() * wv for i in range(nt)])
    ZL = np.array([rp[i, :, :, :ineck[i]].sum() * wv for i in range(nt)])
    NR = np.array([rn[i, :, :, ineck[i]:].sum() * wv for i in range(nt)])
    ZR = np.array([rp[i, :, :, ineck[i]:].sum() * wv for i in range(nt)])
    dN, dZ = NL - NL[0], ZL - ZL[0]
    fL = ZL / np.maximum(NL + ZL, 1e-9)
    fR = ZR / np.maximum(NR + ZR, 1e-9)

    fig = plt.figure(figsize=(14.5, 6.0))
    fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0], hspace=0.42,
                          wspace=0.22, left=0.06, right=0.96, top=0.88,
                          bottom=0.10)
    axd = _style(fig.add_subplot(gs[0, 0]))
    axf = _style(fig.add_subplot(gs[1, 0]))
    axt = _style(fig.add_subplot(gs[0, 1]))
    axq = _style(fig.add_subplot(gs[1, 1]))
    zext = [ext[2], ext[3], ext[0], ext[1]]

    dcm = matplotlib.colormaps['RdBu_r'].copy()
    dcm.set_bad(BG)
    imd = axd.imshow(dif0, origin='lower', extent=zext, cmap=dcm,
                     vmin=-vlim, vmax=vlim, interpolation='bilinear',
                     aspect='equal')
    axd.set_ylabel('x  (fm)', color=DIM, fontsize=9)
    axd.set_title(r'$\rho_n-\rho_p$   (red neutron-rich)', color=FG,
                  fontsize=10, family='monospace')
    cb = fig.colorbar(imd, ax=axd, fraction=0.030, pad=0.02)
    cb.ax.tick_params(colors=DIM, labelsize=7)
    cb.outline.set_edgecolor('#3a3a44')

    fcm = matplotlib.colormaps['coolwarm'].copy()
    fcm.set_bad(BG)
    imf = axf.imshow(frac0, origin='lower', extent=zext, cmap=fcm,
                     vmin=0.35, vmax=0.55, interpolation='bilinear',
                     aspect='equal')
    axf.set_xlabel('z  (fm)   — beam axis', color=DIM, fontsize=9)
    axf.set_ylabel('x  (fm)', color=DIM, fontsize=9)
    axf.set_title('proton fraction Z/A   (masked below ρ = 0.02)', color=FG,
                  fontsize=10, family='monospace')
    cb2 = fig.colorbar(imf, ax=axf, fraction=0.030, pad=0.02)
    cb2.ax.tick_params(colors=DIM, labelsize=7)
    cb2.outline.set_edgecolor('#3a3a44')

    axt.plot(t, dN, color='#5ac8fa', lw=1.6, label='ΔN')
    axt.plot(t, dZ, color='#ff9f0a', lw=1.6, label='ΔZ')
    axt.axhline(0, color=DIM, lw=0.7, alpha=0.5)
    mkt1, = axt.plot([t[0]], [dN[0]], 'o', color='#ffd60a', ms=5)
    mkt2, = axt.plot([t[0]], [dZ[0]], 'o', color='#ffd60a', ms=5)
    axt.set_ylabel('net transfer\nonto left side', color=DIM, fontsize=9)
    # Floor the axis at +/- half a nucleon.  Autoscaling makes 1e-4 of numerical
    # noise fill the panel and read as a real trend, which is exactly the wrong
    # impression before the fragments have even touched.
    lim = max(float(np.abs(np.concatenate([dN, dZ])).max()) * 1.25, 0.5)
    axt.set_ylim(-lim, lim)
    axt.grid(alpha=0.13, color=DIM)
    axt.legend(facecolor=BG, edgecolor='#3a3a44', labelcolor=FG, fontsize=8)

    axq.plot(t, fL, color='#5ac8fa', lw=1.6, label='left')
    axq.plot(t, fR, color='#ff453a', lw=1.6, label='right')
    mkq1, = axq.plot([t[0]], [fL[0]], 'o', color='#ffd60a', ms=5)
    mkq2, = axq.plot([t[0]], [fR[0]], 'o', color='#ffd60a', ms=5)
    axq.set_xlabel('t  (fm/c)', color=DIM, fontsize=9)
    axq.set_ylabel('Z/A of each side', color=DIM, fontsize=9)
    axq.grid(alpha=0.13, color=DIM)
    axq.legend(facecolor=BG, edgecolor='#3a3a44', labelcolor=FG, fontsize=8)

    sup = fig.suptitle('', color=FG, fontsize=12, family='monospace')
    nline = axd.axvline(0.0, color='#8a8f98', lw=0.9, ls='--', alpha=0.8)
    zgrid = np.linspace(ext[2], ext[3], nz)

    def update(i):
        dd, ff = slices(i)
        imd.set_data(dd)
        imf.set_data(ff)
        nline.set_xdata([zgrid[ineck[i]], zgrid[ineck[i]]])
        mkt1.set_data([t[i]], [dN[i]]); mkt2.set_data([t[i]], [dZ[i]])
        mkq1.set_data([t[i]], [fL[i]]); mkq2.set_data([t[i]], [fR[i]])
        sup.set_text(f"{_title(d)}    t = {t[i]:6.1f} fm/c")
        return imd, imf, nline, mkt1, mkt2, mkq1, mkq2, sup

    FuncAnimation(fig, update, frames=nt, interval=1000 // fps, blit=False).save(
        out, writer=PillowWriter(fps=fps), dpi=90,
        savefig_kwargs={'facecolor': BG})
    plt.close(fig)
    print(f"  wrote {out}  ({nt} frames)")
    print(f"    net transfer by end: ΔN = {dN[-1]:+.2f}, ΔZ = {dZ[-1]:+.2f}")
    print(f"    Z/A  left {fL[0]:.4f} -> {fL[-1]:.4f}   "
          f"right {fR[0]:.4f} -> {fR[-1]:.4f}")


# ── isosurface ───────────────────────────────────────────────────────────────

def raycast(vol, level, light=(-0.55, 0.40, 0.73), paint=None, inset=3):
    """
    Shade the first crossing of `level` along axis 0.

    Returns (shade, alpha, painted), each of shape vol.shape[1:].  `paint`, if
    given, is a co-registered scalar volume (here the proton fraction) sampled
    `inset` voxels behind the surface and returned for the caller to colour by;
    `painted` is None otherwise.

    The crossing depth is refined to sub-voxel by linear interpolation between
    the last voxel below the level and the first above it.  Without that the
    depth is an integer and the depth cue banks in visible terraces -- which on
    a nucleus only ~6 voxels across is most of what you see.

    The normal is -grad(rho): density falls off outward, so the negative
    gradient points out of the surface.  It is sampled AT the hit voxel, not at
    a fixed depth -- sampling anywhere else gives a surface that still looks
    lit but puts the highlights in the wrong places on concave features, and
    the neck is exactly such a feature.
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

    # Sub-voxel crossing depth.
    kp = np.maximum(first - 1, 0)
    v_hi = np.take_along_axis(vol, first[None], 0)[0]
    v_lo = np.take_along_axis(vol, kp[None], 0)[0]
    frac = np.where(v_hi > v_lo, (level - v_lo) / (v_hi - v_lo + 1e-12), 0.0)
    depth = (kp + np.clip(frac, 0.0, 1.0)) / max(vol.shape[0] - 1, 1)

    shade = np.clip(0.18 + 0.76 * diff * (1.0 - 0.42 * depth), 0, 1)

    # Soft silhouette.  Returning NaN outside the surface and leaning on
    # cmap.set_bad gives a hard-edged mask, and imshow's interpolation cannot
    # blend across NaN -- so the staircase survives however finely the volume is
    # resampled, and gets worse the more the crop is magnified on screen.
    # Instead, fade in over a thin density band around the level and let the
    # caller composite; the edge then anti-aliases like any other image.
    peak = vol.max(axis=0)
    alpha = np.clip((peak - level) / (0.30 * level), 0.0, 1.0) * hit

    if paint is None:
        return shade, alpha, None

    # Sample the painted field INSIDE the surface, not on it.  At the half
    # density level the proton fraction is a ratio of two small numbers and is
    # correspondingly noisy; stepping a few voxels in reaches bulk density where
    # it is a meaningful local composition rather than surface noise.
    deep = np.clip(first + inset, 0, vol.shape[0] - 1)
    painted = np.take_along_axis(paint, deep[None], 0)[0]
    return shade, alpha, painted


def _camera_np(rn, rp, rot_z, tilt, upsample, smooth, fill):
    """
    Camera transform for a neutron/proton pair, returning (total, fraction).

    The ratio is formed AFTER resampling, never before.  The proton fraction has
    a step at the vacuum edge -- inside it is ~0.4, outside it is 0/0 and has to
    be filled with something -- and cubic interpolation of a step rings, which
    paints bands of spurious composition across the nuclear interior.  The
    separate densities are smooth and interpolate cleanly, so resampling those
    and dividing afterwards has no such edge to ring around.
    """
    a = _camera(rn, rot_z, tilt, upsample, smooth)
    b = _camera(rp, rot_z, tilt, upsample, smooth)
    tot = a + b
    frac = np.where(tot > 1e-3, b / np.maximum(tot, 1e-3), fill)
    return tot, frac


def _camera(vol, rot_z, tilt, upsample, smooth):
    """
    Resample the volume for one camera and return it oriented for casting.

    Rotating the VOLUME and always casting along a fixed axis is far less code
    than rotating rays, and gives the same picture.  Upsampling first is not
    cosmetic: at 1 fm spacing a 16O is about six voxels across, so the raw
    isosurface is genuinely a staircase, and no amount of shading hides it.

    Returns an array whose axis 0 is the view direction, axis 1 is x (vertical
    on screen) and axis 2 is z (horizontal, the beam axis).  Casting along
    axis 0 of the raw (nx, ny, nz) array would look down x and put the beam
    vertical, which is what the first version of this did.
    """
    from scipy.ndimage import rotate as ndrotate, zoom, gaussian_filter
    v = vol
    if rot_z:
        v = ndrotate(v, rot_z, axes=(0, 1), reshape=False, order=1)   # about z
    if tilt:
        v = ndrotate(v, tilt, axes=(1, 2), reshape=False, order=1)    # about x
    if upsample > 1:
        # order=3, not 1.  Linear upsampling of a nucleus only ~6 voxels across
        # produces piecewise-flat facets, and the resulting staircase along the
        # silhouette survives any amount of shading and smoothing.
        v = zoom(v, upsample, order=3)
    if smooth:
        v = gaussian_filter(v, smooth)
    return np.moveaxis(v, 1, 0)          # look down y -> (ny, nx, nz)


def isosurface(d, out, level=0.08, fps=15, upsample=3, smooth=1.0,
               plain_shading=False, shell_mode=True):
    if not d['has3d']:
        raise SystemExit("--iso needs the 3D density; this npz predates rho3d.  "
                         "Re-run tdhf_collide.py to get it.")
    vol, t = d['rho3d'], d['times']
    nt = vol.shape[0]
    peak = float(vol.max())
    if level >= peak:
        raise SystemExit(f"--iso-level {level} is above the peak density "
                         f"{peak:.4f} fm^-3; nothing would be drawn.")

    # (label, rotation about z, tilt about x).  A 90 deg turn about z carries y
    # into x, so the second camera looks down what was originally x.
    cams = [('down y', 0.0, 0.0), ('down x', 90.0, 0.0),
            ('oblique', 35.0, 24.0)]

    # One crop for the whole run, taken from the union of all occupancy so the
    # object does not jump around between frames.
    # ever is (nx, ny, nz); reduce over the OTHER two axes for each extent.
    # `ever.any(axis=0)` is still 2-D, and np.where on it returns a tuple whose
    # first element is y, not z -- which silently produced a window in the wrong
    # place rather than an error.
    ever = vol.max(axis=0) >= level
    xs = np.where(ever.any(axis=(1, 2)))[0]
    zs = np.where(ever.any(axis=(0, 1)))[0]
    pad = 4
    x0, x1 = max(xs[0] - pad, 0), min(xs[-1] + pad, vol.shape[1] - 1)
    z0, z1 = max(zs[0] - pad, 0), min(zs[-1] + pad, vol.shape[3] - 1)

    bg_rgb = np.array(matplotlib.colors.to_rgb(BG))
    plain = matplotlib.colormaps['copper']
    # Diverging, centred on the fraction the composite would reach if the two
    # fragments fully equilibrated -- so blue/red reads directly as "still
    # neutron-rich / still proton-rich relative to equilibrium".
    iso_cmap = matplotlib.colormaps['coolwarm']

    paint_on = d['hasnp'] and not plain_shading
    if paint_on:
        pn, pp = d['rho3d_n'], d['rho3d_p']
        z_tot = float(pp[0].sum())
        a_tot = float(pn[0].sum()) + z_tot
        f_eq = z_tot / a_tot
        span = 0.10
        fnorm = matplotlib.colors.Normalize(f_eq - span, f_eq + span)
        # Contours every 2% in proton fraction, straddling equilibrium.
        flevels = [f_eq + s for s in (-0.08, -0.06, -0.04, -0.02, 0.0,
                                      0.02, 0.04, 0.06, 0.08)]
    else:
        f_eq = None

    sl = (slice(x0 * upsample, (x1 + 1) * upsample),
          slice(z0 * upsample, (z1 + 1) * upsample))

    # Nested shells, outermost (lowest density) first, composited front-to-back.
    # A ray entering the nucleus crosses them in exactly this order, so the
    # standard over-operator applies with no sorting: each shell contributes
    # (1-A) of what is left of the ray.  This is the whole reason to prefer
    # translucent regions to contour lines here -- the diffuse surfaces of the
    # two fragments touch and bridge well before the dense cores do, and an
    # opaque single surface hides that completely.
    # Outer shells must be genuinely transparent, not merely translucent.  Under
    # over-compositing each layer dims everything behind it by (1-a), so three
    # outer shells at ~0.3 leave the dense core contributing under a fifth of the
    # final pixel and it never reads as a bright core however hot its colour is.
    shells = [(0.02, 0.10), (0.06, 0.18), (0.10, 0.30), (0.145, 0.97)]
    shell_cmap = matplotlib.colormaps['inferno']
    # Sample the colormap well up its range.  Starting near the bottom gives
    # every shell a near-black base colour, and once each is further multiplied
    # by its opacity and by the Lambertian term the composite comes out uniformly
    # dim -- the layers are all still there, but nothing is bright enough to read.
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
        if shell_mode:
            return frame_shells(i, rot_z, tilt)
        if paint_on:
            cv, cf = _camera_np(pn[i], pp[i], rot_z, tilt, upsample, smooth,
                                f_eq)
            # Sample ~1 fm inside the half-density surface: far enough in to
            # escape the outermost shell where the ratio of two small densities
            # is noisy, shallow enough that this is still the composition OF THE
            # SURFACE.  Going deeper (the first attempt used 2 fm) mixes in the
            # radial composition gradient -- real physics, the neutron skin, but
            # it makes the colour depend on how thick the nucleus is along the
            # ray rather than on what the surface is made of.
            sh, al, painted = raycast(cv, level, paint=cf, inset=upsample)
            sh, al, painted = sh[sl], al[sl][..., None], painted[sl]
            base = iso_cmap(fnorm(painted))[..., :3]
            # Multiply the composition colour by the Lambertian shade so the
            # picture still reads as a lit solid rather than a flat map.
            img = base * (0.25 + 0.92 * sh[..., None])
            # Contours only where the surface is solidly in view.  At the
            # silhouette the ray grazes the surface and the sampled fraction is
            # unreliable, so lines drawn there trace the aliasing, not the
            # composition.
            cm = contour_mask(painted, flevels) & (al[..., 0] > 0.75)
            img[cm] *= 0.45
            return np.clip(img, 0, 1) * al + bg_rgb * (1.0 - al)
        sh, al, _ = raycast(_camera(vol[i], rot_z, tilt, upsample, smooth),
                            level)
        sh, al = sh[sl], al[sl][..., None]
        return plain(sh)[..., :3] * al + bg_rgb * (1.0 - al)

    # Hero layout: the oblique camera large on the left, the two axis-aligned
    # views stacked small on the right.  Three equal panels spend two thirds of
    # the frame on views that are identical for a head-on collision and nearly
    # so otherwise; the oblique one is the only camera that reads the shape as a
    # solid, so it gets the room.
    fig = plt.figure(figsize=(14.5, 5.6))
    fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(2, 2, width_ratios=[2.05, 1.0], hspace=0.16,
                          wspace=0.06, left=0.02, right=0.98, top=0.88,
                          bottom=0.09)
    order = [(cams[2], gs[:, 0]), (cams[0], gs[0, 1]), (cams[1], gs[1, 1])]
    ims = []
    for (name, rz, tl), cell in order:
        ax = fig.add_subplot(cell)
        ax.set_facecolor(BG)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color('#3a3a44')
        im = ax.imshow(frame(0, rz, tl), origin='lower',
                       interpolation='bilinear', aspect='equal')
        ax.set_title(name, color=FG, fontsize=10, family='monospace', pad=4)
        ims.append((im, rz, tl))
    sup = fig.suptitle('', color=FG, fontsize=12, family='monospace')

    if shell_mode:
        fig.text(0.5, 0.022,
                 'nested translucent shells at ρ = '
                 + ', '.join(f'{lv:g}' for lv, _ in shells)
                 + ' fm⁻³  (outer diffuse → inner dense core)',
                 color=DIM, fontsize=8, family='monospace', ha='center')
    elif paint_on:
        fig.text(0.5, 0.022,
                 f'surface ρ = {level:g} fm⁻³, coloured by proton '
                 f'fraction Z/A  (blue neutron-rich → red proton-rich, '
                 f'equilibrium {f_eq:.3f}); contours every 0.02',
                 color=DIM, fontsize=8, family='monospace', ha='center')

    def update(i):
        for im, rz, tl in ims:
            im.set_data(frame(i, rz, tl))
        sup.set_text(f"{_title(d)}    "
                     f"$\\rho$ = {level:g} fm$^{{-3}}$    t = {t[i]:6.1f} fm/c")
        return [im for im, _, _ in ims] + [sup]

    FuncAnimation(fig, update, frames=nt, interval=1000 // fps, blit=False).save(
        out, writer=PillowWriter(fps=fps), dpi=90,
        savefig_kwargs={'facecolor': BG})
    plt.close(fig)
    print(f"  wrote {out}  ({nt} frames)")


def main():
    p = argparse.ArgumentParser(
        description="Alternative views of a TDHF collision npz",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument('npz')
    p.add_argument('--all', action='store_true')
    p.add_argument('--spacetime', action='store_true')
    p.add_argument('--threeview', action='store_true')
    p.add_argument('--iso', action='store_true')
    p.add_argument('--isospin', action='store_true')
    p.add_argument('--iso-level', type=float, default=0.08,
                   help='isosurface density, fm^-3 (default: half saturation)')
    p.add_argument('--iso-upsample', type=int, default=3,
                   help='resample factor before raycasting; raise for a '
                        'smoother silhouette, cost grows as the cube')
    p.add_argument('--iso-mode', choices=['shells', 'frac', 'plain'],
                   default='shells',
                   help="shells (default): nested translucent density shells, "
                        "so the diffuse surfaces are visible touching before "
                        "the cores merge.  frac: one surface coloured by proton "
                        "fraction (needs rho3d_n/p; only says anything when "
                        "N != Z).  plain: one lit surface.")
    p.add_argument('--flat-slices', action='store_true',
                   help='three-plane view with a continuous colour scale '
                        'instead of filled density bands')
    p.add_argument('--fps', type=int, default=15)
    args = p.parse_args()

    if not (args.all or args.spacetime or args.threeview or args.iso
            or args.isospin):
        args.all = True

    d = load(args.npz)
    stem = str(Path(args.npz).with_suffix(''))
    print(f"  {args.npz}: {len(d['times'])} frames, "
          f"3D density {'present' if d['has3d'] else 'ABSENT (slice only)'}")

    if args.all or args.spacetime:
        spacetime(d, stem + '_spacetime.png')
    if args.all or args.threeview:
        threeview(d, stem + '_3view.gif', args.fps, bands=not args.flat_slices)
    if args.isospin or (args.all and d['hasnp']):
        isospin(d, stem + '_isospin.gif', args.fps)
    if args.all or args.iso:
        if args.iso_mode == 'frac' and not d['hasnp']:
            raise SystemExit("--iso-mode frac needs rho3d_n / rho3d_p; this "
                             "npz predates the neutron/proton split.  Re-run "
                             "tdhf_collide.py to get them.")
        isosurface(d, stem + '_iso.gif', args.iso_level, args.fps,
                   upsample=args.iso_upsample,
                   plain_shading=(args.iso_mode == 'plain'),
                   shell_mode=(args.iso_mode == 'shells'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
