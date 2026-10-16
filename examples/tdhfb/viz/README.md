# TDHFB visualization

Self-contained HTML viewers (one file, no server) for the runs of `run.py`,
`collide.py` and `fission.py`. They publish as Claude artifacts or open locally
in any browser with WebGL 2.

## 1. Record frames during the run

All three drivers record through `frames.py` (`FrameRecorder`):

| Output | When | Contents |
|---|---|---|
| `OUT_frames.npz` | every record (`--every`) | y = 0 slices of ρ_n, ρ_p (float32), the complex pair density ξ_q and pair field Δ_q (complex64) |
| `OUT_levels.npz` | every record (`--levels`, default on) | canonical levels: occupation v² and ⟨h⟩ of each eigen-orbital of ρ̃, with isospin |
| `OUT_3d/t<time>.npz` | every `--frames3d-every` fm/c | full 3D ρ_q (float32) and Δ_q (complex64), plus v0, v0p, rho0pr |

Sizes for U238 on a 1 fm grid: about 3–4 MB per 3D frame, so a 1,000 fm/c run
with 3D frames every 2 fm/c is 1.5–2 GB. Write it to scratch. Everything is
written atomically and survives `--resume`.

```
python run.py     --static U238_relaxed.npz --max-defect 1e-2 --kick 0 --T 300 --dt 0.2 --every 2 \
                  --frames3d-every 2 --out out/rest.npz
python collide.py --frag U238_relaxed.npz --frag2 Ca48_relaxed.npz --nz 121 --sep 20 --e-asym 220 \
                  --dt 0.2 --every 2 --frames3d-every 2 --out out/ca48_u238.npz
python fission.py --static U238_Q30000.npz --nz 121 --frames3d-every 2 --out out/fis.npz
```

## 2. Build the 3D viewer

```
python viz/build_3d.py OUT.html --run rest "U238 at rest" out/rest \
                                --run kick "Quadrupole kick" out/kick --title "U238 in 3D"
```

Each `--run ID LABEL STEM` becomes a tab. The page raymarches each frame's 3D
density and pair density on the GPU (WebGL 2):

- **Pairing glow** (default): an opaque core density surface, a translucent
  cloud surface, and the pair density ξ as light. Brightness goes as |ξ|^γ
  ("glow contrast"), above a threshold ("glow above"), coloured by the phase of
  ξ (rainbow or twilight). ξ is where the orbitals are fractionally occupied,
  so the glow marks the pairing skin.
- **Coloured surfaces** and **core only**: the same surfaces painted with ξ.
- Readout and charts adapt to the run: Q20, neck and fragments (fission),
  separation (collisions), pairing energies, energy error, and the canonical
  levels (⟨h⟩ against v²) when `OUT_levels.npz` exists.
- Controls: orbit (drag), zoom (wheel or pinch), playback 0.25–240 frames/s,
  rotate, density thresholds, cloud opacity, glow strength, contrast and
  threshold, neutron or proton pairing.

Data are cropped to the region the run touches, quantized to 8 bits per
channel, gzipped and base64-embedded. The browser inflates them with
`DecompressionStream`. Budget: about 9–15 KB per frame for U238, so 400 frames
per run (the `--max-frames` default thins longer runs evenly) is about 5 MB. An
artifact page must stay under 16 MB.

## 3. The 2D fission-study viewer

`build_fission_2d.py` builds the y = 0 slice viewer of the 2026-10-06 U238
study: the static path, the releases with the old-step runs overlaid, and the
pairing energy against Q20 from the ground-state side to scission. Its input
paths are those of that study (`/mnt/scratch/knigh137/fission`). Run it there.
