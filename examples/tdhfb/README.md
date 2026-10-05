# TDHFB examples

Real-time TDHFB built on `jax_hfbfft.TDHFB` and `jax_hfbfft.physics.tdhfb`
(the global in-span matrix scheme; the module docstring describes the step).

| Script | What it does |
|---|---|
| `static.py` | Paired static ground state on an odd grid, all state weights 1, saved as `.npz` |
| `run.py` | One nucleus at rest or after a quadrupole kick; energy, N, Z, pairing, Q20, local continuity |
| `collide.py` | Head-on collision of two paired nuclei at a given energy and relative pairing phase |

`--scheme ebata` in `run.py` and `collide.py` runs Ebata's canonical-basis
Cb-TDHFB (PRC 82, 034306) from the same initial state, for comparison.

## Workflow

```bash
python static.py --nucleus O20 --nx 13 --dx 1.2 --v0 300 --iters 3000
python run.py --static out/O20_13x13x13_dx1.2_v300.npz --kick 0.005 --T 60 --check-at 0 20 60

python static.py --nucleus Sn120 --nx 33 --dx 0.8 --v0 362 --iters 20000
python collide.py --frag out/Sn120_33x33x33_dx0.8_v362.npz --nz 69 --sep 16 --e-asym 277 \
    --phase 3.1416 --T 240 --ckpt ck.npz --resume --out out/SnSn_pi.npz
```

`static.py` takes minutes on a GPU for 20O and hours for Sn120. A Sn+Sn
step (480 orbitals, 33x33x69) takes about 2.6 s on an A100; U238 needs an H200.

Minimal use of the API:

```python
from jax_hfbfft import TDHFB

td = TDHFB.from_static("out/O20_13x13x13_dx1.2_v300.npz")   # or an HFBFFT after run()
td.kick_quadrupole(0.005)
for rec in td.iterate(1200, record_every=20):
    print(rec['time'], rec['E'] - td.history[0]['E'], rec['N'], rec['Q20'])
print(td.continuity())          # local continuity defect vs its floor
```

## What to check

- **N and Z** drift by ~5e-9 over 60 fm/c, and the impurity of the generalized
  density stays below ~1e-12 (20O, 13³).
- **Local continuity** (`--check-at`, `TDHFB.continuity()`) should sit at its
  floor, the same measure for the pairing-free motion of the same orbitals.
  At rest dρ/dt ≈ 0 and the fraction means nothing.
- **Energy**: ~9e-3 MeV over 60 fm/c for kicked 20O (60 orbitals, 13³) at dt = 0.05;
  Sn+Sn near the barrier drifts ~0.06 MeV over 240 fm/c.

## The recipe, and why

| Setting | Why |
|---|---|
| Odd grid sizes | No Nyquist mode, so the kinetic operator is the exact gradient of the energy. On even grids TDHFB loses energy at grid-scale momenta. |
| `state_cutoff = 0` (all weights 1) | A pair with soft weight 0 < w < 1 is never a stationary TDHFB state. `from_static` refuses such statics. |
| `uv_floor = 0` (default) | Rebuilds κ = √(v²(1−v²)) instead of the static solver's floored value, which gave nearly empty orbitals κ/ρ ~ 10⁶. |
| Purify after setup (default) | Removes the static's ~1e-6 impurity, which the out-of-span force amplifies in nearly empty orbitals. |
| dt 0.05, eps 1e-7, n_comp 4, pred_n_comp 1 | The validated settings. Check dt/2 when pairing collapses in a collision (fast regrowth near 300 fm/c). |

## Known limits

- Orbitals with occupation below ~1e-5 have unconverged tails that the 1/ρ in
  the out-of-span force amplifies: U238 at rest pumps neutron pairing.
  `--drop-below 1e-4` removes them (about 9x less pumping). Treat results
  beyond ~300 fm/c in collisions with care.
- The Ebata arm advances its amplitudes by explicit Euler (as in all
  comparison runs) and uses fixed rest-frame Kramers partners. Re-finding
  partners by time reversal in a moving frame is wrong; that artifact was
  the retracted "Galilean failure".
