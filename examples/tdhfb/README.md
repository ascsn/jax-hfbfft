# TDHFB examples

Real-time TDHFB built on `jax_hfbfft.TDHFB` and `jax_hfbfft.physics.tdhfb`
(the global in-span matrix scheme; the module docstring describes the step).

| Script | What it does |
|---|---|
| `static.py` | Paired static ground state on an odd grid, all state weights 1, saved as `.npz` |
| `relax.py` | Relaxes a static into the minimum of the TDHFB functional (needed before propagating) |
| `run.py` | One nucleus at rest or after a quadrupole kick; energy, N, Z, pairing, Q20, local continuity |
| `collide.py` | Head-on collision of two paired nuclei at a given energy and relative pairing phase |
| `fission.py` | Release from a relaxed constrained state; tracks neck, fragments, Q20/Q30; stops after scission |

`--scheme ebata` in `run.py` and `collide.py` runs Ebata's canonical-basis
Cb-TDHFB (PRC 82, 034306) from the same initial state, for comparison.

## Workflow

```bash
python static.py --nucleus O20 --nx 13 --dx 1.2 --v0 300 --iters 3000 --uv-floor 0
python relax.py --static out/O20_13x13x13_dx1.2_v300.npz --iters 2000
python run.py --static out/O20_13x13x13_dx1.2_v300_relaxed.npz --kick 0.005 --T 60 --check-at 0 20 60

python static.py --nucleus Sn120 --nx 33 --dx 0.8 --v0 362 --iters 20000 --uv-floor 0
python relax.py --static out/Sn120_33x33x33_dx0.8_v362.npz --iters 3000
python collide.py --frag out/Sn120_33x33x33_dx0.8_v362_relaxed.npz --nz 69 --sep 16 --e-asym 277 \
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
- **Energy**: ~9e-3 MeV over 60 fm/c for kicked 20O (60 orbitals, 13³) at dt = 0.05 with the first-order step;
  Sn+Sn near the barrier drifts ~0.06 MeV over 240 fm/c.

## The recipe, and why

| Setting | Why |
|---|---|
| Odd grid sizes | No Nyquist mode, so the kinetic operator is the exact gradient of the energy. On even grids TDHFB loses energy at grid-scale momenta. |
| `state_cutoff = 0` (all weights 1) | A pair with soft weight 0 < w < 1 is never a stationary TDHFB state. `from_static` refuses such statics. |
| `uv_floor = 0` (default) | Rebuilds κ = √(v²(1−v²)) instead of the static solver's floored value, which gave nearly empty orbitals κ/ρ ~ 10⁶. |
| Relax the static (`relax.py`) | The canonical solver's result is not the TDHFB functional's minimum (diagonal gaps only, nearly empty orbitals unconverged); TDHFB at rest then drifts toward it. 20O at rest: E_pair +0.02% over 60 fm/c relaxed, +5.9% unrelaxed. Fit v0 on relaxed states (pairing grows a few % at fixed v0). |
| Purify after setup (default) | Removes the static's ~1e-6 impurity, which the out-of-span force amplifies in nearly empty orbitals. |
| Second-order step (`strang`, `midpoint`, n_comp 1; the defaults), eps 1e-7, pred_n_comp 1 | Energy error falls 4.0x per halving of dt. dt 0.1 in general; dt 0.2 on 1.0 fm grids is validated for U238 fission (10 fm/c against dt 0.025: density 6e-7, E_pair 2 keV, 1.4x faster than the old first-order dt 0.1, which was off by 35 keV). After the pairing collapses in a collision it revives without bound while N drifts (~1e-3), at dt 0.1 and 0.05 alike: results past a collapse are not valid (open problem). The 20O, Sn120 and LISE results used the first-order step (`--no-strang --no-midpoint`, dt 0.05, n_comp 4). |

## Pairing in heavy nuclei: regularise it

With bare zero-range pairing the orbital space is the pairing cutoff, and once
the orbitals are free (in `relax`, and in the dynamics) the gap follows the
size of that space: 20O on 13^3 @ 1.2 fm, v0 = 300, mean neutron gap 1.16 /
1.38 / 1.63 MeV with the basis reaching 9 / 34 / 40 MeV above lambda; U238 on
37x37x61 @ 1 fm goes from 0.2 MeV at v0 = 280 to 7.6 MeV at v0 = 320.

`relax.py --pair-reg auto` switches to the regularised coupling of Bulgac and
Yu (Force.pair_reg_ecut, physics/edf.regularized_coupling): v_eff = v /
(1 + v K) with K from the local Fermi momentum (3 pi^2 rho_q)^(1/3) and the
cutoff E_c (auto: the top of the static's neutron basis above lambda_n). The
coupling is a local function of the densities, so (i) the pair field stays
local and real and adds no source to the local continuity equation, and (ii)
it is part of the energy functional, so the mean field carries its
rearrangement and the energy is conserved. 20O, n = 30, quadrupole kick, 20
fm/c: continuity defect 4.5e-7 / 3.1e-7 against floors 4.1e-7 / 2.3e-7,
energy drift 7e-8 MeV; mean gap 1.23 / 1.08 / 1.18 MeV for the three basis
sizes above (bare: +40%). Regularised pairing is weaker at the same v0:
refit v0 on relaxed states.

`relax.py --pair-kcut` (a momentum filter on the pair channel) is the older
alternative. It also tames the gap, but the filtered pair field is nonlocal
and breaks local continuity by ~1e4 times its floor (20O as above: 4.3e-3 /
8.8e-3); N and E stay conserved. U238 at k_c = 2.0: v0 = 360 (n) / 415 (p)
gives mean gaps 0.56 / 0.66 MeV (experiment, 3-point: 0.54 / 0.67).

## Known limits

- Statics from `static.py` alone are not stationary under TDHFB; always relax.
  Check `stationarity` (printed by `relax.py`): the out-of-span velocity of
  orbitals with occupation above 1e-5 should be well below 1 MeV.
- Collisions beyond ~300 fm/c, after the pairing collapses, have not been
  validated with relaxed statics; check against dt/2.
- The Ebata arm advances its amplitudes by explicit Euler (as in all
  comparison runs) and uses fixed rest-frame Kramers partners. Partners must
  not be re-found by time reversal in a moving frame: time reversal flips the
  momentum, so the re-found partners are wrong for a moving nucleus.

## Frames and the 3D viewer

`run.py`, `collide.py` and `fission.py` record animation frames through
`frames.py`: y = 0 slices and canonical levels at every record, and with
`--frames3d-every T` the full 3D density and pair field every T fm/c. The
canonical levels are the occupations and single-particle energies of the
eigen-orbitals of ρ̃ (`--no-levels` to skip). `viz/build_3d.py` turns any set
of runs into one self-contained raymarched viewer page: a density core and
cloud with the pair density as a phase-coloured glow, plus the canonical
levels. See `viz/README.md`.
