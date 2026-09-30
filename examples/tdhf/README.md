# TDHF examples

Real-time TDHF runs built on `jax_hfbfft.TDHF` and `jax_hfbfft.physics.collisions`.
Install the package with the visualization extra first:

```bash
pip install -e ".[viz]"
```

| Script | What it does |
|---|---|
| `collision.py` | Heavy-ion collision of two nuclei (default 16O + 16O at E_cm = 34 MeV) |
| `oscillations.py` | Collective modes of one nucleus: release from a Q20 constraint, giant dipole, quadrupole, monopole |
| `render_views.py` | Extra views of a saved collision: spacetime diagram, three planes, 3D isosurface, isospin transfer |

Each run writes an `.npz` under `out/` and an animated GIF; `--render <npz>`
re-renders without re-running. `--help` lists all options.

Minimal use of the API:

```python
from jax_hfbfft import HFBFFT, Nucleus, Force, TDHF, TDConfig

calc = HFBFFT(nucleus=Nucleus(protons=8, neutrons=8),
              force=Force.from_name("SLy4", ipair=0), npsi=(20, 20))
calc.initialize_wavefunctions(method="harmonic_oscillator")
calc.run(max_iterations=300)

td = TDHF.from_static(calc, config=TDConfig(dt=0.2, diag_interval=10))
td.kick_quadrupole(eta=1e-3)
for rec in td.iterate(1500):
    print(rec['time'], rec['Q20'], rec['dE_rel'])
```

The relative energy drift `dE_rel` is the main accuracy check; it should stay
at the 1e-6 level for the default time step. TDHF has no pairing, so the static
calculation must use `ipair=0`.
