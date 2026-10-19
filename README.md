# HFBFFT: Coordinate Space Nuclear DFT with JAX

`HFBFFT` (Hartree-Fock-Bogoliubov with Fast Fourier Transforms) is a nuclear density functional theory (DFT) code written in Python with [JAX](https://github.com/google/jax). It solves the Skyrme HFB equations on a 3D Cartesian grid in coordinate space, with pairing treated in the natural-orbital (canonical) basis, and propagates the solutions in time with TDHF and TDHFB.

## Key Features

- **3D coordinate space**: no spatial symmetry is imposed and the orbitals live on the grid, with no oscillator basis; derivatives and the kinetic energy are evaluated spectrally with FFTs.
- **HFB in the natural-orbital basis**: pairing correlations are carried by the occupations and pair amplitudes of the canonical orbitals.
- **JAX**: the solver is jit-compiled with XLA and runs on CPUs and NVIDIA GPUs (Apple Metal is experimental), in 64-bit by default.
- **Skyrme functionals**: SLy4/5/6, SkM, SkM*, SkP, SIII and UNEDF0/1/2, with J² (tensor) terms where fitted (see [Functionals](#functionals)), with the centre-of-mass treatment each was fitted with, up to the microscopic ⟨P²⟩/2mA.
- **Pairing**: volume and density-dependent delta pairing (VDI/DDDI), the Lipkin-Nogami particle-number correction, and finite temperature (thermal BCS).
- **Time-Dependent Hartree-Fock (TDHF)**: Real-time propagation of converged static states for collective modes, linear response and heavy-ion collisions (see `examples/tdhf`).
- **Time-Dependent HFB (TDHFB)**: Real-time propagation of paired states (global in-span scheme) for collective modes, collisions and fission (see [TDHFB](#time-dependent-hfb) and `examples/tdhfb`).
- **Checkpoints**: static runs save and restart from `.npz` files, which also seed TDHFB.
- **Constrained Calculations**: multipole (Q10 to Q40) and neck (QN) constraints, and deformation scans with an outer/inner augmented-Lagrangian solver.
- **Chains of nuclei**: isotopic chains with warm starts and two-nucleon separation energies (`run_chain`).
- **Configuration files**: a commented `config.yml` covers every option of the `hfbfft` CLI.

## Project Structure

```
jax-hfbfft/
├── src/jax_hfbfft/       # Modern OOP implementation (use this)
│   ├── core/             # Core classes (HFBFFT, TDHF, TDHFB, Force, Nucleus, Grid, Constraint)
│   ├── physics/          # Physics modules (densities, meanfield, pairing, dynamics, etc.)
│   ├── viz/              # Optional plotting (TDHF renders)
│   ├── forces/           # Force presets
│   ├── data/             # Skyrme parameter sets (_forces.yml)
│   ├── gui/              # Web GUI backend (FastAPI)
│   ├── utils/            # Utility functions
│   ├── cli.py            # `hfbfft` command-line interface
│   ├── jax_config.py     # JAX configuration (64-bit by default)
│   └── __init__.py       # Package exports
├── gui-frontend/         # Web GUI frontend (built during pip install)
├── gui-desktop/          # Electron desktop app
├── legacy/               # Archived legacy implementation
├── tests/                # Test suite (pytest)
├── examples/             # Example scripts (examples/tdhf, examples/tdhfb)
├── run_hfb.py            # Classic config-file driver
├── config.yml            # Fully commented configuration template
├── CHANGELOG.md          # Corrections and new features by release
└── pyproject.toml        # Package configuration
```

### Key Modules

- `src/jax_hfbfft/core/hfbfft.py`: Main HFBFFT class - entry point for calculations
- `src/jax_hfbfft/physics/solver.py`: HFB iteration solver
- `src/jax_hfbfft/physics/meanfield.py`: Skyrme mean-field potentials
- `src/jax_hfbfft/physics/densities.py`: Nuclear density computations
- `src/jax_hfbfft/physics/pairing.py`: BCS/HFB pairing and Lipkin-Nogami
- `src/jax_hfbfft/physics/cm_correction.py`: microscopic centre-of-mass correction
- `src/jax_hfbfft/physics/coulomb.py`: Coulomb potential (FFT-based)
- `src/jax_hfbfft/physics/energies.py`: Energy functional calculations
- `src/jax_hfbfft/physics/constraints.py`: Multipole constraints (augmented Lagrangian)
- `src/jax_hfbfft/physics/dynamics.py`: TDHF time propagation
- `src/jax_hfbfft/physics/collisions.py`: Initial states for heavy-ion collisions
- `src/jax_hfbfft/core/tdhf.py`: TDHF class - entry point for time-dependent runs
- `src/jax_hfbfft/physics/tdhfb.py`, `src/jax_hfbfft/core/tdhfb.py`: TDHFB (global in-span scheme) and its class

## Installation

Requires Python 3.9 or newer. The core dependencies are `jax`, `numpy`, `scipy` and `PyYAML`.

```bash
git clone https://github.com/ascsn/jax-hfbfft.git
cd jax-hfbfft

# Python API and CLI only
SKIP_FRONTEND_BUILD=1 pip install -e .

# With the GUI (needs Node.js/npm to build the frontend)
pip install -e ".[gui]"

# GPU support (CUDA 12); combine extras as needed, e.g. ".[gui,cuda]"
pip install -e ".[cuda]"

# Apple Silicon (Metal)
pip install -e ".[metal]"
```

Other extras: `viz` (matplotlib, for TDHF renders), `dev` (pytest and linters), `all`.

## Quick Start

### Command-Line Interface

```bash
# 1. Generate a sample configuration file
hfbfft init

# 2. Edit the configuration
nano config.yml

# 3. Run the calculation
hfbfft run

# Or just run if config.yml exists:
hfbfft
```

`hfbfft init` writes a fully commented `config.yml` template. Other commands: `hfbfft info --forces` lists the available Skyrme forces, `hfbfft info --devices` shows the JAX devices (CPU/GPU), and `hfbfft version`. Run `hfbfft --help` for everything.

### Python API

```python
from jax_hfbfft import HFBFFT, Nucleus, Force

# Define the nucleus
nucleus = Nucleus(protons=50, neutrons=82)  # Sn-132

# Choose the nuclear force
force = Force.from_name("SLy4")

# Create and run the calculation
calc = HFBFFT(nucleus=nucleus, force=force)
results = calc.run(max_iterations=1000)

print(f"Binding energy: {results.total_energy:.3f} MeV")
```

### Classic HFB Code Interface

For users familiar with traditional HFB codes, we provide a configuration file-based interface:

```bash
# 1. Edit configuration file
nano config.yml

# 2. Run calculation
python run_hfb.py

# 3. Results are saved to hfb_results/nucleus_name_timestamp/
```

This provides the classic HFB code workflow:
- Edit input file with all parameters
- Run executable
- Output written to a directory of result files (energies, radii, moments, pairing, spectra, densities)
- Easy benchmarking against other codes (HFBTHO, Sky3D, etc.)
- **Multipole constraints** (Q20, Q22, Q30, Q32, Q40 and the centre of mass, Q10)

`run_hfb.py` reads the same `config.yml` as `hfbfft run` but performs the static solve only: the `dynamics:` section (TDHF/TDHFB) and the `use_ln`, `precond_floor` and `uv_floor` settings are read by `hfbfft run`.

**Example config.yml:**
```yaml
nucleus:
  protons: 50
  neutrons: 82
  name: "Sn132"

force:
  name: "SLy4"
  ipair: 6          # DDDI pairing
  v0_neutron: 362   # Pairing strengths (required for ipair 5/6)
  v0_proton: 362

grid:
  nx: 32
  ny: 32
  nz: 32

iteration:
  max_iterations: 1000
  convergence_threshold: 1.0e-6   # stops early only if reached; otherwise runs all iterations

# Optional: Constrain multipole moments
constraints:
  multipoles:
    Q20: 10.0   # Quadrupole (fm²)
    Q30: 2.0    # Octupole (fm³)
```

Run multiple configurations:
```bash
python run_hfb.py config_O16.yml
python run_hfb.py config_Ca40.yml
python run_hfb.py config_Sn132.yml
```

### Multipole Constraints

The code supports **multipole moment constraints** Q_λμ for exploring deformation and potential energy surfaces. Constraints can be specified using either:

1. **Direct multipole moments** (Q₂₀, Q₃₀, etc. in fm^λ)
2. **Beta-gamma parameters** (β₂, γ - Hill-Wheeler parameterization)

#### Direct Multipoles

```python
from jax_hfbfft import Constraint

# Simple Q20 constraint
constraint = Constraint.from_multipoles({'Q20': 10.0})

# Multiple simultaneous constraints
constraint = Constraint.from_multipoles({
    'Q20': 10.0,   # Quadrupole
    'Q30': 2.0,    # Octupole (pear shape)
    'Q22': 1.0,    # Triaxial
    (4, 0): 1.0,   # Hexadecapole (using tuple notation)
})

calc = HFBFFT(nucleus, force, grid, constraint=constraint)
```

#### Beta-Gamma Parameters

The standard β-γ parameterization commonly used in publications:

```python
from jax_hfbfft import HFBFFT, Nucleus, Force, Grid, Constraint

# Prolate deformation of U-238 (football shape)
nucleus = Nucleus.from_symbol("U", 238)
grid = Grid.create(nx=48, ny=48, nz=48, dx=0.8, dy=0.8, dz=0.8)   # 38.4 fm box
constraint = Constraint.from_beta_gamma(
    mass_number=238,
    beta2=0.25,     # > 0 prolate, < 0 oblate (both about the z axis)
    gamma=0,        # triaxiality angle in degrees
    damprad=16.0,   # see the note below
)
calc = HFBFFT(nucleus, Force.from_name("SLy4"), grid, constraint=constraint)

# Octupole deformation (pear shape, e.g., Ra-224)
constraint = Constraint.from_beta_gamma(
    mass_number=224,
    beta2=0.10,
    beta3=0.08,     # Octupole
    gamma=0,
    damprad=16.0,
)
```

`from_beta_gamma` converts to Q20, Q22, Q30 and Q40 targets for the solver's operators (Q20 = 2z² − x² − y², Q22 = x² − y², ...). β₂ < 0 gives an oblate shape about the z axis; β₂ > 0 with γ = 60° is also oblate, but about the y axis. A single constrained solve only fixes the lab-frame moments, so a deformed nucleus can satisfy them by rotating; pass `principal_axes=True` to lock the orientation, or use `run_constrained_scan` (below), which does this automatically.

**Damping radius.** The multipole operators grow like r^λ, so they are damped beyond `damprad`. By default it is chosen from the nucleus and the requested deformation (about 2.2 R₀ along the long axis, e.g. 9 fm for spherical ⁴⁰Ca and 16 fm for ²³⁸U) and kept 3 fm inside the shortest box side; a warning is printed when the box is too small for the requested shape. The density must stay inside the mask for the moment to be controlled, so enlarge the grid rather than shrink `damprad`. Passing `damprad` overrides the default.

Config file format:
```yaml
constraints:
  beta_gamma:
    beta2: 0.25
    gamma: 0     # Prolate
    beta3: 0.05  # Optional octupole
  damprad: 16.0  # Damping radius (fm); optional, the default scales with A and beta2
```

Supported multipoles: Q10 (the centre of mass, `<z>`), Q20, Q22, Q30, Q32 and Q40, plus the optional principal-axes constraints.

**Collective inertia.** `physics.cranking_inertia(state, grid, [multipole_operator(grid, 'Q20')])` gives the ATDHFB perturbative-cranking mass tensor, the GCM metric and the zero-point energy of a constrained solution, as HFBTHO defines them (masses in ħ² MeV⁻¹ per unit of Q²; multiply by 10⁴ for b⁻² with Q20 in fm²). The unoccupied states, mostly box continuum, are summed implicitly by linear solves with h, so the result does not depend on the number of orbitals. For SLy4 ²⁴Mg at Q20 = 1 b it agrees with HFBTHO to 3% (mass 0.198 vs 0.2035 MeV⁻¹ b⁻²). `physics.moment_of_inertia(state, grid, 'x')` gives the Inglis-Belyaev rotational moment of inertia (ħ² MeV⁻¹) the same way, with the rigid-body value for comparison.

**Fission paths.** On a surface from a multi-dimensional `run_constrained_scan`, `physics.minimum_energy_path(E, start, end)` finds the path whose highest point is lowest (the saddle and the barrier), and `physics.least_action_path(E, coords, start, end, mass=...)` the WKB least-action path with the cranking mass tensor, its action and penetrability. Unconverged points (NaN) are walls.

**Particle-number projection.** `calc.project_particle_number()` (or `physics.project_particle_number(state, grid, force)`) projects a converged even-even solution on its neutron and proton numbers after variation, with the mixed-density prescription (HFBTHO's `PNP_is_on = 1`) on Fomenko gauge points. It returns the projected total energy, the correction, and the particle-number distributions |⟨N'|Φ⟩|². The density-dependent terms are not free of the self-interaction poles at v² = ½, so check the result against `n_gauge`.

**Form factors.** `physics.form_factors(state.densities, grid, q)` gives the orientation-averaged point, charge and weak form factors at momentum transfers `q` (fm⁻¹), folded with nucleon form factors consistent with the reported charge radius, and the weak radius and weak skin R_W − R_ch (as measured by PREX at q = 0.3978 fm⁻¹ in ²⁰⁸Pb and CREX at q = 0.8733 fm⁻¹ in ⁴⁸Ca).

From the CLI, the `analysis:` block of `config.yml` runs the projection and the form factors after the static solve (`analysis.res`), and a kicked dynamics run writes its strength function (`strength.dat`).

**Neck.** `QN` constrains the number of nucleons near the plane z = `neck_z` (default 0, the centre of mass when Q10 is fixed), Q_N = ∫ρ exp(−((z − neck_z)/neck_width)²) with `neck_width` = 1 fm as in HFBTHO; a small Q_N means a thin neck, near scission. Combine it with Q20 along a fission path, e.g. `Constraint.from_multipoles({'Q20': 3000.0, 'QN': 4.0})`, or drive it with `run_constrained_scan(..., multipole='QN')`. For asymmetric shapes set `neck_z` to the density minimum between the fragments. Multipoles can also be given as (λ, μ) tuples. See `examples/constrained_calculation.py`.

#### Deformation Scans

A single constrained solve updates the Lagrange multiplier every iteration, which is slow to reach the target and can stall. For accurate constrained points, or an energy curve, use `run_constrained_scan`: it converges each point with the multiplier frozen and updates the multiplier between solves by a secant step. For an axial multipole (Q20, Q30, Q40) it also constrains Q22 = 0 and the principal axes by default (`axial=True`); without them a deformed nucleus can meet a lab-frame Q20 target by tilting its symmetry axis instead of changing shape.

```python
from jax_hfbfft import Force, Grid
from jax_hfbfft.physics import run_constrained_scan
from jax_hfbfft.physics.solver import apply_cm_correction

grid = Grid.create(nx=24, ny=24, nz=24, dx=1.0, dy=1.0, dz=1.0)
force = apply_cm_correction(Force.from_name("SLy4", ipair=0), 16)
results, state = run_constrained_scan(grid, force, nucleus_z=8, nucleus_n=8, npsi_n=20,
                                      targets=[10.0, 15.0, 20.0, 25.0, 30.0])   # Q20 in fm^2
for r in results:
    print(r['target'], r['achieved'], r['E'])
```

Each target warm-starts from the previous one, so keep the steps small (here about β₂ = 0.05). Several multipoles can be driven at once for a deformation surface: pass `multipole=('Q20', 'Q30')` and targets as tuples, ordered so each point is next to the previous one, e.g. `[(q20, q30) for i, q20 in enumerate(q20s) for q30 in (q30s if i % 2 == 0 else q30s[::-1])]`. The scan prints each point as `[hit]` or `[short]` and ends with a check that dE/dQ between neighbouring points matches λ (ratio ≈ 1); treat `[short]` points, or a ratio far from 1, as unconverged.

## Functionals

| Name | Reference | c.m. correction (`zpe`) | Notes |
|---|---|---|---|
| SLy4 | Chabanat et al., NPA 635, 231 (1998) | 0 | |
| SLy5 | Chabanat et al. (1998) | 0 | With the J² terms of its central force (`j2: central`) |
| SLy6 | Chabanat et al. (1998) | 3 | Fitted with the full (two-body) c.m. correction |
| SkM | Krivine et al., NPA 336, 155 (1980) | 0 | |
| SkM* | Bartel et al., NPA 386, 79 (1982) | 0 | |
| SkP | Dobaczewski et al., NPA 422, 103 (1984) | 0 | With the J² terms of its central force (`j2: central`) |
| SIII | Beiner et al., NPA 238, 29 (1975) | 0 | |
| UNEDF0 | Kortelainen et al., PRC 82, 024313 (2010) | 0 | Parameters from HFBTHO's UNE0; fitted with Lipkin-Nogami |
| UNEDF1 | Kortelainen et al., PRC 85, 024304 (2012) | 2 | Parameters from HFBTHO's UNE1; fitted with Lipkin-Nogami |
| UNEDF2 | Kortelainen et al., PRC 89, 054314 (2014) | 2 | Parameters and tensor couplings from HFBTHO's UNE2; fitted with Lipkin-Nogami |

`tests/test_forces.py` checks every entry against its published parameters (and the UNEDF coupling constants against HFBTHO). Several entries were corrected in October 2026; see [CHANGELOG.md](CHANGELOG.md) before reusing results from earlier versions.

**J² (tensor) terms** (`j2:` in the force file) enter as Σ_t CJ_t (Σ_μν J_t,μν² − s_t·T_t): the full spin-current tensor, with CJ as in HFBTHO, and its time-odd partner s·T, which keeps the functional Galilean invariant. They act on statics, TDHF and TDHFB. Against HFBTHO with UNEDF2 the total energies agree to 1 keV for deformed ²⁴Mg and ²⁸Si and to 25 keV for ⁴⁰,⁴⁸Ca.

The centre-of-mass treatment is set per force by `zpe`:

| `zpe` | Correction |
|---|---|
| 0 | ħ²/2m scaled by (1 − 1/A) (one-body) |
| 1 | constant 17.3/A^0.2 MeV |
| 2 | none |
| 3 | microscopic ⟨P²⟩/2mA on the final state (the 17.3/A^0.2 estimate during the iterations, as in SkyAx) |

**Lipkin-Nogami.** Set `use_ln: true` under `iteration:` in `config.yml` (or `calc.use_ln = True`, or `SolverConfig(use_ln=True)`). The UNEDF functionals were fitted with it, and it costs about five times a plain HFB iteration. λ₂ and E_LN are printed with the results; E_LN is included in the total energy. Validated against HFBTHO+LN on the calcium chain.

**Temperature.** `calc.temperature = T` (MeV), or `physics: temperature:` in `config.yml`, occupies each quasiparticle thermally, f = 1/(1 + e^{E/T}), and the solver minimises the free energy F = E − TS. The results report the entropy and F. It needs the pairing solver (`ipair` 5/6); a zero pairing strength gives finite-temperature Hartree-Fock. For ²⁰O (SLy4, Δₙ = 1.48 MeV at T = 0) the neutron gap vanishes near T = 0.75 MeV, close to the BCS value 0.57 Δ(0).

**Charge radius** is r²_ch = r²_p + 0.8409² − 0.1155·N/Z + 0.0332 fm² (proton, neutron and Darwin-Foldy terms; no spin-orbit term).

## Chains of nuclei

`run_chain` solves a list of (Z, N) in order with one shared basis, starting each nucleus from its neighbour's orbitals, and `separation_energies` adds B, S2n and S2p:

```python
from jax_hfbfft import Force, Grid, run_chain, separation_energies

force = Force.from_name("SLy4").with_pairing(ipair=6, v0neut=362.0, v0prot=362.0)
points = run_chain([(20, n) for n in range(20, 32, 2)], force, Grid.create_cubic(n=32, d=0.8))
for row in separation_energies(points):
    print(row['N'], row['B'], row['S2n'])
```

`examples/isotopic_chain.py` does the same from the command line and writes a CSV.

## Checkpoints and restarts

```python
calc.run(max_iterations=3000, checkpoint_file="ca48.npz", checkpoint_interval=500)

calc2 = HFBFFT(nucleus, force, grid)                     # same grid, nucleus and basis
calc2.initialize_wavefunctions(method="restart", filename="ca48.npz")
calc2.run(max_iterations=2000)                           # continues from the checkpoint
```

`save_checkpoint` / `load_checkpoint` do the same by hand. The file is the static format TDHFB reads, so `TDHFB.from_static("ca48.npz")` also works.

## Time-Dependent Hartree-Fock

`TDHF` propagates a converged static state in real time. Pairing is not included, so the static calculation must use `ipair=0`.

```python
from jax_hfbfft import HFBFFT, Nucleus, Force, TDHF, TDConfig

calc = HFBFFT(nucleus=Nucleus(protons=8, neutrons=8),
              force=Force.from_name("SLy4", ipair=0), npsi=(20, 20))
calc.initialize_wavefunctions(method="harmonic_oscillator")
calc.run(max_iterations=300)

td = TDHF.from_static(calc, config=TDConfig(dt=0.2, diag_interval=10))
td.kick_quadrupole(eta=1e-3)
for rec in td.iterate(1500):          # time, energy drift, moments, ...
    print(rec['time'], rec['Q20'], rec['dE_rel'])
```

After `kick_monopole`, `kick_quadrupole` or `kick_isovector_dipole` and a run, `E, S = td.strength(gamma=1.0)` gives the strength function in fm^2l/MeV (from the recorded `r2`, `Q20` or `Dz_iv`; `minus=` a second run kicked with −η cancels the offset from nucleons the kick emits). `td.ewsr(kind)` gives the energy-weighted sum rule m1 that its first moment should reach (including the TRK enhancement for the dipole); `dynamics.strength_function` transforms any other signal. The relative energy drift `dE_rel` is the main accuracy check. `examples/tdhf` has heavy-ion collisions, collective modes and rendering (`pip install -e ".[viz]"`); the CLI runs TDHF after the static solve when `dynamics: enabled: true` is set in `config.yml`.

## Time-Dependent HFB

`TDHFB` propagates a paired static state. It uses the global in-span scheme: one block per isospin, with the generalized density and pairing tensor of the orbital space evolved as full matrices; particle number and the purity of the generalized density are conserved to round-off (²⁰O: N drifts ~5×10⁻⁹ over 60 fm/c, impurity below 10⁻¹²).

```python
from jax_hfbfft import TDHFB

td = TDHFB.from_static(calc)          # an HFBFFT after run(), or a static .npz
td.relax(2000)                        # move into the TDHFB functional's minimum first
td.kick_quadrupole(0.005)
for rec in td.iterate(1200, record_every=20):
    print(rec['time'], rec['E'], rec['N'], rec['Epair_n'], rec['Q20'])
print(td.continuity())                # local continuity defect vs its floor
```

From the CLI, `dynamics: method: tdhfb` (or `auto`, which picks TDHFB for a paired static) with the options in the `tdhfb:` block of `config.yml`.

The static has to follow a recipe, explained in `examples/tdhfb/README.md`: an **odd** grid (no Nyquist mode), `state_cutoff: [0, 0]` (`from_static` refuses soft state weights), `uv_floor: 0`, a well-converged solve (Kramers partners with equal occupations), and a relax before propagating. An unrelaxed static drifts at rest. A static solved with Lipkin-Nogami or `zpe: 3` contains terms the propagation does not, and `from_static` warns.

**Known limits.**
- *Local continuity* holds at its numerical floor with zero-range (contact) pairing. With the momentum-cutoff pair channel needed for heavy nuclei (`pair_kcut`), the pair source is nonlocal, and dρ/dt is no longer balanced by the local current (²³⁸U at k_c = 3 fm⁻¹: about 10⁴ times the floor), while N and E remain conserved. A toy model traces the defect to pairing each orbital with its time-reversed partner. Treat local currents and fragment fluxes from cutoff runs with care.
- Collisions beyond ~300 fm/c, after the pairing collapses, have not been validated; check them against dt/2.

## Object-Oriented API

Each calculation is an independent `HFBFFT` object holding its own grid, force, orbitals and results.

### Several Nuclei in One Script

```python
from jax_hfbfft import HFBFFT, Nucleus, Force

# Define multiple nuclei
nuclei = [
    Nucleus.from_symbol("Ca", 40),
    Nucleus.from_symbol("Ca", 48),
    Nucleus.from_symbol("Sn", 132),
]

force = Force.from_name("SLy4")

# Each instance holds its own state, so they do not interfere
calculations = [HFBFFT(nucleus=n, force=force) for n in nuclei]

# Solve them one after another
for calc in calculations:
    calc.run()
    print(f"{calc.nucleus}: E = {calc.results.total_energy:.3f} MeV")
```

### Precision Configuration

The package uses **64-bit precision by default** for numerical accuracy. This can be configured:

```python
from jax_hfbfft import set_precision, Precision

# Use 64-bit (default)
set_precision(Precision.FLOAT64)

# Switch to 32-bit for faster GPU calculations
set_precision(Precision.FLOAT32)

# Check current precision
from jax_hfbfft import check_precision
check_precision()  # Prints current settings
```

### Using Legacy Implementation

For compatibility or debugging, you can use the legacy implementation:

```python
from jax_hfbfft import HFBFFT, Nucleus, Force

calc = HFBFFT(
    nucleus=Nucleus(protons=20, neutrons=20),
    force=Force.from_name("SLy4")
)

# Use legacy implementation
results = calc.run(max_iterations=100, use_legacy=True)
```

See the [examples/](examples/) directory for more detailed usage examples.

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Development Status

`HFBFFT` is currently under active development. Planned features include:
- Odd-A nuclei (equal-filling blocking).
- Fayans functionals.

## Graphical User Interface

HFBFFT includes both web-based and desktop GUI interfaces for interactive calculations.

### Web GUI

Start the web interface with:

```bash
hfbfft gui
```

This opens a browser with:
- **Calculator**: Configure and run HFB calculations interactively
- **Results Viewer**: View energies, radii, deformation, and single-particle spectra
- **3D Visualization**: Interactive density distribution plots
- **Run Manager**: Browse and filter calculation history

Options:
```bash
hfbfft gui --port 8080        # Custom port
hfbfft gui --no-browser       # Don't auto-open browser
hfbfft gui --debug            # Enable debug mode
```

### Desktop App (Electron)

For a native desktop experience:

```bash
cd gui-desktop
npm install
npm start
```

The desktop app wraps the web GUI with:
- Native window with system menus
- Automatic backend management
- Cross-platform support (Linux, macOS, Windows)

### JIT Warmup

On first launch, the GUI pre-warms JAX's JIT compilation with a large configuration (256 states, 32³ grid). This takes ~60-90 seconds but ensures subsequent calculations start instantly.



