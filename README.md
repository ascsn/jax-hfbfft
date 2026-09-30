# HFBFFT: Coordinate Space Nuclear DFT with JAX

`HFBFFT` (Hartree-Fock-Bogoliubov with Fast Fourier Transforms) is a high-performance nuclear density functional theory (DFT) code implemented in Python using [JAX](https://github.com/google/jax). It solves the HFB equations in coordinate space on a 3D Cartesian grid, utilizing a natural orbital representation for efficient treatment of pairing correlations.

## Key Features

- **HFB with Natural Orbitals**: Implements a robust HFB scheme using natural orbitals to handle pairing correlations in nuclear systems.
- **Coordinate Space Solver**: Performs calculations on a 3D Cartesian grid, avoiding basis set expansion errors and enabling flexible boundary conditions.
- **FFT-Accelerated Operators**: Uses Fast Fourier Transforms for highly accurate and efficient numerical derivatives and kinetic energy operations.
- **Powered by JAX**: 
  - **Performance**: Just-In-Time (JIT) compilation via XLA for near-native performance.
  - **Portability**: Runs seamlessly on CPUs, GPUs, and TPUs without code changes.
  - **Differentiability**: Ready for future applications involving automatic differentiation.
- **Time-Dependent Hartree-Fock (TDHF)**: Real-time propagation of converged static states for collective modes, linear response and heavy-ion collisions (see `examples/tdhf`).
- **Constrained Calculations**: Multipole constraints (Q10 to Q40) and deformation scans with an outer/inner augmented-Lagrangian solver.
- **Flexible Configuration**: Easily configure nuclei, forces, and grid parameters using YAML files.
- **Extensible Architecture**: Designed to be modular, allowing for easy integration of new functionals or physical observables.

## Project Structure

```
jax-hfbfft/
├── src/jax_hfbfft/       # Modern OOP implementation (use this)
│   ├── core/             # Core classes (HFBFFT, TDHF, Force, Nucleus, Grid, Constraint)
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
├── tests/                # Test suite (75 tests)
├── examples/             # Example scripts (TDHF examples in examples/tdhf)
├── run_hfb.py            # Classic config-file driver
├── config.yml            # Fully commented configuration template
└── pyproject.toml        # Package configuration
```

### Key Modules

- `src/jax_hfbfft/core/hfbfft.py`: Main HFBFFT class - entry point for calculations
- `src/jax_hfbfft/physics/solver.py`: HFB iteration solver
- `src/jax_hfbfft/physics/meanfield.py`: Skyrme mean-field potentials
- `src/jax_hfbfft/physics/densities.py`: Nuclear density computations
- `src/jax_hfbfft/physics/pairing.py`: BCS/HFB pairing
- `src/jax_hfbfft/physics/coulomb.py`: Coulomb potential (FFT-based)
- `src/jax_hfbfft/physics/energies.py`: Energy functional calculations
- `src/jax_hfbfft/physics/constraints.py`: Multipole constraints (augmented Lagrangian)
- `src/jax_hfbfft/physics/dynamics.py`: TDHF time propagation
- `src/jax_hfbfft/physics/collisions.py`: Initial states for heavy-ion collisions
- `src/jax_hfbfft/core/tdhf.py`: TDHF class - entry point for time-dependent runs

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

### Command-Line Interface (Recommended for beginners)

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
- Output written to directory with comprehensive result files
- Easy benchmarking against other codes (HFBTHO, Sky3D, etc.)
- **Multipole constraints** (Q20, Q22, Q30, Q32, Q40 and the centre of mass, Q10)
- **Optional TDHF run** after the static solve (`dynamics:` section)

The commented `config.yml` in the repository root documents every option.

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

**Damping radius.** The multipole operators grow like r^λ, so they are damped beyond `damprad` (default 6 fm). That default only suits light nuclei; for heavy or strongly deformed nuclei it cuts into the nucleus itself and the constraint pulls on the wrong shape. Set `damprad` well past the nuclear surface, up to about half the box length minus 3 fm (the box must extend beyond it). `run_constrained_scan` (below) chooses it automatically.

Config file format:
```yaml
constraints:
  beta_gamma:
    beta2: 0.25
    gamma: 0     # Prolate
    beta3: 0.05  # Optional octupole
  damprad: 16.0  # Damping radius (fm); the 6 fm default is for light nuclei
```

Supported multipoles: Q10 (the centre of mass, `<z>`), Q20, Q22, Q30, Q32 and Q40, plus the optional principal-axes constraints. Multipoles can also be given as (λ, μ) tuples. See `examples/constrained_calculation.py`.

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

Each target warm-starts from the previous one, so keep the steps small (here about β₂ = 0.05). The scan prints each point as `[hit]` or `[short]` and ends with a check that dE/dQ between neighbouring points matches λ (ratio ≈ 1); treat `[short]` points, or a ratio far from 1, as unconverged.

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

The relative energy drift `dE_rel` is the main accuracy check. `examples/tdhf` has heavy-ion collisions, collective modes and rendering (`pip install -e ".[viz]"`); the CLI runs TDHF after the static solve when `dynamics: enabled: true` is set in `config.yml`.

## Object-Oriented API

The package provides a modern object-oriented API with fully encapsulated, independent calculation instances.

### Multiple Parallel Calculations

```python
from jax_hfbfft import HFBFFT, Nucleus, Force

# Define multiple nuclei
nuclei = [
    Nucleus.from_symbol("Ca", 40),
    Nucleus.from_symbol("Ca", 48),
    Nucleus.from_symbol("Sn", 132),
]

force = Force.from_name("SLy4")

# Create independent calculations - each instance is fully isolated
calculations = [HFBFFT(nucleus=n, force=force) for n in nuclei]

# Run all calculations
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
- Support for a wider range of Skyrme functionals.
- Advanced constraint options for multi-dimensional potential energy surfaces.

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



