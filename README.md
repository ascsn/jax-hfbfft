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
- **Flexible Configuration**: Easily configure nuclei, forces, and grid parameters using YAML files.
- **Extensible Architecture**: Designed to be modular, allowing for easy integration of new functionals or physical observables.

## Project Structure

```
jax-hfbfft/
├── src/jax_hfbfft/       # Modern OOP implementation (use this)
│   ├── core/             # Core classes (HFBFFT, Force, Nucleus, Grid)
│   ├── physics/          # Physics modules (densities, meanfield, pairing, etc.)
│   ├── forces/           # Force parameter definitions
│   ├── utils/            # Utility functions
│   ├── jax_config.py     # JAX configuration (64-bit by default)
│   └── __init__.py       # Package exports
├── legacy/               # Archived legacy implementation
├── tests/                # Test suite (58 tests)
├── examples/             # Example scripts
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

## Installation

Ensure you have a modern Python environment. You can install the required dependencies via pip:

```bash
pip install -r requirements.txt
```

Common dependencies include:
- `jax`
- `jaxlib`
- `PyYAML`

## Quick Start

**👉 See the [Quickstart Guide](docs/Quickstart.md) for detailed installation and usage instructions.**

### Installation

```bash
# Install from source with GUI
git clone https://github.com/your-org/jax-hfbfft.git
cd jax-hfbfft
pip install -e ".[gui]"

# With GPU support (CUDA 12)
pip install -e ".[gui,cuda]"

# Apple Silicon (Metal)
pip install -e ".[gui,metal]"
```

**Note:** The frontend automatically builds during installation. Set `SKIP_FRONTEND_BUILD=1` if you only need the Python API.

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

See [docs/CLI_WORKFLOW.md](docs/CLI_WORKFLOW.md) for complete CLI documentation.

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
- **Arbitrary multipole constraints** (Q20, Q30, Q40, etc.)

See [RUN_INSTRUCTIONS.md](RUN_INSTRUCTIONS.md) for complete documentation and example configurations.

**Example config.yml:**
```yaml
nucleus:
  protons: 50
  neutrons: 82
  name: "Sn132"

force:
  name: "SLy4"
  ipair: 6  # DDDI pairing

grid:
  nx: 32
  ny: 32
  nz: 32

iteration:
  max_iterations: 500
  convergence_threshold: 1.0e-6

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

The code supports **arbitrary multipole moment constraints** Q_λμ for exploring deformation and potential energy surfaces. Constraints can be specified using either:

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
# Prolate deformation (football shape)
constraint = Constraint.from_beta_gamma(
    mass_number=238,
    beta2=0.25,    # Deformation parameter
    gamma=0        # 0° = prolate, 60° = oblate
)

# Octupole deformation (pear shape, e.g., Ra-224)
constraint = Constraint.from_beta_gamma(
    mass_number=224,
    beta2=0.10,
    beta3=0.08,    # Octupole
    gamma=0
)

calc = HFBFFT(nucleus, force, grid, constraint=constraint)
```

Config file format:
```yaml
constraints:
  beta_gamma:
    beta2: 0.25
    gamma: 0     # Prolate
    beta3: 0.05  # Optional octupole
```

Available multipole names: Q00, Q10, Q20, Q21, Q22, Q30, Q40, Q50, Q60, etc., or specify directly as (λ, μ) tuples for arbitrary multipoles.

See [docs/multipole_constraints.md](docs/multipole_constraints.md), `examples/constrained_multipoles.py`, and `examples/beta_gamma_constraints.py` for details.

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
- Time-dependent HFB (TDHFB) for nuclear dynamics.
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

See [docs/GUI_IMPLEMENTATION_PLAN.md](docs/GUI_IMPLEMENTATION_PLAN.md) for architecture details.


