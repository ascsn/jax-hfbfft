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

### Installation

```bash
# Install in development mode
pip install -e .

# Or with GPU support
pip install -e ".[cuda]"
```

### Basic Calculation

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


