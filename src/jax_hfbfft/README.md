# JAX-HFBFFT

A JAX-based Hartree-Fock-Bogoliubov (HFB) solver with Fast Fourier Transform for nuclear structure calculations.

## Overview

JAX-HFBFFT provides a modern, object-oriented framework for performing nuclear mean-field calculations using the Hartree-Fock-Bogoliubov method with Skyrme-type effective interactions. The package is designed for:

- **Object-oriented design**: Encapsulated calculation objects that can run independently
- **Parallel calculations**: Multiple nuclei, forces, or constraints can be computed simultaneously
- **GPU acceleration**: Optional GPU support through JAX for significant speedups
- **Modular architecture**: Forces, constraints, and other components can be flexibly combined

## Installation

### Basic Installation (CPU)

```bash
pip install jax-hfbfft
```

### With GPU Support (CUDA 12)

```bash
pip install "jax-hfbfft[cuda]"
```

### With GPU Support (CUDA 11)

```bash
pip install "jax-hfbfft[cuda11]"
```

### With Apple Silicon GPU Support

```bash
pip install "jax-hfbfft[metal]"
```

### Development Installation

```bash
git clone https://github.com/ascsn/jax-hfbfft.git
cd jax-hfbfft
pip install -e ".[dev]"
```

## Quick Start

### Basic Calculation

```python
from jax_hfbfft import HFBFFT, Nucleus, Force

# Define the nucleus
nucleus = Nucleus(protons=50, neutrons=82)  # Sn-132

# Choose the nuclear force
force = Force.from_name("SLy4")

# Create and run the calculation
calc = HFBFFT(nucleus=nucleus, force=force)
calc.initialize_wavefunctions(method="harmonic_oscillator")
results = calc.run(max_iterations=1000)

# Access results
print(f"Binding energy: {results.total_energy:.3f} MeV")
print(f"RMS radius: {results.rms_radius_total:.3f} fm")
```

### Using Element Symbols

```python
from jax_hfbfft import Nucleus

# Create nuclei using element symbols
ca40 = Nucleus.from_symbol("Ca", 40)
sn132 = Nucleus.from_symbol("Sn", 132)
pb208 = Nucleus.from_symbol("Pb", 208)
```

### With Pairing

```python
from jax_hfbfft import HFBFFT, Nucleus, Force

# Create force with DDDI pairing
force = Force.from_name("SLy4").with_pairing(
    ipair=6,  # DDDI pairing
    v0prot=362.0,
    v0neut=362.0,
    tbcs=True
)

calc = HFBFFT(
    nucleus=Nucleus(protons=50, neutrons=82),
    force=force
)
```

### Constrained Calculations

```python
from jax_hfbfft import HFBFFT, Nucleus, Force, Constraint

# Prolate deformation constraint
constraint = Constraint.prolate(beta2=0.3)

calc = HFBFFT(
    nucleus=Nucleus(protons=92, neutrons=146),  # U-238
    force=Force.from_name("SLy4"),
    constraint=constraint
)
```

### Multiple Parallel Calculations

```python
from jax_hfbfft import HFBFFT, Nucleus, Force

# Define multiple nuclei
nuclei = [
    Nucleus.from_symbol("Ca", 40),
    Nucleus.from_symbol("Ca", 48),
    Nucleus.from_symbol("Ni", 56),
    Nucleus.from_symbol("Ni", 68),
    Nucleus.from_symbol("Sn", 100),
    Nucleus.from_symbol("Sn", 132),
]

force = Force.from_name("SLy4")

# Create independent calculations
calculations = [HFBFFT(nucleus=n, force=force) for n in nuclei]

# Run all calculations
for calc in calculations:
    calc.initialize_wavefunctions()
    calc.run(max_iterations=500)

# Collect results
for calc in calculations:
    print(f"{calc.nucleus}: E = {calc.results.total_energy:.3f} MeV")
```

## Configuration

### From YAML Files

```python
from jax_hfbfft import Config, HFBFFT

# Load configuration from file
config = Config.from_yaml("my_calculation.yml")

# Create calculation from config
calc = HFBFFT.from_config(config)
```

Example YAML configuration:

```yaml
nucleus:
  protons: 50
  neutrons: 82

force:
  name: SLy4
  ipair: 6
  v0prot: 362.0
  v0neut: 362.0

grid:
  nx: 32
  ny: 32
  nz: 32
  dx: 0.8
  dy: 0.8
  dz: 0.8

static:
  max_iterations: 1000
  convergence_threshold: 1.0e-6
  x0dmp: 0.45
  e0dmp: 100.0
```

## Available Forces

The following Skyrme parameterizations are available:

- `SLy4` - Standard Skyrme force
- `SkM*` - Skyrme force with improved fission barriers

Load custom forces from YAML:

```python
force = Force.from_name("MyForce", forces_file="custom_forces.yml")
```

## GPU Acceleration

JAX-HFBFFT automatically uses available GPUs when JAX is configured with GPU support:

```python
import jax
print(f"Available devices: {jax.devices()}")

# Force CPU-only execution if needed
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
```

## Package Structure

```
jax_hfbfft/
├── core/           # Core calculation classes
│   ├── hfbfft.py   # Main HFBFFT class
│   ├── nucleus.py  # Nucleus specification
│   ├── force.py    # Nuclear force definitions
│   ├── constraint.py # Constraint handling
│   └── grid.py     # Spatial discretization
├── forces/         # Force presets and utilities
├── utils/          # Utility functions
│   ├── math.py     # Mathematical utilities
│   └── io.py       # I/O utilities
└── config.py       # Configuration management
```

## Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use JAX-HFBFFT in your research, please cite:

```bibtex
@software{jax_hfbfft,
  title = {JAX-HFBFFT: A JAX-based HFB solver for nuclear structure},
  author = {ASCSN},
  year = {2024},
  url = {https://github.com/ascsn/jax-hfbfft}
}
```

## Acknowledgments

This code is based on the HFBFFT code developed for nuclear structure calculations.
