# Legacy Code Archive

This directory contains the original (legacy) implementation of the JAX-HFBFFT code.

## Status: Archived

These files have been replaced by the modern object-oriented implementation in `src/jax_hfbfft/`. The legacy code is kept here for:

1. **Reference**: To understand the original algorithms and physics implementations
2. **Validation**: For comparing results between old and new implementations
3. **Fallback**: In case the `use_legacy=True` option is needed

## Files

| File | Description | Lines |
|------|-------------|-------|
| `static.py` | Main HFB iteration loop | ~2950 |
| `levels.py` | Single-particle wavefunctions and properties | ~436 |
| `constraint.py` | Constraint handling | ~430 |
| `meanfield.py` | Skyrme mean-field potentials | ~390 |
| `output.py` | Output formatting | ~376 |
| `energies.py` | Energy calculations | ~371 |
| `densities.py` | Density computations | ~336 |
| `pairs.py` | BCS pairing | ~220 |
| `forces.py` | Force parameter definitions | ~156 |
| `grids.py` | Spatial grid setup | ~154 |
| `inout.py` | I/O utilities | ~150 |
| `params.py` | Parameter handling | ~95 |
| `coulomb.py` | Coulomb potential solver | ~60 |
| `trivial.py` | Helper functions | ~30 |
| `moment.py` | Moment calculations | ~10 |

## Using Legacy Code

To use the legacy implementation via the new API:

```python
from jax_hfbfft import HFBFFT, Nucleus, Force

calc = HFBFFT(
    nucleus=Nucleus(protons=50, neutrons=82),
    force=Force.from_name("SLy4")
)

# Run with legacy implementation
calc.run(max_iterations=1000, use_legacy=True)
```

## Migration Notes

The modern implementation in `src/jax_hfbfft/` provides:

- **Object-oriented design**: Clean separation of concerns
- **Type hints**: Full type annotations for better IDE support
- **64-bit by default**: Configurable precision via `jax_config.py`
- **Modular physics**: Independent physics modules in `physics/`
- **Better testing**: Comprehensive test suite
- **Documentation**: Docstrings and examples

## Do Not Modify

These files are archived and should not be modified. If bugs are found, they should be fixed in the modern implementation.
