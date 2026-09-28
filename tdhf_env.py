"""
Environment setup for the TDHF scripts on this cluster.

    import tdhf_env; tdhf_env.setup()      # BEFORE importing jax or matplotlib

Call it first thing.  Two of the four settings only take effect if they are in
os.environ before JAX is imported, so importing this after `import jax` silently
does nothing useful.

WHAT THIS HANDLES
-----------------
  sys.path      Adds ./src so `jax_hfbfft` is importable without installing it,
                and -- only if the import actually fails -- the Python-bundle
                module that carries pyparsing.  matplotlib imports pyparsing and
                ~/.local/lib/python3.11/site-packages does not have it.
                `module load matplotlib/3.7.2-gfbf-2023a` is the official route
                but that module lives in the `generic` arch tree and Lmod
                refuses it on the login node, so the path is added directly.
                Nothing here depends on Lmod.

  JAX_ENABLE_X64    64-bit everywhere.  TDHF energy conservation is meaningless
                    in single precision.

  OMP_NUM_THREADS / XLA_FLAGS
                    JAX otherwise grabs every core it can see, which on a shared
                    node means oversubscribing whatever SLURM actually gave you.
                    Taken from SLURM_CPUS_PER_TASK when present.

WHAT THIS CANNOT HANDLE
-----------------------
LD_LIBRARY_PATH for libpython3.11.so.  The dynamic linker reads it at exec time,
so by the time any Python code runs the question is already settled -- if the
interpreter started, the variable was not needed; if it did not start, no Python
file can help.  A login shell that fails with

    python3: error while loading shared libraries: libpython3.11.so.1.0

needs the variable set before python3 is invoked, which is what tdhf_env.sh is
still there for.  Use the wrapper in that case, or export it from your profile:

    export LD_LIBRARY_PATH=/opt/software-current/2023.06/x86_64/amd/zen2/\\
software/Python/3.11.3-GCCcore-12.3.0/lib:$LD_LIBRARY_PATH
"""

import os
import sys
from pathlib import Path

# Python-bundle-PyPI built against the same GCCcore-12.3.0 toolchain as the
# interpreter in PATH.  Matching the toolchain matters: a bundle from a
# different one can pull in extensions compiled against the wrong libstdc++.
_PYBUNDLE = Path(
    '/opt/software-current/2023.06/x86_64/generic/software/'
    'Python-bundle-PyPI/2023.06-GCCcore-12.3.0/lib/python3.11/site-packages'
)

LIBPYTHON_HINT = (
    '/opt/software-current/2023.06/x86_64/amd/zen2/software/'
    'Python/3.11.3-GCCcore-12.3.0/lib'
)


def _have(module: str) -> bool:
    import importlib.util
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def setup(threads=None, quiet=True):
    """
    Prepare sys.path and the JAX environment variables.  Idempotent.

    Returns the list of paths added, so a caller that wants to report what it
    did can.  Safe to call on a machine that has none of the module tree: every
    addition is guarded on the path existing and on the import actually being
    broken.
    """
    added = []

    # ./src -- the package is used from the checkout, not installed.
    here = Path(__file__).resolve().parent
    src = here / 'src'
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
        added.append(str(src))

    # pyparsing, and only if it is genuinely missing.  Adding a whole
    # site-packages unconditionally would shadow the user's own installs.
    if not _have('pyparsing') and _PYBUNDLE.is_dir():
        if str(_PYBUNDLE) not in sys.path:
            sys.path.append(str(_PYBUNDLE))
            added.append(str(_PYBUNDLE))

    # Must precede `import jax`.
    os.environ.setdefault('JAX_ENABLE_X64', 'True')

    n = threads or os.environ.get('SLURM_CPUS_PER_TASK')
    if n:
        os.environ.setdefault('OMP_NUM_THREADS', str(n))
        os.environ.setdefault(
            'XLA_FLAGS',
            f'--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={n}',
        )

    if 'jax' in sys.modules and not quiet:
        print('  [warning] tdhf_env.setup() ran after jax was imported; '
              'JAX_ENABLE_X64 and XLA_FLAGS will have had no effect.',
              file=sys.stderr)

    return added


def check():
    """Import the things that usually break and say which ones did. Returns bool."""
    ok = True
    try:
        import jax
        print(f'  jax {jax.__version__}  devices={jax.devices()}')
    except Exception as exc:                       # noqa: BLE001
        print(f'  jax FAILED: {exc}')
        ok = False
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot        # noqa: F401
        from matplotlib.animation import PillowWriter   # noqa: F401
        print(f'  matplotlib {matplotlib.__version__} (Agg, PillowWriter ok)')
    except Exception as exc:                       # noqa: BLE001
        print(f'  matplotlib FAILED: {exc}')
        print(f'    if this is pyparsing, the bundle path was not found:\n'
              f'      {_PYBUNDLE}')
        ok = False
    try:
        from jax_hfbfft.physics.dynamics import tdhf_step   # noqa: F401
        print('  jax_hfbfft importable')
    except Exception as exc:                       # noqa: BLE001
        print(f'  jax_hfbfft FAILED: {exc}')
        ok = False
    return ok


if __name__ == '__main__':
    added = setup()
    print('paths added:')
    for p in added or ['  (none needed)']:
        print(f'  {p}')
    print('imports:')
    sys.exit(0 if check() else 1)
