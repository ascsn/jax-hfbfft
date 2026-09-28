"""
High-level interface for time-dependent Hartree-Fock (TDHF).

TDHF is the dynamical counterpart of HFBFFT: it takes a converged static
solution and propagates it in real time. The low-level functions live in
jax_hfbfft.physics.dynamics.

Example:
    >>> from jax_hfbfft import HFBFFT, Nucleus, Force, TDHF, TDConfig
    >>> calc = HFBFFT(nucleus=Nucleus(protons=8, neutrons=8),
    ...               force=Force.from_name("SLy4", ipair=0), npsi=(20, 20))
    >>> calc.initialize_wavefunctions(method="harmonic_oscillator")
    >>> calc.run(max_iterations=300)
    >>> td = TDHF.from_static(calc, config=TDConfig(dt=0.2, n_steps=1500))
    >>> td.kick_quadrupole(eta=1e-3)
    >>> for rec in td.iterate():          # one record per diag_interval steps
    ...     print(rec['time'], rec['Q20'], rec['dE_rel'])
"""

import dataclasses
from typing import Callable, Iterator, List, Optional

import jax
import jax.numpy as jnp
import numpy as np

from jax_hfbfft.core.force import Force
from jax_hfbfft.core.grid import Grid
from jax_hfbfft.physics import dynamics as D


class TDHF:
    """
    Real-time TDHF evolution with recorded observables.

    Records are dicts with time, step, E, dE_rel (energy drift relative to the
    first record after the last excitation), ortho, A, cm_x/y/z, r2, r2_int,
    Q20 and Q22; see physics.dynamics.observables.
    """

    def __init__(self, state: D.TDState, grid: Grid, force: Force,
                 config: Optional[D.TDConfig] = None):
        self.state = state
        self.grid = grid
        self.force = force
        self.config = config or D.TDConfig()
        self.history: List[dict] = []
        self._E0: Optional[float] = None

    @classmethod
    def from_static(cls, static, grid: Optional[Grid] = None,
                    force: Optional[Force] = None,
                    config: Optional[D.TDConfig] = None, **prepare_kwargs) -> "TDHF":
        """
        Start from a converged static calculation.

        Args:
            static: An HFBFFT instance after run(), or a SolverState from run_hfb.
            grid: Required when `static` is a SolverState.
            force: Propagation Hamiltonian; defaults to the force the static
                state was relaxed in.
            config: TDConfig for the run.
            **prepare_kwargs: Passed to prepare_tdhf_state.
        """
        from jax_hfbfft.core.hfbfft import HFBFFT
        if isinstance(static, HFBFFT):
            if static.solver_state is None:
                raise ValueError("The HFBFFT calculation has not been run yet.")
            grid = grid or static.grid
            static = static.solver_state
        if grid is None:
            raise ValueError("grid is required when starting from a SolverState.")
        config = config or D.TDConfig()
        prepare_kwargs.setdefault("use_coulomb", config.use_coulomb)
        prepare_kwargs.setdefault("density_chunk", config.density_chunk)
        state = D.prepare_tdhf_state(static, grid, force=force, **prepare_kwargs)
        force = force or static.force
        return cls(state, grid, force, config)

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def time(self) -> float:
        """Current time in fm/c."""
        return float(self.state.time)

    @property
    def Z(self) -> int:
        return int(round(float(jnp.sum(self.state.wocc * (self.state.isospin == 1)))))

    @property
    def N(self) -> int:
        return int(round(float(jnp.sum(self.state.wocc * (self.state.isospin == 0)))))

    # ── Excitations ─────────────────────────────────────────────────────────

    def _set_psi(self, psi: jax.Array) -> None:
        d, mf, wc = D.build_fields(
            psi, self.state.wocc, self.state.isospin, self.grid, self.force,
            self.state.coulomb_solver, self.config.use_coulomb,
            self.config.density_chunk,
        )
        self.state = dataclasses.replace(self.state, psi=psi, densities=d,
                                         meanfield=mf, wcoul=wc)
        self._E0 = None   # energy drift is measured from the excited state

    def boost(self, k) -> "TDHF":
        """Give every nucleon momentum hbar*k, k = (kx, ky, kz) in fm^-1."""
        self._set_psi(D.apply_boost(self.state.psi, self.grid, tuple(k)))
        return self

    def kick(self, operator: jax.Array, eta: float) -> "TDHF":
        """Apply exp(-i eta Q) for a one-body operator Q given on the grid."""
        self._set_psi(D.apply_multipole_boost(self.state.psi, self.grid, operator, eta))
        return self

    def kick_quadrupole(self, eta: float) -> "TDHF":
        """Isoscalar quadrupole kick, Q = 2z^2 - x^2 - y^2."""
        X, Y, Z = self.grid.get_meshgrid()
        return self.kick(2 * Z**2 - X**2 - Y**2, eta)

    def kick_monopole(self, eta: float) -> "TDHF":
        """Isoscalar monopole (breathing) kick, Q = r^2."""
        X, Y, Z = self.grid.get_meshgrid()
        return self.kick(X**2 + Y**2 + Z**2, eta)

    def kick_isovector_dipole(self, eta: float) -> "TDHF":
        """Isovector dipole kick along z (centre of mass not excited)."""
        self._set_psi(D.apply_isovector_dipole_boost(
            self.state.psi, self.state.isospin, self.grid, eta, self.Z, self.N))
        return self

    # ── Propagation ─────────────────────────────────────────────────────────

    def observables(self) -> dict:
        """Observables of the current state (not added to history)."""
        return D.observables(self.state, self.grid, self.force, self.config.use_coulomb)

    def _record(self) -> dict:
        rec = self.observables()
        if self._E0 is None:
            self._E0 = rec['E']
        rec['dE_rel'] = (rec['E'] - self._E0) / abs(self._E0)
        self.history.append(rec)
        return rec

    def iterate(self, n_steps: Optional[int] = None) -> Iterator[dict]:
        """
        Propagate n_steps (default config.n_steps), yielding a record every
        config.diag_interval steps. The first record is the starting state
        unless it has already been recorded.

        Stopping the iteration early (break) leaves the object in a valid
        state; iterate() or run() can then continue from there.
        """
        n_steps = self.config.n_steps if n_steps is None else n_steps
        if not self.history or self._E0 is None \
                or self.history[-1]['step'] != int(self.state.step):
            yield self._record()
        done = 0
        while done < n_steps:
            n = min(self.config.diag_interval, n_steps - done)
            self.state = D.advance(self.state, self.grid, self.force, self.config, n)
            done += n
            yield self._record()

    def run(self, n_steps: Optional[int] = None,
            callback: Optional[Callable[["TDHF", dict], None]] = None) -> List[dict]:
        """Propagate n_steps and return the full history."""
        for rec in self.iterate(n_steps):
            if callback is not None:
                callback(self, rec)
        return self.history

    def history_arrays(self) -> dict:
        """The history as a dict of NumPy arrays, one per observable."""
        if not self.history:
            return {}
        return {k: np.array([r[k] for r in self.history]) for k in self.history[0]}
