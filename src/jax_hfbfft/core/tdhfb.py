"""
High-level interface for time-dependent HFB (TDHFB).

TDHFB propagates a converged, paired static solution in real time with the
global in-span matrix scheme of jax_hfbfft.physics.tdhfb, or with Ebata's
canonical-basis Cb-TDHFB as a comparison (scheme='ebata').

The static must be solved on an odd grid with state_cutoff = 0 (every state
weight 1); see physics/tdhfb.py for why.

Example:
    >>> from jax_hfbfft import HFBFFT, Nucleus, Force, Grid, TDHFB
    >>> force = Force.from_name("SLy4").with_pairing(ipair=6, v0neut=300., v0prot=300., rho0pr=0.32)
    >>> calc = HFBFFT(nucleus=Nucleus(protons=8, neutrons=12), force=force,
    ...               grid=Grid.create_cubic(n=13, d=1.2), npsi=(40, 20))
    >>> calc.run(max_iterations=3000)
    >>> td = TDHFB.from_static(calc)
    >>> td.kick_quadrupole(0.005)
    >>> for rec in td.iterate(2000, record_every=20):
    ...     print(rec['time'], rec['E'] - td.history[0]['E'], rec['Q20'])
"""

import dataclasses
import os
from typing import Iterator, List, Optional

import jax.numpy as jnp
import numpy as np

from jax_hfbfft.core.force import Force
from jax_hfbfft.core.grid import Grid
from jax_hfbfft.physics import tdhfb as T
from jax_hfbfft.physics.coulomb import CoulombSolver
import warnings

from jax_hfbfft.physics.solver import apply_cm_correction, statics_only_terms


# ── Static files ─────────────────────────────────────────────────────────────

def save_static(path: str, state, grid: Grid, functional: str = "SLy4", **meta) -> None:
    """
    Save a static solution (a SolverState) with everything TDHFB needs to
    rebuild it: orbitals, occupations, the grid and the pairing force.
    Pass the pairing parameters as meta (v0, v0p, ipair, rho0pr).
    """
    meta = {k: v for k, v in meta.items() if v is not None}
    pairing, sforce = getattr(state, 'pairing', None), getattr(state, 'force', None)
    if pairing is not None and getattr(pairing, 'lambda2', None) is not None:
        meta.setdefault('lambda2', np.asarray(pairing.lambda2))   # Lipkin-Nogami, see statics_only_terms
    if sforce is not None:
        meta.setdefault('zpe', int(np.asarray(sforce.zpe)))
    np.savez_compressed(
        path, psi=np.asarray(state.psi), wocc=np.asarray(state.wocc), wguv=np.asarray(state.wguv),
        pairwg=np.asarray(state.pairwg), wstates=np.asarray(state.wstates),
        isospin=np.asarray(state.isospin), sp_energy=np.asarray(state.sp_energy),
        E=float(state.energies.ehfint), efluct=float(state.efluct),
        nx=grid.nx, ny=grid.ny, nz=grid.nz, dx=grid.dx, functional=functional, **meta)


def load_static(path: str, nz: Optional[int] = None):
    """
    A static .npz (save_static, or the cb_static.py format) as (data, grid, force).

    nz > the file's nz embeds the nucleus in a longer box by zero-padding the
    orbitals symmetrically in z (collisions). The force carries the
    centre-of-mass correction for this nucleus alone.
    """
    d = dict(np.load(path))
    nx, ny, nz0, dx = int(d['nx']), int(d['ny']), int(d['nz']), float(d['dx'])
    if nz is not None and nz > nz0:
        p0 = (nz - nz0) // 2
        d['psi'] = np.pad(d['psi'], [(0, 0)] * 4 + [(p0, nz - nz0 - p0)])
        d['nz'] = nz
    grid = Grid.create(nx=nx, ny=ny, nz=d['psi'].shape[-1], dx=dx, dy=dx, dz=dx)
    return d, grid, apply_cm_correction(_force_of(d), int(round(float(np.sum(d['wocc'])))))


def _force_of(d) -> Force:
    """The pairing force a static file was solved with, without the c.m. correction."""
    v0 = float(d.get('v0', 362.0))
    return Force.from_name(str(d.get('functional', 'SLy4'))).with_pairing(
        ipair=int(d.get('ipair', 6)), v0neut=v0, v0prot=float(d.get('v0p', v0)),
        rho0pr=float(d.get('rho0pr', 0.32)), tbcs=False, pair_cutoff=(0.0, 0.0),
        state_cutoff=(0.0, 0.0), pair_kcut=float(d.get('pair_kcut', 0.0)),
        pair_kwidth=float(d.get('pair_kwidth', 0.1)))


def _static_arrays(static):
    """(psi, wocc, wguv, isospin, wstates, pairwg) of a SolverState or loaded dict."""
    get = (lambda k: static[k]) if isinstance(static, dict) else (lambda k: getattr(static, k))
    return tuple(get(k) for k in ('psi', 'wocc', 'wguv', 'isospin', 'wstates', 'pairwg'))


def _initial_state(static, grid, scheme, uv_floor, max_defect=1e-6, kramers_tol=1e-7):
    if scheme == 'global' and isinstance(static, dict) and 'kap' in static:     # relaxed (relax.py)
        return T.GState(psi=jnp.asarray(static['psi']), rho=jnp.asarray(static['rho']),
                        kap=jnp.asarray(static['kap']), isospin=jnp.asarray(static['isospin']))
    psi, wocc, wguv, iso, wst, pwg = _static_arrays(static)
    if scheme == 'global':
        return T.from_static(psi, wocc, wguv, iso, grid, wstates=wst, pairwg=pwg, uv_floor=uv_floor,
                             max_defect=max_defect, kramers_tol=kramers_tol)
    return T.ebata_state(psi, wocc, wguv, iso, grid, wstates=wst, pairwg=pwg, uv_floor=uv_floor)


# ── TDHFB ────────────────────────────────────────────────────────────────────

class TDHFB:
    """
    Real-time TDHFB evolution with recorded observables.

    Records are dicts with time, E, Epair_n, Epair_p, N, Z, Q20, r2 (<r^2>) and, for the
    global scheme, purity (|rho~ - rho~^2 - kap~ kap~^+|, conserved exactly).
    """

    def __init__(self, state, grid: Grid, force: Force, dt: float = 0.1, eps: float = 1e-7,
                 n_comp: int = 1, pred_n_comp: Optional[int] = 1, taylor_order: int = 6,
                 use_coulomb: bool = True, strang: bool = True, midpoint: bool = True,
                 absorb_width: float = 0.0, absorb_strength: float = 10.0):
        self.state = state
        self.grid = grid
        self.force = force
        self.cs = CoulombSolver.create(grid)
        self.dt = dt
        self.eps, self.n_comp, self.pred_n_comp = eps, n_comp, pred_n_comp
        self.taylor_order = taylor_order
        self.use_coulomb = use_coulomb
        self.strang, self.midpoint = strang, midpoint
        self.absorb_width, self.absorb_strength = absorb_width, absorb_strength   # absorbing boundaries (fm, MeV)
        self.history: List[dict] = []
        self._step = None
        self._ref_step = None

    @property
    def scheme(self) -> str:
        return 'ebata' if isinstance(self.state, T.EbataState) else 'global'

    @property
    def time(self) -> float:
        return float(self.state.time)

    # ── Construction ────────────────────────────────────────────────────────

    @classmethod
    def from_static(cls, static, grid: Optional[Grid] = None, force: Optional[Force] = None,
                    scheme: str = 'global', uv_floor: Optional[float] = 0.0, purify: bool = True,
                    max_defect: float = 1e-6, kramers_tol: float = 1e-7, nz: Optional[int] = None,
                    **kwargs) -> "TDHFB":
        """
        Start from a converged static solution.

        Args:
            static: An HFBFFT instance after run(), a SolverState, or the path of
                a static .npz (save_static / cb_static.py format).
            grid, force: Needed for a SolverState; default to the static's own.
            scheme: 'global' (the method) or 'ebata' (comparison arm).
            uv_floor: 0 rebuilds kappa = sqrt(v2(1-v2)); None keeps the static's wguv.
            purify: McWeeny-purify the global-scheme state (removes the static's
                ~1e-6 impurity, which the complement force amplifies).
            max_defect: tolerance on the static's time-reversal symmetry
                (physics.tdhfb.kramers_basis).
            kramers_tol: occupation tolerance for grouping Kramers partners.
            nz: embed a static file in a box this long in z (zero padding),
                e.g. to give fission fragments room to separate.
            **kwargs: Passed to TDHFB (dt, eps, n_comp, ...).
        """
        from jax_hfbfft.core.hfbfft import HFBFFT
        if isinstance(static, str):
            static, g, f = load_static(static, nz)
            grid, force = grid or g, force or f
        elif isinstance(static, HFBFFT):
            if static.solver_state is None:
                raise ValueError("The HFBFFT calculation has not been run yet.")
            grid, force = grid or static.grid, force or static.solver_force
            static = static.solver_state
        if grid is None or force is None:
            raise ValueError("grid and force are required when starting from a SolverState.")
        for msg in statics_only_terms(static, force):
            warnings.warn("TDHFB.from_static: " + msg, stacklevel=2)
        td = cls(_initial_state(static, grid, scheme, uv_floor, max_defect, kramers_tol), grid, force, **kwargs)
        if purify and td.scheme == 'global':
            td.state = T.purify(td.state)
        return td

    @classmethod
    def collision(cls, static_a: str, static_b: Optional[str] = None, nz: Optional[int] = None,
                  sep: float = 16.0, e_asym: float = 277.0, phase: float = 0.0,
                  scheme: str = 'global', uv_floor: Optional[float] = 0.0, band: float = 6.0,
                  **kwargs) -> "TDHFB":
        """
        Head-on collision of two static nuclei (static .npz paths; B defaults to A).

        Each fragment is embedded in a box of length nz (zero padding in z),
        placed at -/+ sep/2, boosted to the asymptotic c.m. energy e_asym (MeV,
        Coulomb included) and given the relative pairing phase `phase` on B.
        The composite force carries the c.m. correction for A1 + A2.
        The placement is stored in self.info.
        """
        da, grid, _ = load_static(static_a, nz)
        db = load_static(static_b, nz)[0] if static_b else da
        nums = []
        for d in (da, db):
            iso, w = np.asarray(d['isospin']), np.asarray(d['wocc'])
            nums.append((int(round(w[iso == 1].sum())), int(round(w.sum()))))
        (ZA, AA), (ZB, AB) = nums
        force = apply_cm_correction(_force_of(da), AA + AB)
        frags = [_initial_state(d, grid, scheme, uv_floor) for d in (da, db)]
        build = T.make_collision if scheme == 'global' else T.make_ebata_collision
        state, info = build(*frags, grid, ZA, AA, ZB, AB, sep, e_asym, phase, band)
        td = cls(state, grid, force, **kwargs)
        if td.scheme == 'global':
            td.state = T.purify(td.state)
        td.info = info
        return td

    # ── Excitations and state maintenance ───────────────────────────────────

    def kick(self, operator, eta: float) -> "TDHFB":
        """exp(i eta Q(r)) on every orbital (rho~ and kap~ are unchanged)."""
        self.state = T.kick(self.state, operator, eta)
        return self

    def kick_quadrupole(self, eta: float) -> "TDHFB":
        """Isoscalar quadrupole kick, Q = 2z^2 - x^2 - y^2."""
        X, Y, Z = self.grid.get_meshgrid()
        return self.kick(2 * Z**2 - X**2 - Y**2, eta)

    def kick_monopole(self, eta: float) -> "TDHFB":
        """Isoscalar monopole (breathing) kick, Q = r^2."""
        X, Y, Z = self.grid.get_meshgrid()
        return self.kick(X**2 + Y**2 + Z**2, eta)

    def relax(self, n_iter: int, step: float = 1.0, mix: float = 0.5, verbose: int = 0) -> "TDHFB":
        """
        Relax the state into the minimum of the TDHFB functional (global scheme;
        physics.tdhfb.relax).  A static from the canonical solver is not that
        minimum, and at rest it then drifts toward it (20O: E_pair +5.9% over
        60 fm/c unrelaxed, +0.02% relaxed).  Do this before kicking.
        """
        if self.scheme != 'global':
            raise ValueError("relax() is implemented for the global scheme")
        self.state, _ = T.relax(self.state, self.grid, self.force, self.cs, n_iter=n_iter,
                                step=step, mix=mix, use_coulomb=self.use_coulomb, verbose=verbose)
        self._step = self._ref_step = None
        self.history = []
        return self

    def drop_empty(self, below: float):
        """Remove orbitals with occupation below `below` (global scheme). Returns (count, particles)."""
        self.state, n, lost = T.drop_empty(self.state, below)
        if n:
            self._step = self._ref_step = None
        return n, lost

    # ── Propagation ─────────────────────────────────────────────────────────

    def _get_step(self):
        if self._step is None:
            if self.scheme == 'global':
                W = (T.absorbing_potential(self.grid, self.absorb_width, self.absorb_strength)
                     if self.absorb_width > 0 else None)
                self._step = T.make_step(self.state, self.grid, self.force, self.cs, self.eps,
                                         self.n_comp, self.pred_n_comp, self.taylor_order,
                                         self.use_coulomb, self.strang, self.midpoint, absorb=W)
            else:
                self._step = T.make_ebata_step(self.state, self.grid, self.force, self.cs,
                                               self.taylor_order, use_coulomb=self.use_coulomb)
        return self._step

    def fields(self):
        """(densities, meanfield, wcoul, delta, xi) of the current state."""
        return T.fields(self.state, self.grid, self.force, self.cs, self.use_coulomb)

    def observables(self, flds=None) -> dict:
        """Observables of the current state (not added to history)."""
        flds = flds or self.fields()
        E, ep = T.energy_of_fields(flds, self.grid, self.force, self.use_coulomb)
        X, Y, Z = self.grid.get_meshgrid()
        rho = flds[0].rho[0] + flds[0].rho[1]
        N, Zp = T.particle_numbers(self.state)
        rec = dict(time=self.time, E=E, Epair_n=float(ep[0]), Epair_p=float(ep[1]), N=N, Z=Zp,
                   Q20=float(self.grid.wxyz * jnp.sum(rho * (2 * Z**2 - X**2 - Y**2))),
                   r2=float(self.grid.wxyz * jnp.sum(rho * (X**2 + Y**2 + Z**2))) / (N + Zp))
        if self.scheme == 'global':
            rec['purity'] = T.purity(self.state)
        return rec

    def iterate(self, n_steps: int, record_every: int = 1, extra=None) -> Iterator[dict]:
        """
        Propagate n_steps, yielding a record every record_every steps (and the
        starting state if it has not been recorded). extra(td, fields) -> dict
        adds driver-specific entries to each record.
        """
        def record():
            flds = self.fields()
            rec = self.observables(flds)
            if extra is not None:
                rec.update(extra(self, flds))
            if not np.isfinite(rec['E']):
                raise FloatingPointError(f"non-finite state at t = {self.time:.2f} fm/c")
            self.history.append(rec)
            return rec

        if not self.history or abs(self.history[-1]['time'] - self.time) > 1e-9:
            yield record()
        for i in range(1, n_steps + 1):
            self.state = self._get_step()(self.state, self.dt)    # rebuilt after drop_empty
            if i % record_every == 0 or i == n_steps:
                yield record()

    def run(self, n_steps: int, record_every: int = 1) -> List[dict]:
        """Propagate n_steps and return the full history."""
        for _ in self.iterate(n_steps, record_every):
            pass
        return self.history

    def continuity(self) -> dict:
        """Local continuity defect against its floor (global scheme); see physics.tdhfb.continuity_defect."""
        if self.scheme != 'global':
            raise ValueError("continuity() is implemented for the global scheme")
        if self._ref_step is None:
            self._ref_step = T.make_step(self.state, self.grid, self.force, self.cs, self.eps,
                                         self.n_comp, None, self.taylor_order, self.use_coulomb,
                                         self.strang, self.midpoint)
        return T.continuity_defect(self.state, self._ref_step, self.grid, self.force, self.cs)

    # ── Checkpoints ─────────────────────────────────────────────────────────

    def save(self, path: str, **extra) -> None:
        """Write the state (and any extra arrays) to an .npz."""
        st = {f.name: np.asarray(getattr(self.state, f.name)) for f in dataclasses.fields(self.state)}
        tmp = path + '.tmp.npz'                 # atomic: a job killed mid-write keeps the previous checkpoint
        np.savez(tmp, scheme=self.scheme, **st, **extra)
        os.replace(tmp, path)

    def load(self, path: str) -> "TDHFB":
        """Replace the state by one written with save() (same grid and force)."""
        d = np.load(path)
        cls = T.GState if str(d['scheme']) == 'global' else T.EbataState
        names = [f.name for f in dataclasses.fields(cls)]
        self.state = cls(**{k: (float(d[k]) if k == 'time' else jnp.asarray(d[k])) for k in names})
        self._step = self._ref_step = None
        return self

    def history_arrays(self) -> dict:
        """The history as a dict of NumPy arrays, one per observable."""
        if not self.history:
            return {}
        return {k: np.array([r[k] for r in self.history]) for k in self.history[0]}
