"""
Odd nuclei in the equal-filling approximation (EFA).

The odd nucleon is shared equally by a Kramers doublet {b, Tb}: both members get
v^2 = 1/2 and no pairing, so the state stays time-even.  The density depends only
on the two-dimensional space the doublet spans, not on which orbitals carry it, so
the blocked level is tracked as that space.  The reference spinor r is the blocking
candidate of the even-even core and stays fixed; each iteration

  1. the orbital with the largest weight <i|P|i>, P = |r><r| + |Tr><Tr|, fixes the
     shell: the orbitals degenerate with it (to 1e-4 MeV), i.e. its Kramers partner,
     or the whole j-shell of a spherical core;
  2. the shell is rotated so that two of its orbitals span the projection of {r, Tr}
     on it (a rotation inside a degenerate shell leaves h unchanged), and those two
     are blocked.

The blocked pair is always closed under time reversal, so the state stays time-even.
Choosing the two largest weights over all orbitals instead picks members of
different Kramers pairs whenever a near-degenerate shell is mixed (17O on a cubic
grid: the pair hopped between levels and efluct stalled at 5e-3), and updating r to
the current doublet lets such mixings accumulate.

EFA is a time-even approximation, so in EFA runs the time-odd densities (current,
spin, spin-kinetic) are set to zero (time_even); otherwise numerical noise in them
grows through the time-odd terms of the functional and breaks the Kramers
degeneracy the tracking relies on (17O: the 1p1/2 pair split by 0.14 MeV within 40
iterations).

run_odd starts from the even-even core, takes the shells within `window` MeV of
the lowest quasiparticle energy, splits each by |Omega| (eigenvalues of J_z in the
shell), and solves the odd nucleus for every candidate.  In a nearly spherical odd
nucleus the Omega label is not kept: on the cubic grid the 17O Omega = 5/2 candidate
ends in the Omega = 1/2 state, 20 keV above HFBTHO's 5/2 one (axial=True, which
constrains Q22 and the off-diagonal quadrupole moments to zero, does not cure it).
"""

import dataclasses
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np


def time_reverse(phi):
    """T phi = i sigma_y K phi for a spinor (2, nx, ny, nz)."""
    return jnp.stack([-jnp.conj(phi[1]), jnp.conj(phi[0])])


def track_doublet(psi, sp_energy, deltaf, lagrange, isospin, ref, q: int, wxyz, tol: float = 1e-4):
    """
    One tracking step (see the module docstring) for the blocked isospin q.

    Returns (psi, sp_energy, deltaf, lagrange, blocked): the arrays with the shell
    containing the doublet rotated, and the (nstates,) 0/1 mask of the blocked pair.
    """
    n = psi.shape[0]
    v = jnp.stack([ref, time_reverse(ref)])
    o = wxyz * jnp.einsum('ksxyz,vsxyz->kv', jnp.conj(psi), v)               # <k|v>
    inq = isospin == q
    weight = jnp.where(inq, jnp.sum(jnp.abs(o) ** 2, axis=1), -1.0)
    shell = inq & (jnp.abs(sp_energy - sp_energy[jnp.argmax(weight)]) < tol)

    # eigenvectors of P inside the shell; distinct negative diagonal elsewhere keeps
    # every other orbital as it is (an eigenvector e_i, put back at position i)
    both = shell[:, None] & shell[None, :]
    A = jnp.where(both, o @ jnp.conj(o.T), 0.0) + jnp.diag(jnp.where(shell, 0.0, -10.0 - jnp.arange(n)))
    lam, U = jnp.linalg.eigh(A)
    home = jnp.argmax(jnp.abs(U), axis=0)
    in_shell = lam > -5.0
    phase = U[home, jnp.arange(n)]
    U = U * jnp.where(in_shell, 1.0, jnp.conj(phase) / jnp.abs(phase))[None, :]
    cols = jnp.argsort(jnp.where(in_shell, -lam, 2.0 * n + home))             # shell first, by weight
    slots = jnp.argsort(jnp.where(shell, jnp.arange(n), n + jnp.arange(n)))   # shell positions first
    R = jnp.zeros_like(U).at[:, slots].set(U[:, cols])                        # psi_i' = sum_j R_ji psi_j

    rot = lambda x: jnp.einsum('ji,j...->i...', R, x)
    psi, lagrange = rot(psi), rot(lagrange)
    W = jnp.abs(R) ** 2
    sp_energy, deltaf = W.T @ sp_energy, W.T @ deltaf

    o = wxyz * jnp.einsum('ksxyz,vsxyz->kv', jnp.conj(psi), v)
    w = jnp.where(inq, jnp.sum(jnp.abs(o) ** 2, axis=1), -2.0)
    pool = jnp.where(jnp.sum(shell) >= 2, shell, inq)        # no degenerate partner: best two overall
    top = jnp.argsort(-jnp.where(pool, w, -1.0))[:2]
    return psi, sp_energy, deltaf, lagrange, jnp.zeros(n).at[top].set(1.0)


def time_even(dens):
    """The densities with the time-odd ones (current, spin, spin-kinetic) set to zero."""
    zero = lambda x: None if x is None else jnp.zeros_like(x)
    return dataclasses.replace(dens, current=zero(dens.current), sdens=zero(dens.sdens), tspin=zero(dens.tspin))


def efa_hf_occupations(wocc, sp_energy, isospin, blocked, q: int, n_q: int):
    """Unpaired occupations: the doublet at 1/2 each, the n_q - 1 lowest other orbitals of q filled."""
    others = (isospin == q) & (blocked < 0.5)
    rank = jnp.argsort(jnp.argsort(jnp.where(others, sp_energy, jnp.inf)))
    filled = others & (rank < n_q - 1)
    return jnp.where(isospin == q, jnp.where(blocked > 0.5, 0.5, filled.astype(wocc.dtype)), wocc)


def _fermi(state, q):
    """lambda of isospin q: the pairing Fermi energy, or mid-gap without pairing."""
    from jax_hfbfft.physics.inertia import fermi_levels
    return float(fermi_levels(state)[q])


def efa_candidates(core, grid, q: int, window: float = 2.0, tol: float = 1e-2) -> list:
    """
    Blocking candidates of the even-even core for isospin q: (near-)degenerate shells
    with a quasiparticle energy within `window` MeV of the lowest, one candidate per
    |Omega| of each shell.  Returns dicts (ref, e, Eqp, omega), sorted by Eqp.
    """
    from jax_hfbfft.physics.inertia import angular_momentum_operator
    iso, e, d = (np.asarray(x) for x in (core.isospin, core.sp_energy, core.deltaf))
    lam = _fermi(core, q)
    idx = np.nonzero(iso == q)[0]
    eqp = np.sqrt((e[idx] - lam) ** 2 + d[idx] ** 2)
    idx = idx[eqp <= eqp.min() + window]
    jz = angular_momentum_operator(grid, 'z')
    out = []
    while len(idx):
        shell = idx[np.abs(e[idx] - e[idx[0]]) < tol]
        idx = np.setdiff1d(idx, shell)
        psi = core.psi[shell]
        J = np.asarray(grid.wxyz * jnp.einsum('ksxyz,lsxyz->kl', jnp.conj(psi), jax.vmap(jz)(psi)))
        m, V = np.linalg.eigh(0.5 * (J + J.conj().T))
        for omega in sorted({round(abs(x) * 2) / 2 for x in m}):
            k = int(np.argmin(np.abs(m - omega)))
            ref = jnp.einsum('k,ksxyz->sxyz', jnp.asarray(V[:, k]), psi)
            ref = ref / jnp.sqrt(grid.wxyz * jnp.sum(jnp.abs(ref) ** 2))
            out.append(dict(ref=ref, e=float(e[shell].mean()),
                            Eqp=float(np.sqrt((e[shell].mean() - lam) ** 2 + d[shell].mean() ** 2)),
                            omega=omega))
    return sorted(out, key=lambda c: c['Eqp'])


def run_odd(grid, force, nucleus_z: int, nucleus_n: int, npsi_n: int, config=None,
            window: float = 2.0, core=None, axial: bool = False, use_coulomb: bool = True,
            verbose: bool = True):
    """
    EFA ground state of an odd-A nucleus: the even-even core (Z, N-1) or (Z-1, N)
    (solved here unless given as `core`), then one run per blocking candidate.
    axial: keep the shapes axial about z (see the module docstring); a given core
    should then be axial about z too.

    Returns (best, results): the lowest-energy SolverState and a list of dicts
    (state, e, Eqp, omega, E, converged) sorted by E.
    """
    from jax_hfbfft.core.constraint import Constraint
    from jax_hfbfft.physics.solver import SolverConfig, run_hfb
    if nucleus_n % 2 == nucleus_z % 2:
        raise ValueError("run_odd needs exactly one of Z, N odd")
    q = 0 if nucleus_n % 2 else 1
    cz, cn = (nucleus_z, nucleus_n - 1) if q == 0 else (nucleus_z - 1, nucleus_n)
    config = config or SolverConfig()
    constraint = Constraint.from_multipoles({'Q22': 0.0}, principal_axes=True) if axial else None
    if core is None:
        core = run_hfb(grid, force, nucleus_z=cz, nucleus_n=cn, npsi_n=npsi_n, config=config,
                       constraint=constraint, use_coulomb=use_coulomb)
    odd_config = dataclasses.replace(config, block_isospin=q)
    results = []
    for c in efa_candidates(core, grid, q, window):
        st = run_hfb(grid, force, nucleus_z=nucleus_z, nucleus_n=nucleus_n, npsi_n=npsi_n, config=odd_config,
                     initial_state=dataclasses.replace(core, block_ref=c['ref']), constraint=constraint,
                     use_coulomb=use_coulomb)
        results.append(dict(state=st, e=c['e'], Eqp=c['Eqp'], omega=c['omega'],
                            E=float(st.energies.ehfint), converged=bool(st.converged)))
        if verbose:
            print(f"[EFA] Omega {c['omega']:.1f}  e {c['e']:8.3f}  Eqp {c['Eqp']:6.3f}  ->  "
                  f"E {results[-1]['E']:.4f} MeV  efluct {float(st.efluct):.1e}", flush=True)
    results.sort(key=lambda r: r['E'])
    return results[0]['state'], results
