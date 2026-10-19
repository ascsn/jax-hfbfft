"""
Density calculations for HFB.

This module computes nuclear densities from single-particle wavefunctions:
- rho: particle density
- tau: kinetic density
- current: current density vector
- sdens: spin density vector
- sodens: spin-orbit density vector
- chi: pairing density (anomalous density)
"""

import jax
import jax.numpy as jnp
from functools import partial
from dataclasses import dataclass
from typing import Optional, Tuple

from jax_hfbfft.jax_config import get_dtypes
from jax_hfbfft.core.grid import deriv_x, deriv_y, deriv_z


@jax.tree_util.register_dataclass
@dataclass
class Densities:
    """
    Container for nuclear densities.

    All densities have shape (2, nx, ny, nz) where the first index is isospin:
    - 0: neutrons
    - 1: protons

    Vector densities (current, sdens, sodens) have shape (2, 3, nx, ny, nz)
    where the second index is the spatial component (x, y, z).
    """
    rho: jax.Array      # Particle density
    tau: jax.Array      # Kinetic density
    chi: jax.Array      # Pairing density (anomalous)
    current: jax.Array  # Current density (vector)
    sdens: jax.Array    # Spin density (vector)
    sodens: jax.Array   # Spin-orbit density (vector)
    # Symmetric part of the spin-current tensor J_mu,nu = Im psi^+ sigma_nu d_mu psi,
    # components (xx, yy, zz, xy, xz, yz), shape (2, 6, nx, ny, nz); only computed
    # for forces with J^2 terms (compute_densities(tensor=True)), else None.
    jsym: Optional[jax.Array] = None
    # Spin-kinetic density T_nu = sum_mu (d_mu psi)^+ sigma_nu d_mu psi, (2, 3, nx, ny, nz),
    # the time-odd partner of J_mu,nu; computed with jsym, else None.
    tspin: Optional[jax.Array] = None

    @classmethod
    def zeros(cls, nx: int, ny: int, nz: int) -> "Densities":
        """Create zero-initialized densities."""
        dtypes = get_dtypes()
        shape4d = (2, nx, ny, nz)
        shape5d = (2, 3, nx, ny, nz)

        return cls(
            rho=jnp.zeros(shape4d, dtype=dtypes.float),
            tau=jnp.zeros(shape4d, dtype=dtypes.float),
            chi=jnp.zeros(shape4d, dtype=dtypes.float),
            current=jnp.zeros(shape5d, dtype=dtypes.float),
            sdens=jnp.zeros(shape5d, dtype=dtypes.float),
            sodens=jnp.zeros(shape5d, dtype=dtypes.float),
        )


def compute_densities(
    psi: jax.Array,
    wocc: jax.Array,
    wguv: jax.Array,
    pairwg: jax.Array,
    wstates: jax.Array,
    isospin: jax.Array,
    grid,
    chunk: int = 0,
    tensor: bool = False,
) -> Densities:
    """
    Compute all nuclear densities from wavefunctions.

    Uses fully vectorized operations for GPU efficiency.

    Args:
        psi: Wavefunctions array with shape (nstates, 2, nx, ny, nz)
        wocc: Occupation weights (BCS v^2) with shape (nstates,)
        wguv: Pairing weights (u*v) with shape (nstates,)
        pairwg: Pairing cutoff weights with shape (nstates,)
        isospin: Isospin indices (0=neutron, 1=proton) with shape (nstates,)
        grid: Grid object with dx, dy, dz
        chunk: If > 0, reduce over states in chunks of this size. Peak memory
            for the per-state intermediates then scales with `chunk` instead of
            the number of states, at a small cost in speed.
        tensor: also compute the symmetric spin-current tensor (Densities.jsym).

    Returns:
        Densities object containing all computed densities
    """
    return _compute_densities_vectorized(
        psi, wocc, wguv, pairwg, wstates, isospin,
        grid.dx, grid.dy, grid.dz, grid.nx, grid.ny, grid.nz,
        int(chunk), bool(tensor),
    )


@partial(jax.jit, static_argnums=(12, 13))
def _compute_densities_vectorized(
    psi: jax.Array,
    wocc: jax.Array,
    wguv: jax.Array,
    pairwg: jax.Array,
    wstates: jax.Array,
    isospin: jax.Array,
    dx: float,
    dy: float,
    dz: float,
    nx: int,
    ny: int,
    nz: int,
    chunk: int = 0,
    tensor: bool = False,
) -> Densities:
    """
    Vectorized density computation using vmap and segment reduction.
    """
    dtypes = get_dtypes()
    nstates = psi.shape[0]

    # Compute weights
    weights = wocc * wstates
    weightsuv = wguv * pairwg * wstates

    # Compute per-state contributions using vmap
    def single_state_densities(psi_n, weight, weightuv):
        """Compute density contributions from a single state (not accumulated yet)."""
        psi0 = psi_n[0]  # Spin-up
        psi1 = psi_n[1]  # Spin-down
        psi0_conj = jnp.conjugate(psi0)
        psi1_conj = jnp.conjugate(psi1)

        # Particle density contribution
        rho_contrib = weight * jnp.real(psi0_conj * psi0 + psi1_conj * psi1)

        # Pairing density contribution
        chi_contrib = 0.5 * weightuv * jnp.real(psi0_conj * psi0 + psi1_conj * psi1)

        # Spin density contributions
        sdens_x = 2.0 * weight * jnp.real(psi0_conj * psi1)
        sdens_y = 2.0 * weight * jnp.imag(psi0_conj * psi1)
        sdens_z = weight * jnp.real(psi0_conj * psi0 - psi1_conj * psi1)
        sdens_contrib = jnp.stack([sdens_x, sdens_y, sdens_z], axis=0)

        # Derivatives
        dpsi_dx = deriv_x(psi_n, dx)
        dpsi_dy = deriv_y(psi_n, dy)
        dpsi_dz = deriv_z(psi_n, dz)

        dpsi0_dx, dpsi1_dx = dpsi_dx[0], dpsi_dx[1]
        dpsi0_dy, dpsi1_dy = dpsi_dy[0], dpsi_dy[1]
        dpsi0_dz, dpsi1_dz = dpsi_dz[0], dpsi_dz[1]

        # Kinetic density
        tau_contrib = weight * jnp.real(
            jnp.conjugate(dpsi0_dx) * dpsi0_dx + jnp.conjugate(dpsi1_dx) * dpsi1_dx +
            jnp.conjugate(dpsi0_dy) * dpsi0_dy + jnp.conjugate(dpsi1_dy) * dpsi1_dy +
            jnp.conjugate(dpsi0_dz) * dpsi0_dz + jnp.conjugate(dpsi1_dz) * dpsi1_dz
        )

        # Current density
        current_x = weight * jnp.imag(psi0_conj * dpsi0_dx + psi1_conj * dpsi1_dx)
        current_y = weight * jnp.imag(psi0_conj * dpsi0_dy + psi1_conj * dpsi1_dy)
        current_z = weight * jnp.imag(psi0_conj * dpsi0_dz + psi1_conj * dpsi1_dz)
        current_contrib = jnp.stack([current_x, current_y, current_z], axis=0)

        # Spin-orbit density (matching legacy implementation)
        # J_x = Im(psi* sy dz psi - psi* sz dy psi)
        # J_y = Im(psi* sz dx psi - psi* sx dz psi)
        # J_z = Im(psi* sx dy psi - psi* sy dx psi)

        # sodens_x = sodens_x_y + sodens_x_z
        sodens_x_y = weight * jnp.imag(dpsi0_dy * psi0_conj - dpsi1_dy * psi1_conj)
        sodens_x_z = weight * jnp.real(dpsi1_dz * psi0_conj - dpsi0_dz * psi1_conj)
        sodens_x = sodens_x_y + sodens_x_z

        # sodens_y = sodens_y_x + sodens_y_z
        sodens_y_x = -weight * jnp.imag(dpsi0_dx * psi0_conj - dpsi1_dx * psi1_conj)
        sodens_y_z = weight * jnp.imag(dpsi1_dz * psi0_conj + dpsi0_dz * psi1_conj)
        sodens_y = sodens_y_x + sodens_y_z

        # sodens_z = sodens_z_x + sodens_z_y
        sodens_z_x = -weight * jnp.real(psi0 * jnp.conjugate(dpsi1_dx) - psi1 * jnp.conjugate(dpsi0_dx))
        sodens_z_y = -weight * jnp.imag(dpsi1_dy * psi0_conj + dpsi0_dy * psi1_conj)
        sodens_z = sodens_z_x + sodens_z_y

        sodens_contrib = jnp.stack([sodens_x, sodens_y, sodens_z], axis=0)

        out = (rho_contrib, tau_contrib, chi_contrib, current_contrib, sdens_contrib, sodens_contrib)
        if not tensor:
            return out

        def j_row(d0, d1):   # (J_mu,x, J_mu,y, J_mu,z) for one derivative direction mu
            return (jnp.imag(psi0_conj * d1 + psi1_conj * d0),
                    jnp.real(psi1_conj * d0 - psi0_conj * d1),
                    jnp.imag(psi0_conj * d0 - psi1_conj * d1))
        jx, jy, jz = j_row(dpsi0_dx, dpsi1_dx), j_row(dpsi0_dy, dpsi1_dy), j_row(dpsi0_dz, dpsi1_dz)
        jsym_contrib = weight * jnp.stack([jx[0], jy[1], jz[2], 0.5 * (jx[1] + jy[0]),
                                           0.5 * (jx[2] + jz[0]), 0.5 * (jy[2] + jz[1])])

        def spin(b0, b1):    # b^+ sigma b for a spinor (b0, b1)
            return jnp.stack([2.0 * jnp.real(jnp.conj(b0) * b1), 2.0 * jnp.imag(jnp.conj(b0) * b1),
                              jnp.abs(b0)**2 - jnp.abs(b1)**2])
        tspin_contrib = weight * (spin(dpsi0_dx, dpsi1_dx) + spin(dpsi0_dy, dpsi1_dy)
                                  + spin(dpsi0_dz, dpsi1_dz))
        return out + (jsym_contrib, tspin_contrib)

    def per_isospin(x, iso):
        """(nstates, ...) per-state contributions -> (2, ...) neutron/proton sums."""
        m = (iso == 0).reshape((-1,) + (1,) * (x.ndim - 1)).astype(x.dtype)
        return jnp.stack([jnp.sum(x * m, axis=0), jnp.sum(x * (1 - m), axis=0)])

    if chunk <= 0 or chunk >= nstates:
        dens = [per_isospin(x, isospin) for x in jax.vmap(single_state_densities)(psi, weights, weightsuv)]
    else:
        pad = (-nstates) % chunk

        def _pad(a):
            if pad == 0:
                return a
            return jnp.concatenate(
                [a, jnp.zeros((pad,) + a.shape[1:], dtype=a.dtype)], axis=0)

        psi_c = _pad(psi).reshape((-1, chunk) + psi.shape[1:])
        w_c = _pad(weights).reshape(-1, chunk)
        wuv_c = _pad(weightsuv).reshape(-1, chunk)
        # padded entries get isospin 0, harmless: their weights are zero
        iso_c = _pad(isospin).reshape(-1, chunk)

        shapes = jax.eval_shape(single_state_densities, psi[0], weights[0], weightsuv[0])
        zeros = tuple(jnp.zeros((2,) + x.shape, dtype=dtypes.float) for x in shapes)

        def _body(carry, xs):
            p, w, wuv, iq = xs
            c = jax.vmap(single_state_densities)(p, w, wuv)
            return tuple(t + per_isospin(x, iq) for t, x in zip(carry, c)), None

        dens, _ = jax.lax.scan(_body, zeros, (psi_c, w_c, wuv_c, iso_c))

    return Densities(*dens[:6], *(dens[6:] if tensor else (None, None)))
