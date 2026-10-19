"""
Charge and weak form factors of a static solution, orientation averaged
(F(q) = Int rho(r) j0(q r) / Int rho, the monopole part for a deformed nucleus).

The point densities are folded with nucleon form factors whose slopes give the
charge radius of compute_radii, r_ch^2 = r_p^2 + <r^2>_P + (N/Z) <r^2>_N + DF:
a dipole G_E^p of radius 0.8409 fm, the Galster G_E^n with slope -<r^2>_N / 6
(<r^2>_N = -0.1155 fm^2), and the Darwin-Foldy term on the protons. The weak
charges q_p = 0.0721, q_n = -0.9878 and nucleon weak form factors
G^Z_p = q_p G_E^p + q_n G_E^n, G^Z_n = q_n G_E^p + q_p G_E^n follow Horowitz
et al., PRC 85, 032501 (2012). Spin-orbit and strange-quark contributions are
left out.
"""

import jax
import jax.numpy as jnp
import numpy as np

HBARC, M_N = 197.3269804, 938.919          # MeV fm, MeV
R_P2, R_N2, DARWIN_FOLDY = 0.8409**2, -0.1155, 0.0332   # fm^2
Q_P, Q_N = 0.0721, -0.9878


def nucleon_form_factors(q):
    """Electric form factors (G_E^p, G_E^n) at momentum transfer q (fm^-1)."""
    q2 = jnp.asarray(q) ** 2
    tau = q2 * HBARC**2 / (4.0 * M_N**2)
    dipole = (1.0 + q2 * HBARC**2 / 0.71e6) ** -2
    return (1.0 + q2 * R_P2 / 12.0) ** -2, -R_N2 / 6.0 * q2 * dipole / (1.0 + 5.6 * tau)


def form_factors(densities, grid, q) -> dict:
    """
    Point (F_p, F_n), charge (F_ch) and weak (F_W) form factors at the momentum
    transfers q (fm^-1), each 1 at q = 0, with the radii their slopes give
    (R_ch, R_W, about the origin as in compute_radii) and the skins R_n - R_p,
    R_W - R_ch.
    """
    q = jnp.atleast_1d(jnp.asarray(q, dtype=float))
    X, Y, Z = grid.get_meshgrid()
    r = jnp.sqrt(X**2 + Y**2 + Z**2)
    rho_n, rho_p = densities.rho
    F = jax.vmap(lambda k: grid.wxyz * jnp.stack([jnp.sum(rho_n * jnp.sinc(k * r / jnp.pi)),
                                                   jnp.sum(rho_p * jnp.sinc(k * r / jnp.pi))]))(q)
    Fn, Fp = F[:, 0], F[:, 1]                                  # F_q(0) = N, Z
    N, Zp = (float(grid.wxyz * jnp.sum(x)) for x in (rho_n, rho_p))
    Gp, Gn = nucleon_form_factors(q)
    df = jnp.exp(-q**2 * DARWIN_FOLDY / 6.0)
    F_ch = (Gp * df * Fp + Gn * Fn) / Zp
    Q_W = Zp * Q_P + N * Q_N
    F_W = ((Q_P * Gp + Q_N * Gn) * Fp + (Q_N * Gp + Q_P * Gn) * Fn) / Q_W
    r2 = r**2
    Rp2, Rn2 = (float(grid.wxyz * jnp.sum(x * r2)) / n for x, n in ((rho_p, Zp), (rho_n, N)))
    R_ch = np.sqrt(Rp2 + R_P2 + N / Zp * R_N2 + DARWIN_FOLDY)
    R_W = np.sqrt((Zp * Q_P * (Rp2 + R_P2) + Zp * Q_N * R_N2
                   + N * Q_N * (Rn2 + R_P2) + N * Q_P * R_N2) / Q_W)
    return dict(q=np.asarray(q), F_p=np.asarray(Fp / Zp), F_n=np.asarray(Fn / N), F_ch=np.asarray(F_ch),
                F_W=np.asarray(F_W), R_ch=float(R_ch), R_W=float(R_W),
                skin=float(np.sqrt(Rn2) - np.sqrt(Rp2)), weak_skin=float(R_W - R_ch))
