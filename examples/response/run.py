"""
QRPA strength, dipole polarizability and parameter sensitivities of 48Ca (SLy4, HF) from one linear-response
operator (physics/response.py). About 15 minutes on one GPU.

    python run.py [--steps 3000]

Prints the isoscalar monopole centroid and peak, alpha_D, and the exact derivatives of the neutron skin with
respect to every Skyrme parameter; writes ca48_isgmr.npz (E, S).
"""
import argparse
import jax.numpy as jnp
import numpy as np
from jax_hfbfft import Force
from jax_hfbfft.core.grid import Grid
from jax_hfbfft.physics.response import Response, energy_gradient
from jax_hfbfft.physics.solver import run_hfb, apply_cm_correction, two_basis_config

ap = argparse.ArgumentParser()
ap.add_argument('--steps', type=int, default=3000)
a = ap.parse_args()
Z, N = 20, 28
g = Grid.create_cubic(n=35, d=1.0)
force = apply_cm_correction(Force.from_name("SLy4").with_pairing(ipair=0), Z + N)
state = run_hfb(g, force, nucleus_z=Z, nucleus_n=N, npsi_n=34, config=two_basis_config(etol=1e-7))
print(f"E = {float(state.energies.ehfint):.4f} MeV")
R = Response(state, force, g)
X, Y, Zc = g.get_meshgrid()
r2, w, rho = X ** 2 + Y ** 2 + Zc ** 2, g.wxyz, state.densities.rho

# isoscalar monopole strength (Gamma = 1 MeV) from one parity-even Lanczos chain
chain = R.lanczos(jnp.stack([r2, r2]), steps=a.steps, parity=1, verbose=False)
E = np.linspace(0.05, 40, 1600)
S = Response.spectrum(chain, E, gamma=1.0)
print(f"ISGMR: peak {E[np.argmax(S)]:.2f} MeV, m1/m0 {np.trapezoid(E * S, E) / np.trapezoid(S, E):.2f} MeV")
np.savez("ca48_isgmr.npz", E=E, S=S)

# electric dipole polarizability, c.m. held fixed
m, _ = R.polarizability(jnp.stack([-(Z / (Z + N)) * Zc, (N / (Z + N)) * Zc]), constraints=(jnp.stack([Zc, Zc]),))
print(f"alpha_D = {2 * 1.4399784 * m:.3f} fm^3")

# exact sensitivities of the neutron skin (one adjoint solve) and of the energy (Hellmann-Feynman)
Nq = [float(w * jnp.sum(rho[q])) for q in (0, 1)]
rq = [float(jnp.sqrt(w * jnp.sum(r2 * rho[q]) / Nq[q])) for q in (0, 1)]
skin = R.sensitivities({'skin': jnp.stack([r2 / (2 * rq[0] * Nq[0]), -r2 / (2 * rq[1] * Nq[1])])}, verbose=False)['skin']
dE = energy_gradient(state, force, g)
print(f"skin {rq[0] - rq[1]:.4f} fm")
for k in dE:
    print(f"  d/d{k:7s}  skin {skin[k]:+.4e} fm   E {dE[k]:+.4e} MeV")
