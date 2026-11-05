#!/usr/bin/env python3
"""
Analyses of a converged static solution: particle-number projection, charge and
weak form factors, deformations, and the Inglis-Belyaev moment of inertia.

    python static_analysis.py           # 20Ne (prolate), SLy4 + DDDI pairing
"""

from jax_hfbfft import HFBFFT, Force, Grid, Nucleus
from jax_hfbfft.physics import form_factors, moment_of_inertia


def main():
    force = Force.from_name("SLy4").with_pairing(ipair=6, v0neut=350.0, v0prot=350.0, rho0pr=0.32,
                                                 pair_cutoff=(0.0, 0.0), state_cutoff=(0.0, 0.0))
    calc = HFBFFT(nucleus=Nucleus(protons=10, neutrons=10), force=force,
                  grid=Grid.create_cubic(n=20, d=1.0), npsi=(20, 20))
    r = calc.run(max_iterations=1500, convergence_threshold=1e-6, print_interval=250)
    print(f"\nE = {r.total_energy:.4f} MeV, gap_n = {r.pairing_gap_n:.3f} MeV, "
          f"beta2 = {r.beta2:.3f}, beta3 = {r.beta3:.3f}, beta4 = {r.beta4:.3f}")

    pav = calc.project_particle_number(n_gauge=11)
    print(f"Projected: E = {pav['total_energy']:.4f} MeV (correction {pav['correction']:+.4f} MeV)")
    print("  P(N) = " + ", ".join(f"{n}: {p:.3f}" for n, p in pav['distribution']['n'].items() if p > 1e-3))

    ff = form_factors(calc.solver_state.densities, calc.grid, [0.5, 1.0, 1.5])
    print(f"R_ch = {ff['R_ch']:.4f} fm, R_W = {ff['R_W']:.4f} fm, neutron skin {ff['skin']:.4f} fm")
    for q, fch, fw in zip(ff['q'], ff['F_ch'], ff['F_W']):
        print(f"  q = {q:.2f} fm^-1: F_ch = {fch:+.5f}, F_W = {fw:+.5f}")

    for axis in 'xyz':     # zero about the symmetry axis of an axial shape
        J = moment_of_inertia(calc.solver_state, calc.grid, axis)
        print(f"Moment of inertia J_{axis} = {J['J']:.3f} hbar^2/MeV (rigid body {J['J_rigid']:.3f})")


if __name__ == "__main__":
    main()
