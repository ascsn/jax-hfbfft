# Changelog

## Unreleased

- Finite temperature: thermal BCS occupations, entropy and free energy
  (`temperature`; zero pairing strength gives finite-temperature HF).
- Neck constraint `QN` (Gaussian slab of width `neck_width` at `neck_z`).
- ATDHFB cranking inertia, GCM metric and zero-point energy
  (`physics.cranking_inertia`), with the continuum summed by Sternheimer solves;
  validated against HFBTHO; Inglis-Belyaev moments of inertia
  (`physics.moment_of_inertia`). `physics.multipole_operator` gives the
  constraint operators on the grid.
  The inertia sums diagonalise h within groups of (nearly) equal occupation and
  leave orbitals with v² < 1e-4 to the complement: with weak pairing the solver
  keeps mixed occupied and unconverged empty orbitals, and the moment of inertia
  of ²⁰Ne came out 2.5x too small (2.20 instead of 5.69 ħ²/MeV).
- Particle-number projection after variation (`HFBFFT.project_particle_number`,
  `physics.project_particle_number`), mixed-density prescription; the energy
  functional accepts complex (gauge-rotated) densities.
- Charge and weak form factors, weak radius and skins (`physics.form_factors`).
- The results report Q30, Q40, β₃ and β₄ (along z about the centre of mass).
- Minimum-energy and least-action paths on a deformation surface, with the
  barrier, saddle, action and penetrability (`physics.minimum_energy_path`,
  `physics.least_action_path`).
- A constrained `HFBFFT.run()` warns when a driven moment ends more than 5% off
  its goal (the in-loop multiplier update can stall).
- `run_constrained_scan` drives several multipoles at once (deformation surfaces,
  e.g. Q20 x Q30), with a diagonal secant on the multiplier vector; 1D scans are
  unchanged.
- `run_chain` / `separation_energies` for chains of nuclei with warm starts,
  and `examples/isotopic_chain.py`.
- TDHFB isovector dipole kick and boost; the isovector dipole moment `Dz_iv`
  is recorded by TDHF and TDHFB, and `examples/tdhf/oscillations.py --mode gdr`
  now records it (it recorded Q20).
- Energy-weighted sum rules `ewsr` (with the Skyrme TRK enhancement), printed by
  the CLI for kicked runs.
- `TDHF.strength()` / `TDHFB.strength()`: the strength function of the last kick,
  with Lorentzian damping (`gamma`) and the ±η symmetrization (`minus=`).
  `dynamics.strength_function` returned −ħc times the strength for TDHF kicks
  (sign and a missing 1/ħc; `strength_moments` hid the sign by taking |S|).
- J² (tensor) terms of the full spin-current tensor J_mu,nu with their time-odd
  partner s·T (`j2:` in the force file): SLy5 and SkP now carry the J² terms of
  their central force, and UNEDF2 is available. Validated against HFBTHO (UNEDF2
  ²⁴Mg, ²⁸Si, ⁴⁰,⁴⁸Ca); Galilean invariance checked.
- β₂ in the results was the quadrupole component along the longest axis, half
  the Bohr β for an axial oblate shape (and too small for any triaxial one); it
  is now β = √(β₂₀² + 2β₂₂²), consistent with the reported γ.
- A static run with as many orbitals as particles (e.g. ¹⁶O HF with 8 + 8, the
  default basis without pairing) stopped at iteration 1 and returned the initial
  guess whenever the convergence threshold was finite: equal occupations make the Lagrange
  asymmetry, the convergence measure, vanish identically. Such blocks now measure
  |(1 − P) h ψ|; runs with unoccupied orbitals are unchanged.
- CLI: an `analysis:` block (particle-number projection, form factors, written to
  `analysis.res`), and kicked TDHF/TDHFB runs write `strength.dat` and print m1
  against the EWSR (`dynamics.strength_gamma`, `strength_emax`).
- The CLI reads `physics: include_coulomb` (it was ignored); the unused
  `time_reversal` and `time_odd` options were removed from `config.yml`.

## 2026-10 (canonical-tdhfb)

### Corrected force parameters

Results computed with the affected forces before this release should be redone.

- **SLy5, SLy6, SkM, SkM\*, SkP, SIII**: the spin-orbit strength was stored as
  `t4 = 0, b4p = W0` instead of `t4 = W0, b4p = W0/2`, which gave
  C^∇J₀ = −W0/2 (correct: −3W0/4) and doubled C^∇J₁. SLy4 was correct.
- **UNEDF0**: t2, the density exponent, t4 and b4p were wrong; all parameters now
  follow HFBTHO's UNE0.
- **UNEDF1**: t4, b4p and the density exponent were wrong, and `zpe` was 0
  (HFBTHO's UNE1 has no c.m. correction, now `zpe: 2`).
- **SLy6** now uses `zpe: 3` (it was fitted with the full two-body c.m.
  correction). SkP parameters are given to full precision.
- `tests/test_forces.py` checks every entry against independent values.

### Statics

- Microscopic centre-of-mass correction ⟨P²⟩/2mA (`zpe: 3`).
- Lipkin-Nogami particle-number correction (`use_ln`), validated against
  HFBTHO+LN. Its λ₂ is the self-consistent root of λ₂ = F(λ₂) with exact LN
  occupations; earlier development versions had the sign of λ₂ inverted,
  summed the traces over both Kramers partners, iterated the occupations by a
  divergent substitution, and missed the open-shell root above F(0).
- The BCS occupations are floored at 1e-10 (u·v at √uv_floor) as in the
  FORTRAN code. The previous 1e-6 floor on v² added up to ~1e-4 particles from
  the empty states.
- The constraint damping radius defaults to a value scaled with the nucleus and
  the requested deformation (it was a fixed 6 fm, which cut into heavy nuclei).
- Checkpoint and restart of static runs (`save_checkpoint`, `load_checkpoint`,
  `initialize_wavefunctions("restart")`, `run(checkpoint_file=...)`); a second
  `run()` now resumes with the current pairing amplitudes.
- Charge radius with current proton, neutron and Darwin-Foldy terms (it used
  √(r_p² + 0.64)).
- The results files report per-isospin pairing energies, both gap definitions,
  λ₂, E_LN and the c.m. treatment.

### Dynamics

- TDHFB from the CLI (`dynamics.method: tdhfb`), `TDHFB.relax()`, and warnings
  when a static with Lipkin-Nogami or `zpe: 3` seeds TDHF/TDHFB.
- Second-order TDHFB step (Strang splitting with a midpoint field), absorbing
  boundaries, monopole kicks, fission and fragment tools, frame recording and a
  3D viewer.
- Eigendecompositions in the TDHFB step are checked and fall back to host
  LAPACK when the GPU result is not finite or not orthonormal (seen on the
  degenerate matrices of a collapsed-pairing state).
