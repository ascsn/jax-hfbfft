# Changelog

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
