# Validation

The checks behind the accuracy statements in the README. Grids are N³ points at spacing d (fm); "FD" is a central
finite difference of re-converged solver states.

## Two-basis HFB solver

Against HFBTHO, `examples/two_basis_benchmark/run.py` (A100; pairing strengths of the 8 MeV window tuned to the
HFBTHO gap of one nucleus per set):

| Set | Grid | E − E(HFBTHO) | Neutron gaps (ours / HFBTHO) | Time per nucleus |
|---|---|---|---|---|
| Ca42-46 + odd, UNEDF1 | 33³ @ 0.8 | −10 to −37 keV | 0.781/0.779, 0.869/0.869, 0.749/0.748 | 1.7-6.1 min |
| Ca40-44, UNEDF2 | 33³ @ 0.8 | 0 to −69 keV | 0.910/0.911, 1.012/1.014 | 2.2-3.5 min |
| Sn120, UNEDF1 | 33³ @ 0.8 | −264 keV | 0.834/0.823 | 7.2 min |
| Dy162-164 + odd, SLy4 (deformed) | 37³ @ 0.8 | −447 to −457 keV | 1.285/1.275, 1.218/1.217 | 6.9-27 min |

The offsets are constant along each chain (spread ≤ 69 keV), so separation energies and odd-even staggering agree
to tens of keV; ground-state Ω of the odd nuclei and β₂ (to 6×10⁻⁴) match. Without pairing the solver reproduces
the gradient solver's HF state (`tests/test_two_basis.py`).

**Constrained scans** (`run_constrained_scan` with `two_basis_config()`, exact-response Newton step): ²⁰Ne SLy4,
25³ @ 1.0, Q20 = 60 fm²: HF reaches the target to 6×10⁻⁶ in 4 outer steps (10⁻⁸ in 13); paired, to 7×10⁻⁷ in 8.
Derivatives at fixed Q20 by the constrained response
(`Response(..., constraint_potential='auto').derivatives(constraints=...)`): dE/dt0 and dE/dt3 equal FD to
1×10⁻⁸ (HF) and 3×10⁻⁵ (paired); the radius derivatives differ from FD by exactly the FD states' constraint miss
(the same dr/dQ for t0 and t3), i.e. the analytic values are the fixed-Q ones.

## Functionals

- `skyrme_ad` (autodiff fields) equals the hand-coded Skyrme mean fields to 10⁻⁹ (`tests/test_edf.py`).
- Fayans FyDr/FyStd/FyDrHFB total energies equal SkyAx's to 4-13 keV (¹⁶O, ⁴⁰Ca, ⁴⁸Ca, no pairing; 0.5 fm here,
  0.4 fm in SkyAx). At 1.0 fm they are off by 0.1-0.25 MeV: use 0.7 fm or finer.
- J² terms against HFBTHO (UNEDF2): 1 keV for deformed ²⁴Mg and ²⁸Si, 25 keV for ⁴⁰,⁴⁸Ca.

## Linear response (`physics/response.py`)

SLy4, HF, 35³ @ 1.0:

- ⁴⁸Ca isoscalar monopole strength equals TDHF (±kick, 4000 fm/c, same grid and ground state) to 0.85% of the
  peak; energy-weighted sum to 0.03%.
- Derivatives of E, r_n, r_p and the skin equal FD to 5-6 digits for every parameter tried; adjoint equals forward.
- α_D(⁴⁸Ca) = 2.43 fm³.

Paired (²⁰O, SLy4): moving-window derivatives of E, radii and pair densities equal FD to 10⁻⁷-3×10⁻⁴; the frozen
window is off by 0.5-2.6%. With the default `uv_floor` (u·v ≥ 10⁻³ on every orbital) the state is not stationary
(pair density off by 0.5%): use `uv_floor=0` for derivative work.

Cost (⁴⁸Ca, one GPU): ~17 s per parameter (forward), ~30 s per observable for all parameters (adjoint), ~10 min
for a 3000-step Lanczos chain.

## Second order (`physics/second_order.py`, `second_order_hfb.py`)

- Hessian polarizability equals the response operator's to 2×10⁻⁸ (⁴⁸Ca), 5×10⁻¹² (¹⁶O); dα_D/dθ equals FD of
  re-converged α_D for all ten Skyrme parameters to 10⁻⁸-4×10⁻⁵ (⁴⁸Ca, 237 s for all ten).
- Paired (²⁰O, mixed pairing, `uv_floor=0`, 19³ @ 1.0): dP/dθ of the r² polarizability equals FD of the QRPA value
  at re-solved states to 1.5×10⁻⁵-1.1×10⁻⁴; the variational (frozen-window) adjoint is off by 0.4-1.8%.
- On a 1 fm grid the centred HF state is a translation saddle (egg-box effect): second-order sensitivities carry
  ~0.5% grid error.

## TDHFB

Scheme checks (conservation, order of the step, continuity, the LISE TD-SLDA cross-check) are in
`examples/tdhfb/README.md`. Regularised pairing (`Force.pair_reg_ecut`), ²⁰O, 13³ @ 1.2, v0 = 300, quadrupole
kick, 20 fm/c:

| Pair channel | Mean neutron gap for basis tops 9 / 34 / 40 MeV | Continuity defect (n, p) / floor | Energy drift |
|---|---|---|---|
| bare contact | 1.16 / 1.38 / 1.63 MeV | at floor | 4.5×10⁻⁷ MeV |
| momentum cutoff `pair_kcut` 3 fm⁻¹ | — | 4.3×10⁻³, 8.8×10⁻³ (~10⁴ × floor) | conserved |
| regularised, E_c = basis top | 1.23 / 1.08 / 1.18 MeV | 4.5×10⁻⁷ / 4.1×10⁻⁷, 3.1×10⁻⁷ / 2.3×10⁻⁷ | 7×10⁻⁸ MeV |

¹²⁰Sn + ¹²⁰Sn (33×33×69 @ 0.8, 480 orbitals, regularised pairing): before contact the fragment separation follows
the point-Coulomb trajectory to 0.01 fm; until the neutron pairing collapses in contact, the energy drifts by
≤ 3×10⁻⁴ MeV, N is conserved to 10⁻⁸ and continuity stays at its floor (runs that re-separate before a collapse:
to 600 fm/c). After a collapse the pairing revives and grows without bound (to 3× the ground-state pairing energy
within ~100 fm/c) while N drifts by ~10⁻³, the same at dt 0.1 and 0.05: the scheme is not valid after a pairing
collapse (open problem).
