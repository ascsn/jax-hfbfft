"""
Two-basis HFB against HFBTHO (reference.json): energies, gaps, deformations, S2n / Delta3 and wall times.

    python run.py SET [--fast]        SET: ca_unedf1, sn120_unedf1, ca_unedf2, dy_sly4 (or all)

--fast uses a 1.0 fm grid of about the same box (absolute energies shift by ~20-30 keV, separation
energies by a few keV) and, for SLy4, density mixing 0.4. Even chains are warm-started from their
lighter neighbour; odd nuclei run every EFA candidate within the given quasiparticle window on the
even core. Writes results_SET[_fast].json next to this file. Pairing strengths of the two-basis
model (window 8 MeV) are not HFBTHO's: they were tuned to its gap of one nucleus per set.
"""
import dataclasses, json, os, sys, time

import jax

from jax_hfbfft import Force
from jax_hfbfft.core.grid import Grid
from jax_hfbfft.physics.blocking import run_odd
from jax_hfbfft.physics.energies import compute_radii
from jax_hfbfft.physics.solver import run_hfb, apply_cm_correction, two_basis_config

HERE = os.path.dirname(os.path.abspath(__file__))
SYMBOL = {'Ca': 20, 'Sn': 50, 'Dy': 66}


def nucleus(name):
    sym = name.rstrip('0123456789')
    Z = SYMBOL[sym]
    return Z, int(name[len(sym):]) - Z


def run_set(name, spec, fast):
    nx, dx = spec['grid']
    if fast:
        nx, dx = int(round(nx * dx)) | 1, 1.0
    g = Grid.create(nx=nx, ny=nx, nz=nx, dx=dx, dy=dx, dz=dx)
    v0p = spec.get('v0p', spec['v0n'] * spec.get('v0p_ratio', 1.0))
    force = lambda A: apply_cm_correction(Force.from_name(spec['functional']).with_pairing(
        ipair=6, v0neut=spec['v0n'], v0prot=v0p, rho0pr=0.32, pair_cutoff=(0.0, 0.0), state_cutoff=(8.0, 8.0)), A)
    mixing = 0.4 if fast and spec['functional'] == 'SLy4' else spec['mixing']
    cold = two_basis_config(density_mixing=mixing, axial_symmetry=spec.get('axial', False),
                            initial_beta2=spec.get('initial_beta2', 0.0))
    warm = dataclasses.replace(cold, bcs_start=0, diag_start=0, iteranneal=0)
    res, states = {}, {}

    def record(key, st, t0, **extra):
        r = compute_radii(st.densities, g)
        res[key] = dict(E=float(st.energies.ehfint), gap_n=float(st.pairing.avdeltv2[0]), gap_p=float(st.pairing.avdeltv2[1]),
                        beta2=float(r.beta2), iters=int(st.iteration), converged=bool(st.converged),
                        minutes=round((time.time() - t0) / 60, 2), **extra)
        print(key, json.dumps(res[key]), flush=True)

    prev = None
    for key in spec['chain']:
        Z, N = nucleus(key)
        t0 = time.time()
        if prev is not None and float(prev.pairing.avdeltv2[0]) < 0.05:
            prev = None                  # no warm start from a collapsed gap: it can stay collapsed
        st = run_hfb(g, force(Z + N), nucleus_z=Z, nucleus_n=N, npsi_n=spec['npsi'],
                     config=cold if prev is None else warm, initial_state=prev)
        record(key, st, t0)
        states[key] = prev = st
    for key, (core, window) in spec.get('odd', {}).items():
        Z, N = nucleus(key)
        t0 = time.time()
        _, cands = run_odd(g, force(Z + N), Z, N, spec['npsi'], config=warm, window=window,
                           core=states[core], axial=spec.get('axial', False), verbose=False)
        record(key, cands[0]['state'], t0, omega=cands[0]['omega'], candidates=len(cands))
    return res, dict(grid=[nx, dx], mixing=mixing, gpu=jax.devices()[0].device_kind)


def report(spec, res):
    ref = spec['hfbtho']
    print(f"{'':6s} {'E':>11s} {'dE keV':>8s} {'gap_n':>7s} {'d':>7s} {'beta2':>7s} {'d':>8s} {'it':>5s} {'min':>6s} {'THO min':>7s}")
    for k, r in res.items():
        t = ref[k]
        d = lambda f: f"{r[f] - t[f]:+7.3f}" if f in t else f"{'':>7s}"
        print(f"{k:6s} {r['E']:11.4f} {1e3 * (r['E'] - t['E']):+8.1f} {r['gap_n']:7.3f} {d('gap_n')} {r['beta2']:7.4f} "
              f"{d('beta2'):>8s} {r['iters']:5d} {r['minutes']:6.2f} {t.get('time', float('nan')):7.2f}"
              + ("" if r['converged'] else "  NOT CONVERGED"))
    A = {nucleus(k)[1]: k for k in res}
    for N in sorted(A):
        if N - 2 in A and N % 2 == 0:
            ff, th = res[A[N - 2]]['E'] - res[A[N]]['E'], ref[A[N - 2]]['E'] - ref[A[N]]['E']
            print(f"S2n({A[N]}) {ff:8.4f}  HFBTHO {th:8.4f}  diff {1e3 * (ff - th):+6.1f} keV")
        if N % 2 and N - 1 in A and N + 1 in A:
            d3 = lambda E: -0.5 * (E[A[N - 1]]['E'] - 2 * E[A[N]]['E'] + E[A[N + 1]]['E'])
            print(f"Delta3({A[N]}) {d3(res):7.4f}  HFBTHO {d3(ref):7.4f}  diff {1e3 * (d3(res) - d3(ref)):+6.1f} keV")


if __name__ == '__main__':
    sets = json.load(open(os.path.join(HERE, 'reference.json')))['sets']
    fast = '--fast' in sys.argv
    names = list(sets) if sys.argv[1] == 'all' else [sys.argv[1]]
    for name in names:
        print(f"===== {name}{' (fast)' if fast else ''}", flush=True)
        res, meta = run_set(name, sets[name], fast)
        report(sets[name], res)
        json.dump(dict(meta=meta, results=res), open(os.path.join(HERE, f"results_{name}{'_fast' if fast else ''}.json"), 'w'), indent=1)
