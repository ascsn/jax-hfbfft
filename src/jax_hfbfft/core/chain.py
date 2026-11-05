"""
Chains of nuclei (isotopes, isotones, mass tables) and their separation energies.

    from jax_hfbfft import Force, Grid
    from jax_hfbfft.core.chain import run_chain, separation_energies

    force = Force.from_name("SLy4").with_pairing(ipair=6, v0neut=362.0, v0prot=362.0)
    points = run_chain([(20, n) for n in range(20, 30, 2)], force, Grid.create_cubic(n=32, d=0.8))
    for row in separation_energies(points):
        print(row)

Every nucleus uses the same basis size, so each one starts from the converged
orbitals of the previous one (`warm_start`), which typically saves most of the
iterations along a chain.
"""

import dataclasses
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from jax_hfbfft.core.force import Force
from jax_hfbfft.core.grid import Grid
from jax_hfbfft.core.hfbfft import HFBFFT, default_npsi
from jax_hfbfft.core.nucleus import Nucleus


@dataclasses.dataclass
class ChainPoint:
    """Ground-state properties of one nucleus of a chain (energies in MeV, radii in fm).
    E is the integrated energy functional (ehfint)."""
    Z: int
    N: int
    E: float
    converged: bool
    efluct: float
    gap_n: float
    gap_p: float
    lambda2_n: float
    lambda2_p: float
    r_ch: float
    r_n: float
    r_p: float
    beta2: float
    gamma: float

    @property
    def A(self) -> int:
        return self.Z + self.N


def chain_basis(nuclei: Sequence[Tuple[int, int]], force: Force) -> Tuple[int, int]:
    """The largest default basis (npsi_n, npsi_p) over the chain."""
    sizes = [default_npsi(Nucleus(protons=z, neutrons=n), force) for z, n in nuclei]
    return max(s[0] for s in sizes), max(s[1] for s in sizes)


def run_chain(
    nuclei: Sequence[Tuple[int, int]],
    force: Force,
    grid: Grid,
    npsi: Optional[Tuple[int, int]] = None,
    max_iterations: int = 1000,
    convergence_threshold: float = 1e-6,
    warm_start: bool = True,
    checkpoint_dir: Optional[str] = None,
    configure: Optional[Callable[[HFBFFT], None]] = None,
) -> List[ChainPoint]:
    """
    Solve each (Z, N) of `nuclei` in order and return their ChainPoints.

    Args:
        nuclei: (Z, N) pairs, ordered so neighbours are similar (e.g. along N).
        force, grid: shared by every nucleus.
        npsi: basis size for the whole chain; defaults to chain_basis().
        warm_start: start each nucleus from the previous one's orbitals.
        checkpoint_dir: save each solution as <dir>/Z<Z>_N<N>.npz (HFBFFT.save_checkpoint).
        configure: called on every HFBFFT before it runs, to set solver options
            (e.g. lambda c: setattr(c, 'use_ln', True)).
    """
    npsi = npsi or chain_basis(nuclei, force)
    points, prev = [], None
    for z, n in nuclei:
        calc = HFBFFT(Nucleus(protons=z, neutrons=n), force=force, grid=grid, npsi=npsi)
        calc.diag_start = calc.bcs_start = 30
        calc.tvaryx_0 = True
        if configure is not None:
            configure(calc)
        if warm_start and prev is not None:
            calc.state.psi = prev.state.psi            # occupations stay those of (Z, N)
        else:
            calc.initialize_wavefunctions(method="harmonic_oscillator")
        r = calc.run(max_iterations=max_iterations, convergence_threshold=convergence_threshold,
                     print_interval=max(max_iterations // 4, 1), sinfo_interval=10**9)
        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)
            calc.save_checkpoint(os.path.join(checkpoint_dir, f"Z{z}_N{n}.npz"))
        points.append(ChainPoint(
            Z=z, N=n, E=float(calc.solver_state.energies.ehfint), converged=r.converged, efluct=r.final_fluctuation,
            gap_n=r.pairing_gap_n, gap_p=r.pairing_gap_p, lambda2_n=r.lambda2_n,
            lambda2_p=r.lambda2_p, r_ch=r.charge_radius, r_n=r.rms_radius_n,
            r_p=r.rms_radius_p, beta2=r.beta2, gamma=r.gamma))
        prev = calc
    return points


def separation_energies(points: Sequence[ChainPoint]) -> List[Dict[str, float]]:
    """
    Per nucleus: binding energy B = -E, and the two-nucleon separation energies
    S2n = E(Z, N-2) - E(Z, N) and S2p = E(Z-2, N) - E(Z, N) where the neighbour
    is in `points` (NaN otherwise).
    """
    E = {(p.Z, p.N): p.E for p in points}
    nan = float('nan')
    return [dict(Z=p.Z, N=p.N, A=p.A, B=-p.E,
                 S2n=E[(p.Z, p.N - 2)] - p.E if (p.Z, p.N - 2) in E else nan,
                 S2p=E[(p.Z - 2, p.N)] - p.E if (p.Z - 2, p.N) in E else nan)
            for p in points]
