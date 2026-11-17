"""
Pairing calculations for HFB.

This module implements BCS pairing with:
- Volume-delta interaction (VDI)
- Density-dependent delta interaction (DDDI)
- Fermi energy search via Brent's method
- Soft cutoff functions for pairing windows
- Lipkin-Nogami approximate particle-number projection
- Finite temperature (thermal BCS): quasiparticles occupied with f = 1/(1 + exp(E/T))
"""

import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Tuple, Dict, Any

from jax_hfbfft.jax_config import get_dtypes
from jax_hfbfft.core.force import Force


@jax.tree_util.register_dataclass
@dataclass
class Pairing:
    """
    Container for pairing results.

    Attributes:
        eferm: Fermi energy per isospin (2,)
        epair: Pairing energy per isospin (2,)
        avdelt: Average pairing gap weighted by uv (2,)
        avdeltv2: Average pairing gap weighted by v^2 (2,)
        avg: Average pairing force (2,)
    """
    eferm: jax.Array    # Fermi energies
    epair: jax.Array    # Pairing energies
    avdelt: jax.Array   # <uv*Delta>/<uv>
    avdeltv2: jax.Array # <v^2*Delta>/<v^2>
    avg: jax.Array      # Average force
    lambda2: jax.Array  # Lipkin-Nogami lambda_2 per isospin (zero if LN off)
    e_ln: jax.Array     # Lipkin-Nogami energy correction per isospin
    entropy: jax.Array  # Quasiparticle entropy per isospin (zero at T = 0)

    @classmethod
    def zeros(cls) -> "Pairing":
        """Create zero-initialized pairing data."""
        dtypes = get_dtypes()
        return cls(
            eferm=jnp.zeros(2, dtype=dtypes.float),
            epair=jnp.zeros(2, dtype=dtypes.float),
            avdelt=jnp.zeros(2, dtype=dtypes.float),
            avdeltv2=jnp.zeros(2, dtype=dtypes.float),
            avg=jnp.zeros(2, dtype=dtypes.float),
            lambda2=jnp.zeros(2, dtype=dtypes.float),
            e_ln=jnp.zeros(2, dtype=dtypes.float),
            entropy=jnp.zeros(2, dtype=dtypes.float),
        )


def thermal_factor(equasi: jax.Array, temperature) -> jax.Array:
    """1 - 2 f(E) = tanh(E / 2T) for quasiparticle energies E; exactly 1 at T = 0."""
    t = jnp.maximum(temperature, 1e-12)
    return jnp.where(temperature > 0.0, jnp.tanh(equasi / (2.0 * t)), 1.0)


def qp_entropy(equasi: jax.Array, wstates: jax.Array, temperature) -> jax.Array:
    """S = -sum_k w_k [f ln f + (1 - f) ln(1 - f)], f = 1 / (1 + exp(E_k / T)); zero at T = 0."""
    x = equasi / jnp.maximum(temperature, 1e-12)
    # f ln f + (1-f) ln(1-f) = -(log(1 + e^-x) + x f), stable for large x
    f = jax.nn.sigmoid(-x)
    s = jnp.sum(wstates * (jnp.log1p(jnp.exp(-x)) + x * f))
    return jnp.where(temperature > 0.0, s, 0.0)


@jax.jit
def find_fermi_energy_jax(
    target_n: float,
    sp_energy: jax.Array,
    deltaf: jax.Array,
    wstates: jax.Array,
    emin: float = -200.0,
    emax: float = 200.0,
    max_iter: int = 40,
    temperature: float = 0.0,
) -> float:
    """
    Find Fermi energy using JAX-native bisection search.

    This is much faster than scipy's brentq as it avoids CPU-GPU synchronization.
    """
    def body_fun(state):
        low, high, i = state
        mid = (low + high) / 2.0
        n = bcs_occupation(mid, sp_energy, deltaf, wstates, temperature)
        low = jnp.where(n < target_n, mid, low)
        high = jnp.where(n >= target_n, mid, high)
        return low, high, i + 1

    # Initial brackets - check if expansion is needed
    f_low = bcs_occupation(emin, sp_energy, deltaf, wstates, temperature) - target_n
    f_high = bcs_occupation(emax, sp_energy, deltaf, wstates, temperature) - target_n

    # Simple bracket expansion
    emin = jnp.where(f_low > 0, emin - 100.0, emin)
    emax = jnp.where(f_high < 0, emax + 100.0, emax)

    # Bisection loop
    low, high, _ = jax.lax.while_loop(
        lambda s: s[2] < max_iter,
        body_fun,
        (emin, emax, 0)
    )

    return (low + high) / 2.0


def bcs_occupation(
    eferm: jax.Array,
    sp_energy: jax.Array,
    deltaf: jax.Array,
    wstates: jax.Array,
    temperature: float = 0.0,
) -> jax.Array:
    """
    Calculate particle number for given Fermi energy.

    BCS occupation: v^2 = 0.5 * (1 - (e - ef) / E * tanh(E / 2T)), E = sqrt((e - ef)^2 + Delta^2)
    (tanh -> 1 at T = 0).
    """
    edif = sp_energy - eferm
    equasi = jnp.sqrt(edif**2 + deltaf**2)
    equasi_safe = jnp.maximum(equasi, 1.0e-20)

    # BCS v^2
    v2 = 0.5 * (1.0 - edif / equasi_safe * thermal_factor(equasi, temperature))
    v2 = jnp.clip(v2, 1.0e-10, 1.0 - 1.0e-10)

    return jnp.sum(v2 * wstates)


def find_fermi_energy(
    target_n: float,
    sp_energy: jax.Array,
    deltaf: jax.Array,
    wstates: jax.Array,
    emin: float = -200.0,
    emax: float = 200.0,
    temperature: float = 0.0,
) -> float:
    """
    Find Fermi energy that gives target particle number.

    Args:
        target_n: Target particle number (N or Z)
        sp_energy: Single-particle energies
        deltaf: Pairing gaps
        wstates: State degeneracy weights
        emin, emax: Search bounds
        temperature: T in MeV (thermal occupations)

    Returns:
        Fermi energy
    """
    return find_fermi_energy_jax(target_n, sp_energy, deltaf, wstates, emin, emax,
                                 temperature=temperature)


def soft_cutoff(energy: jax.Array, cutoff: float, width: float) -> jax.Array:
    """
    Fermi-function soft cutoff.

    f(e) = 1 / (1 + exp((e - cutoff) / width))

    Args:
        energy: Single-particle energies
        cutoff: Cutoff energy
        width: Cutoff width

    Returns:
        Cutoff factors in [0, 1]
    """
    return 1.0 / (1.0 + jnp.exp((energy - cutoff) / width))


def compute_bcs_occupations(
    sp_energy: jax.Array,
    deltaf: jax.Array,
    eferm: float,
) -> Tuple[jax.Array, jax.Array]:
    """
    Compute BCS occupation factors v^2 and uv.

    Args:
        sp_energy: Single-particle energies
        deltaf: Pairing gaps
        eferm: Fermi energy

    Returns:
        (v2, uv) - occupation and anomalous occupation
    """
    edif = sp_energy - eferm
    equasi = jnp.sqrt(edif**2 + deltaf**2)
    equasi_safe = jnp.maximum(equasi, 1.0e-20)

    v2 = 0.5 * (1.0 - edif / equasi_safe)
    v2 = jnp.clip(v2, 1.0e-10, 1.0 - 1.0e-10)

    # uv = sqrt(v^2 * (1 - v^2))
    uv = jnp.sqrt(jnp.maximum(v2 * (1.0 - v2), 1.0e-10))

    return v2, uv


def compute_lambda2(
    v2: jax.Array,
    wstates: jax.Array,
    avdelt: jax.Array,
    epair: jax.Array,
) -> jax.Array:
    """
    Lipkin-Nogami lambda_2 coefficient.

    Transcribed from the reference implementation in HFBTHO's
    `hfbtho_lipkin.f90::tracesln`, which computes

        lambda_2 = -G_eff * SNtor / SDtor
        SNtor    =  8 * ( S_u3v1 * S_u1v3 - S_u4v4 )
        SDtor    = 32 * ( S_u2v2^2         - S_u4v4 )
        G_eff    = <Delta>^2 / E_pair

    with the sums running over paired states.  The leading minus is dropped
    because HFBTHO's E_pair is negative and this code's is positive (see the
    sign note below).  SNtor/SDtor reduces to
    (1/4)(Su3v1*Su1v3 - Su4v4)/(Su2v2^2 - Su4v4), matching the closed form in
    that file's doc comment.  The expression is invariant under u <-> v, so
    the u/v naming inversion inside the reference routine is immaterial.

    The degenerate case SDtor -> 0 (vanishing pairing) is handled by returning
    zero rather than the reference's logarithmic branch: with no pairing there
    is no particle-number fluctuation to correct, so lambda_2 = 0 is both the
    physical limit and numerically safe.  HFBTHO's extra branch exists to keep
    a Broyden iteration alive through transient collapse; the same guard here
    would need its own validation, so it is deliberately not reproduced.

    Args:
        v2: Occupation probabilities v^2 (nstates,)
        wstates: State weights (nstates,)
        avdelt: Average pairing gap for this isospin (scalar)
        epair: Pairing energy for this isospin (scalar)

    Returns:
        lambda_2 in MeV, positive; zero when pairing has collapsed.
    """
    v2c = jnp.clip(v2, 0.0, 1.0)
    u = jnp.sqrt(jnp.maximum(1.0 - v2c, 0.0))
    v = jnp.sqrt(v2c)

    # HFBTHO's traces run over the canonical states with Omega > 0, one member
    # of each Kramers pair.  Here both members are explicit states, so every
    # sum carries a factor 1/2 (the S_u4v4 terms do not scale like the squares,
    # so this is not an overall normalisation).
    uv = u * v
    half = 0.5
    s_u3v1 = half * jnp.sum(u**3 * v * wstates)
    s_u1v3 = half * jnp.sum(u * v**3 * wstates)
    s_u2v2 = half * jnp.sum(uv**2 * wstates)
    s_u4v4 = half * jnp.sum(uv**4 * wstates)

    sn = 8.0 * (s_u3v1 * s_u1v3 - s_u4v4)
    sd = 32.0 * (s_u2v2**2 - s_u4v4)

    # Collapsed pairing: every sum vanishes together, so SN/SD and G_eff are
    # 0/0 and the ratio is noise.  LN corrects the particle-number fluctuation
    # 4 S_u2v2 = <(Delta N)^2>; below 1e-3 particles (a gap of ~0.1 MeV) there
    # is nothing to correct and lambda_2 = 0 exactly.  Tests for numerical
    # zeros alone are too loose: a collapsed 16O (gap 0.008 MeV) passes them.
    number_fluct = 4.0 * s_u2v2
    paired = ((number_fluct > 1e-3) & (jnp.abs(epair) > 1e-8)
              & (jnp.abs(avdelt) > 1e-6) & (jnp.abs(sd) > 1e-13))
    epair_safe = jnp.where(paired, epair, 1.0)
    sd_safe = jnp.where(paired, sd, 1.0)
    g_eff = avdelt**2 / epair_safe

    # Sign: HFBTHO writes ala2 = -Geff*(SNtor/SDtor) with Geff = del^2/ept,
    # where its ept is negative (attractive coupling), so its lambda_2 comes
    # out positive.  Here epair is a positive magnitude, so the minus is
    # dropped.  lambda_2 > 0 is required: E_LN = -4 lambda_2 S_u2v2 must lower
    # the energy.
    lam2 = jnp.where(paired, g_eff * sn / sd_safe, 0.0)

    # HFBTHO's guard: lambda_2 >= 10 MeV is numerical, reset to 4.  There is
    # deliberately no negative clip, so a negative lambda_2 stays visible.
    lam2 = jnp.where(lam2 >= 10.0, 4.0, lam2)
    return lam2


def ln_occupations(
    eferm: jax.Array,
    sp_energy: jax.Array,
    deltaf: jax.Array,
    lambda2: jax.Array,
    n_bisect: int = 42,
) -> jax.Array:
    """
    Self-consistent Lipkin-Nogami occupations at fixed Fermi energy and lambda_2.

    Each state's v^2 solves  v2 = f(e + 4*lambda2*v2 - eferm),  f(x) = (1 - x/sqrt(x^2 + Delta^2))/2.
    The right-hand side falls monotonically with v2 (lambda2 >= 0), so the root is unique and
    bisection on [0, 1] always finds it.

    The plain substitution v2 <- f(e + 4*lambda2*v2) does not converge near the Fermi surface: its
    slope there is ~4*lambda2/(2*Delta), about 1.4 for UNEDF1 Ca (lambda2 0.7, Delta 1).  Unconverged
    occupations amplify small level splittings and deform spherical nuclei.
    """
    lam2 = jnp.maximum(lambda2, 0.0)

    def f(v):
        x = sp_energy + 4.0 * lam2 * v - eferm
        return 0.5 * (1.0 - x / jnp.maximum(jnp.sqrt(x**2 + deltaf**2), 1.0e-20))

    def body(_, lo_hi):
        lo, hi = lo_hi
        mid = 0.5 * (lo + hi)
        above = mid - f(mid) > 0.0          # g(v) = v - f(v) is increasing
        return jnp.where(above, lo, mid), jnp.where(above, mid, hi)

    # eferm may carry leading axes (a batch of trial Fermi energies, find_fermi_energy_ln)
    shape = jnp.broadcast_shapes(jnp.shape(eferm), jnp.shape(sp_energy))
    lo, hi = jax.lax.fori_loop(0, n_bisect, body,
                               (jnp.zeros(shape, sp_energy.dtype), jnp.ones(shape, sp_energy.dtype)))
    return jnp.clip(0.5 * (lo + hi), 1.0e-10, 1.0 - 1.0e-10)


def find_fermi_energy_ln(
    target_n: float,
    sp_energy: jax.Array,
    deltaf: jax.Array,
    wstates: jax.Array,
    lambda2: jax.Array,
    emin: float = -500.0,
    emax: float = 500.0,
    n_trial: int = 64,
    n_rounds: int = 6,
) -> jax.Array:
    """
    Fermi energy for Lipkin-Nogami occupations (ln_occupations); N(eferm) is monotonic.

    Each round evaluates N at n_trial Fermi energies at once and shrinks the bracket to the cell holding
    target_n, so the bracket falls by n_trial-1 per round (1000 MeV -> ~2e-8 MeV in 6 rounds).  This
    keeps the number of sequential steps small; a scalar bisection nested around the occupation
    bisection is about 10x slower.
    """
    frac = jnp.linspace(0.0, 1.0, n_trial)

    def body(_, lo_hi):
        lo, hi = lo_hi
        ef = lo + (hi - lo) * frac                                       # (n_trial,)
        n = jnp.sum(ln_occupations(ef[:, None], sp_energy[None, :], deltaf[None, :], lambda2)
                    * wstates[None, :], axis=1)
        k = jnp.clip(jnp.sum(n < target_n) - 1, 0, n_trial - 2)           # last trial below target
        return ef[k], ef[k + 1]

    lo, hi = jax.lax.fori_loop(0, n_rounds, body, (jnp.asarray(emin, sp_energy.dtype),
                                                   jnp.asarray(emax, sp_energy.dtype)))
    return 0.5 * (lo + hi)


def lipkin_nogami_energy(lambda2: jax.Array, v2: jax.Array, wstates: jax.Array) -> jax.Array:
    """
    Lipkin-Nogami energy correction, E_LN = -4 * lambda_2 * Tr rho(1-rho).

    Matches `etr(it) = -four*ala2(it)*S_U2V2` in the reference, where S_U2V2
    runs over one member of each Kramers pair: 4*S_U2V2 = <(Delta N)^2>.  Both
    members are explicit states here, hence the 1/2.
    """
    v2c = jnp.clip(v2, 0.0, 1.0)
    s_u2v2 = 0.5 * jnp.sum(v2c * (1.0 - v2c) * wstates)
    return -4.0 * lambda2 * s_u2v2


def solve_pairing_isospin(
    iq: int,
    particle_number: float,
    sp_energy: jax.Array,
    deltaf: jax.Array,
    wstates: jax.Array,
    pairwg: jax.Array,
    pair_cutoff: float = 0.0,
    state_cutoff: float = 0.0,
    softcut_range: float = 0.1,
    smallp: float = 1.0e-6,
    use_ln: bool = False,        # Lipkin-Nogami particle-number correction
    ln_passes: int = 12,         # Illinois iterations for lambda_2 = F(lambda_2)
    temperature: float = 0.0,    # MeV; thermal BCS (FT-HF when the gaps vanish)
    blocked=None,                # EFA: 1 for the blocked doublet (v^2 = 1/2, no pairing)
):
    wstates_in = wstates         # Fermi-search weights (before the state cutoff update)
    free = 1.0 if blocked is None else 1.0 - blocked
    if blocked is not None:      # the doublet holds one particle outside the pairing pool
        if use_ln:
            raise ValueError("Lipkin-Nogami with blocking is not implemented")
        particle_number = particle_number - 0.5 * jnp.sum(blocked * wstates)

    # FORTRAN convention (pairs.f90): pairwg is applied to the gap exactly once,
    # in pairgap/diagstep. The BCS solve does NOT reapply it — see pairs.f90:223,
    # "there is already pairwg in deltaf and chi, so no pairwg here." The incoming
    # deltaf already carries pairwg, so use it as-is.
    deltaf_eff = deltaf

    # Find Fermi energy using effective gaps (pairwg baked in)
    eferm = find_fermi_energy(
        particle_number, sp_energy, deltaf_eff * free, wstates * free,
        emin=-500.0, emax=500.0, temperature=temperature,
    )

    # Update cutoffs
    pairwg = jax.lax.cond(
        pair_cutoff > 0.0,
        lambda _: soft_cutoff(
            sp_energy,
            eferm + pair_cutoff,
            jnp.maximum(softcut_range * pair_cutoff, 1e-6)
        ),
        lambda _: pairwg,  # The fallback value if pair_cutoff <= 0.0
        operand=None
    )

    # 2. Update wstates
    wstates = jax.lax.cond(
        state_cutoff > 0.0,
        lambda _: soft_cutoff(
            sp_energy,
            eferm + state_cutoff,
            jnp.maximum(softcut_range * state_cutoff, 1e-6)
        ),
        lambda _: wstates,  # The fallback value if state_cutoff <= 0.0
        operand=None
    )

    # Final occupations with effective gaps
    edif = sp_energy - eferm
    equasi = jnp.sqrt(edif**2 + deltaf_eff**2)
    equasi_safe = jnp.maximum(equasi, 1.0e-20)

    # At T > 0 each quasiparticle is occupied with f = 1/(1 + exp(E/T)):
    # rho_k = v^2 (1 - f) + u^2 f and kappa_k = u v (1 - 2f), with 1 - 2f = tanh(E/2T).
    th = thermal_factor(equasi, temperature)
    v2_bcs = 0.5 * (1.0 - edif / equasi_safe)
    v2 = 0.5 * (1.0 - edif / equasi_safe * th)
    # FORTRAN floors v2 at smal = 1e-10 (pairs.f90 bcs_occupation) and u*v at
    # sqrt(smallp) = sqrt(1e-6) (pairdn); smallp is not a floor on v2.  vfloor
    # keeps uv_floor = 0 (TDHFB statics) exactly unfloored.
    vfloor = jnp.minimum(smallp, 1.0e-10)
    v2 = jnp.clip(v2, vfloor, 1.0 - vfloor)

    # wguv = u v tanh(E/2T), no pairwg here (already in deltaf_eff)
    wguv = jnp.sqrt(jnp.maximum(v2_bcs * (1.0 - v2_bcs), smallp)) * th
    if blocked is not None:
        v2, wguv, pairwg = jnp.where(blocked > 0.5, 0.5, v2), wguv * free, pairwg * free

    # Statistics — vol uses wstates only, NOT pairwg (FORTRAN comment)
    vol = 0.5 * wguv * wstates
    sumuv = jnp.maximum(jnp.sum(vol), 1.0e-20)
    sumduv = jnp.sum(vol * deltaf_eff)   # deltaf_eff consistent with gaps
    sumv2 = jnp.sum(v2 * wstates)
    sumdv2 = jnp.sum(deltaf_eff * v2 * wstates)

    stats = {
        'eferm':   eferm,
        'epair':   sumduv,
        'avdelt':  sumduv / sumuv,
        'avdeltv2': sumdv2 / jnp.maximum(sumv2, 1.0e-20),
        'avg':     sumduv / sumuv**2,
    }

    # ── Lipkin-Nogami ────────────────────────────────────────────────────────
    # The LN occupations solve  v2 = f(e + 4 lambda_2 v2 - eferm)  exactly
    # (ln_occupations), and lambda_2 is the root of  lambda_2 = F(lambda_2),
    # F = compute_lambda2 evaluated on the LN solution AT that lambda_2.  The
    # bracket starts at [0, F(0)] and is doubled while the residual is still
    # negative (in an open shell F rises with lambda_2, so the root can lie
    # above F(0)); Illinois regula falsi then converges in a few steps.
    # E_LN = -4 lambda_2 Tr rho(1-rho) depends on the orbitals only through v2,
    # so the wave-function step is unchanged.
    def ln_solution(lam2):
        ef = find_fermi_energy_ln(particle_number, sp_energy, deltaf_eff, wstates_in, lam2)
        vv = jnp.clip(ln_occupations(ef, sp_energy, deltaf_eff, lam2), vfloor, 1.0 - vfloor)
        gg = jnp.sqrt(jnp.maximum(vv * (1.0 - vv), smallp))
        vol_ = 0.5 * gg * wstates
        suv = jnp.maximum(jnp.sum(vol_), 1.0e-20)
        sduv = jnp.sum(vol_ * deltaf_eff)
        sv2 = jnp.sum(vv * wstates)
        sdv2 = jnp.sum(deltaf_eff * vv * wstates)
        st = {
            'eferm':    ef,
            'epair':    sduv,
            'avdelt':   sduv / suv,
            'avdeltv2': sdv2 / jnp.maximum(sv2, 1.0e-20),
            'avg':      sduv / suv**2,
        }
        return ef, vv, gg, st

    def resid(lam2):                      # h(lam2) = lam2 - F(lam2)
        _, vv, _, st = ln_solution(lam2)
        return lam2 - compute_lambda2(vv, wstates, st['avdelt'], st['epair'])

    lambda2 = jnp.zeros((), dtype=v2.dtype)
    if use_ln:
        f0 = compute_lambda2(v2, wstates, stats['avdelt'], stats['epair'])    # F(0)
        a, fa = jnp.zeros((), f0.dtype), -f0
        b = jnp.maximum(f0, 1.0e-6)
        fb = resid(b)

        def expand(_, c):
            def grow(c):
                a, fa, b, fb = c
                b2 = 2.0 * b
                return b, fb, b2, resid(b2)
            return jax.lax.cond(c[3] < 0.0, grow, lambda c: c, c)

        a, fa, b, fb = jax.lax.fori_loop(0, 6, expand, (a, fa, b, fb))

        def illinois(_, c):
            a, fa, b, fb, side = c
            denom = jnp.where(jnp.abs(fb - fa) > 1e-14, fb - fa, 1e-14)
            x = jnp.clip(b - fb * (b - a) / denom, jnp.minimum(a, b), jnp.maximum(a, b))
            fx = resid(x)
            same = fx * fb > 0.0              # x replaces b; keep a, halve fa if it stuck twice
            a2 = jnp.where(same, a, b)
            fa2 = jnp.where(same, jnp.where(side == 1, 0.5 * fa, fa), fb)
            return (a2, fa2, x, fx, jnp.where(same, 1, -1))

        a, fa, b, fb, _ = jax.lax.fori_loop(0, ln_passes, illinois, (a, fa, b, fb, 0))
        lambda2 = jnp.where(f0 > 0.0, b, 0.0)      # unpaired: no correction
        eferm, v2, wguv, stats = ln_solution(lambda2)

    stats['lambda2'] = lambda2
    stats['e_ln'] = lipkin_nogami_energy(lambda2, v2, wstates)
    stats['entropy'] = qp_entropy(equasi, wstates, temperature)
    return eferm, v2, wguv, pairwg, wstates, stats

def compute_pairing_gaps(
    psi: jax.Array,
    v_pair: jax.Array,
    isospin: jax.Array,
    pairwg: jax.Array,
    wxyz: float,
    iteration: int,
    mass_number: int,
) -> jax.Array:
    """
    Compute pairing gaps from wavefunctions and pairing field.

    For early iterations, uses a simple estimate. Otherwise computes:
    Delta_i = pairwg_i * integral(v_pair(r) * |psi_i(r)|^2)

    Args:
        psi: Wavefunctions (nstates, 2, nx, ny, nz)
        v_pair: Pairing potential (2, nx, ny, nz)
        isospin: Isospin indices (nstates,)
        pairwg: Pairing cutoff weights (nstates,)
        wxyz: Integration weight
        iteration: Current iteration number
        mass_number: Total mass number

    Returns:
        Pairing gaps (nstates,)
    """
    nstates = psi.shape[0]
    dtypes = get_dtypes()

    # For early iterations, use constant gap estimate
    gap_estimate = 11.2 / jnp.sqrt(jnp.maximum(1.0, mass_number))
    initial_gaps = gap_estimate * jnp.ones(nstates, dtype=dtypes.float)

    # Compute |psi|^2 summed over spin
    psi_sq = jnp.real(psi * jnp.conjugate(psi))
    density = jnp.sum(psi_sq, axis=1)  # (nstates, nx, ny, nz)

    # Compute gaps using vectorized operations instead of vmap for speed
    # Select v_pair for each state (nstates, nx, ny, nz)
    # isospin is (nstates,)
    v_pair_states = jnp.where(isospin[:, None, None, None] == 0, v_pair[0], v_pair[1])

    # Gap_i = wxyz * sum_r (v_pair(r) * density_i(r))
    calculated_gaps = wxyz * jnp.sum(v_pair_states * density, axis=(1, 2, 3))

    calculated_gaps = calculated_gaps * pairwg

    # Switch between initial estimate and calculated gaps
    return jnp.where(iteration <= 10, initial_gaps, calculated_gaps)


def solve_pairing(
    sp_energy: jax.Array,
    deltaf: jax.Array,
    wstates: jax.Array,
    pairwg: jax.Array,
    isospin: jax.Array,
    nstates_n: int,
    nstates_p: int,
    target_n: int,
    target_p: int,
    force: Force,
    use_ln: bool = False,
    ln_passes: int = 12,
    temperature: float = 0.0,
    blocked=None,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, Pairing]:
    """
    Solve BCS pairing for both isospins.

    Args:
        sp_energy: Single-particle energies (nstates,)
        deltaf: Pairing gaps (nstates,)
        wstates: State weights (nstates,)
        pairwg: Pairing cutoff weights (nstates,)
        isospin: Isospin indices (nstates,)
        nstates_n: Number of neutron states
        nstates_p: Number of proton states
        target_n: Target neutron number
        target_p: Target proton number
        force: Force parameters
        use_ln: Apply the Lipkin-Nogami correction (static)
        ln_passes: Illinois iterations for lambda_2 (static)
        temperature: T in MeV for thermal occupations
        blocked: (nstates,) EFA mask of the blocked doublet (physics.blocking), or None

    Returns:
        (wocc, wguv, pairwg_new, wstates_new, pairing)
    """
    dtypes = get_dtypes()
    nstates = sp_energy.shape[0]

    # Process each isospin using slices for JIT compatibility
    # This assumes neutrons come first in the state arrays
    sp_n = sp_energy[:nstates_n]
    delta_n = deltaf[:nstates_n]
    wst_n = wstates[:nstates_n]
    pwg_n = pairwg[:nstates_n]

    eferm_n, v2_n, uv_n, pwg_new_n, wst_new_n, stats_n = solve_pairing_isospin(
        0, float(target_n), sp_n, delta_n, wst_n, pwg_n,
        pair_cutoff=force.pair_cutoff[0],
        state_cutoff=force.state_cutoff[0],
        softcut_range=force.softcut_range,
        smallp=force.uv_floor,
        use_ln=use_ln,
        ln_passes=ln_passes,
        temperature=temperature,
        blocked=None if blocked is None else blocked[:nstates_n],
    )

    sp_p = sp_energy[nstates_n:]
    delta_p = deltaf[nstates_n:]
    wst_p = wstates[nstates_n:]
    pwg_p = pairwg[nstates_n:]

    eferm_p, v2_p, uv_p, pwg_new_p, wst_new_p, stats_p = solve_pairing_isospin(
        1, float(target_p), sp_p, delta_p, wst_p, pwg_p,
        pair_cutoff=force.pair_cutoff[1],
        state_cutoff=force.state_cutoff[1],
        softcut_range=force.softcut_range,
        smallp=force.uv_floor,
        use_ln=use_ln,
        ln_passes=ln_passes,
        temperature=temperature,
        blocked=None if blocked is None else blocked[nstates_n:],
    )

    # Reassemble arrays
    wocc = jnp.concatenate([v2_n, v2_p])
    wguv = jnp.concatenate([uv_n, uv_p])

    # For cutoff weights, conditionally update
    pairwg_new = jnp.concatenate([
        jnp.where(force.pair_cutoff[0] > 0.0, pwg_new_n, pwg_n),
        jnp.where(force.pair_cutoff[1] > 0.0, pwg_new_p, pwg_p)
    ])

    wstates_new = jnp.concatenate([
        jnp.where(force.state_cutoff[0] > 0.0, wst_new_n, wst_n),
        jnp.where(force.state_cutoff[1] > 0.0, wst_new_p, wst_p)
    ])

    # Create result container
    pairing = Pairing(
        eferm=jnp.array([stats_n['eferm'], stats_p['eferm']]),
        epair=jnp.array([stats_n['epair'], stats_p['epair']]),
        avdelt=jnp.array([stats_n['avdelt'], stats_p['avdelt']]),
        avdeltv2=jnp.array([stats_n['avdeltv2'], stats_p['avdeltv2']]),
        avg=jnp.array([stats_n['avg'], stats_p['avg']]),
        lambda2=jnp.array([stats_n['lambda2'], stats_p['lambda2']]),
        e_ln=jnp.array([stats_n['e_ln'], stats_p['e_ln']]),
        entropy=jnp.array([stats_n['entropy'], stats_p['entropy']]),
    )

    return wocc, wguv, pairwg_new, wstates_new, pairing
