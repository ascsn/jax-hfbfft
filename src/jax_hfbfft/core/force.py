"""
Force class for nuclear interaction definitions.

This module provides the Force class which encapsulates Skyrme-type
effective nuclear interactions for HFB calculations.
"""

import jax
import jax.numpy as jnp
import dataclasses
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import yaml

# Nuclear saturation density (fm^-3), the reference for surface/mixed pairing.
RHO_SAT = 0.16

# EDF coupling constants accepted as primary parameters (parametrization 'C'); see Force
C_PARAMS = ('Crho0', 'Crho1', 'Crho0D', 'Crho1D', 'Ctau0', 'Ctau1', 'Cdrho0', 'Cdrho1', 'CdJ0', 'CdJ1',
            'CJ0', 'CJ1')


@jax.tree_util.register_dataclass
@dataclass
class Force:
    """
    Skyrme-type effective nuclear interaction.

    This class encapsulates the parameters of a Skyrme interaction including
    the standard Skyrme parameters (t0, t1, t2, t3, x0, x1, x2, x3) and
    derived coefficients (b0, b1, b2, b3, etc.).

    Two independent choices (both static):

    parametrization  'tx' (default): t0..t4, x0..x3, b4p are the inputs and the C couplings are derived.
                     'C': the EDF coupling constants are the inputs (Bender, Heenen & Reinhard, RMP 75, 121):
                     C^rho_t[rho0] = Crho_t + CrhoD_t rho0^power, C^tau_t = Ctau_t, C^Delta rho_t = Cdrho_t,
                     C^nabla J_t = CdJ_t, C^J_t = CJ_t (t = 0 isoscalar, 1 isovector). b and t/x follow from them;
                     derivatives (response.params_of) are then taken with respect to the C's. Time-odd partners
                     follow Galilean invariance (C^j = -C^tau, C^nabla j = C^nabla J, C^T = -C^J); the s-channel
                     couplings c_s*, c_ds*, c_sDs* stay independent. Force.from_couplings, Force.to_couplings.
    functional       'skyrme' (default): hand-coded Skyrme fields (meanfield.py) and energy (energies.py).
                     'skyrme_ad' or 'fayans': the functional is an energy density in physics/edf.py and every
                     mean field is its derivative by automatic differentiation.

    Attributes:
        name (str): Name of the force parameterization
        t0, t1, t2, t3, t4 (float): Skyrme interaction strengths
        x0, x1, x2, x3 (float): Skyrme exchange parameters
        power (float): Density-dependence power (typically 1/6 to 1)

    Examples:
        >>> # Load a predefined force
        >>> force = Force.from_name("SLy4")

        >>> # Create custom force with parameters
        >>> force = Force(name="custom", t0=-2488.91, t1=486.82, ...)

        >>> # Modify pairing parameters
        >>> force = force.with_pairing(v0prot=362.0, v0neut=362.0)
    """

    # Force identification
    name: str = field(metadata=dict(static=True))

    # Exchange type
    ex: int = field(metadata=dict(static=True))

    # Zero-point energy correction flag
    zpe: int = 0

    # Effective mass (h-bar^2 / 2m for each isospin)
    h2m: Optional[jax.Array] = None

    # Standard Skyrme parameters
    t0: float = 0.0
    t1: float = 0.0
    t2: float = 0.0
    t3: float = 0.0
    t4: float = 0.0
    x0: float = 0.0
    x1: float = 0.0
    x2: float = 0.0
    x3: float = 0.0

    # Spin-orbit parameter
    b4p: float = 0.0

    # Density-dependence power
    power: float = 1.0

    # Pairing parameters
    ipair: int = field(default=0, metadata=dict(static=True))  # Pairing type
    v0prot: float = 0.0  # Proton pairing strength
    v0neut: float = 0.0  # Neutron pairing strength
    rho0pr: float = 0.16  # Pairing reference density
    pair_reg: bool = False  # Regularization flag
    delta_fit: Optional[jax.Array] = None  # Fitted gap parameters
    pair_cutoff: Optional[jax.Array] = None  # Energy cutoff for pairing
    state_cutoff: Optional[jax.Array] = None  # State cutoff
    softcut_range: float = 0.1  # Soft cutoff width parameter
    # u*v is floored at sqrt(uv_floor) (FORTRAN smallp = 1e-6) and v2 at
    # min(uv_floor, 1e-10) (FORTRAN smal). Statics for TDHFB want 0: the floored
    # uv is inconsistent with v2 for nearly empty states.
    uv_floor: float = 1.0e-6
    # Regularised contact pairing (Bulgac & Yu, PRL 88, 042504): the coupling v = v0 g(rho) becomes
    # v_eff = v / (1 + v K), K = k_c/(4 pi^2 B) [1 - (k_F/2k_c) ln((k_c + k_F)/(k_c - k_F))], with the local
    # k_F = (3 pi^2 rho_q)^(1/3), k_c^2 = k_F^2 + E_c / B_q and B_q = dE/dtau_q: a local function of the
    # densities, so the pair field stays local (TDHFB local continuity) and the gap no longer depends on
    # the cutoff E_c (MeV above the Fermi surface) of the pairing space. E_c must be large against the local
    # Fermi energy: where k_F/k_c > 0.83, K < 0 and the coupling is enhanced (SLy4 DDDI at E_c = 26 MeV: by at
    # most 3%, at rho_n ~ 0.13). 0: bare coupling. Autodiff functionals only (Force.functional, e.g. 'skyrme_ad').
    pair_reg_ecut: float = field(default=0.0, metadata=dict(static=True))
    # Momentum cutoff of the TDHFB pair channel (physics/tdhfb.py): pairing acts on
    # the orbital components below |k| = pair_kcut (fm^-1), smoothed over
    # pair_kwidth. Regularises zero-range pairing, whose gap otherwise grows into
    # grid-scale states once the orbitals are free. 0 = bare contact pairing.
    # The canonical static solver ignores it; relax with physics.tdhfb.relax.
    pair_kcut: float = field(default=0.0, metadata=dict(static=True))
    pair_kwidth: float = field(default=0.1, metadata=dict(static=True))
    tbcs: bool = False  # Use BCS approximation

    # Physical constants
    h2ma: float = 20.73553  # h-bar^2 / (2 * nucleon mass)
    nucleon_mass: float = 938.9  # Nucleon mass in MeV

    # Derived Skyrme coefficients (calculated from t, x parameters)
    b0: float = 0.0
    b0p: float = 0.0
    b1: float = 0.0
    b1p: float = 0.0
    b2: float = 0.0
    b2p: float = 0.0
    b3: float = 0.0
    b3p: float = 0.0
    b4: float = 0.0

    # Time-odd spin-channel couplings (all zero by default)
    c_s0: float = 0.0    # s^2, isoscalar
    c_s1: float = 0.0    # s^2, isovector
    c_ds0: float = 0.0   # rho^alpha s^2, isoscalar
    c_ds1: float = 0.0   # rho^alpha s^2, isovector
    c_sDs0: float = 0.0  # s.Lap(s), isoscalar
    c_sDs1: float = 0.0  # s.Lap(s), isovector

    # Slater determinant parameter
    slate: float = 0.0

    # EDF coupling constants (derived)
    Crho0: float = 0.0
    Crho1: float = 0.0
    Crho0D: float = 0.0
    Crho1D: float = 0.0
    Cdrho0: float = 0.0
    Cdrho1: float = 0.0
    Ctau0: float = 0.0
    Ctau1: float = 0.0
    CdJ0: float = 0.0
    CdJ1: float = 0.0
    # J^2 (tensor) couplings, coefficients of sum_mu,nu J_mu,nu^2 (HFBTHO's CJ). The vector
    # (antisymmetric) part of J_mu,nu, 1/2 J^2, always acts; the symmetric part needs
    # `tensor`, set for forces with j2 in the force file.
    CJ0: float = 0.0
    CJ1: float = 0.0
    tensor: bool = field(default=False, metadata=dict(static=True))

    # Which parameters are primary ('tx' or 'C', see the class docstring) and which code builds the
    # energy and fields ('skyrme' hand-coded, or an autodiff functional of physics/edf.py)
    parametrization: str = field(default='tx', metadata=dict(static=True))
    functional: str = field(default='skyrme', metadata=dict(static=True))
    # Fayans parameters (physics/edf.py, FY_KEYS order), functional 'fayans'
    fy: Optional[jax.Array] = None
    # Parameters of a user-defined functional (physics/edf.register): name -> value, differentiable as 'extra.<name>'
    extra: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Calculate derived coefficients after initialization."""
        # Initialize arrays if not set
        if self.delta_fit is None:
            object.__setattr__(self, 'delta_fit', jnp.array([-1.0, -1.0]))
        if self.pair_cutoff is None:
            object.__setattr__(self, 'pair_cutoff', jnp.array([0.0, 0.0]))
        if self.state_cutoff is None:
            object.__setattr__(self, 'state_cutoff', jnp.array([0.0, 0.0]))
        if self.h2m is None:
            object.__setattr__(self, 'h2m', jnp.array([self.h2ma, self.h2ma]))

    @classmethod
    def from_name(
        cls,
        name: str,
        forces_file: Optional[str] = None,
        **kwargs
    ) -> "Force":
        """
        Create a Force from a predefined parameterization.

        Args:
            name: Name of the force (e.g., "SLy4", "SkM*", "UNEDF1")
            forces_file: Path to YAML file with force definitions
            **kwargs: Override parameters (e.g., pairing strengths)

        Returns:
            Initialized Force object

        Examples:
            >>> force = Force.from_name("SLy4")
            >>> force = Force.from_name("SLy4", v0prot=362.0, v0neut=362.0)
        """
        # Try to find forces file
        if forces_file is None:
            # Look in package data or current directory
            possible_paths = [
                Path(__file__).parent.parent / "data" / "_forces.yml",
                Path("_forces.yml"),
            ]
            for path in possible_paths:
                if path.exists():
                    forces_file = str(path)
                    break

        if forces_file is None:
            raise FileNotFoundError(
                "Could not find forces definition file. "
                "Please provide forces_file argument."
            )

        # Load force parameters from YAML
        with open(forces_file, 'r') as f:
            forces_data = yaml.safe_load(f)

        if name not in forces_data:
            available = ", ".join(forces_data.keys())
            raise ValueError(
                f"Force '{name}' not found. Available forces: {available}"
            )

        force_params = forces_data[name]

        # Build initialization parameters
        init_kwargs = {
            'name': name,
            'ex': force_params.get('ex', 1),
            'zpe': force_params.get('zpe', 0),
            'h2m': jnp.array(force_params.get('h2m', [20.73553, 20.73553])),
            't0': force_params.get('t0', 0.0),
            't1': force_params.get('t1', 0.0),
            't2': force_params.get('t2', 0.0),
            't3': force_params.get('t3', 0.0),
            't4': force_params.get('t4', 0.0),
            'x0': force_params.get('x0', 0.0),
            'x1': force_params.get('x1', 0.0),
            'x2': force_params.get('x2', 0.0),
            'x3': force_params.get('x3', 0.0),
            'b4p': force_params.get('b4p', 0.0),
            'power': force_params.get('power', 1.0),
            'c_s0': force_params.get('c_s0', 0.0),
            'c_s1': force_params.get('c_s1', 0.0),
            'c_ds0': force_params.get('c_ds0', 0.0),
            'c_ds1': force_params.get('c_ds1', 0.0),
            'c_sDs0': force_params.get('c_sDs0', 0.0),
            'c_sDs1': force_params.get('c_sDs1', 0.0),
            'pair_reg_ecut': float(force_params.get('pair_reg_ecut', 0.0)),
        }
        # Pairing: an optional 'vdi:' / 'dddi:' block in the force file supplies
        # default strengths for ipair=5/6.  Explicit kwargs always win.
        ipair = int(kwargs.get('ipair', 0))
        block = {5: 'vdi', 6: 'dddi'}.get(ipair)
        pair_params = force_params.get(block) if block else None
        if pair_params is not None:
            init_kwargs['v0prot'] = pair_params.get('v0prot', 0.0)
            init_kwargs['v0neut'] = pair_params.get('v0neut', 0.0)
            init_kwargs['rho0pr'] = pair_params.get('rho0pr', 0.16)
        elif block is not None and not {'v0prot', 'v0neut'} <= kwargs.keys():
            raise ValueError(
                f"Force '{name}' has no '{block}:' block; pass v0prot/v0neut "
                f"explicitly for ipair={ipair}."
            )
        # J^2 terms: 'central' = those of the Skyrme central force (SLy5, SkP),
        # or explicit [CJ0, CJ1]
        j2 = force_params.get('j2')
        if j2 == 'central':
            t1, t2 = init_kwargs['t1'], init_kwargs['t2']
            x1, x2 = init_kwargs['x1'], init_kwargs['x2']
            init_kwargs['CJ0'] = (t1 - t2) / 16 - (t1 * x1 + t2 * x2) / 8
            init_kwargs['CJ1'] = (t1 - t2) / 16
        elif j2 is not None:
            init_kwargs['CJ0'], init_kwargs['CJ1'] = (float(c) for c in j2)
        init_kwargs['tensor'] = j2 is not None

        # EDF coupling constants as the primary parameters (parametrization 'C')
        cblock = force_params.get('couplings')
        if cblock is not None:
            unknown = set(cblock) - set(C_PARAMS)
            if unknown:
                raise ValueError(f"Force '{name}': unknown couplings {sorted(unknown)}; expected {C_PARAMS}")
            for k in ('t0', 't1', 't2', 't3', 't4', 'x0', 'x1', 'x2', 'x3', 'b4p'):
                init_kwargs.pop(k, None)
            init_kwargs.update({k: float(v) for k, v in cblock.items()}, parametrization='C')
            if abs(float(cblock.get('CJ0', 0.0))) + abs(float(cblock.get('CJ1', 0.0))) > 0.0:
                init_kwargs['tensor'] = True
        # Fayans functional (physics/edf.py): volume/surface and pairing terms from a 'fayans:' block,
        # spin-orbit on t4/b4p, no effective mass
        fy_block = force_params.get('fayans')
        if fy_block is not None:
            from jax_hfbfft.physics import edf as _edf
            fy, t4, b4p = _edf.fayans_parameters(fy_block, init_kwargs['h2m'])
            init_kwargs.update(functional='fayans', fy=fy, t4=t4, b4p=b4p)
        if 'functional' in force_params:
            init_kwargs['functional'] = force_params['functional']

        # Apply user overrides
        init_kwargs.update(kwargs)
        # A negative strength is a file's "not fitted" placeholder.
        if block is not None and min(init_kwargs['v0prot'], init_kwargs['v0neut']) < 0.0:
            raise ValueError(
                f"Force '{name}' has placeholder pairing strengths "
                f"(v0prot={init_kwargs['v0prot']}, v0neut={init_kwargs['v0neut']}); "
                f"supply v0prot/v0neut explicitly."
            )

        # Create force and calculate derived coefficients
        force = cls(**init_kwargs)
        return force._calculate_derived_coefficients()

    def _calculate_derived_coefficients(self) -> "Force":
        """Calculate derived Skyrme coefficients from primary parameters."""
        if self.parametrization == 'C':
            return self._from_C()

        # Calculate b coefficients from t and x parameters
        b0 = self.t0 * (1 + 0.5 * self.x0)
        b0p = self.t0 * (0.5 + self.x0)

        b1 = (self.t1 + 0.5 * self.x1 * self.t1 +
              self.t2 + 0.5 * self.x2 * self.t2) / 4
        b1p = (self.t1 * (0.5 + self.x1) -
               self.t2 * (0.5 + self.x2)) / 4

        b2 = (3 * self.t1 * (1 + 0.5 * self.x1) -
              self.t2 * (1 + 0.5 * self.x2)) / 8
        b2p = (3 * self.t1 * (0.5 + self.x1) +
               self.t2 * (0.5 + self.x2)) / 8

        b3 = self.t3 * (1 + 0.5 * self.x3) / 4
        b3p = self.t3 * (0.5 + self.x3) / 4

        b4 = self.t4 / 2

        # Calculate EDF coupling constants using the b parameters
        Crho0 = 0.5 * b0 - 0.25 * b0p
        Crho1 = -0.25 * b0p

        Crho0D = (1.0 / 3.0) * b3 - (1.0 / 6.0) * b3p
        Crho1D = -(1.0 / 6.0) * b3p

        Cdrho0 = -0.5 * b2 + 0.25 * b2p
        Cdrho1 = 0.25 * b2p

        Ctau0 = b1 - 0.5 * b1p
        Ctau1 = -0.5 * b1p

        CdJ0 = -b4 - 0.5 * self.b4p
        CdJ1 = -0.5 * self.b4p

        # Coulomb constant integration
        e2 = 1.4399784085965135   # MeV·fm
        slate = (3.0 / jnp.pi) ** (1.0 / 3.0) * e2

        return dataclasses.replace(
            self,
            b0=b0, b0p=b0p,
            b1=b1, b1p=b1p,
            b2=b2, b2p=b2p,
            b3=b3, b3p=b3p,
            b4=b4,
            slate=slate,
            Crho0=Crho0, Crho1=Crho1,
            Crho0D=Crho0D, Crho1D=Crho1D,
            Cdrho0=Cdrho0, Cdrho1=Cdrho1,
            Ctau0=Ctau0, Ctau1=Ctau1,
            CdJ0=CdJ0, CdJ1=CdJ1,
        )

    def _from_C(self) -> "Force":
        """b coefficients (used by the hand-coded fields) and t/x (informational) from the C couplings."""
        b0p = -4.0 * self.Crho1
        b0 = 2.0 * self.Crho0 + 0.5 * b0p
        b3p = -6.0 * self.Crho1D
        b3 = 3.0 * self.Crho0D + 0.5 * b3p
        b2p = 4.0 * self.Cdrho1
        b2 = 0.5 * b2p - 2.0 * self.Cdrho0
        b1p = -2.0 * self.Ctau1
        b1 = self.Ctau0 + 0.5 * b1p
        b4p = -2.0 * self.CdJ1
        b4 = -self.CdJ0 - 0.5 * b4p
        # the equivalent Skyrme force (b -> t, x), for reference only
        safe = lambda t: jnp.where(t == 0.0, 1.0, t)
        t0 = 4.0 / 3.0 * (b0 - 0.5 * b0p)
        t3 = 16.0 / 3.0 * (b3 - 0.5 * b3p)
        A1, A2 = b1 + 2.0 * b2, 3.0 * b1 - 2.0 * b2          # t_i (1 + x_i/2)
        B1, B2 = b1p + 2.0 * b2p, 2.0 * b2p - 3.0 * b1p      # t_i (1/2 + x_i)
        t1, t2 = 4.0 / 3.0 * (A1 - 0.5 * B1), 4.0 / 3.0 * (A2 - 0.5 * B2)
        x = lambda b, t: jnp.where(t == 0.0, 0.0, b / safe(t) - 0.5)
        e2 = 1.4399784085965135
        return dataclasses.replace(
            self, b0=b0, b0p=b0p, b1=b1, b1p=b1p, b2=b2, b2p=b2p, b3=b3, b3p=b3p, b4=b4, b4p=b4p,
            t0=t0, t1=t1, t2=t2, t3=t3, t4=2.0 * b4,
            x0=x(b0p, t0), x1=x(B1, t1), x2=x(B2, t2), x3=x(4.0 * b3p, t3),
            slate=(3.0 / jnp.pi) ** (1.0 / 3.0) * e2)

    @classmethod
    def from_couplings(cls, name: str, couplings: Dict[str, float], power: float,
                       h2m=(20.73553, 20.73553), **kwargs) -> "Force":
        """
        A force defined by its EDF coupling constants (parametrization 'C').

        couplings: any of C_PARAMS (missing ones are zero), e.g. {'Crho0': -933.3, 'Crho1': 830.0, ...}.
        power: the density exponent alpha of C^rho_t[rho0] = Crho_t + CrhoD_t rho0^alpha.
        kwargs: other Force fields (ex, zpe, pairing, c_s*, functional, ...).
        """
        unknown = set(couplings) - set(C_PARAMS)
        if unknown:
            raise ValueError(f"unknown couplings {sorted(unknown)}; expected {C_PARAMS}")
        tensor = abs(float(couplings.get('CJ0', 0.0))) + abs(float(couplings.get('CJ1', 0.0))) > 0.0
        kw = dict(name=name, ex=kwargs.pop('ex', 1), h2m=jnp.asarray(h2m), power=power,
                  parametrization='C', tensor=kwargs.pop('tensor', tensor))
        kw.update({k: float(v) for k, v in couplings.items()})
        kw.update(kwargs)
        return cls(**kw)._calculate_derived_coefficients()

    def to_couplings(self) -> "Force":
        """The same force with the C couplings as primary parameters (derivatives then by C)."""
        return dataclasses.replace(self, parametrization='C')._calculate_derived_coefficients()

    def with_functional(self, functional: str) -> "Force":
        """The same parameters evaluated by another implementation ('skyrme', 'skyrme_ad', 'fayans')."""
        return dataclasses.replace(self, functional=functional)

    def couplings(self) -> Dict[str, float]:
        """The EDF coupling constants of this force (C_PARAMS), whichever parametrization is primary."""
        return {k: float(getattr(self, k)) for k in C_PARAMS}

    def with_pairing(
        self,
        ipair: Optional[int] = None,
        v0prot: Optional[float] = None,
        v0neut: Optional[float] = None,
        rho0pr: Optional[float] = None,
        tbcs: Optional[bool] = None,
        pair_cutoff: Optional[Tuple[float, float]] = None,
        state_cutoff: Optional[Tuple[float, float]] = None,
        uv_floor: Optional[float] = None,
        pair_kcut: Optional[float] = None,
        pair_kwidth: Optional[float] = None,
    ) -> "Force":
        """
        Create a new Force with modified pairing parameters.

        Args:
            ipair: Pairing type (0=none, 5=VDI, 6=DDDI)
            v0prot: Proton pairing strength (MeV·fm³)
            v0neut: Neutron pairing strength (MeV·fm³)
            rho0pr: Reference density for DDDI pairing
            tbcs: Use BCS approximation
            pair_cutoff: Energy cutoff (neutron, proton) in MeV
            state_cutoff: State cutoff (neutron, proton) in MeV
            uv_floor: Floor on v2(1 - v2) in the BCS solution (default 1e-6)
            pair_kcut, pair_kwidth: Momentum cutoff of the TDHFB pair channel (fm^-1)

        Returns:
            New Force with updated pairing parameters
        """

        updates = {}
        if ipair is not None:
            updates['ipair'] = ipair
        if v0prot is not None:
            updates['v0prot'] = v0prot
        if v0neut is not None:
            updates['v0neut'] = v0neut
        if rho0pr is not None:
            updates['rho0pr'] = rho0pr
        if tbcs is not None:
            updates['tbcs'] = tbcs
        if pair_cutoff is not None:
            updates['pair_cutoff'] = jnp.array(pair_cutoff)
        if state_cutoff is not None:
            updates['state_cutoff'] = jnp.array(state_cutoff)
        if uv_floor is not None:
            updates['uv_floor'] = uv_floor
        if pair_kcut is not None:
            updates['pair_kcut'] = float(pair_kcut)
        if pair_kwidth is not None:
            updates['pair_kwidth'] = float(pair_kwidth)

        return dataclasses.replace(self, **updates)

    def with_pairing_type(self, kind: str, v0: float = 362.0) -> "Force":
        """
        Create a new Force with a density-dependent delta pairing interaction.
        Args:
            kind: 'volume' (no density dependence), 'surface' (rho0pr = rho_sat)
                or 'mixed' (rho0pr = 2 rho_sat).
            v0: Pairing strength for both isospins (MeV fm^3).
        Returns:
            New Force with ipair=6 (DDDI) and the requested rho0pr.
        """
        kind = kind.lower()
        if kind == 'volume':
            rho0pr = 1.0e30  # makes 1 - rho/rho0pr equal to 1 everywhere
        elif kind == 'surface':
            rho0pr = RHO_SAT
        elif kind == 'mixed':
            rho0pr = 2.0 * RHO_SAT
        else:
            raise ValueError(
                f"Unknown pairing type {kind!r}. Use 'volume', 'surface', or 'mixed'."
            )
        return dataclasses.replace(
            self, ipair=6, v0neut=v0, v0prot=v0, rho0pr=rho0pr,
        )

    def with_time_odd(
        self,
        c_s0: float = 0.0,
        c_s1: float = 0.0,
        c_ds0: float = 0.0,
        c_ds1: float = 0.0,
        c_sDs0: float = 0.0,
        c_sDs1: float = 0.0,
    ) -> "Force":
        """
        Create a new Force with explicit time-odd spin-channel couplings.

        Args:
            c_s0, c_s1: s^2 couplings, isoscalar / isovector (MeV fm^3)
            c_ds0, c_ds1: rho^alpha s^2 couplings, isoscalar / isovector
            c_sDs0, c_sDs1: s.Lap(s) couplings, isoscalar / isovector (MeV fm^5)

        Returns:
            New Force with all six couplings set (unspecified ones to zero).
        """

        return dataclasses.replace(
            self, c_s0=c_s0, c_s1=c_s1, c_ds0=c_ds0, c_ds1=c_ds1,
            c_sDs0=c_sDs0, c_sDs1=c_sDs1,
        )

    def with_time_odd_from_skyrme(self) -> "Force":
        """
        Create a new Force with the time-odd spin couplings implied by the
        Skyrme t/x parameters (as in HFBTHO's hfbtho_unedf.f90, C_from_t).

        Returns:
            New Force with c_s0, c_s1, c_ds0, c_ds1, c_sDs0, c_sDs1 set.
        """

        cs0 = -(1.0 / 4.0) * self.t0 * (0.5 - self.x0)
        cs1 = -(1.0 / 8.0) * self.t0
        cds0 = -(1.0 / 24.0) * self.t3 * (0.5 - self.x3)
        cds1 = -(1.0 / 48.0) * self.t3
        csds0 = (3.0 / 32.0) * self.t1 * (0.5 - self.x1) + (1.0 / 32.0) * self.t2 * (0.5 + self.x2)
        csds1 = (3.0 / 64.0) * self.t1 + (1.0 / 64.0) * self.t2

        return dataclasses.replace(
            self, c_s0=cs0, c_s1=cs1, c_ds0=cds0, c_ds1=cds1,
            c_sDs0=csds0, c_sDs1=csds1,
        )

    @property
    def has_time_odd(self) -> bool:
        """True if any time-odd spin coupling is non-zero."""
        return any(abs(c) > 0.0 for c in (self.c_s0, self.c_s1, self.c_ds0,
                                          self.c_ds1, self.c_sDs0, self.c_sDs1))

    def __str__(self) -> str:
        return f"Force({self.name})"

    def __repr__(self) -> str:
        return (
            f"Force(name={self.name!r}, t0={self.t0:.2f}, t1={self.t1:.2f}, "
            f"t2={self.t2:.2f}, t3={self.t3:.2f}, power={self.power:.3f})"
        )


# Dictionary of commonly used forces with their parameters
BUILTIN_FORCES = {
    "SLy4": {
        "ex": 1, "zpe": 0,
        "h2m": [20.73553, 20.73553],
        "t0": -2488.91, "t1": 486.82, "t2": -546.39, "t3": 13777.0, "t4": 123.0,
        "x0": 0.834, "x1": -0.344, "x2": -1.0, "x3": 1.354,
        "b4p": 61.5, "power": 1.0/6.0,
    },
    "SkM*": {
        "ex": 1, "zpe": 0,
        "h2m": [20.73553, 20.73553],
        "t0": -2645.0, "t1": 410.0, "t2": -135.0, "t3": 15595.0, "t4": 130.0,
        "x0": 0.09, "x1": 0.0, "x2": 0.0, "x3": 0.0,
        "b4p": 65.0, "power": 1.0/6.0,
    },
}
