"""
Pydantic models for GUI API.

This module defines all the data models used for API requests and responses.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from enum import Enum
import uuid


class CalculationPhase(str, Enum):
    """Current phase of a calculation."""
    PENDING = "pending"
    WARMUP = "warmup"
    INITIALIZING = "initializing"
    ITERATING = "iterating"
    CONVERGED = "converged"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PairingType(str, Enum):
    """Pairing interaction type."""
    NONE = "none"
    VDI = "vdi"
    DDDI = "dddi"


class ConstraintType(str, Enum):
    """Constraint type for constrained HFB."""
    NONE = "none"
    MULTIPOLE = "multipole"
    BETA_GAMMA = "beta_gamma"


class RunType(str, Enum):
    """Type of run stored in history/activity logs."""
    CALCULATION = "calculation"
    SURFACE = "surface"


# ============================================================================
# Request Models
# ============================================================================

class NucleusInput(BaseModel):
    """Input specification for a nucleus."""
    protons: int = Field(..., ge=1, le=120, description="Number of protons (Z)")
    neutrons: int = Field(..., ge=1, le=200, description="Number of neutrons (N)")
    
    @property
    def mass_number(self) -> int:
        return self.protons + self.neutrons
    
    @property
    def symbol(self) -> str:
        """Get element symbol (simplified)."""
        # Common elements for nuclear physics
        symbols = {
            2: "He", 6: "C", 8: "O", 10: "Ne", 12: "Mg", 14: "Si",
            16: "S", 18: "Ar", 20: "Ca", 22: "Ti", 24: "Cr", 26: "Fe",
            28: "Ni", 30: "Zn", 32: "Ge", 34: "Se", 36: "Kr", 38: "Sr",
            40: "Zr", 42: "Mo", 44: "Ru", 46: "Pd", 48: "Cd", 50: "Sn",
            52: "Te", 54: "Xe", 56: "Ba", 58: "Ce", 60: "Nd", 62: "Sm",
            64: "Gd", 66: "Dy", 68: "Er", 70: "Yb", 72: "Hf", 74: "W",
            76: "Os", 78: "Pt", 80: "Hg", 82: "Pb", 84: "Po", 86: "Rn",
            88: "Ra", 90: "Th", 92: "U", 94: "Pu",
        }
        return symbols.get(self.protons, f"Z{self.protons}")


class GridConfig(BaseModel):
    """Grid configuration."""
    nx: int = Field(24, ge=8, le=64, description="Grid points in x")
    ny: int = Field(24, ge=8, le=64, description="Grid points in y")
    nz: int = Field(24, ge=8, le=64, description="Grid points in z")
    dx: float = Field(1.0, ge=0.5, le=2.0, description="Grid spacing in x (fm)")
    dy: float = Field(1.0, ge=0.5, le=2.0, description="Grid spacing in y (fm)")
    dz: float = Field(1.0, ge=0.5, le=2.0, description="Grid spacing in z (fm)")
    auto: bool = Field(True, description="Auto-calculate grid from nucleus size")


class PairingConfig(BaseModel):
    """Pairing configuration."""
    type: PairingType = Field(PairingType.NONE, description="Pairing type")
    v0_neutron: float = Field(-200.0, description="Neutron pairing strength (MeV fm³)")
    v0_proton: float = Field(-200.0, description="Proton pairing strength (MeV fm³)")


class ConstraintConfig(BaseModel):
    """Constraint configuration."""
    type: ConstraintType = Field(ConstraintType.NONE, description="Constraint type")
    q20: Optional[float] = Field(None, description="Q20 multipole moment (fm²)")
    q30: Optional[float] = Field(None, description="Q30 multipole moment (fm³)")
    q40: Optional[float] = Field(None, description="Q40 multipole moment (fm⁴)")
    beta2: Optional[float] = Field(None, description="Beta2 deformation")
    gamma: Optional[float] = Field(None, description="Gamma angle (degrees)")


class IterationConfig(BaseModel):
    """Iteration parameters."""
    max_iterations: int = Field(200, ge=10, le=5000, description="Maximum iterations")
    convergence_threshold: float = Field(1e-6, ge=1e-10, le=1e-2, description="Convergence threshold")
    print_interval: int = Field(10, ge=1, le=100, description="Print interval")


class CalculationRequest(BaseModel):
    """Request to start a new calculation."""
    nucleus: NucleusInput
    force_name: str = Field("SLy4", description="Skyrme force name")
    grid: GridConfig = Field(default_factory=GridConfig)
    pairing: PairingConfig = Field(default_factory=PairingConfig)
    constraint: ConstraintConfig = Field(default_factory=ConstraintConfig)
    iteration: IterationConfig = Field(default_factory=IterationConfig)
    
    @field_validator("force_name")
    @classmethod
    def validate_force_name(cls, v):
        # Will be validated against available forces on the server
        return v


class BetaSurfaceRequest(BaseModel):
    """Request to compute a 1D beta deformation surface."""
    nucleus: NucleusInput
    force_name: str = Field("SLy4", description="Skyrme force name")
    grid: GridConfig = Field(default_factory=GridConfig)
    pairing: PairingConfig = Field(default_factory=PairingConfig)
    iteration: IterationConfig = Field(default_factory=IterationConfig)
    beta_min: float = Field(-0.3, description="Minimum beta2 value (oblate)")
    beta_max: float = Field(0.3, description="Maximum beta2 value (prolate)")
    beta_steps: int = Field(13, ge=3, le=101, description="Number of beta points")
    gamma: float = Field(0.0, description="Gamma angle in degrees")
    hot_start: bool = Field(False, description="Use nearest beta solution to initialize")


# ============================================================================
# Response Models
# ============================================================================

class EnergyBreakdown(BaseModel):
    """Detailed energy breakdown."""
    total: float = Field(..., description="Total binding energy (MeV)")
    kinetic: float = Field(0.0, description="Kinetic energy (MeV)")
    potential: float = Field(0.0, description="Potential energy (MeV)")
    coulomb: float = Field(0.0, description="Coulomb energy (MeV)")
    pairing: float = Field(0.0, description="Pairing energy (MeV)")
    cm_correction: float = Field(0.0, description="Center-of-mass correction (MeV)")
    
    # Skyrme decomposition
    ehf0: float = Field(0.0, description="t0 term (MeV)")
    ehf1: float = Field(0.0, description="Current term (MeV)")
    ehf2: float = Field(0.0, description="Laplacian term (MeV)")
    ehf3: float = Field(0.0, description="Density-dependent term (MeV)")
    ehfls: float = Field(0.0, description="Spin-orbit term (MeV)")


class RadiiResults(BaseModel):
    """RMS radii results."""
    neutron: float = Field(..., description="Neutron RMS radius (fm)")
    proton: float = Field(..., description="Proton RMS radius (fm)")
    total: float = Field(..., description="Total RMS radius (fm)")
    charge: float = Field(..., description="Charge radius (fm)")


class DeformationResults(BaseModel):
    """Deformation parameters."""
    beta2: float = Field(0.0, description="Beta2 deformation")
    gamma: float = Field(0.0, description="Gamma angle (degrees)")
    q20: float = Field(0.0, description="Q20 moment (fm²)")
    q22: float = Field(0.0, description="Q22 moment (fm²)")


class PairingResults(BaseModel):
    """Pairing calculation results."""
    gap_neutron: float = Field(0.0, description="Neutron pairing gap (MeV)")
    gap_proton: float = Field(0.0, description="Proton pairing gap (MeV)")
    fermi_neutron: float = Field(0.0, description="Neutron Fermi energy (MeV)")
    fermi_proton: float = Field(0.0, description="Proton Fermi energy (MeV)")


class SingleParticleLevel(BaseModel):
    """Single-particle level information."""
    index: int = Field(..., description="State index")
    isospin: Literal["neutron", "proton"] = Field(..., description="Particle type")
    energy: float = Field(..., description="Single-particle energy (MeV)")
    occupation: float = Field(..., description="Occupation probability")
    parity: int = Field(1, description="Parity (+1 or -1)")
    label: Optional[str] = Field(None, description="State label (e.g., '1s1/2')")


class CalculationResults(BaseModel):
    """Complete calculation results."""
    energies: EnergyBreakdown
    radii: RadiiResults
    deformation: DeformationResults
    pairing: PairingResults
    single_particle_spectrum: List[SingleParticleLevel] = Field(default_factory=list)
    
    # Convergence info
    converged: bool = Field(False, description="Whether calculation converged")
    iterations: int = Field(0, description="Number of iterations performed")
    final_fluctuation: float = Field(0.0, description="Final energy fluctuation")
    
    # Timing
    total_time_seconds: float = Field(0.0, description="Total calculation time")
    jit_time_seconds: float = Field(0.0, description="JIT compilation time")


class BetaSurfacePoint(BaseModel):
    """Single point on a beta deformation surface."""
    beta2: float = Field(..., description="Beta2 deformation")
    energy: float = Field(..., description="Total energy (MeV)")
    converged: bool = Field(False, description="Whether the point converged")
    iterations: int = Field(0, description="Iterations performed")
    q20: float = Field(0.0, description="Q20 moment (fm²)")
    q22: float = Field(0.0, description="Q22 moment (fm²)")


class BetaSurfaceResult(BaseModel):
    """Result for a 1D beta deformation surface."""
    id: Optional[str] = Field(None, description="Surface scan history ID")
    nucleus: NucleusInput
    force_name: str
    points: List[BetaSurfacePoint]


class CalculationProgress(BaseModel):
    """Progress update for a running calculation."""
    calculation_id: str
    phase: CalculationPhase
    iteration: int = 0
    max_iterations: int = 200
    fluctuation: float = 0.0
    energy: float = 0.0
    message: Optional[str] = None


class CalculationStatus(BaseModel):
    """Current status of a calculation."""
    id: str = Field(..., description="Unique calculation ID")
    nucleus: NucleusInput
    force_name: str
    phase: CalculationPhase
    progress: CalculationProgress
    started_at: datetime
    completed_at: Optional[datetime] = None
    results: Optional[CalculationResults] = None
    error_message: Optional[str] = None
    run_type: RunType = Field(RunType.CALCULATION, description="Run type")
    surface_results: Optional[BetaSurfaceResult] = None


class CalculationSummary(BaseModel):
    """Summary of a calculation for listing."""
    id: str
    nucleus_symbol: str
    nucleus_a: int
    force_name: str
    phase: CalculationPhase
    energy: Optional[float] = None
    started_at: datetime
    completed_at: Optional[datetime] = None
    run_type: RunType = Field(RunType.CALCULATION, description="Run type")


class ForceInfo(BaseModel):
    """Information about a Skyrme force."""
    name: str
    description: Optional[str] = None
    t0: float
    t1: float
    t2: float
    t3: float
    x0: float
    x1: float
    x2: float
    x3: float
    b4: float
    b4p: float
    power: float


class SystemStatus(BaseModel):
    """System status information."""
    backend: str = Field(..., description="JAX backend (cpu/gpu)")
    device_name: str = Field(..., description="Device name")
    device_info: Optional[str] = Field(None, description="Additional device info")
    precision: str = Field(..., description="Floating point precision")
    jax_version: Optional[str] = Field(None, description="JAX version")
    jit_warmed_up: bool = Field(False, description="Whether JIT is warmed up")
    warmup_progress: float = Field(0.0, ge=0.0, le=1.0, description="Warmup progress")
    warmup_status: str = Field("not_started", description="Warmup status (not_started/in_progress/completed/failed)")
    warmup_error: Optional[str] = Field(None, description="Warmup error message if failed")
    active_calculations: int = Field(0, description="Number of active calculations")
    memory_used_gb: Optional[float] = Field(None, description="Memory usage (GB)")
    memory_total_gb: Optional[float] = Field(None, description="Total memory (GB)")
    gpu_memory_used_gb: Optional[float] = Field(None, description="GPU memory usage (GB)")
    gpu_memory_total_gb: Optional[float] = Field(None, description="GPU total memory (GB)")


class HistoryFilter(BaseModel):
    """Filters for history query."""
    element: Optional[str] = Field(None, description="Filter by element symbol")
    min_a: Optional[int] = Field(None, description="Minimum mass number")
    max_a: Optional[int] = Field(None, description="Maximum mass number")
    force_name: Optional[str] = Field(None, description="Filter by force name")
    status: Optional[CalculationPhase] = Field(None, description="Filter by status")
    from_date: Optional[datetime] = Field(None, description="From date")
    to_date: Optional[datetime] = Field(None, description="To date")
    run_type: Optional[RunType] = Field(None, description="Filter by run type")


class HistoryResponse(BaseModel):
    """Response for history query."""
    calculations: List[CalculationSummary]
    total: int
    page: int
    page_size: int
    total_pages: int

class GridInfo(BaseModel):
    """Grid information for density data."""
    nx: int
    ny: int
    nz: int
    dx: float
    dy: float
    dz: float


class DensityMetadata(BaseModel):
    """Metadata for density arrays."""
    min_value: float
    max_value: float
    units: str = "fm^-3"


class DensityData(BaseModel):
    """Density data for visualization."""
    density: List[List[List[float]]]  # 3D array as nested lists
    grid: GridInfo
    type: str  # "total", "neutron", "proton", "tau_n", "tau_p", "tau_total"
    metadata: DensityMetadata