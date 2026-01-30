// API Types - matching the Python Pydantic models

export type CalculationPhase = 
  | 'pending' 
  | 'warmup' 
  | 'initializing' 
  | 'iterating' 
  | 'converged' 
  | 'failed' 
  | 'cancelled'

export type PairingType = 'none' | 'vdi' | 'dddi'

export type ConstraintType = 'none' | 'multipole' | 'beta_gamma'
export type RunType = 'calculation' | 'surface'

// Request types
export interface NucleusInput {
  protons: number
  neutrons: number
}

export interface GridConfig {
  nx: number
  ny: number
  nz: number
  spacing?: number
  dx?: number
  dy?: number
  dz?: number
  auto?: boolean
}

export interface PairingConfig {
  type: PairingType
  v0_neutron: number
  v0_proton: number
}

export interface ConstraintConfig {
  type: ConstraintType
  q20?: number
  q30?: number
  q40?: number
  beta2?: number
  gamma?: number
}

export interface IterationConfig {
  max_iterations: number
  convergence_threshold: number
  print_interval?: number
  x0dmp?: number           // Gradient descent damping
  e0dmp?: number           // Preconditioning energy scale
  density_mixing?: number  // Density mixing parameter
  diag_start?: number      // When to start diagonalization
  bcs_start?: number       // When to start BCS iterations
}

export interface PhysicsConfig {
  include_coulomb?: boolean
  time_reversal?: boolean
  time_odd?: boolean
  include_cm_correction?: boolean
}

export interface InitializationConfig {
  method?: 'harmonic_oscillator' | 'woods_saxon' | 'random'
  ho_length_x?: number
  ho_length_y?: number
  ho_length_z?: number
}

export interface CalculationRequest {
  nucleus: NucleusInput
  force_name: string
  grid?: GridConfig
  pairing?: PairingConfig
  constraint?: ConstraintConfig
  iteration?: IterationConfig
  physics?: PhysicsConfig
  initialization?: InitializationConfig
}

export interface BetaSurfaceRequest {
  nucleus: NucleusInput
  force_name: string
  grid?: GridConfig
  pairing?: PairingConfig
  iteration?: IterationConfig
  beta_min: number
  beta_max: number
  beta_steps: number
  gamma?: number
  hot_start?: boolean
}

// Response types
export interface EnergyBreakdown {
  total: number
  kinetic: number
  potential: number
  coulomb: number
  pairing: number
  cm_correction: number
  ehf0: number
  ehf1: number
  ehf2: number
  ehf3: number
  ehfls: number
}

export interface RadiiResults {
  neutron: number
  proton: number
  total: number
  charge: number
}

export interface DeformationResults {
  beta2: number
  gamma: number
  q20: number
  q22: number
}

export interface PairingResults {
  gap_neutron: number
  gap_proton: number
  fermi_neutron: number
  fermi_proton: number
}

export interface SingleParticleLevel {
  index: number
  isospin: 'neutron' | 'proton'
  energy: number
  occupation: number
  parity: number
  label?: string
}

export interface CalculationResults {
  energies: EnergyBreakdown
  radii: RadiiResults
  deformation: DeformationResults
  pairing: PairingResults
  single_particle_levels: SingleParticleLevel[]
  converged: boolean
  iterations: number
  final_fluctuation: number
  total_time_seconds: number
  jit_time_seconds: number
}

export interface BetaSurfacePoint {
  beta2: number
  energy: number
  converged: boolean
  iterations: number
  q20: number
  q22: number
}

export interface BetaSurfaceResult {
  id?: string
  nucleus: NucleusInput
  force_name: string
  points: BetaSurfacePoint[]
}

export interface CalculationProgress {
  calculation_id: string
  phase: CalculationPhase
  iteration: number
  max_iterations: number
  fluctuation: number
  energy: number
  message?: string
}

export interface CalculationStatus {
  id: string
  nucleus: NucleusInput
  force_name: string
  phase: CalculationPhase
  progress: CalculationProgress
  started_at: string
  completed_at?: string
  results?: CalculationResults
  error_message?: string
  run_type?: RunType
  surface_results?: BetaSurfaceResult
}

export interface CalculationSummary {
  id: string
  nucleus_symbol: string
  nucleus_a: number
  force_name: string
  phase: CalculationPhase
  energy?: number
  started_at: string
  completed_at?: string
  run_type?: RunType
}

export interface ForceInfo {
  name: string
  description?: string
  t0: number
  t1: number
  t2: number
  t3: number
  x0: number
  x1: number
  x2: number
  x3: number
  b4: number
  b4p: number
  power: number
}

export interface SystemStatus {
  backend: string
  device_name: string
  device_info?: string
  precision: string
  jax_version?: string
  jit_warmed_up: boolean
  warmup_progress: number
  warmup_status: string  // NEW: 'not_started' | 'in_progress' | 'completed' | 'failed'
  warmup_error?: string  // NEW: Error message if warmup failed
  active_calculations: number
  memory_used_gb?: number
  memory_total_gb?: number
  gpu_memory_used_gb?: number  // NEW: GPU memory usage
  gpu_memory_total_gb?: number  // NEW: GPU total memory
}

export interface WarmupStatus {
  status: 'not_started' | 'in_progress' | 'completed' | 'failed'
  progress: number
  message: string
  elapsed_seconds?: number
  is_ready: boolean
  error?: string
}

export interface HistoryResponse {
  calculations: CalculationSummary[]
  total: number
  page: number
  page_size: number
  total_pages: number
}

export interface Element {
  symbol: string
  z: number
  name: string
}

export interface NucleusPreset {
  symbol: string
  a: number
  z: number
  n: number
  label: string
}

// WebSocket message types
export interface WSMessage {
  type: string
  data?: unknown
  message?: string
  calculation_id?: string
}

export interface WSProgressMessage extends WSMessage {
  type: 'progress'
  data: CalculationProgress
}

export interface WSStatusMessage extends WSMessage {
  type: 'status'
  data: CalculationStatus
}

export interface WSWarmupMessage extends WSMessage {
  type: 'warmup_status'
  data: WarmupStatus
}

// System Diagnostics
export interface DeviceDetail {
  id: number
  platform: string
  device_kind: string
  memory_stats?: {
    bytes_in_use_gb: number
    bytes_limit_gb: number
    peak_bytes_in_use_gb: number
  }
  memory_stats_error?: string
}

export interface MemoryInfo {
  total_gb: number
  available_gb: number
  used_gb: number
  percent: number
}

export interface GPUInfo {
  index: number
  name: string
  driver_version: string
  memory_total_mb: number
  memory_used_mb: number
  memory_free_mb: number
  temperature_c?: number
  utilization_percent?: number
}

export interface SystemDiagnostics {
  python: {
    version: string
    platform: string
    architecture: string
  }
  jax: {
    version: string
    backend: string
    devices: string[]
    device_count: number
    x64_enabled: boolean
    device_details: DeviceDetail[]
  } | { error: string }
  system_memory: MemoryInfo | { error: string }
  nvidia_gpus?: GPUInfo[] | { error: string }
  jit_warmup: WarmupStatus
  packages: Record<string, string> | { error: string }
}
