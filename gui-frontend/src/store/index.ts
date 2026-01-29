import { create } from 'zustand'
import { 
  SystemStatus, 
  WarmupStatus, 
  CalculationStatus, 
  CalculationProgress,
  CalculationRequest,
  GridConfig,
  PairingConfig,
  ConstraintConfig,
  IterationConfig,
  PhysicsConfig,
  InitializationConfig,
} from '@/types'

// Default form values
const defaultGrid: GridConfig = {
  nx: 24,
  ny: 24,
  nz: 24,
  spacing: 1.0,
  auto: true,
}

const defaultPairing: PairingConfig = {
  type: 'none',
  v0_neutron: -200.0,
  v0_proton: -200.0,
}

const defaultConstraint: ConstraintConfig = {
  type: 'none',
}

const defaultIteration: IterationConfig = {
  max_iterations: 200,
  convergence_threshold: 1e-6,
  print_interval: 10,
  x0dmp: 0.45,
  e0dmp: 20.0,
  density_mixing: 0.5,
  diag_start: 30,
  bcs_start: 30,
}

const defaultPhysics: PhysicsConfig = {
  include_coulomb: true,
  time_reversal: true,
  time_odd: false,
  include_cm_correction: true,
}

const defaultInitialization: InitializationConfig = {
  method: 'harmonic_oscillator',
  ho_length_x: 3.0,
  ho_length_y: 3.0,
  ho_length_z: 3.0,
}

interface CalculationFormState {
  protons: number
  neutrons: number
  forceName: string
  grid: GridConfig
  pairing: PairingConfig
  constraint: ConstraintConfig
  iteration: IterationConfig
  physics: PhysicsConfig
  initialization: InitializationConfig
}

interface AppState {
  // System status
  systemStatus: SystemStatus | null
  warmupStatus: WarmupStatus | null
  
  // Active calculations
  activeCalculations: Map<string, CalculationStatus>
  
  // Currently selected calculation (for results view)
  selectedCalculationId: string | null
  
  // Form state
  form: CalculationFormState
  
  // WebSocket connection state
  wsConnected: boolean
  
  // UI state
  showAdvancedOptions: boolean
  
  // Actions
  setSystemStatus: (status: SystemStatus) => void
  setWarmupStatus: (status: WarmupStatus) => void
  setWsConnected: (connected: boolean) => void
  
  addCalculation: (calc: CalculationStatus) => void
  updateCalculation: (id: string, update: Partial<CalculationStatus>) => void
  updateCalculationProgress: (progress: CalculationProgress) => void
  removeCalculation: (id: string) => void
  
  setSelectedCalculation: (id: string | null) => void
  
  // Form actions
  setFormField: <K extends keyof CalculationFormState>(
    field: K, 
    value: CalculationFormState[K]
  ) => void
  setNucleus: (protons: number, neutrons: number) => void
  resetForm: () => void
  getCalculationRequest: () => CalculationRequest
  
  setShowAdvancedOptions: (show: boolean) => void
}

export const useStore = create<AppState>((set, get) => ({
  // Initial state
  systemStatus: null,
  warmupStatus: null,
  activeCalculations: new Map(),
  selectedCalculationId: null,
  wsConnected: false,
  showAdvancedOptions: false,
  
  form: {
    protons: 8,
    neutrons: 8,
    forceName: 'SLy4',
    grid: { ...defaultGrid },
    pairing: { ...defaultPairing },
    constraint: { ...defaultConstraint },
    iteration: { ...defaultIteration },
    physics: { ...defaultPhysics },
    initialization: { ...defaultInitialization },
  },
  
  // Actions
  setSystemStatus: (status) => set({ systemStatus: status }),
  setWarmupStatus: (status) => set({ warmupStatus: status }),
  setWsConnected: (connected) => set({ wsConnected: connected }),
  
  addCalculation: (calc) => set((state) => {
    const newMap = new Map(state.activeCalculations)
    newMap.set(calc.id, calc)
    return { activeCalculations: newMap }
  }),
  
  updateCalculation: (id, update) => set((state) => {
    const existing = state.activeCalculations.get(id)
    if (!existing) return state
    
    const newMap = new Map(state.activeCalculations)
    newMap.set(id, { ...existing, ...update })
    return { activeCalculations: newMap }
  }),
  
  updateCalculationProgress: (progress) => set((state) => {
    const existing = state.activeCalculations.get(progress.calculation_id)
    if (!existing) return state
    
    const newMap = new Map(state.activeCalculations)
    newMap.set(progress.calculation_id, {
      ...existing,
      phase: progress.phase,
      progress,
    })
    return { activeCalculations: newMap }
  }),
  
  removeCalculation: (id) => set((state) => {
    const newMap = new Map(state.activeCalculations)
    newMap.delete(id)
    return { activeCalculations: newMap }
  }),
  
  setSelectedCalculation: (id) => set({ selectedCalculationId: id }),
  
  setFormField: (field, value) => set((state) => ({
    form: { ...state.form, [field]: value }
  })),
  
  setNucleus: (protons, neutrons) => set((state) => ({
    form: { ...state.form, protons, neutrons }
  })),
  
  resetForm: () => set(() => ({
    form: {
      protons: 8,
      neutrons: 8,
      forceName: 'SLy4',
      grid: { ...defaultGrid },
      pairing: { ...defaultPairing },
      constraint: { ...defaultConstraint },
      iteration: { ...defaultIteration },
      physics: { ...defaultPhysics },
      initialization: { ...defaultInitialization },
    }
  })),
  
  getCalculationRequest: () => {
    const { form } = get()
    return {
      nucleus: {
        protons: form.protons,
        neutrons: form.neutrons,
      },
      force_name: form.forceName,
      grid: form.grid,
      pairing: form.pairing,
      constraint: form.constraint,
      iteration: form.iteration,
      physics: form.physics,
      initialization: form.initialization,
    }
  },
  
  setShowAdvancedOptions: (show) => set({ showAdvancedOptions: show }),
}))
