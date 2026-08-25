import { useStore } from '@/store'
import { usePresets, useForces } from '@/hooks'
import { 
  Card, CardHeader, CardTitle, CardDescription, CardContent, 
  Button, Input, Label, Select,
  Accordion, AccordionItem, AccordionTrigger, AccordionContent 
} from '@/components/ui'
import { Tooltip, TooltipTrigger, TooltipContent, TooltipProvider } from '@/components/ui'
import { HelpCircle, Settings2, Grid3X3, Repeat, Atom, Waves, Target, Cog } from 'lucide-react'
import { formatNucleus, cn } from '@/lib/utils'

function HelpTip({ children }: { children: React.ReactNode }) {
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger>
          <HelpCircle className="w-3 h-3 text-muted-foreground ml-1" />
        </TooltipTrigger>
        <TooltipContent className="max-w-xs">
          {children}
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  )
}

export function CalculationForm() {
  const { form, setFormField, setNucleus } = useStore()
  const { data: presets } = usePresets()
  const { data: forces } = useForces()
  
  const massNumber = form.protons + form.neutrons
  const nucleusName = formatNucleus(form.protons, form.neutrons)
  
  return (
    <Card className="animate-fade-in">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Settings2 className="w-5 h-5" />
          Calculation Setup
        </CardTitle>
        <CardDescription>
          Configure <strong className="text-foreground">{nucleusName}</strong> (A = {massNumber}) calculation
        </CardDescription>
      </CardHeader>
      
      <CardContent className="space-y-6">
        {/* Nucleus Presets */}
        <div className="space-y-2">
          <Label>Quick Presets</Label>
          <div className="flex flex-wrap gap-2">
            {presets?.slice(0, 10).map((preset, idx) => (
              <Button
                key={`${preset.symbol}-${preset.a}`}
                variant={form.protons === preset.z && form.neutrons === preset.n ? 'default' : 'outline'}
                size="sm"
                onClick={() => setNucleus(preset.z, preset.n)}
                className={cn(
                  "interactive-scale animate-scale-in",
                  `stagger-${Math.min(idx + 1, 10)}`
                )}
              >
                {preset.symbol}-{preset.a}
              </Button>
            ))}
          </div>
        </div>
        
        {/* Basic Settings - Always visible */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="space-y-2">
            <Label htmlFor="protons" className="flex items-center">
              Protons (Z)
              <HelpTip>Atomic number, typically 1-92</HelpTip>
            </Label>
            <Input
              id="protons"
              type="number"
              min={1}
              max={120}
              value={form.protons}
              onChange={(e) => setFormField('protons', parseInt(e.target.value) || 1)}
              className="interactive-glow"
              aria-label="Number of protons"
            />
          </div>
          
          <div className="space-y-2">
            <Label htmlFor="neutrons" className="flex items-center">
              Neutrons (N)
              <HelpTip>Neutron number, typically N ≥ Z for stable nuclei</HelpTip>
            </Label>
            <Input
              id="neutrons"
              type="number"
              min={1}
              max={200}
              value={form.neutrons}
              onChange={(e) => setFormField('neutrons', parseInt(e.target.value) || 1)}
              className="interactive-glow"
              aria-label="Number of neutrons"
            />
          </div>
          
          <div className="space-y-2 col-span-2">
            <Label htmlFor="force" className="flex items-center">
              Skyrme Force
              <HelpTip>Nuclear interaction parameterization. SLy4 is a popular general-purpose choice.</HelpTip>
            </Label>
            <Select
              id="force"
              value={form.forceName}
              onChange={(e) => setFormField('forceName', e.target.value)}
              options={
                forces?.map((f) => ({ value: f.name, label: f.name })) || 
                [{ value: 'SLy4', label: 'SLy4' }]
              }
              className="interactive-glow"
              aria-label="Skyrme force selection"
            />
          </div>
        </div>
        
        {/* Advanced Settings Accordion */}
        <Accordion type="multiple" className="w-full">
          {/* Grid Settings */}
          <AccordionItem value="grid">
            <AccordionTrigger className="text-sm hover:no-underline">
              <div className="flex items-center gap-2">
                <Grid3X3 className="w-4 h-4 transition-transform duration-300 group-hover:rotate-180" />
                Grid Configuration
              </div>
            </AccordionTrigger>
            <AccordionContent className="animate-slide-down">
              <div className="space-y-4 pt-2">
                <div className="grid grid-cols-3 gap-3">
                  <div className="space-y-1">
                    <Label htmlFor="nx" className="text-xs">Nx</Label>
                    <Input
                      id="nx"
                      type="number"
                      min={8}
                      max={64}
                      step={2}
                      value={form.grid.nx}
                      onChange={(e) => setFormField('grid', { ...form.grid, nx: parseInt(e.target.value) || 24, auto: false })}
                    />
                  </div>
                  <div className="space-y-1">
                    <Label htmlFor="ny" className="text-xs">Ny</Label>
                    <Input
                      id="ny"
                      type="number"
                      min={8}
                      max={64}
                      step={2}
                      value={form.grid.ny}
                      onChange={(e) => setFormField('grid', { ...form.grid, ny: parseInt(e.target.value) || 24, auto: false })}
                    />
                  </div>
                  <div className="space-y-1">
                    <Label htmlFor="nz" className="text-xs">Nz</Label>
                    <Input
                      id="nz"
                      type="number"
                      min={8}
                      max={64}
                      step={2}
                      value={form.grid.nz}
                      onChange={(e) => setFormField('grid', { ...form.grid, nz: parseInt(e.target.value) || 24, auto: false })}
                    />
                  </div>
                </div>
                <div className="space-y-1">
                  <Label htmlFor="spacing" className="text-xs flex items-center">
                    Grid Spacing (fm)
                    <HelpTip>Smaller spacing gives more accuracy but needs larger box. Typical: 0.8-1.0 fm</HelpTip>
                  </Label>
                  <Input
                    id="spacing"
                    type="number"
                    min={0.5}
                    max={2.0}
                    step={0.1}
                    value={form.grid.spacing ?? 1.0}
                    onChange={(e) => setFormField('grid', { ...form.grid, spacing: parseFloat(e.target.value) || 1.0, auto: false })}
                  />
                </div>
              </div>
            </AccordionContent>
          </AccordionItem>
          
          {/* Iteration Settings */}
          <AccordionItem value="iteration">
            <AccordionTrigger className="text-sm">
              <div className="flex items-center gap-2">
                <Repeat className="w-4 h-4" />
                Iteration Control
              </div>
            </AccordionTrigger>
            <AccordionContent>
              <div className="space-y-4 pt-2">
                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-1">
                    <Label htmlFor="maxIterations" className="text-xs">Max Iterations</Label>
                    <Input
                      id="maxIterations"
                      type="number"
                      min={10}
                      max={5000}
                      value={form.iteration.max_iterations}
                      onChange={(e) => setFormField('iteration', { ...form.iteration, max_iterations: parseInt(e.target.value) || 200 })}
                    />
                  </div>
                  <div className="space-y-1">
                    <Label htmlFor="threshold" className="text-xs flex items-center">
                      Convergence
                      <HelpTip>Energy fluctuation threshold in MeV. Typical: 1e-5 to 1e-7</HelpTip>
                    </Label>
                    <Input
                      id="threshold"
                      type="number"
                      step="any"
                      value={form.iteration.convergence_threshold}
                      onChange={(e) => setFormField('iteration', { ...form.iteration, convergence_threshold: parseFloat(e.target.value) || 1e-6 })}
                    />
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-1">
                    <Label htmlFor="x0dmp" className="text-xs flex items-center">
                      Damping (x0dmp)
                      <HelpTip>Gradient descent damping. Smaller = more stable but slower. Typical: 0.3-0.5</HelpTip>
                    </Label>
                    <Input
                      id="x0dmp"
                      type="number"
                      min={0.1}
                      max={1.0}
                      step={0.05}
                      value={form.iteration.x0dmp ?? 0.45}
                      onChange={(e) => setFormField('iteration', { ...form.iteration, x0dmp: parseFloat(e.target.value) || 0.45 })}
                    />
                  </div>
                  <div className="space-y-1">
                    <Label htmlFor="densityMixing" className="text-xs flex items-center">
                      Density Mixing
                      <HelpTip>Fraction of new density mixed with old. Smaller = more stable. Typical: 0.3-0.7</HelpTip>
                    </Label>
                    <Input
                      id="densityMixing"
                      type="number"
                      min={0.1}
                      max={1.0}
                      step={0.1}
                      value={form.iteration.density_mixing ?? 0.5}
                      onChange={(e) => setFormField('iteration', { ...form.iteration, density_mixing: parseFloat(e.target.value) || 0.5 })}
                    />
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-1">
                    <Label htmlFor="diagStart" className="text-xs flex items-center">
                      Diag Start
                      <HelpTip>Iteration to start full diagonalization. Before this, gradient descent only.</HelpTip>
                    </Label>
                    <Input
                      id="diagStart"
                      type="number"
                      min={0}
                      max={200}
                      value={form.iteration.diag_start ?? 30}
                      onChange={(e) => setFormField('iteration', { ...form.iteration, diag_start: parseInt(e.target.value) || 30 })}
                    />
                  </div>
                  <div className="space-y-1">
                    <Label htmlFor="e0dmp" className="text-xs flex items-center">
                      Precond. Energy
                      <HelpTip>Preconditioning energy scale (MeV). Typical: 20-100</HelpTip>
                    </Label>
                    <Input
                      id="e0dmp"
                      type="number"
                      min={5}
                      max={200}
                      value={form.iteration.e0dmp ?? 20}
                      onChange={(e) => setFormField('iteration', { ...form.iteration, e0dmp: parseFloat(e.target.value) || 20 })}
                    />
                  </div>
                </div>
              </div>
            </AccordionContent>
          </AccordionItem>
          
          {/* Pairing Settings */}
          <AccordionItem value="pairing">
            <AccordionTrigger className="text-sm">
              <div className="flex items-center gap-2">
                <Waves className="w-4 h-4" />
                Pairing Interaction
              </div>
            </AccordionTrigger>
            <AccordionContent>
              <div className="space-y-4 pt-2">
                <div className="space-y-1">
                  <Label htmlFor="pairing" className="text-xs">Pairing Type</Label>
                  <Select
                    id="pairing"
                    value={form.pairing.type}
                    onChange={(e) => setFormField('pairing', { ...form.pairing, type: e.target.value as 'none' | 'vdi' | 'dddi' })}
                    options={[
                      { value: 'none', label: 'None (pure Hartree-Fock)' },
                      { value: 'vdi', label: 'VDI (Volume Delta Interaction)' },
                      { value: 'dddi', label: 'DDDI (Density-Dependent, recommended)' },
                    ]}
                  />
                </div>
                {form.pairing.type !== 'none' && (
                  <div className="grid grid-cols-2 gap-3">
                    <div className="space-y-1">
                      <Label htmlFor="v0n" className="text-xs flex items-center">
                        V₀ Neutron (MeV·fm³)
                        <HelpTip>Neutron pairing strength. Typical: -200 to -400</HelpTip>
                      </Label>
                      <Input
                        id="v0n"
                        type="number"
                        value={form.pairing.v0_neutron}
                        onChange={(e) => setFormField('pairing', { ...form.pairing, v0_neutron: parseFloat(e.target.value) || -200 })}
                      />
                    </div>
                    <div className="space-y-1">
                      <Label htmlFor="v0p" className="text-xs flex items-center">
                        V₀ Proton (MeV·fm³)
                        <HelpTip>Proton pairing strength. Typical: -200 to -400</HelpTip>
                      </Label>
                      <Input
                        id="v0p"
                        type="number"
                        value={form.pairing.v0_proton}
                        onChange={(e) => setFormField('pairing', { ...form.pairing, v0_proton: parseFloat(e.target.value) || -200 })}
                      />
                    </div>
                  </div>
                )}
              </div>
            </AccordionContent>
          </AccordionItem>
          
          {/* Constraints */}
          <AccordionItem value="constraints">
            <AccordionTrigger className="text-sm">
              <div className="flex items-center gap-2">
                <Target className="w-4 h-4" />
                Constraints
              </div>
            </AccordionTrigger>
            <AccordionContent>
              <div className="space-y-4 pt-2">
                <div className="space-y-1">
                  <Label htmlFor="constraintType" className="text-xs">Constraint Type</Label>
                  <Select
                    id="constraintType"
                    value={form.constraint.type}
                    onChange={(e) => setFormField('constraint', { ...form.constraint, type: e.target.value as 'none' | 'multipole' | 'beta_gamma' })}
                    options={[
                      { value: 'none', label: 'None (unconstrained)' },
                      { value: 'multipole', label: 'Multipole Moments (Q₂₀, Q₃₀, ...)' },
                      { value: 'beta_gamma', label: 'β-γ Deformation' },
                    ]}
                  />
                </div>
                {form.constraint.type === 'multipole' && (
                  <div className="grid grid-cols-3 gap-3">
                    <div className="space-y-1">
                      <Label htmlFor="q20" className="text-xs">Q₂₀ (fm²)</Label>
                      <Input
                        id="q20"
                        type="number"
                        value={form.constraint.q20 ?? 0}
                        onChange={(e) => setFormField('constraint', { ...form.constraint, q20: parseFloat(e.target.value) || 0 })}
                      />
                    </div>
                    <div className="space-y-1">
                      <Label htmlFor="q30" className="text-xs">Q₃₀ (fm³)</Label>
                      <Input
                        id="q30"
                        type="number"
                        value={form.constraint.q30 ?? 0}
                        onChange={(e) => setFormField('constraint', { ...form.constraint, q30: parseFloat(e.target.value) || 0 })}
                      />
                    </div>
                    <div className="space-y-1">
                      <Label htmlFor="q40" className="text-xs">Q₄₀ (fm⁴)</Label>
                      <Input
                        id="q40"
                        type="number"
                        value={form.constraint.q40 ?? 0}
                        onChange={(e) => setFormField('constraint', { ...form.constraint, q40: parseFloat(e.target.value) || 0 })}
                      />
                    </div>
                  </div>
                )}
                {form.constraint.type === 'beta_gamma' && (
                  <div className="grid grid-cols-2 gap-3">
                    <div className="space-y-1">
                      <Label htmlFor="beta2" className="text-xs flex items-center">
                        β₂
                        <HelpTip>Quadrupole deformation parameter</HelpTip>
                      </Label>
                      <Input
                        id="beta2"
                        type="number"
                        step={0.05}
                        value={form.constraint.beta2 ?? 0}
                        onChange={(e) => setFormField('constraint', { ...form.constraint, beta2: parseFloat(e.target.value) || 0 })}
                      />
                    </div>
                    <div className="space-y-1">
                      <Label htmlFor="gamma" className="text-xs flex items-center">
                        γ (degrees)
                        <HelpTip>Triaxiality parameter (0=prolate, 60=oblate)</HelpTip>
                      </Label>
                      <Input
                        id="gamma"
                        type="number"
                        min={0}
                        max={60}
                        value={form.constraint.gamma ?? 0}
                        onChange={(e) => setFormField('constraint', { ...form.constraint, gamma: parseFloat(e.target.value) || 0 })}
                      />
                    </div>
                  </div>
                )}
              </div>
            </AccordionContent>
          </AccordionItem>
          
          {/* Physics Options */}
          <AccordionItem value="physics">
            <AccordionTrigger className="text-sm">
              <div className="flex items-center gap-2">
                <Cog className="w-4 h-4" />
                Physics Options
              </div>
            </AccordionTrigger>
            <AccordionContent>
              <div className="space-y-3 pt-2">
                <label className="flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={form.physics.include_coulomb ?? true}
                    onChange={(e) => setFormField('physics', { ...form.physics, include_coulomb: e.target.checked })}
                    className="rounded"
                  />
                  Include Coulomb interaction
                  <HelpTip>Include electrostatic repulsion between protons</HelpTip>
                </label>
                <label className="flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={form.physics.include_cm_correction ?? true}
                    onChange={(e) => setFormField('physics', { ...form.physics, include_cm_correction: e.target.checked })}
                    className="rounded"
                  />
                  Center-of-mass correction
                  <HelpTip>Correct for spurious center-of-mass motion</HelpTip>
                </label>
                <label className="flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={form.physics.time_reversal ?? true}
                    onChange={(e) => setFormField('physics', { ...form.physics, time_reversal: e.target.checked })}
                    className="rounded"
                  />
                  Time-reversal symmetry
                  <HelpTip>Standard for even-even nuclei</HelpTip>
                </label>
                <label className="flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={form.physics.time_odd ?? false}
                    onChange={(e) => setFormField('physics', { ...form.physics, time_odd: e.target.checked })}
                    className="rounded"
                  />
                  Time-odd terms
                  <HelpTip>Include for odd-mass or rotating nuclei</HelpTip>
                </label>
              </div>
            </AccordionContent>
          </AccordionItem>
          
          {/* Initialization */}
          <AccordionItem value="initialization">
            <AccordionTrigger className="text-sm">
              <div className="flex items-center gap-2">
                <Atom className="w-4 h-4" />
                Initialization
              </div>
            </AccordionTrigger>
            <AccordionContent>
              <div className="space-y-4 pt-2">
                <div className="space-y-1">
                  <Label htmlFor="initMethod" className="text-xs">Initialization Method</Label>
                  <Select
                    id="initMethod"
                    value={form.initialization.method ?? 'harmonic_oscillator'}
                    onChange={(e) => setFormField('initialization', { ...form.initialization, method: e.target.value as 'harmonic_oscillator' | 'woods_saxon' | 'random' })}
                    options={[
                      { value: 'harmonic_oscillator', label: 'Harmonic Oscillator (recommended)' },
                      { value: 'woods_saxon', label: 'Woods-Saxon' },
                      { value: 'random', label: 'Random (testing only)' },
                    ]}
                  />
                </div>
                {form.initialization.method === 'harmonic_oscillator' && (
                  <div className="grid grid-cols-3 gap-3">
                    <div className="space-y-1">
                      <Label htmlFor="hoX" className="text-xs">HO Length X (fm)</Label>
                      <Input
                        id="hoX"
                        type="number"
                        min={1}
                        max={10}
                        step={0.1}
                        value={form.initialization.ho_length_x ?? 3.0}
                        onChange={(e) => setFormField('initialization', { ...form.initialization, ho_length_x: parseFloat(e.target.value) || 3.0 })}
                      />
                    </div>
                    <div className="space-y-1">
                      <Label htmlFor="hoY" className="text-xs">HO Length Y (fm)</Label>
                      <Input
                        id="hoY"
                        type="number"
                        min={1}
                        max={10}
                        step={0.1}
                        value={form.initialization.ho_length_y ?? 3.0}
                        onChange={(e) => setFormField('initialization', { ...form.initialization, ho_length_y: parseFloat(e.target.value) || 3.0 })}
                      />
                    </div>
                    <div className="space-y-1">
                      <Label htmlFor="hoZ" className="text-xs">HO Length Z (fm)</Label>
                      <Input
                        id="hoZ"
                        type="number"
                        min={1}
                        max={10}
                        step={0.1}
                        value={form.initialization.ho_length_z ?? 3.0}
                        onChange={(e) => setFormField('initialization', { ...form.initialization, ho_length_z: parseFloat(e.target.value) || 3.0 })}
                      />
                    </div>
                  </div>
                )}
              </div>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      </CardContent>
    </Card>
  )
}
