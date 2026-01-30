import { useStore } from '@/store'
import { usePresets, useForces } from '@/hooks'
import { Card, CardHeader, CardTitle, CardDescription, CardContent, Button, Input, Label, Select } from '@/components/ui'
import { Tooltip, TooltipTrigger, TooltipContent, TooltipProvider } from '@/components/ui'
import { HelpCircle, ChevronDown, ChevronUp } from 'lucide-react'
import { formatNucleus } from '@/lib/utils'

export function NucleusInput() {
  const { form, setFormField, setNucleus, showAdvancedOptions, setShowAdvancedOptions } = useStore()
  const { data: presets } = usePresets()
  const { data: forces } = useForces()
  
  const massNumber = form.protons + form.neutrons
  const nucleusName = formatNucleus(form.protons, form.neutrons)
  
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          Nucleus Configuration
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger>
                <HelpCircle className="w-4 h-4 text-muted-foreground" />
              </TooltipTrigger>
              <TooltipContent className="max-w-xs">
                <p>Configure the nucleus to calculate. Select a preset or enter custom proton/neutron numbers.</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </CardTitle>
        <CardDescription>
          Selected: <strong>{nucleusName}</strong> (A = {massNumber})
        </CardDescription>
      </CardHeader>
      
      <CardContent className="space-y-6">
        {/* Presets */}
        <div className="space-y-2">
          <Label>Quick Presets</Label>
          <div className="flex flex-wrap gap-2">
            {presets?.slice(0, 8).map((preset) => (
              <Button
                key={`${preset.symbol}-${preset.a}`}
                variant={form.protons === preset.z && form.neutrons === preset.n ? 'default' : 'outline'}
                size="sm"
                onClick={() => setNucleus(preset.z, preset.n)}
              >
                {preset.symbol}-{preset.a}
              </Button>
            ))}
          </div>
        </div>
        
        {/* Manual input */}
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label htmlFor="protons" className="flex items-center gap-1">
              Protons (Z)
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger>
                    <HelpCircle className="w-3 h-3 text-muted-foreground" />
                  </TooltipTrigger>
                  <TooltipContent>Atomic number, typically 1-92</TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </Label>
            <Input
              id="protons"
              type="number"
              min={1}
              max={120}
              value={form.protons}
              onChange={(e) => {
                const val = e.target.value
                if (val === '') {
                  setFormField('protons', '' as any)
                } else {
                  const num = parseInt(val)
                  setFormField('protons', isNaN(num) ? 1 : Math.max(1, Math.min(120, num)))
                }
              }}
              onBlur={(e) => {
                if (e.target.value === '') {
                  setFormField('protons', 1)
                }
              }}
            />
          </div>
          
          <div className="space-y-2">
            <Label htmlFor="neutrons" className="flex items-center gap-1">
              Neutrons (N)
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger>
                    <HelpCircle className="w-3 h-3 text-muted-foreground" />
                  </TooltipTrigger>
                  <TooltipContent>Neutron number, typically N ≥ Z for stable nuclei</TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </Label>
            <Input
              id="neutrons"
              type="number"
              min={1}
              max={200}
              value={form.neutrons}
              onChange={(e) => {
                const val = e.target.value
                if (val === '') {
                  setFormField('neutrons', '' as any)
                } else {
                  const num = parseInt(val)
                  setFormField('neutrons', isNaN(num) ? 1 : Math.max(1, Math.min(200, num)))
                }
              }}
              onBlur={(e) => {
                if (e.target.value === '') {
                  setFormField('neutrons', 1)
                }
              }}
            />
          </div>
        </div>
        
        {/* Force selection */}
        <div className="space-y-2">
          <Label htmlFor="force" className="flex items-center gap-1">
            Skyrme Force
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger>
                  <HelpCircle className="w-3 h-3 text-muted-foreground" />
                </TooltipTrigger>
                <TooltipContent className="max-w-xs">
                  <p>The nuclear interaction parameterization. SLy4 is a popular general-purpose choice.</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </Label>
          <Select
            id="force"
            value={form.forceName}
            onChange={(e) => setFormField('forceName', e.target.value)}
            options={
              forces?.map((f) => ({ value: f.name, label: f.name })) || 
              [{ value: 'SLy4', label: 'SLy4' }]
            }
          />
        </div>
        
        {/* Advanced options toggle */}
        <Button
          variant="ghost"
          className="w-full justify-between"
          onClick={() => setShowAdvancedOptions(!showAdvancedOptions)}
        >
          <span>Advanced Options</span>
          {showAdvancedOptions ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </Button>
        
        {/* Advanced options */}
        {showAdvancedOptions && (
          <div className="space-y-4 pt-4 border-t">
            {/* Grid configuration */}
            <div className="space-y-2">
              <Label>Grid Size</Label>
              <div className="grid grid-cols-3 gap-2">
                <div>
                  <Label htmlFor="nx" className="text-xs text-muted-foreground">Nx</Label>
                  <Input
                    id="nx"
                    type="number"
                    min={8}
                    max={64}
                    value={form.grid.nx}
                    onChange={(e) => {
                      const val = e.target.value
                      if (val === '') {
                        setFormField('grid', { ...form.grid, nx: '' as any, auto: false })
                      } else {
                        const num = parseInt(val)
                        setFormField('grid', { ...form.grid, nx: isNaN(num) ? 24 : Math.max(8, Math.min(64, num)), auto: false })
                      }
                    }}
                    onBlur={(e) => {
                      if (e.target.value === '') {
                        setFormField('grid', { ...form.grid, nx: 24, auto: false })
                      }
                    }}
                  />
                </div>
                <div>
                  <Label htmlFor="ny" className="text-xs text-muted-foreground">Ny</Label>
                  <Input
                    id="ny"
                    type="number"
                    min={8}
                    max={64}
                    value={form.grid.ny}
                    onChange={(e) => {
                      const val = e.target.value
                      if (val === '') {
                        setFormField('grid', { ...form.grid, ny: '' as any, auto: false })
                      } else {
                        const num = parseInt(val)
                        setFormField('grid', { ...form.grid, ny: isNaN(num) ? 24 : Math.max(8, Math.min(64, num)), auto: false })
                      }
                    }}
                    onBlur={(e) => {
                      if (e.target.value === '') {
                        setFormField('grid', { ...form.grid, ny: 24, auto: false })
                      }
                    }}
                  />
                </div>
                <div>
                  <Label htmlFor="nz" className="text-xs text-muted-foreground">Nz</Label>
                  <Input
                    id="nz"
                    type="number"
                    min={8}
                    max={64}
                    value={form.grid.nz}
                    onChange={(e) => {
                      const val = e.target.value
                      if (val === '') {
                        setFormField('grid', { ...form.grid, nz: '' as any, auto: false })
                      } else {
                        const num = parseInt(val)
                        setFormField('grid', { ...form.grid, nz: isNaN(num) ? 24 : Math.max(8, Math.min(64, num)), auto: false })
                      }
                    }}
                    onBlur={(e) => {
                      if (e.target.value === '') {
                        setFormField('grid', { ...form.grid, nz: 24, auto: false })
                      }
                    }}
                  />
                </div>
              </div>
            </div>
            
            {/* Iteration settings */}
            <div className="space-y-2">
              <Label htmlFor="maxIterations">Max Iterations</Label>
              <Input
                id="maxIterations"
                type="number"
                min={10}
                max={5000}
                value={form.iteration.max_iterations}
                onChange={(e) => {
                  const val = e.target.value
                  if (val === '') {
                    setFormField('iteration', { ...form.iteration, max_iterations: '' as any })
                  } else {
                    const num = parseInt(val)
                    setFormField('iteration', { ...form.iteration, max_iterations: isNaN(num) ? 200 : Math.max(10, Math.min(5000, num)) })
                  }
                }}
                onBlur={(e) => {
                  if (e.target.value === '') {
                    setFormField('iteration', { ...form.iteration, max_iterations: 200 })
                  }
                }}
              />
            </div>
            
            {/* Pairing */}
            <div className="space-y-2">
              <Label htmlFor="pairing">Pairing Type</Label>
              <Select
                id="pairing"
                value={form.pairing.type}
                onChange={(e) => setFormField('pairing', { ...form.pairing, type: e.target.value as 'none' | 'vdi' | 'dddi' })}
                options={[
                  { value: 'none', label: 'None' },
                  { value: 'vdi', label: 'VDI (Volume)' },
                  { value: 'dddi', label: 'DDDI (Density-Dependent)' },
                ]}
              />
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
