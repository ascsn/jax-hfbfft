import { useMemo, useState } from 'react'
import { SingleParticleLevel } from '@/types'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui'
import { Tooltip, TooltipTrigger, TooltipContent, TooltipProvider } from '@/components/ui'
import { Button } from '@/components/ui'
import { X } from 'lucide-react'
import { formatNumber } from '@/lib/utils'

interface SpectrumPlotProps {
  levels: SingleParticleLevel[]
  fermiNeutron?: number
  fermiProton?: number
}

// Helper: Get shell letter from orbital quantum number l
function getShellLetter(l: number): string {
  const shells = ['s', 'p', 'd', 'f', 'g', 'h', 'i', 'j', 'k']
  return l < shells.length ? shells[l] : `l=${l}`
}

// Helper: Get color for orbital shell
function getShellColor(l: number, isNeutron: boolean): { fill: string; stroke: string } {
  const colors = {
    0: { base: '#3b82f6', light: '#93c5fd' },  // s: Blue
    1: { base: '#10b981', light: '#6ee7b7' },  // p: Green
    2: { base: '#f97316', light: '#fdba74' },  // d: Orange
    3: { base: '#ef4444', light: '#fca5a5' },  // f: Red
    default: { base: '#a855f7', light: '#d8b4fe' } // g+: Purple
  }
  
  const colorSet = l <= 3 ? colors[l as 0 | 1 | 2 | 3] : colors.default
  
  // Return stroke color (darker for neutrons, lighter for protons)
  return {
    fill: colorSet.base,
    stroke: isNeutron ? colorSet.base : colorSet.light
  }
}

export function SpectrumPlot({ levels, fermiNeutron = 0, fermiProton = 0 }: SpectrumPlotProps) {
  const [selectedLevel, setSelectedLevel] = useState<SingleParticleLevel | null>(null)
  
  const { neutronLevels, protonLevels, minEnergy, maxEnergy } = useMemo(() => {
    const neutrons = levels.filter(l => l.isospin === 'neutron').sort((a, b) => a.energy - b.energy)
    const protons = levels.filter(l => l.isospin === 'proton').sort((a, b) => a.energy - b.energy)
    
    const allEnergies = levels.map(l => l.energy)
    const min = Math.min(...allEnergies, fermiNeutron, fermiProton) - 5
    const max = Math.max(...allEnergies, fermiNeutron, fermiProton) + 5
    
    return {
      neutronLevels: neutrons,
      protonLevels: protons,
      minEnergy: min,
      maxEnergy: max,
    }
  }, [levels, fermiNeutron, fermiProton])
  
  const energyToY = (energy: number) => {
    const height = 400
    const range = maxEnergy - minEnergy
    return ((maxEnergy - energy) / range) * height
  }
  
  if (levels.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Single-Particle Spectrum</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground">No single-particle data available</p>
        </CardContent>
      </Card>
    )
  }
  
  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
      <Card className={selectedLevel ? 'lg:col-span-2' : 'lg:col-span-3'}>
        <CardHeader>
          <CardTitle>Single-Particle Spectrum</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex justify-center">
            <svg viewBox="-50 -20 300 440" className="w-full max-w-md">
              {/* Energy axis */}
              <line x1="100" y1="0" x2="100" y2="400" stroke="currentColor" strokeWidth="1" opacity="0.2" />
              
              {/* Energy labels */}
              {Array.from({ length: 5 }, (_, i) => {
                const energy = minEnergy + (maxEnergy - minEnergy) * (1 - i / 4)
                const y = energyToY(energy)
                return (
                  <g key={i}>
                    <line x1="95" y1={y} x2="105" y2={y} stroke="currentColor" strokeWidth="1" opacity="0.3" />
                    <text x="90" y={y + 4} textAnchor="end" className="text-xs fill-muted-foreground">
                      {formatNumber(energy, 1)}
                    </text>
                  </g>
                )
              })}
              
              {/* Axis label */}
              <text x="-30" y="200" textAnchor="middle" transform="rotate(-90 -30 200)" className="text-sm fill-muted-foreground">
                Energy (MeV)
              </text>
              
              {/* Column labels */}
              <text x="40" y="420" textAnchor="middle" className="text-sm font-medium fill-current">
                Neutrons
              </text>
              <text x="160" y="420" textAnchor="middle" className="text-sm font-medium fill-current">
                Protons
              </text>
              
              {/* Fermi levels with info icons */}
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <g className="cursor-help">
                      <line 
                        x1="0" y1={energyToY(fermiNeutron)} 
                        x2="80" y2={energyToY(fermiNeutron)} 
                        stroke="#3b82f6" 
                        strokeWidth="2" 
                        strokeDasharray="4,4"
                        opacity="0.6"
                      />
                      <circle 
                        cx="85" 
                        cy={energyToY(fermiNeutron)} 
                        r="5" 
                        fill="#3b82f6" 
                        opacity="0.8"
                      />
                      <text 
                        x="85" 
                        y={energyToY(fermiNeutron) + 3} 
                        textAnchor="middle" 
                        className="text-[8px] fill-white font-bold"
                      >
                        i
                      </text>
                    </g>
                  </TooltipTrigger>
                  <TooltipContent className="max-w-xs">
                    <p className="font-semibold">Neutron Fermi Energy (λ_n)</p>
                    <p className="text-xs mt-1">
                      Chemical potential in BCS theory. Accounts for pairing correlations and determines the occupation probability of single-particle states.
                    </p>
                    <p className="text-xs mt-1 font-mono">λ_n = {formatNumber(fermiNeutron, 3)} MeV</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
              
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <g className="cursor-help">
                      <line 
                        x1="120" y1={energyToY(fermiProton)} 
                        x2="200" y2={energyToY(fermiProton)} 
                        stroke="#ef4444" 
                        strokeWidth="2" 
                        strokeDasharray="4,4"
                        opacity="0.6"
                      />
                      <circle 
                        cx="115" 
                        cy={energyToY(fermiProton)} 
                        r="5" 
                        fill="#ef4444" 
                        opacity="0.8"
                      />
                      <text 
                        x="115" 
                        y={energyToY(fermiProton) + 3} 
                        textAnchor="middle" 
                        className="text-[8px] fill-white font-bold"
                      >
                        i
                      </text>
                    </g>
                  </TooltipTrigger>
                  <TooltipContent className="max-w-xs">
                    <p className="font-semibold">Proton Fermi Energy (λ_p)</p>
                    <p className="text-xs mt-1">
                      Chemical potential in BCS theory. Accounts for pairing correlations and determines the occupation probability of single-particle states.
                    </p>
                    <p className="text-xs mt-1 font-mono">λ_p = {formatNumber(fermiProton, 3)} MeV</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
              
              {/* Neutron levels */}
              <TooltipProvider>
                {neutronLevels.map((level) => {
                  const y = energyToY(level.energy)
                  const width = 60 * Math.max(level.occupation, 0.15)
                  const isSelected = selectedLevel?.index === level.index && selectedLevel?.isospin === level.isospin
                  const colors = getShellColor(level.l, true)
                  const paritySymbol = level.parity > 0 ? '+' : '-'
                  
                  return (
                    <Tooltip key={`${level.isospin}-${level.index}`}>
                      <TooltipTrigger asChild>
                        <g 
                          className="cursor-pointer transition-all duration-200 hover:opacity-80"
                          onClick={() => setSelectedLevel(level)}
                        >
                          <rect
                            x={40 - width / 2}
                            y={y - 4}
                            width={width}
                            height={8}
                            fill={colors.fill}
                            fillOpacity={level.occupation > 0.5 ? 0.8 : 0.4}
                            stroke={isSelected ? '#facc15' : colors.stroke}
                            strokeWidth={isSelected ? 3 : 1.5}
                            rx="2"
                            className="transition-all duration-200"
                          />
                        </g>
                      </TooltipTrigger>
                      <TooltipContent>
                        <div className="space-y-1">
                          <p className="font-bold text-sm">{level.label}</p>
                          <p className="text-xs">n={level.n}, l={level.l}, j={level.j}/2, π={paritySymbol}</p>
                          <p className="text-xs">ε = {formatNumber(level.energy, 3)} MeV</p>
                          <p className="text-xs">v² = {formatNumber(level.occupation, 3)}</p>
                          <p className="text-xs text-muted-foreground mt-1">Click for details</p>
                        </div>
                      </TooltipContent>
                    </Tooltip>
                  )
                })}
              </TooltipProvider>
              
              {/* Proton levels */}
              <TooltipProvider>
                {protonLevels.map((level) => {
                  const y = energyToY(level.energy)
                  const width = 60 * Math.max(level.occupation, 0.15)
                  const isSelected = selectedLevel?.index === level.index && selectedLevel?.isospin === level.isospin
                  const colors = getShellColor(level.l, false)
                  const paritySymbol = level.parity > 0 ? '+' : '-'
                  
                  return (
                    <Tooltip key={`${level.isospin}-${level.index}`}>
                      <TooltipTrigger asChild>
                        <g 
                          className="cursor-pointer transition-all duration-200 hover:opacity-80"
                          onClick={() => setSelectedLevel(level)}
                        >
                          <rect
                            x={160 - width / 2}
                            y={y - 4}
                            width={width}
                            height={8}
                            fill={colors.fill}
                            fillOpacity={level.occupation > 0.5 ? 0.8 : 0.4}
                            stroke={isSelected ? '#facc15' : colors.stroke}
                            strokeWidth={isSelected ? 3 : 1.5}
                            rx="2"
                            className="transition-all duration-200"
                          />
                        </g>
                      </TooltipTrigger>
                      <TooltipContent>
                        <div className="space-y-1">
                          <p className="font-bold text-sm">{level.label}</p>
                          <p className="text-xs">n={level.n}, l={level.l}, j={level.j}/2, π={paritySymbol}</p>
                          <p className="text-xs">ε = {formatNumber(level.energy, 3)} MeV</p>
                          <p className="text-xs">v² = {formatNumber(level.occupation, 3)}</p>
                          <p className="text-xs text-muted-foreground mt-1">Click for details</p>
                        </div>
                      </TooltipContent>
                    </Tooltip>
                  )
                })}
              </TooltipProvider>
            </svg>
          </div>
          
          {/* Enhanced Legend with shell colors */}
          <div className="mt-6 space-y-3">
            <div className="text-sm font-semibold text-center">Orbital Shells</div>
            <div className="flex flex-wrap justify-center gap-4 text-xs">
              <div className="flex items-center gap-1.5">
                <div className="w-6 h-2 rounded" style={{ backgroundColor: '#3b82f6' }} />
                <span>s (l=0)</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="w-6 h-2 rounded" style={{ backgroundColor: '#10b981' }} />
                <span>p (l=1)</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="w-6 h-2 rounded" style={{ backgroundColor: '#f97316' }} />
                <span>d (l=2)</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="w-6 h-2 rounded" style={{ backgroundColor: '#ef4444' }} />
                <span>f (l=3)</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="w-6 h-2 rounded" style={{ backgroundColor: '#a855f7' }} />
                <span>g+ (l≥4)</span>
              </div>
            </div>
            <div className="flex justify-center gap-6 text-xs text-muted-foreground">
              <span>Darker = Neutron | Lighter = Proton</span>
              <span>Width ∝ Occupation (v²)</span>
            </div>
          </div>
        </CardContent>
      </Card>
      
      {/* Level Detail Panel */}
      {selectedLevel && (
        <Card className="lg:col-span-1 animate-in slide-in-from-right duration-300">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-4">
            <CardTitle className="text-lg">Level Details</CardTitle>
            <Button 
              variant="ghost" 
              size="sm"
              onClick={() => setSelectedLevel(null)}
              className="h-8 w-8 p-0"
              aria-label="Close details"
            >
              <X className="h-4 w-4" />
            </Button>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Spectroscopic label */}
            <div className="text-center">
              <div 
                className="inline-block px-4 py-2 rounded-lg text-2xl font-bold"
                style={{ 
                  backgroundColor: getShellColor(selectedLevel.l, selectedLevel.isospin === 'neutron').fill + '20',
                  color: getShellColor(selectedLevel.l, selectedLevel.isospin === 'neutron').fill
                }}
              >
                {selectedLevel.label}
              </div>
            </div>
            
            {/* Quantum Numbers */}
            <div className="space-y-2">
              <h4 className="font-semibold text-sm text-muted-foreground uppercase tracking-wide">
                Quantum Numbers
              </h4>
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-muted/50 rounded-lg p-3">
                  <div className="text-xs text-muted-foreground">Principal (n)</div>
                  <div className="text-2xl font-bold">{selectedLevel.n}</div>
                </div>
                <div className="bg-muted/50 rounded-lg p-3">
                  <div className="text-xs text-muted-foreground">Orbital (l)</div>
                  <div className="text-2xl font-bold">
                    {selectedLevel.l}
                    <span className="text-sm ml-1 text-muted-foreground">({getShellLetter(selectedLevel.l)})</span>
                  </div>
                </div>
                <div className="bg-muted/50 rounded-lg p-3">
                  <div className="text-xs text-muted-foreground">Total ang. (j)</div>
                  <div className="text-2xl font-bold">{selectedLevel.j}/2</div>
                </div>
                <div className="bg-muted/50 rounded-lg p-3">
                  <div className="text-xs text-muted-foreground">Parity (π)</div>
                  <div className="text-2xl font-bold">{selectedLevel.parity > 0 ? '+' : '-'}</div>
                </div>
              </div>
            </div>
            
            {/* Physical Properties */}
            <div className="space-y-2">
              <h4 className="font-semibold text-sm text-muted-foreground uppercase tracking-wide">
                Properties
              </h4>
              <div className="space-y-2">
                <div className="flex justify-between items-center py-2 border-b">
                  <span className="text-sm text-muted-foreground">Energy</span>
                  <span className="font-mono font-semibold">{formatNumber(selectedLevel.energy, 4)} MeV</span>
                </div>
                <div className="flex justify-between items-center py-2 border-b">
                  <span className="text-sm text-muted-foreground">Occupation (v²)</span>
                  <span className="font-mono font-semibold">{formatNumber(selectedLevel.occupation, 4)}</span>
                </div>
                <div className="flex justify-between items-center py-2 border-b">
                  <span className="text-sm text-muted-foreground">Isospin</span>
                  <span className="font-semibold capitalize">{selectedLevel.isospin}</span>
                </div>
                <div className="flex justify-between items-center py-2 border-b">
                  <span className="text-sm text-muted-foreground">Shell Type</span>
                  <span className="font-semibold">
                    {getShellLetter(selectedLevel.l)}-orbital (l={selectedLevel.l})
                  </span>
                </div>
                <div className="flex justify-between items-center py-2">
                  <span className="text-sm text-muted-foreground">State Index</span>
                  <span className="font-mono">{selectedLevel.index}</span>
                </div>
              </div>
            </div>
            
            {/* Occupation visualization */}
            <div className="space-y-2">
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>Empty (u²)</span>
                <span>Occupied (v²)</span>
              </div>
              <div className="h-3 bg-muted rounded-full overflow-hidden flex">
                <div 
                  className="bg-muted-foreground/30 transition-all duration-300"
                  style={{ width: `${(1 - selectedLevel.occupation) * 100}%` }}
                />
                <div 
                  className="transition-all duration-300"
                  style={{ 
                    width: `${selectedLevel.occupation * 100}%`,
                    backgroundColor: getShellColor(selectedLevel.l, selectedLevel.isospin === 'neutron').fill
                  }}
                />
              </div>
              <div className="text-xs text-center text-muted-foreground">
                u² = {formatNumber(1 - selectedLevel.occupation, 3)} | v² = {formatNumber(selectedLevel.occupation, 3)}
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
