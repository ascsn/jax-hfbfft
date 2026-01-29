import { useMemo } from 'react'
import { SingleParticleLevel } from '@/types'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui'
import { Tooltip, TooltipTrigger, TooltipContent, TooltipProvider } from '@/components/ui'
import { formatNumber } from '@/lib/utils'

interface SpectrumPlotProps {
  levels: SingleParticleLevel[]
  fermiNeutron?: number
  fermiProton?: number
}

export function SpectrumPlot({ levels, fermiNeutron = 0, fermiProton = 0 }: SpectrumPlotProps) {
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
    <Card>
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
            <text x="40" y="420" textAnchor="middle" className="text-sm fill-neutron font-medium">
              Neutrons
            </text>
            <text x="160" y="420" textAnchor="middle" className="text-sm fill-proton font-medium">
              Protons
            </text>
            
            {/* Fermi levels */}
            <line 
              x1="0" y1={energyToY(fermiNeutron)} 
              x2="80" y2={energyToY(fermiNeutron)} 
              stroke="#3b82f6" 
              strokeWidth="2" 
              strokeDasharray="4,4"
              opacity="0.5"
            />
            <line 
              x1="120" y1={energyToY(fermiProton)} 
              x2="200" y2={energyToY(fermiProton)} 
              stroke="#ef4444" 
              strokeWidth="2" 
              strokeDasharray="4,4"
              opacity="0.5"
            />
            
            {/* Neutron levels */}
            <TooltipProvider>
              {neutronLevels.map((level) => {
                const y = energyToY(level.energy)
                const width = 60 * level.occupation
                return (
                  <Tooltip key={level.index}>
                    <TooltipTrigger asChild>
                      <g className="cursor-pointer">
                        <rect
                          x={40 - width / 2}
                          y={y - 3}
                          width={width}
                          height={6}
                          fill={level.occupation > 0.5 ? '#3b82f6' : '#93c5fd'}
                          rx="2"
                        />
                      </g>
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>ε = {formatNumber(level.energy, 3)} MeV</p>
                      <p>n = {formatNumber(level.occupation, 3)}</p>
                      {level.label && <p>{level.label}</p>}
                    </TooltipContent>
                  </Tooltip>
                )
              })}
            </TooltipProvider>
            
            {/* Proton levels */}
            <TooltipProvider>
              {protonLevels.map((level) => {
                const y = energyToY(level.energy)
                const width = 60 * level.occupation
                return (
                  <Tooltip key={level.index}>
                    <TooltipTrigger asChild>
                      <g className="cursor-pointer">
                        <rect
                          x={160 - width / 2}
                          y={y - 3}
                          width={width}
                          height={6}
                          fill={level.occupation > 0.5 ? '#ef4444' : '#fca5a5'}
                          rx="2"
                        />
                      </g>
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>ε = {formatNumber(level.energy, 3)} MeV</p>
                      <p>n = {formatNumber(level.occupation, 3)}</p>
                      {level.label && <p>{level.label}</p>}
                    </TooltipContent>
                  </Tooltip>
                )
              })}
            </TooltipProvider>
          </svg>
        </div>
        
        {/* Legend */}
        <div className="flex justify-center gap-6 mt-4 text-sm">
          <div className="flex items-center gap-2">
            <div className="w-4 h-2 bg-neutron rounded" />
            <span>Occupied (n)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-2 bg-proton rounded" />
            <span>Occupied (p)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-0.5 border-t-2 border-dashed border-muted-foreground" />
            <span>Fermi Level</span>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
