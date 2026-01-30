import { useParams, Link } from 'react-router-dom'
import { useCalculation, useDensityData } from '@/hooks'
import { ResultsDisplay, SpectrumPlot, StatusDisplay, DensityVisualizer } from '@/components'
import { Button } from '@/components/ui'
import { ArrowLeft, Download, Loader2 } from 'lucide-react'
import { formatNucleus, getPhaseInfo } from '@/lib/utils'
import { useState } from 'react'
import type { DensityType } from '@/components/DensityVisualizer'

export function ResultsPage() {
  const { id } = useParams<{ id: string }>()
  const { data: calculation, isLoading, error } = useCalculation(id!)
  const [selectedDensityType, setSelectedDensityType] = useState<DensityType>('total')
  
  // Fetch density data for completed calculations
  const calculationComplete = calculation?.phase === 'converged' || calculation?.phase === 'failed'
  const { data: densityData, isLoading: densityLoading } = useDensityData(
    id,
    selectedDensityType,
    { enabled: calculationComplete && calculation?.phase === 'converged' }
  )
  
  if (isLoading) {
    return (
      <div className="flex flex-col items-center justify-center py-20">
        <Loader2 className="w-8 h-8 animate-spin text-primary mb-4" />
        <p className="text-muted-foreground">Loading calculation...</p>
      </div>
    )
  }
  
  if (error || !calculation) {
    return (
      <div className="flex flex-col items-center justify-center py-20">
        <p className="text-destructive mb-4">
          {error?.message || 'Calculation not found'}
        </p>
        <Link to="/history">
          <Button variant="outline">
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to History
          </Button>
        </Link>
      </div>
    )
  }
  
  // Defensive handling for nucleus data
  const protons = calculation.nucleus?.protons ?? 0
  const neutrons = calculation.nucleus?.neutrons ?? 0
  const nucleusName = protons > 0 ? formatNucleus(protons, neutrons) : 'Unknown Nucleus'
  const phaseInfo = getPhaseInfo(calculation.phase)
  const isComplete = calculation.phase === 'converged' || calculation.phase === 'failed'
  
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Link to="/history">
            <Button variant="ghost" size="sm">
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back
            </Button>
          </Link>
          <div>
            <h1 className="text-2xl font-bold">{nucleusName} Calculation</h1>
            <p className="text-muted-foreground">
              {calculation.force_name} • {phaseInfo.label}
            </p>
          </div>
        </div>
        
        {isComplete && calculation.results && (
          <Button variant="outline">
            <Download className="w-4 h-4 mr-2" />
            Export Results
          </Button>
        )}
      </div>
      
      {/* Show live status if still running */}
      {!isComplete && (
        <StatusDisplay />
      )}
      
      {/* Results */}
      {calculation.results ? (
        <div className="space-y-6">
          <ResultsDisplay 
            results={calculation.results} 
            nucleusName={nucleusName}
          />
          
          {/* 3D Density Visualization */}
          <DensityVisualizer 
            initialData={densityData}
            isLoading={densityLoading}
            onDensityTypeChange={setSelectedDensityType}
          />
          
          {/* Spectrum plot */}
          {calculation.results.single_particle_levels && (
            <SpectrumPlot
              levels={calculation.results.single_particle_levels}
              fermiNeutron={calculation.results.pairing.fermi_neutron}
              fermiProton={calculation.results.pairing.fermi_proton}
            />
          )}
        </div>
      ) : calculation.phase === 'failed' ? (
        <div className="text-center py-10">
          <p className="text-destructive text-lg mb-2">Calculation Failed</p>
          <p className="text-muted-foreground">
            {calculation.progress.message || 'An error occurred during the calculation'}
          </p>
        </div>
      ) : (
        <div className="text-center py-10 text-muted-foreground">
          Results will appear here when the calculation completes.
        </div>
      )}
    </div>
  )
}
