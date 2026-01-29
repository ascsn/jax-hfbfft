import { CalculationResults as Results } from '@/types'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui'
import { formatEnergy, formatRadius, formatNumber } from '@/lib/utils'
import { Atom, Target, Waves, Zap } from 'lucide-react'

interface ResultsDisplayProps {
  results: Results
  nucleusName: string
}

export function ResultsDisplay({ results, nucleusName }: ResultsDisplayProps) {
  return (
    <div className="space-y-6">
      {/* Summary card */}
      <Card className="border-primary/20 bg-primary/5">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Atom className="w-6 h-6 text-primary" />
            {nucleusName} Results
          </CardTitle>
          <CardDescription>
            Converged in {results.iterations} iterations
            {results.total_time_seconds > 0 && ` (${formatNumber(results.total_time_seconds, 1)}s)`}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center p-4 rounded-lg bg-background">
              <p className="text-sm text-muted-foreground">Total Energy</p>
              <p className="text-2xl font-bold text-primary">
                {formatNumber(results.energies.total, 2)}
              </p>
              <p className="text-xs text-muted-foreground">MeV</p>
            </div>
            <div className="text-center p-4 rounded-lg bg-background">
              <p className="text-sm text-muted-foreground">Charge Radius</p>
              <p className="text-2xl font-bold">
                {formatNumber(results.radii.charge, 3)}
              </p>
              <p className="text-xs text-muted-foreground">fm</p>
            </div>
            <div className="text-center p-4 rounded-lg bg-background">
              <p className="text-sm text-muted-foreground">β₂</p>
              <p className="text-2xl font-bold">
                {formatNumber(results.deformation.beta2, 3)}
              </p>
              <p className="text-xs text-muted-foreground">deformation</p>
            </div>
            <div className="text-center p-4 rounded-lg bg-background">
              <p className="text-sm text-muted-foreground">Q₂₀</p>
              <p className="text-2xl font-bold">
                {formatNumber(results.deformation.q20, 2)}
              </p>
              <p className="text-xs text-muted-foreground">fm²</p>
            </div>
          </div>
        </CardContent>
      </Card>
      
      {/* Energy breakdown */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-lg">
            <Zap className="w-5 h-5" />
            Energy Breakdown
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            <EnergyRow label="Kinetic Energy" value={results.energies.kinetic} />
            <EnergyRow label="t₀ Term" value={results.energies.ehf0} />
            <EnergyRow label="Current Term" value={results.energies.ehf1} />
            <EnergyRow label="Laplacian Term" value={results.energies.ehf2} />
            <EnergyRow label="Density-Dependent" value={results.energies.ehf3} />
            <EnergyRow label="Spin-Orbit" value={results.energies.ehfls} />
            <EnergyRow label="Coulomb" value={results.energies.coulomb} />
            <EnergyRow label="Pairing" value={results.energies.pairing} />
            <EnergyRow label="C.M. Correction" value={results.energies.cm_correction} />
            <div className="border-t pt-2 mt-2">
              <EnergyRow label="Total" value={results.energies.total} bold />
            </div>
          </div>
        </CardContent>
      </Card>
      
      {/* Radii */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-lg">
            <Target className="w-5 h-5" />
            RMS Radii
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <RadiusItem label="Neutron" value={results.radii.neutron} color="text-neutron" />
            <RadiusItem label="Proton" value={results.radii.proton} color="text-proton" />
            <RadiusItem label="Total" value={results.radii.total} />
            <RadiusItem label="Charge" value={results.radii.charge} color="text-primary" />
          </div>
        </CardContent>
      </Card>
      
      {/* Pairing (if present) */}
      {(results.pairing.gap_neutron > 0 || results.pairing.gap_proton > 0) && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg">
              <Waves className="w-5 h-5" />
              Pairing Properties
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <p className="text-sm text-muted-foreground">Δ_n</p>
                <p className="text-lg font-mono">{formatEnergy(results.pairing.gap_neutron)}</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Δ_p</p>
                <p className="text-lg font-mono">{formatEnergy(results.pairing.gap_proton)}</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">λ_n</p>
                <p className="text-lg font-mono">{formatEnergy(results.pairing.fermi_neutron)}</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">λ_p</p>
                <p className="text-lg font-mono">{formatEnergy(results.pairing.fermi_proton)}</p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

function EnergyRow({ label, value, bold }: { label: string; value: number; bold?: boolean }) {
  return (
    <div className={`flex justify-between ${bold ? 'font-bold' : ''}`}>
      <span className="text-muted-foreground">{label}</span>
      <span className="font-mono">{formatEnergy(value)}</span>
    </div>
  )
}

function RadiusItem({ label, value, color }: { label: string; value: number; color?: string }) {
  return (
    <div className="text-center p-3 rounded-lg bg-muted">
      <p className="text-sm text-muted-foreground">{label}</p>
      <p className={`text-xl font-mono ${color || ''}`}>{formatRadius(value)}</p>
    </div>
  )
}
