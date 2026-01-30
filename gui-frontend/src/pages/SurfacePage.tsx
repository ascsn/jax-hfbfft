import { useMemo, useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle, Button, Input, Label, Select } from '@/components/ui'
import { useForces, useCalculation } from '@/hooks'
import { useHistoricalSurface } from '@/hooks/useHistory'
import { api } from '@/services/api'
import { useStore } from '@/store'
import { BetaSurfaceResult, PairingType } from '@/types'
import { Loader2, LineChart } from 'lucide-react'
import { formatNucleus } from '@/lib/utils'
import { useParams } from 'react-router-dom'

export function SurfacePage() {
  const { id } = useParams()
  const { form, addCalculation, updateCalculation, removeCalculation } = useStore()
  const { data: forces } = useForces()
  const { data: historicalSurface } = useHistoricalSurface(id ?? null)
  const [protons, setProtons] = useState(form.protons)
  const [neutrons, setNeutrons] = useState(form.neutrons)
  const [forceName, setForceName] = useState(form.forceName)
  const [betaMin, setBetaMin] = useState(-0.3)
  const [betaMax, setBetaMax] = useState(0.3)
  const [betaSteps, setBetaSteps] = useState(13)
  const [gamma, setGamma] = useState(0)
  const [hotStart, setHotStart] = useState(false)
  const [gridAuto, setGridAuto] = useState(form.grid.auto ?? true)
  const [gridSize, setGridSize] = useState({ nx: form.grid.nx, ny: form.grid.ny, nz: form.grid.nz })
  const [spacing, setSpacing] = useState(form.grid.spacing ?? 1.0)
  const [pairingType, setPairingType] = useState<PairingType>(form.pairing.type)
  const [v0Neutron, setV0Neutron] = useState(form.pairing.v0_neutron)
  const [v0Proton, setV0Proton] = useState(form.pairing.v0_proton)
  const [maxIterations, setMaxIterations] = useState(Math.max(form.iteration.max_iterations, 600))
  const [convergence, setConvergence] = useState(Math.max(form.iteration.convergence_threshold, 1e-5))
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<BetaSurfaceResult | null>(null)
  const [surfaceId, setSurfaceId] = useState<string | null>(null)
  const surfaceStatus = useCalculation(surfaceId)
  const [selectedPoint, setSelectedPoint] = useState<BetaSurfaceResult['points'][number] | null>(null)

  const nucleusName = formatNucleus(
    result?.nucleus.protons ?? protons,
    result?.nucleus.neutrons ?? neutrons
  )

  const chartData = useMemo(() => {
    if (!result?.points?.length) return null
    const points = [...result.points].sort((a, b) => a.beta2 - b.beta2)
    const energies = points.map((p) => p.energy)
    const minE = Math.min(...energies)
    const maxE = Math.max(...energies)
    const minBeta = Math.min(...points.map((p) => p.beta2))
    const maxBeta = Math.max(...points.map((p) => p.beta2))
    return { points, minE, maxE, minBeta, maxBeta }
  }, [result])

  const selectedGamma = useMemo(() => {
    if (!selectedPoint) return null
    return Math.atan2(Math.sqrt(3) * selectedPoint.q22, selectedPoint.q20) * 180 / Math.PI
  }, [selectedPoint])

  useEffect(() => {
    if (historicalSurface) {
      setResult(historicalSurface)
    }
  }, [historicalSurface])

  useEffect(() => {
    if (surfaceStatus.data?.surface_results) {
      setResult(surfaceStatus.data.surface_results)
    }
  }, [surfaceStatus.data])

  const handleRun = async () => {
    setLoading(true)
    setError(null)
    setResult(null)

    const localId = `surface-${Date.now()}`
    addCalculation({
      id: localId,
      nucleus: { protons, neutrons },
      force_name: forceName,
      phase: 'iterating',
      progress: {
        calculation_id: localId,
        phase: 'iterating',
        iteration: 0,
        max_iterations: betaSteps,
        fluctuation: 0,
        energy: 0,
        message: 'Running beta surface scan...',
      },
      started_at: new Date().toISOString(),
      run_type: 'surface',
    })

    try {
      const request = {
        nucleus: { protons, neutrons },
        force_name: forceName,
        grid: {
          nx: gridSize.nx,
          ny: gridSize.ny,
          nz: gridSize.nz,
          dx: spacing,
          dy: spacing,
          dz: spacing,
          auto: gridAuto,
        },
        pairing: {
          type: pairingType,
          v0_neutron: v0Neutron,
          v0_proton: v0Proton,
        },
        iteration: {
          max_iterations: maxIterations,
          convergence_threshold: convergence,
          print_interval: form.iteration.print_interval ?? 10,
        },
        beta_min: betaMin,
        beta_max: betaMax,
        beta_steps: betaSteps,
        gamma,
        hot_start: hotStart,
      }

      const response = await api.startBetaSurface(request)
      removeCalculation(localId)
      setSurfaceId(response.calculation_id)
      addCalculation({
        id: response.calculation_id,
        nucleus: { protons, neutrons },
        force_name: forceName,
        phase: 'iterating',
        progress: {
          calculation_id: response.calculation_id,
          phase: 'iterating',
          iteration: 0,
          max_iterations: betaSteps,
          fluctuation: 0,
          energy: 0,
          message: 'Running beta surface scan...',
        },
        started_at: new Date().toISOString(),
        run_type: 'surface',
      })
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to run surface scan'
      setError(message)
      updateCalculation(localId, {
        phase: 'failed',
        progress: {
          calculation_id: localId,
          phase: 'failed',
          iteration: 0,
          max_iterations: betaSteps,
          fluctuation: 0,
          energy: 0,
          message: message,
        },
        completed_at: new Date().toISOString(),
        run_type: 'surface',
      })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <div className="lg:col-span-1 space-y-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <LineChart className="w-5 h-5" />
              Constrained Surface
            </CardTitle>
            <CardDescription>
              Scan beta2 from oblate to prolate shapes.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>Protons (Z)</Label>
              <Input type="number" min={1} max={120} value={protons} onChange={(e) => setProtons(parseInt(e.target.value || '1', 10))} />
            </div>
            <div className="space-y-2">
              <Label>Neutrons (N)</Label>
              <Input type="number" min={1} max={200} value={neutrons} onChange={(e) => setNeutrons(parseInt(e.target.value || '1', 10))} />
            </div>
            <div className="space-y-2">
              <Label>Skyrme Force</Label>
              <Select
                value={forceName}
                onChange={(e) => setForceName(e.target.value)}
                options={forces?.map((f) => ({ value: f.name, label: f.name })) || [{ value: 'SLy4', label: 'SLy4' }]}
              />
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-2">
                <Label>β min</Label>
                <Input type="number" step="0.05" value={betaMin} onChange={(e) => setBetaMin(parseFloat(e.target.value))} />
              </div>
              <div className="space-y-2">
                <Label>β max</Label>
                <Input type="number" step="0.05" value={betaMax} onChange={(e) => setBetaMax(parseFloat(e.target.value))} />
              </div>
            </div>

            <div className="space-y-2">
              <Label>Grid</Label>
              <div className="flex items-center gap-2">
                <input
                  id="grid-auto"
                  type="checkbox"
                  checked={gridAuto}
                  onChange={(e) => setGridAuto(e.target.checked)}
                />
                <Label htmlFor="grid-auto" className="text-sm">Auto grid size</Label>
              </div>
              {!gridAuto && (
                <div className="grid grid-cols-3 gap-2">
                  <div>
                    <Label htmlFor="nx" className="text-xs text-muted-foreground">Nx</Label>
                    <Input
                      id="nx"
                      type="number"
                      min={8}
                      max={64}
                      value={gridSize.nx}
                      onChange={(e) => setGridSize({ ...gridSize, nx: parseInt(e.target.value || '24', 10) })}
                    />
                  </div>
                  <div>
                    <Label htmlFor="ny" className="text-xs text-muted-foreground">Ny</Label>
                    <Input
                      id="ny"
                      type="number"
                      min={8}
                      max={64}
                      value={gridSize.ny}
                      onChange={(e) => setGridSize({ ...gridSize, ny: parseInt(e.target.value || '24', 10) })}
                    />
                  </div>
                  <div>
                    <Label htmlFor="nz" className="text-xs text-muted-foreground">Nz</Label>
                    <Input
                      id="nz"
                      type="number"
                      min={8}
                      max={64}
                      value={gridSize.nz}
                      onChange={(e) => setGridSize({ ...gridSize, nz: parseInt(e.target.value || '24', 10) })}
                    />
                  </div>
                </div>
              )}
              <div className="space-y-1">
                <Label htmlFor="spacing" className="text-xs text-muted-foreground">Grid spacing (fm)</Label>
                <Input
                  id="spacing"
                  type="number"
                  step="0.1"
                  min={0.5}
                  max={2.0}
                  value={spacing}
                  onChange={(e) => setSpacing(parseFloat(e.target.value || '1.0'))}
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label>Pairing</Label>
              <Select
                value={pairingType}
                onChange={(e) => setPairingType(e.target.value as PairingType)}
                options={[
                  { value: 'none', label: 'None' },
                  { value: 'vdi', label: 'VDI' },
                  { value: 'dddi', label: 'DDDI' },
                ]}
              />
              {pairingType !== 'none' && (
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <Label htmlFor="v0n" className="text-xs text-muted-foreground">V0 neutron</Label>
                    <Input
                      id="v0n"
                      type="number"
                      value={v0Neutron}
                      onChange={(e) => setV0Neutron(parseFloat(e.target.value || '-200'))}
                    />
                  </div>
                  <div>
                    <Label htmlFor="v0p" className="text-xs text-muted-foreground">V0 proton</Label>
                    <Input
                      id="v0p"
                      type="number"
                      value={v0Proton}
                      onChange={(e) => setV0Proton(parseFloat(e.target.value || '-200'))}
                    />
                  </div>
                </div>
              )}
            </div>

            <div className="flex items-center gap-2">
              <input
                id="hot-start"
                type="checkbox"
                checked={hotStart}
                onChange={(e) => setHotStart(e.target.checked)}
              />
              <Label htmlFor="hot-start" className="text-sm">Hot start (use nearest beta solution)</Label>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-2">
                <Label>β steps</Label>
                <Input type="number" min={3} max={101} value={betaSteps} onChange={(e) => setBetaSteps(parseInt(e.target.value || '3', 10))} />
              </div>
              <div className="space-y-2">
                <Label>γ (deg)</Label>
                <Input type="number" step="5" value={gamma} onChange={(e) => setGamma(parseFloat(e.target.value))} />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-2">
                <Label>Max iterations</Label>
                <Input type="number" min={10} max={5000} value={maxIterations} onChange={(e) => setMaxIterations(parseInt(e.target.value || '200', 10))} />
              </div>
              <div className="space-y-2">
                <Label>Convergence</Label>
                <Input type="number" step="1e-6" value={convergence} onChange={(e) => setConvergence(parseFloat(e.target.value))} />
              </div>
            </div>

            {error && (
              <div className="text-sm text-destructive">{error}</div>
            )}

            <Button className="w-full" onClick={handleRun} disabled={loading}>
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Scanning surface...
                </>
              ) : (
                <>Run β Scan</>
              )}
            </Button>
          </CardContent>
        </Card>
      </div>

      <div className="lg:col-span-2 space-y-6">
        <Card>
          <CardHeader>
            <CardTitle>{nucleusName} β Surface</CardTitle>
            <CardDescription>
              {result ? `Force: ${result.force_name}` : 'Run a scan to visualize oblate/prolate minima.'}
            </CardDescription>
          </CardHeader>
          <CardContent>
            {!result && (
              <div className="text-sm text-muted-foreground">No surface data yet.</div>
            )}

            {chartData && (
              <div className="space-y-4">
                <div className="h-48 w-full rounded-lg border bg-muted/20 p-3">
                  <svg viewBox="0 0 600 200" className="w-full h-full">
                    <text x="20" y="12" className="fill-muted-foreground" fontSize="10">
                      {chartData.maxE.toFixed(2)} MeV
                    </text>
                    <text x="20" y="102" className="fill-muted-foreground" fontSize="10">
                      {((chartData.maxE + chartData.minE) / 2).toFixed(2)} MeV
                    </text>
                    <text x="20" y="192" className="fill-muted-foreground" fontSize="10">
                      {chartData.minE.toFixed(2)} MeV
                    </text>
                    <text x="20" y="198" className="fill-muted-foreground" fontSize="10">
                      {chartData.minBeta.toFixed(2)}
                    </text>
                    <text x="300" y="198" textAnchor="middle" className="fill-muted-foreground" fontSize="10">
                      {(((chartData.minBeta + chartData.maxBeta) / 2).toFixed(2))}
                    </text>
                    <text x="580" y="198" textAnchor="end" className="fill-muted-foreground" fontSize="10">
                      {chartData.maxBeta.toFixed(2)}
                    </text>
                    <polyline
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      className="text-primary"
                      points={chartData.points.map((p) => {
                        const x = ((p.beta2 - chartData.minBeta) / (chartData.maxBeta - chartData.minBeta || 1)) * 560 + 20
                        const y = 180 - ((p.energy - chartData.minE) / (chartData.maxE - chartData.minE || 1)) * 150
                        return `${x},${y}`
                      }).join(' ')}
                    />
                    {chartData.points.map((p) => {
                      const x = ((p.beta2 - chartData.minBeta) / (chartData.maxBeta - chartData.minBeta || 1)) * 560 + 20
                      const y = 180 - ((p.energy - chartData.minE) / (chartData.maxE - chartData.minE || 1)) * 150
                      return (
                        <circle key={p.beta2} cx={x} cy={y} r={3} className={p.converged ? 'fill-primary' : 'fill-destructive'}>
                          <title>{`β2=${p.beta2.toFixed(3)}, E=${p.energy.toFixed(3)} MeV`}</title>
                        </circle>
                      )
                    })}
                  </svg>
                </div>
                <div className="grid grid-cols-3 gap-4 text-sm">
                  <div>
                    <span className="text-muted-foreground">Energy range</span>
                    <div className="font-medium">
                      {chartData.minE.toFixed(3)} to {chartData.maxE.toFixed(3)} MeV
                    </div>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Points</span>
                    <div className="font-medium">{chartData.points.length}</div>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Converged</span>
                    <div className="font-medium">
                      {chartData.points.filter((p) => p.converged).length}/{chartData.points.length}
                    </div>
                  </div>
                </div>

                {chartData.points.some((p) => !p.converged) && (
                  <div className="text-xs text-destructive">
                    Some points did not converge. Increase max iterations or relax the threshold.
                  </div>
                )}
                {surfaceStatus.data?.progress && surfaceStatus.data.phase === 'iterating' && (
                  <div className="text-xs text-muted-foreground">
                    Running point {surfaceStatus.data.progress.iteration}/{surfaceStatus.data.progress.max_iterations}
                  </div>
                )}

                {selectedPoint && (
                  <Card className="bg-muted/20">
                    <CardContent className="pt-4 grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                      <div>
                        <div className="text-muted-foreground">β2 target</div>
                        <div className="font-mono">{selectedPoint.beta2.toFixed(3)}</div>
                      </div>
                      <div>
                        <div className="text-muted-foreground">Energy</div>
                        <div className="font-mono">{selectedPoint.energy.toFixed(4)} MeV</div>
                      </div>
                      <div>
                        <div className="text-muted-foreground">Q20</div>
                        <div className="font-mono">{selectedPoint.q20.toFixed(3)}</div>
                      </div>
                      <div>
                        <div className="text-muted-foreground">Q22</div>
                        <div className="font-mono">{selectedPoint.q22.toFixed(3)}</div>
                      </div>
                      <div>
                        <div className="text-muted-foreground">γ (deg)</div>
                        <div className="font-mono">{selectedGamma?.toFixed(2) ?? '—'}</div>
                      </div>
                      <div>
                        <div className="text-muted-foreground">Iterations</div>
                        <div className="font-mono">{selectedPoint.iterations}</div>
                      </div>
                      <div>
                        <div className="text-muted-foreground">Converged</div>
                        <div className={selectedPoint.converged ? 'text-green-600' : 'text-destructive'}>
                          {selectedPoint.converged ? 'Yes' : 'No'}
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                )}

                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-left text-muted-foreground border-b">
                        <th className="py-2">β2</th>
                        <th className="py-2">Energy (MeV)</th>
                        <th className="py-2">Q20</th>
                        <th className="py-2">Q22</th>
                        <th className="py-2">Converged</th>
                      </tr>
                    </thead>
                    <tbody>
                      {chartData.points.map((p) => (
                        <tr
                          key={p.beta2}
                          className={`border-b last:border-0 cursor-pointer ${selectedPoint?.beta2 === p.beta2 ? 'bg-muted/30' : ''}`}
                          onClick={() => setSelectedPoint(p)}
                        >
                          <td className="py-2 font-mono">{p.beta2.toFixed(3)}</td>
                          <td className="py-2 font-mono">{p.energy.toFixed(4)}</td>
                          <td className="py-2 font-mono">{p.q20.toFixed(3)}</td>
                          <td className="py-2 font-mono">{p.q22.toFixed(3)}</td>
                          <td className="py-2">
                            <span className={p.converged ? 'text-green-600' : 'text-destructive'}>
                              {p.converged ? 'Yes' : 'No'}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
