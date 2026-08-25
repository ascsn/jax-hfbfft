import { useStore } from '@/store'
import { useStartCalculation } from '@/hooks'
import { CalculationForm, StatusDisplay } from '@/components'
import { Card, CardHeader, CardTitle, CardContent, Button } from '@/components/ui'
import { Play, Loader2, Info } from 'lucide-react'
import { formatNucleus } from '@/lib/utils'

export function CalculatorPage() {
  const { form, systemStatus, getCalculationRequest } = useStore()
  const startCalculation = useStartCalculation()
  
  // JIT warmup is disabled - calculations can start immediately
  const isCalculating = startCalculation.isPending
  const canStart = !isCalculating
  const nucleusName = formatNucleus(form.protons, form.neutrons)
  
  const handleStart = () => {
    // Get the full calculation request from the store
    const request = getCalculationRequest()
    startCalculation.mutate(request)
  }
  
  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Left column: Input form */}
      <div className="lg:col-span-2 space-y-6">
        <CalculationForm />
        
        {/* Calculate button */}
        <Card>
          <CardContent className="pt-6">
            <Button
              className="w-full h-14 text-lg"
              disabled={!canStart}
              onClick={handleStart}
            >
              {isCalculating ? (
                <>
                  <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                  Starting calculation...
                </>
              ) : (
                <>
                  <Play className="w-5 h-5 mr-2" />
                  Calculate {nucleusName}
                </>
              )}
            </Button>
          </CardContent>
        </Card>
      </div>
      
      {/* Right column: Status and info */}
      <div className="space-y-6">
        <StatusDisplay />
        
        {/* System info */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg">
              <Info className="w-5 h-5" />
              System Information
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-sm">
            {systemStatus ? (
              <>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">JAX Backend</span>
                  <span className="font-medium capitalize">{systemStatus.backend}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Precision</span>
                  <span className="font-mono">{systemStatus.precision}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">JAX Version</span>
                  <span className="font-mono">{systemStatus.jax_version}</span>
                </div>
                {systemStatus.device_info && (
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Device</span>
                    <span className="font-mono text-xs">{systemStatus.device_info}</span>
                  </div>
                )}
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Memory Used</span>
                  <span className="font-mono">
                    {(systemStatus.memory_used_gb ?? 0).toFixed(1)} / {(systemStatus.memory_total_gb ?? 0).toFixed(1)} GB
                  </span>
                </div>
              </>
            ) : (
              <p className="text-muted-foreground">Loading system info...</p>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
