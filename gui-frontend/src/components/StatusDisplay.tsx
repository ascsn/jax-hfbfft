import { useStore } from '@/store'
import { Card, CardHeader, CardTitle, CardContent, Progress } from '@/components/ui'
import { Activity, CheckCircle, XCircle, Loader2 } from 'lucide-react'
import { cn, formatNumber, getPhaseInfo } from '@/lib/utils'
import { MemoryDisplay } from './MemoryDisplay'

export function StatusDisplay() {
  const { activeCalculations, systemStatus } = useStore()
  
  // Get the most recent active calculation
  const calculations = Array.from(activeCalculations.values())
  const activeCalc = calculations.find(c => 
    ['pending', 'warmup', 'initializing', 'iterating'].includes(c.phase)
  )
  
  if (!activeCalc && !systemStatus) {
    return null
  }
  
  const progress = activeCalc?.progress
  const phaseInfo = progress ? getPhaseInfo(progress.phase) : null
  
  // Calculate progress percentage
  const iterationProgress = progress 
    ? (progress.iteration / progress.max_iterations) * 100 
    : 0
  
  return (
    <div className="space-y-4">
      {/* System Status Card */}
      {systemStatus && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="w-5 h-5" />
              System Status
            </CardTitle>
          </CardHeader>
          
          <CardContent className="space-y-4">
            {/* Memory Usage */}
            <div className="space-y-3">
              <MemoryDisplay
                label="System RAM"
                used={systemStatus.memory_used_gb}
                total={systemStatus.memory_total_gb}
                icon="cpu"
              />
              
              {systemStatus.gpu_memory_total_gb && (
                <MemoryDisplay
                  label="GPU VRAM"
                  used={systemStatus.gpu_memory_used_gb}
                  total={systemStatus.gpu_memory_total_gb}
                  icon="gpu"
                />
              )}
            </div>
            
            {/* Backend Info */}
            <div className="text-xs text-muted-foreground space-y-1">
              <div className="flex justify-between">
                <span>Backend:</span>
                <span className="font-mono">{systemStatus.backend}</span>
              </div>
              <div className="flex justify-between">
                <span>Device:</span>
                <span className="font-mono">{systemStatus.device_name}</span>
              </div>
              <div className="flex justify-between">
                <span>Precision:</span>
                <span className="font-mono">{systemStatus.precision}</span>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
      
      {/* Active Calculation Card */}
      {activeCalc && progress && (
        <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Activity className="w-5 h-5" />
          Active Calculation
        </CardTitle>
      </CardHeader>
      
      <CardContent className="space-y-4">
        {/* Phase indicator */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            {progress.phase === 'converged' ? (
              <CheckCircle className="w-5 h-5 text-green-500" />
            ) : progress.phase === 'failed' ? (
              <XCircle className="w-5 h-5 text-destructive" />
            ) : (
              <Loader2 className="w-5 h-5 animate-spin text-primary" />
            )}
            <span className={cn("font-medium", phaseInfo?.color)}>
              {phaseInfo?.label}
            </span>
          </div>
          <span className="text-sm text-muted-foreground">
            {(activeCalc.nucleus?.protons ?? 0) + (activeCalc.nucleus?.neutrons ?? 0)} nucleons
          </span>
        </div>
        
        {/* Iteration progress */}
        {progress.phase === 'iterating' && (
          <>
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>Iteration {progress.iteration} / {progress.max_iterations}</span>
                <span>{Math.round(iterationProgress)}%</span>
              </div>
              <Progress value={iterationProgress} className="h-3" />
            </div>
            
            {/* Energy and fluctuation */}
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div className="p-3 rounded-lg bg-muted">
                <p className="text-muted-foreground">Energy</p>
                <p className="text-lg font-mono">
                  {formatNumber(progress.energy, 3)} MeV
                </p>
              </div>
              <div className="p-3 rounded-lg bg-muted">
                <p className="text-muted-foreground">Fluctuation</p>
                <p className="text-lg font-mono">
                  {progress.fluctuation.toExponential(2)}
                </p>
              </div>
            </div>
          </>
        )}
        
        {/* Message */}
        {progress.message && (
          <p className="text-sm text-muted-foreground italic">
            {progress.message}
          </p>
        )}
      </CardContent>
    </Card>
      )}
    </div>
  )
}
