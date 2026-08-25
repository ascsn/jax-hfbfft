import { useStore } from '@/store'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui'
import { Activity, ChevronDown, ChevronUp } from 'lucide-react'
import { MemoryDisplay } from './MemoryDisplay'
import { ActiveCalculationCard } from './ActiveCalculationCard'
import { useState } from 'react'

export function StatusDisplay() {
  const { activeCalculations, systemStatus } = useStore()
  const [showSystemStatus, setShowSystemStatus] = useState(false)
  
  // Get all active calculations (not completed or cancelled)
  // Filter out any undefined or incomplete entries
  const calculations = Array.from(activeCalculations.values()).filter(
    (c): c is NonNullable<typeof c> => c != null && c.progress != null
  )
  const runningCalcs = calculations.filter(c => 
    c.phase && ['pending', 'warmup', 'initializing', 'iterating'].includes(c.phase)
  )
  
  if (runningCalcs.length === 0 && !systemStatus) {
    return null
  }
  
  return (
    <div className="space-y-4">
      {/* Active Calculations Card */}
      {runningCalcs.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="w-5 h-5" />
              Active Calculations
              <span className="ml-2 px-2 py-0.5 bg-primary text-primary-foreground rounded-full text-xs font-semibold">
                {runningCalcs.length}
              </span>
            </CardTitle>
          </CardHeader>
          
          <CardContent className="space-y-3">
            {runningCalcs.map(calc => (
              <ActiveCalculationCard 
                key={calc.id} 
                calculation={calc}
              />
            ))}
          </CardContent>
        </Card>
      )}
      
      {/* System Status Card (Collapsible) */}
      {systemStatus && (
        <Card>
          <CardHeader 
            className="cursor-pointer hover:bg-accent/50 transition-colors"
            onClick={() => setShowSystemStatus(!showSystemStatus)}
          >
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center gap-2">
                <Activity className="w-5 h-5" />
                System Status
              </CardTitle>
              {showSystemStatus ? (
                <ChevronUp className="w-4 h-4 text-muted-foreground" />
              ) : (
                <ChevronDown className="w-4 h-4 text-muted-foreground" />
              )}
            </div>
          </CardHeader>
          
          {showSystemStatus && (
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
          )}
        </Card>
      )}
    </div>
  )
}
