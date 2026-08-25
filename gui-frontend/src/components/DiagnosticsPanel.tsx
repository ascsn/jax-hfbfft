import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '@/services/api'
import { SystemDiagnostics } from '@/types'
import { 
  Card, 
  CardHeader, 
  CardTitle, 
  CardContent,
  Button,
  Accordion,
  AccordionItem,
  AccordionTrigger,
  AccordionContent
} from '@/components/ui'
import { Info, RefreshCw, Copy, CheckCircle } from 'lucide-react'

export function DiagnosticsPanel() {
  const [copied, setCopied] = useState(false)
  
  const { data: diagnostics, isLoading, refetch } = useQuery<SystemDiagnostics>({
    queryKey: ['diagnostics'],
    queryFn: () => api.getDiagnostics() as Promise<SystemDiagnostics>,
    staleTime: 30000, // 30 seconds
  })
  
  const handleCopy = () => {
    if (diagnostics) {
      navigator.clipboard.writeText(JSON.stringify(diagnostics, null, 2))
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }
  }
  
  if (isLoading) {
    return (
      <Card>
        <CardContent className="p-6 text-center text-muted-foreground">
          Loading diagnostics...
        </CardContent>
      </Card>
    )
  }
  
  if (!diagnostics) {
    return null
  }
  
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Info className="w-5 h-5" />
            System Diagnostics
          </CardTitle>
          <div className="flex items-center gap-2">
            <Button
              size="sm"
              variant="outline"
              onClick={handleCopy}
              className="gap-2"
            >
              {copied ? <CheckCircle className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
              {copied ? 'Copied!' : 'Copy'}
            </Button>
            <Button
              size="sm"
              variant="outline"
              onClick={() => refetch()}
              className="gap-2"
            >
              <RefreshCw className="w-3 h-3" />
              Refresh
            </Button>
          </div>
        </div>
      </CardHeader>
      
      <CardContent className="space-y-4">
        <Accordion type="multiple" className="w-full">
          {/* Python Info */}
          <AccordionItem value="python">
            <AccordionTrigger>Python Environment</AccordionTrigger>
            <AccordionContent>
              {diagnostics.python ? (
                <div className="space-y-2 text-sm">
                  <InfoRow label="Version" value={diagnostics.python.version} />
                  <InfoRow label="Platform" value={diagnostics.python.platform} />
                  <InfoRow label="Architecture" value={diagnostics.python.architecture} />
                </div>
              ) : (
                <p className="text-sm text-muted-foreground">Python information unavailable</p>
              )}
            </AccordionContent>
          </AccordionItem>
          
          {/* JAX Info */}
          <AccordionItem value="jax">
            <AccordionTrigger>JAX Configuration</AccordionTrigger>
            <AccordionContent>
              {!diagnostics.jax ? (
                <p className="text-sm text-muted-foreground">JAX information unavailable</p>
              ) : 'error' in diagnostics.jax ? (
                <p className="text-sm text-destructive">{diagnostics.jax.error}</p>
              ) : (
                <div className="space-y-4">
                  <div className="space-y-2 text-sm">
                    <InfoRow label="Version" value={diagnostics.jax.version} />
                    <InfoRow label="Backend" value={diagnostics.jax.backend} />
                    <InfoRow label="Device Count" value={diagnostics.jax.device_count} />
                    <InfoRow label="64-bit Precision" value={diagnostics.jax.x64_enabled ? 'Enabled' : 'Disabled'} />
                    <InfoRow label="Devices" value={diagnostics.jax.devices.join(', ')} />
                  </div>
                  
                  {diagnostics.jax.device_details.map((device, idx) => (
                    <div key={idx} className="p-3 rounded-lg bg-muted space-y-2">
                      <p className="font-medium text-sm">Device {device.id}</p>
                      <div className="space-y-1 text-xs">
                        <InfoRow label="Platform" value={device.platform} />
                        <InfoRow label="Kind" value={device.device_kind} />
                        {device.memory_stats && (
                          <>
                            <InfoRow 
                              label="Memory Used" 
                              value={`${device.memory_stats.bytes_in_use_gb.toFixed(2)} GB`} 
                            />
                            <InfoRow 
                              label="Memory Limit" 
                              value={`${device.memory_stats.bytes_limit_gb.toFixed(2)} GB`} 
                            />
                            <InfoRow 
                              label="Peak Memory" 
                              value={`${device.memory_stats.peak_bytes_in_use_gb.toFixed(2)} GB`} 
                            />
                          </>
                        )}
                        {device.memory_stats_error && (
                          <p className="text-xs text-muted-foreground">{device.memory_stats_error}</p>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </AccordionContent>
          </AccordionItem>
          
          {/* System Memory */}
          <AccordionItem value="memory">
            <AccordionTrigger>System Memory</AccordionTrigger>
            <AccordionContent>
              {!diagnostics.system_memory ? (
                <p className="text-sm text-muted-foreground">Memory information unavailable</p>
              ) : 'error' in diagnostics.system_memory ? (
                <p className="text-sm text-destructive">{diagnostics.system_memory.error}</p>
              ) : (
                <div className="space-y-2 text-sm">
                  <InfoRow label="Total" value={`${diagnostics.system_memory.total_gb.toFixed(2)} GB`} />
                  <InfoRow label="Available" value={`${diagnostics.system_memory.available_gb.toFixed(2)} GB`} />
                  <InfoRow label="Used" value={`${diagnostics.system_memory.used_gb.toFixed(2)} GB`} />
                  <InfoRow label="Percent" value={`${diagnostics.system_memory.percent.toFixed(1)}%`} />
                </div>
              )}
            </AccordionContent>
          </AccordionItem>
          
          {/* GPU Info */}
          {diagnostics.nvidia_gpus && (
            <AccordionItem value="gpu">
              <AccordionTrigger>NVIDIA GPUs</AccordionTrigger>
              <AccordionContent>
                {'error' in diagnostics.nvidia_gpus ? (
                  <p className="text-sm text-destructive">{diagnostics.nvidia_gpus.error}</p>
                ) : Array.isArray(diagnostics.nvidia_gpus) ? (
                  <div className="space-y-3">
                    {diagnostics.nvidia_gpus.map((gpu) => (
                      <div key={gpu.index} className="p-3 rounded-lg bg-muted space-y-2">
                        <p className="font-medium text-sm">GPU {gpu.index}: {gpu.name}</p>
                        <div className="space-y-1 text-xs">
                          <InfoRow label="Driver" value={gpu.driver_version} />
                          <InfoRow 
                            label="Memory" 
                            value={`${(gpu.memory_used_mb / 1024).toFixed(2)} / ${(gpu.memory_total_mb / 1024).toFixed(2)} GB`} 
                          />
                          {gpu.temperature_c && (
                            <InfoRow label="Temperature" value={`${gpu.temperature_c}°C`} />
                          )}
                          {gpu.utilization_percent !== undefined && gpu.utilization_percent !== null && (
                            <InfoRow label="Utilization" value={`${gpu.utilization_percent}%`} />
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground">GPU information unavailable</p>
                )}
              </AccordionContent>
            </AccordionItem>
          )}
          
          {/* Packages */}
          <AccordionItem value="packages">
            <AccordionTrigger>Package Versions</AccordionTrigger>
            <AccordionContent>
              {!diagnostics.packages ? (
                <p className="text-sm text-muted-foreground">Package information unavailable</p>
              ) : 'error' in diagnostics.packages ? (
                <p className="text-sm text-destructive">{diagnostics.packages.error}</p>
              ) : (
                <div className="space-y-2 text-sm">
                  {Object.entries(diagnostics.packages).map(([pkg, version]) => (
                    <InfoRow key={pkg} label={pkg} value={version} />
                  ))}
                </div>
              )}
            </AccordionContent>
          </AccordionItem>
          
          {/* JIT Warmup Status */}
          <AccordionItem value="warmup">
            <AccordionTrigger>JIT Warmup Status</AccordionTrigger>
            <AccordionContent>
              {diagnostics.jit_warmup ? (
                <div className="space-y-2 text-sm">
                  <InfoRow label="Status" value={diagnostics.jit_warmup.status} />
                  <InfoRow label="Progress" value={`${((diagnostics.jit_warmup.progress ?? 0) * 100).toFixed(1)}%`} />
                  <InfoRow label="Is Ready" value={diagnostics.jit_warmup.is_ready ? 'Yes' : 'No'} />
                  {diagnostics.jit_warmup.message && (
                    <InfoRow label="Message" value={diagnostics.jit_warmup.message} />
                  )}
                  {diagnostics.jit_warmup.elapsed_seconds && (
                    <InfoRow label="Elapsed" value={`${diagnostics.jit_warmup.elapsed_seconds.toFixed(1)}s`} />
                  )}
                  {diagnostics.jit_warmup.error && (
                    <div className="mt-2 p-2 rounded bg-destructive/10 text-destructive text-xs">
                      {diagnostics.jit_warmup.error}
                    </div>
                  )}
                </div>
              ) : (
                <p className="text-sm text-muted-foreground">JIT warmup information unavailable</p>
              )}
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      </CardContent>
    </Card>
  )
}

function InfoRow({ label, value }: { label: string; value: string | number | boolean }) {
  return (
    <div className="flex justify-between items-center">
      <span className="text-muted-foreground">{label}:</span>
      <span className="font-mono text-right">{String(value)}</span>
    </div>
  )
}
