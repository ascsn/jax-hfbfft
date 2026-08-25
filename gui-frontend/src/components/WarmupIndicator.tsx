import { Progress, Button, Alert, AlertDescription } from '@/components/ui'
import { Zap, CheckCircle, XCircle, AlertTriangle, RefreshCw } from 'lucide-react'
import { api } from '@/services/api'
import { SystemStatus } from '@/types'
import { cn } from '@/lib/utils'

interface WarmupIndicatorProps {
  status: SystemStatus
  className?: string
}

export function WarmupIndicator({ status, className }: WarmupIndicatorProps) {
  const handleRetry = async () => {
    try {
      await api.triggerWarmup(true)
    } catch (error) {
      console.error('Failed to retry warmup:', error)
    }
  }
  
  // Warmup is complete
  if (status.jit_warmed_up && status.warmup_status === 'completed') {
    return (
      <div className={cn('flex items-center gap-2 text-sm text-green-600', className)}>
        <CheckCircle className="w-4 h-4" />
        <span className="font-medium">JIT Ready</span>
      </div>
    )
  }
  
  // Warmup failed
  if (status.warmup_status === 'failed') {
    return (
      <Alert variant="destructive" className={className}>
        <AlertTriangle className="w-4 h-4" />
        <AlertDescription>
          <div className="flex items-start justify-between gap-4">
            <div className="flex-1">
              <p className="font-medium mb-1">JIT Warmup Failed</p>
              {status.warmup_error && (
                <p className="text-xs opacity-90 mt-1">{status.warmup_error}</p>
              )}
            </div>
            <Button 
              size="sm" 
              variant="outline"
              onClick={handleRetry}
              className="shrink-0"
            >
              <RefreshCw className="w-3 h-3 mr-1" />
              Retry
            </Button>
          </div>
        </AlertDescription>
      </Alert>
    )
  }
  
  // Warmup in progress
  if (status.warmup_status === 'in_progress') {
    return (
      <div className={cn('p-4 rounded-lg bg-blue-500/10 border border-blue-500/20', className)}>
        <div className="flex items-center gap-2 mb-2">
          <Zap className="w-4 h-4 text-blue-500 animate-pulse" />
          <span className="font-medium text-blue-600">Warming up JIT compiler...</span>
        </div>
        <Progress value={status.warmup_progress * 100} className="h-2 mb-2" />
        <p className="text-xs text-muted-foreground">
          {Math.round(status.warmup_progress * 100)}% complete
        </p>
      </div>
    )
  }
  
  // Not started
  if (status.warmup_status === 'not_started') {
    return (
      <div className={cn('flex items-center gap-2 text-sm text-muted-foreground', className)}>
        <XCircle className="w-4 h-4" />
        <span>JIT not warmed up</span>
        <Button 
          size="sm" 
          variant="ghost"
          onClick={handleRetry}
          className="ml-auto"
        >
          <Zap className="w-3 h-3 mr-1" />
          Start Warmup
        </Button>
      </div>
    )
  }
  
  return null
}
