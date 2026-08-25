import { useNavigate } from 'react-router-dom'
import { CalculationStatus as CalcStatus } from '@/types'
import { Card, CardContent } from '@/components/ui'
import { formatNucleus, formatNumber, getPhaseInfo, cn, getTagStyle } from '@/lib/utils'
import { Loader2, CheckCircle, XCircle, AlertCircle, Ban } from 'lucide-react'
import { Progress } from '@/components/ui'

interface ActiveCalculationCardProps {
  calculation: CalcStatus
  onClick?: () => void
}

export function ActiveCalculationCard({ calculation, onClick }: ActiveCalculationCardProps) {
  const navigate = useNavigate()
  
  // Defensive check: if calculation or progress is undefined, don't render
  if (!calculation || !calculation.progress) {
    return null
  }
  
  const progress = calculation.progress
  const phaseInfo = getPhaseInfo(progress.phase)
  
  const handleClick = () => {
    if (onClick) {
      onClick()
      return
    }

    if (calculation.run_type === 'surface') {
      const surfaceId = calculation.surface_results?.id
      navigate(surfaceId ? `/surface/${surfaceId}` : '/surface')
      return
    }

    navigate(`/results/${calculation.id}`)
  }
  
  // Calculate iteration progress percentage
  const iterationProgress = progress.max_iterations > 0
    ? (progress.iteration / progress.max_iterations) * 100
    : 0
  
  // Format nucleus name
  const nucleusName = formatNucleus(
    calculation.nucleus.protons,
    calculation.nucleus.neutrons
  )
  const runTypeLabel = calculation.run_type === 'surface' ? 'Surface' : 'Single'
  
  // Calculate elapsed time
  const startTime = new Date(calculation.started_at).getTime()
  const now = Date.now()
  const elapsedSeconds = (now - startTime) / 1000
  
  // Estimate time per iteration
  const secondsPerIter = progress.iteration > 0
    ? elapsedSeconds / progress.iteration
    : 0
  
  // Phase-specific icon
  const PhaseIcon = () => {
    switch (progress.phase) {
      case 'converged':
        return <CheckCircle className="w-4 h-4 text-green-500 animate-scale-in" />
      case 'failed':
        return <XCircle className="w-4 h-4 text-destructive animate-shake" />
      case 'cancelled':
        return <Ban className="w-4 h-4 text-orange-500" />
      case 'iterating':
        return <Loader2 className="w-4 h-4 text-primary animate-spin" />
      case 'warmup':
      case 'initializing':
        return <AlertCircle className="w-4 h-4 text-yellow-500 animate-pulse-slow" />
      default:
        return <Loader2 className="w-4 h-4 text-muted-foreground animate-spin" />
    }
  }
  
  return (
    <Card 
      className={cn(
        "cursor-pointer transition-all duration-300",
        "hover:bg-accent/50 hover:-translate-y-1 hover:shadow-lg",
        "active:scale-98",
        "animate-scale-in"
      )}
      onClick={handleClick}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault()
          handleClick()
        }
      }}
      aria-label={`${nucleusName} calculation - ${phaseInfo.label}`}
    >
      <CardContent className="p-3 space-y-2">
        {/* Header: Nucleus, Force, Status */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <PhaseIcon />
            <span className="font-semibold text-sm">{nucleusName}</span>
            <span className="text-[10px] uppercase tracking-wide text-muted-foreground">{runTypeLabel}</span>
            <span className="text-xs text-muted-foreground">{calculation.force_name}</span>
          </div>
          <span className={`text-xs font-medium ${phaseInfo.color}`}>
            {phaseInfo.label}
          </span>
        </div>
        
        {/* Tags display */}
        {calculation.tags && calculation.tags.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {calculation.tags.map((tag, idx) => (
              <span
                key={idx}
                className={cn(
                  "inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium",
                  getTagStyle(tag)
                )}
                title={tag}
              >
                {tag}
              </span>
            ))}
          </div>
        )}
        
        {/* Progress bar (only for iterating phase) */}
        {progress.phase === 'iterating' && (
          <div className="space-y-1">
            <Progress 
              value={iterationProgress} 
              className="h-2 transition-all duration-300"
            />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span className="font-medium">Iter {progress.iteration}/{progress.max_iterations}</span>
              <span className="font-mono">{Math.round(iterationProgress)}%</span>
            </div>
          </div>
        )}
        
        {/* Details row (context-dependent) */}
        {progress.phase === 'iterating' && progress.iteration > 0 && (
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div>
              <span className="text-muted-foreground">Energy: </span>
              <span className="font-mono">{formatNumber(progress.energy, 1)} MeV</span>
            </div>
            <div>
              <span className="text-muted-foreground">Δ: </span>
              <span className="font-mono">{progress.fluctuation.toExponential(1)}</span>
            </div>
          </div>
        )}
        
        {/* Timing info */}
        {progress.phase === 'iterating' && secondsPerIter > 0 && (
          <div className="text-xs text-muted-foreground">
            {formatNumber(secondsPerIter, 2)}s/iter
            {progress.iteration < progress.max_iterations && (
              <span className="ml-2">
                • ~{formatNumber((progress.max_iterations - progress.iteration) * secondsPerIter, 0)}s remaining
              </span>
            )}
          </div>
        )}
        
        {/* Message for non-iterating phases */}
        {(progress.phase === 'warmup' || progress.phase === 'initializing' || progress.phase === 'pending') && progress.message && (
          <p className="text-xs text-muted-foreground italic">
            {progress.message}
          </p>
        )}
        
        {/* Elapsed time for all phases */}
        {elapsedSeconds > 1 && (
          <div className="text-xs text-muted-foreground">
            Elapsed: {formatNumber(elapsedSeconds, 1)}s
          </div>
        )}
      </CardContent>
    </Card>
  )
}
