import { Cpu, Activity } from 'lucide-react'
import { cn } from '@/lib/utils'

interface MemoryDisplayProps {
  label: string
  used?: number
  total?: number
  icon?: 'cpu' | 'gpu'
  className?: string
}

export function MemoryDisplay({ label, used, total, icon = 'cpu', className }: MemoryDisplayProps) {
  if (!total || !used) {
    return null
  }
  
  const percentage = (used / total) * 100
  const Icon = icon === 'gpu' ? Activity : Cpu
  
  const getColor = (percent: number) => {
    if (percent < 70) return 'text-green-600'
    if (percent < 90) return 'text-yellow-600'
    return 'text-red-600'
  }
  
  const getProgressColor = (percent: number) => {
    if (percent < 70) return 'bg-green-500'
    if (percent < 90) return 'bg-yellow-500'
    return 'bg-red-500'
  }
  
  return (
    <div className={cn('space-y-2', className)}>
      <div className="flex items-center justify-between text-sm">
        <div className="flex items-center gap-2">
          <Icon className="w-4 h-4" />
          <span className="font-medium">{label}</span>
        </div>
        <span className={cn('font-mono', getColor(percentage))}>
          {used.toFixed(1)} / {total.toFixed(1)} GB
        </span>
      </div>
      
      <div className="relative h-2 w-full overflow-hidden rounded-full bg-muted">
        <div 
          className={cn('h-full transition-all duration-300', getProgressColor(percentage))}
          style={{ width: `${Math.min(percentage, 100)}%` }}
        />
      </div>
      
      <p className="text-xs text-muted-foreground text-right">
        {percentage.toFixed(1)}% used
      </p>
    </div>
  )
}
