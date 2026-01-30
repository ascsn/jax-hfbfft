import { useState } from 'react'
import { Link } from 'react-router-dom'
import { useHistory, useForces, useDeleteFromHistory } from '@/hooks'
import { Card, CardContent, Button, Input, Select } from '@/components/ui'
import { Search, Eye, ChevronLeft, ChevronRight, Download, Trash2 } from 'lucide-react'
import { formatNumber, formatDate, getPhaseInfo, cn } from '@/lib/utils'

export function HistoryPage() {
  const [filters, setFilters] = useState({
    nucleus: '',
    force: '',
    status: '',
  })
  const [page, setPage] = useState(1)
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null)
  const pageSize = 20
  
  const { data: forces } = useForces()
  const { data: history, isLoading, refetch } = useHistory({
    ...filters,
    page,
    page_size: pageSize,
  })
  const deleteFromHistory = useDeleteFromHistory()
  
  const handleDelete = (id: string) => {
    if (deleteConfirm === id) {
      deleteFromHistory.mutate(id, {
        onSuccess: () => {
          setDeleteConfirm(null)
          refetch()
        }
      })
    } else {
      setDeleteConfirm(id)
      // Auto-cancel confirm after 3 seconds
      setTimeout(() => setDeleteConfirm(prev => prev === id ? null : prev), 3000)
    }
  }
  
  const statusOptions = [
    { value: '', label: 'All Statuses' },
    { value: 'converged', label: 'Converged' },
    { value: 'iterating', label: 'Running' },
    { value: 'failed', label: 'Failed' },
    { value: 'cancelled', label: 'Cancelled' },
  ]
  
  const forceOptions = [
    { value: '', label: 'All Forces' },
    ...(forces?.map(f => ({ value: f.name, label: f.name })) || []),
  ]
  
  const handleFilterChange = (key: string, value: string) => {
    setFilters(prev => ({ ...prev, [key]: value }))
    setPage(1)
  }
  
  const totalPages = history ? Math.ceil(history.total / pageSize) : 1
  
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Run History</h1>
        <Button variant="outline" onClick={() => refetch()}>
          Refresh
        </Button>
      </div>
      
      {/* Filters */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex flex-wrap gap-4">
            <div className="flex-1 min-w-[200px]">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <Input
                  placeholder="Search by nucleus (e.g. O-16, Ca40)..."
                  className="pl-10"
                  value={filters.nucleus}
                  onChange={(e) => handleFilterChange('nucleus', e.target.value)}
                />
              </div>
            </div>
            
            <Select
              className="w-40"
              value={filters.force}
              onChange={(e) => handleFilterChange('force', e.target.value)}
              options={forceOptions}
            />
            
            <Select
              className="w-40"
              value={filters.status}
              onChange={(e) => handleFilterChange('status', e.target.value)}
              options={statusOptions}
            />
          </div>
        </CardContent>
      </Card>
      
      {/* Results table */}
      <Card>
        <CardContent className="p-0">
          {isLoading ? (
            <div className="p-8 text-center text-muted-foreground">
              <div className="animate-pulse space-y-3">
                <div className="h-4 bg-muted rounded w-1/4 mx-auto"></div>
                <div className="h-4 bg-muted rounded w-1/3 mx-auto"></div>
              </div>
              <p className="mt-4">Loading history...</p>
            </div>
          ) : !history?.calculations?.length ? (
            <div className="p-12 text-center">
              <Search className="w-12 h-12 text-muted-foreground/50 mx-auto mb-4" />
              <p className="text-lg font-medium mb-2">No calculations found</p>
              <p className="text-muted-foreground">
                {filters.nucleus || filters.force || filters.status
                  ? 'Try adjusting your filters or clearing them.'
                  : 'Start a new calculation to see it here.'}
              </p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b bg-muted/50">
                    <th className="text-left p-4 font-medium">Nucleus</th>
                    <th className="text-left p-4 font-medium">Force</th>
                    <th className="text-left p-4 font-medium">Type</th>
                    <th className="text-left p-4 font-medium">Status</th>
                    <th className="text-left p-4 font-medium">Energy (MeV)</th>
                    <th className="text-left p-4 font-medium">Iterations</th>
                    <th className="text-left p-4 font-medium">Date</th>
                    <th className="text-left p-4 font-medium">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {history.calculations.map((calc) => {
                    const phaseInfo = getPhaseInfo(calc.phase)
                    // Backend returns nucleus_symbol and nucleus_a from CalculationSummary
                    const nucleusName = calc.nucleus_symbol && calc.nucleus_a > 0 
                      ? `${calc.nucleus_symbol}-${calc.nucleus_a}`
                      : 'Unknown'
                    const massNumber = calc.nucleus_a ?? 0
                    const runType = calc.run_type ?? 'calculation'
                    
                    return (
                      <tr key={calc.id} className="border-b hover:bg-muted/30">
                        <td className="p-4">
                          <span className="font-medium">{nucleusName}</span>
                          {massNumber > 0 && (
                            <span className="text-muted-foreground text-sm ml-1">
                              (A={massNumber})
                            </span>
                          )}
                        </td>
                        <td className="p-4 font-mono text-sm">{calc.force_name}</td>
                        <td className="p-4">
                          <span className={cn(
                            "inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium",
                            runType === 'surface' && "bg-purple-500/10 text-purple-600",
                            runType === 'calculation' && "bg-muted text-muted-foreground"
                          )}>
                            {runType === 'surface' ? 'Surface' : 'Single'}
                          </span>
                        </td>
                        <td className="p-4">
                          <span className={cn(
                            "inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium",
                            calc.phase === 'converged' && "bg-green-500/10 text-green-600",
                            calc.phase === 'failed' && "bg-red-500/10 text-red-600",
                            calc.phase === 'cancelled' && "bg-yellow-500/10 text-yellow-600",
                            calc.phase === 'iterating' && "bg-blue-500/10 text-blue-600",
                          )}>
                            {phaseInfo.label}
                          </span>
                        </td>
                        <td className="p-4 font-mono">
                          {calc.energy != null
                            ? formatNumber(calc.energy, 2)
                            : '—'
                          }
                        </td>
                        <td className="p-4 font-mono text-sm">
                          {/* Summary doesn't have progress - just show status */}
                          —
                        </td>
                        <td className="p-4 text-sm text-muted-foreground">
                          {calc.started_at ? formatDate(calc.started_at) : '—'}
                        </td>
                        <td className="p-4">
                          <div className="flex items-center gap-1">
                            {runType === 'calculation' ? (
                              <Link to={`/results/${calc.id}`}>
                                <Button variant="ghost" size="sm" title="View details">
                                  <Eye className="w-4 h-4" />
                                </Button>
                              </Link>
                            ) : (
                              <Link to={`/surface/${calc.id}`}>
                                <Button variant="outline" size="sm" title="View surface scan">
                                  <Eye className="w-4 h-4" />
                                </Button>
                              </Link>
                            )}
                            {calc.phase === 'converged' && (
                              <Button variant="ghost" size="sm" title="Download results">
                                <Download className="w-4 h-4" />
                              </Button>
                            )}
                            <Button 
                              variant={deleteConfirm === calc.id ? "destructive" : "ghost"} 
                              size="sm" 
                              title={deleteConfirm === calc.id ? "Click again to confirm" : "Delete"}
                              onClick={() => handleDelete(calc.id)}
                              disabled={deleteFromHistory.isPending}
                            >
                              <Trash2 className="w-4 h-4" />
                            </Button>
                          </div>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
      
      {/* Pagination */}
      {history && history.total > pageSize && (
        <div className="flex items-center justify-between">
          <p className="text-sm text-muted-foreground">
            Showing {((page - 1) * pageSize) + 1} - {Math.min(page * pageSize, history.total)} of {history.total} calculations
          </p>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              disabled={page === 1}
              onClick={() => setPage(p => p - 1)}
            >
              <ChevronLeft className="w-4 h-4 mr-1" />
              Previous
            </Button>
            <span className="text-sm text-muted-foreground">
              Page {page} of {totalPages}
            </span>
            <Button
              variant="outline"
              size="sm"
              disabled={page >= totalPages}
              onClick={() => setPage(p => p + 1)}
            >
              Next
              <ChevronRight className="w-4 h-4 ml-1" />
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}
