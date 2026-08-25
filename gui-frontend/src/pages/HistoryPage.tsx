import { useState, useMemo } from 'react'
import { Link } from 'react-router-dom'
import { useHistory, useForces, useDeleteFromHistory } from '@/hooks'
import { Card, CardContent, Button, Input, Select } from '@/components/ui'
import { Search, Eye, ChevronLeft, ChevronRight, Download, Trash2, X, Tags, ArrowUpDown, ArrowUp, ArrowDown, FileDown, CheckSquare, Square } from 'lucide-react'
import { formatNumber, formatDate, getPhaseInfo, cn, getTagStyle } from '@/lib/utils'

type SortField = 'date' | 'energy' | 'mass' | 'force' | 'status'
type SortDirection = 'asc' | 'desc'

export function HistoryPage() {
  const [filters, setFilters] = useState({
    nucleus: '',
    force: '',
    status: '',
    tags: [] as string[],
  })
  const [page, setPage] = useState(1)
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null)
  const [sortField, setSortField] = useState<SortField>('date')
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc')
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
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
  
  const handleTagFilterToggle = (tag: string) => {
    setFilters(prev => ({
      ...prev,
      tags: prev.tags.includes(tag)
        ? prev.tags.filter(t => t !== tag)
        : [...prev.tags, tag]
    }))
    setPage(1)
  }
  
  const clearTagFilters = () => {
    setFilters(prev => ({ ...prev, tags: [] }))
    setPage(1)
  }
  
  // Extract unique tags from all calculations
  const allUniqueTags = history?.calculations
    ? [...new Set(history.calculations.flatMap(calc => calc.tags || []))].sort()
    : []
  
  const totalPages = history ? Math.ceil(history.total / pageSize) : 1
  
  // Sorting handler
  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(prev => prev === 'asc' ? 'desc' : 'asc')
    } else {
      setSortField(field)
      setSortDirection('desc')
    }
  }
  
  // Sort calculations
  const sortedCalculations = useMemo(() => {
    if (!history?.calculations) return []
    
    const sorted = [...history.calculations]
    sorted.sort((a, b) => {
      let comparison = 0
      
      switch (sortField) {
        case 'date':
          comparison = new Date(a.started_at || 0).getTime() - new Date(b.started_at || 0).getTime()
          break
        case 'energy':
          comparison = (a.energy ?? Infinity) - (b.energy ?? Infinity)
          break
        case 'mass':
          comparison = (a.nucleus_a ?? 0) - (b.nucleus_a ?? 0)
          break
        case 'force':
          comparison = a.force_name.localeCompare(b.force_name)
          break
        case 'status':
          comparison = a.phase.localeCompare(b.phase)
          break
      }
      
      return sortDirection === 'asc' ? comparison : -comparison
    })
    
    return sorted
  }, [history?.calculations, sortField, sortDirection])
  
  // Selection handlers
  const handleToggleSelection = (id: string) => {
    setSelectedIds(prev => {
      const newSet = new Set(prev)
      if (newSet.has(id)) {
        newSet.delete(id)
      } else {
        newSet.add(id)
      }
      return newSet
    })
  }
  
  const handleToggleAllSelection = () => {
    if (selectedIds.size === sortedCalculations.length) {
      setSelectedIds(new Set())
    } else {
      setSelectedIds(new Set(sortedCalculations.map(c => c.id)))
    }
  }
  
  const handleClearSelection = () => {
    setSelectedIds(new Set())
  }
  
  // CSV Export
  const handleExportCSV = () => {
    if (!sortedCalculations.length) return
    
    // Define CSV headers
    const headers = [
      'ID',
      'Nucleus',
      'Mass Number',
      'Force',
      'Type',
      'Status',
      'Energy (MeV)',
      'E/A (MeV)',
      'Tags',
      'Duration',
      'Started At',
      'Completed At'
    ]
    
    // Generate CSV rows
    const rows = sortedCalculations.map(calc => {
      const nucleusName = calc.nucleus_symbol && calc.nucleus_a > 0 
        ? `${calc.nucleus_symbol}-${calc.nucleus_a}`
        : 'Unknown'
      const massNumber = calc.nucleus_a ?? 0
      const energyPerNucleon = calc.energy && massNumber > 0
        ? (calc.energy / massNumber).toFixed(3)
        : ''
      const duration = calc.started_at && calc.completed_at
        ? formatDuration(new Date(calc.completed_at).getTime() - new Date(calc.started_at).getTime())
        : ''
      const tags = calc.tags?.join('; ') || ''
      
      return [
        calc.id,
        nucleusName,
        massNumber,
        calc.force_name,
        calc.run_type ?? 'calculation',
        calc.phase,
        calc.energy?.toFixed(2) ?? '',
        energyPerNucleon,
        tags,
        duration,
        calc.started_at || '',
        calc.completed_at || ''
      ]
    })
    
    // Create CSV content
    const csvContent = [
      headers.join(','),
      ...rows.map(row => row.map(cell => `"${cell}"`).join(','))
    ].join('\n')
    
    // Download CSV
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' })
    const link = document.createElement('a')
    const url = URL.createObjectURL(blob)
    link.setAttribute('href', url)
    link.setAttribute('download', `hfb-history-${new Date().toISOString().split('T')[0]}.csv`)
    link.style.visibility = 'hidden'
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }
  
  // Format duration helper
  const formatDuration = (ms: number): string => {
    const seconds = Math.floor(ms / 1000)
    if (seconds < 60) return `${seconds}s`
    const minutes = Math.floor(seconds / 60)
    const remainingSeconds = seconds % 60
    if (minutes < 60) return `${minutes}m ${remainingSeconds}s`
    const hours = Math.floor(minutes / 60)
    const remainingMinutes = minutes % 60
    return `${hours}h ${remainingMinutes}m`
  }
  
  // Sort icon component
  const SortIcon = ({ field }: { field: SortField }) => {
    if (sortField !== field) {
      return <ArrowUpDown className="w-3 h-3 opacity-40" />
    }
    return sortDirection === 'asc' 
      ? <ArrowUp className="w-3 h-3" />
      : <ArrowDown className="w-3 h-3" />
  }
  
  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Run History</h1>
        <div className="flex items-center gap-2">
          <Button 
            variant="outline" 
            onClick={handleExportCSV}
            disabled={!sortedCalculations.length}
            className="interactive-scale"
            title="Export history as CSV"
          >
            <FileDown className="w-4 h-4 mr-2" />
            Export CSV
          </Button>
          <Button variant="outline" onClick={() => refetch()} className="interactive-scale">
            Refresh
          </Button>
        </div>
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
          
          {/* Tag filter */}
          {allUniqueTags.length > 0 && (
            <div className="flex flex-wrap items-center gap-2 pt-4 border-t">
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Tags className="w-4 h-4" />
                <span>Filter by tags:</span>
              </div>
              {allUniqueTags.map(tag => (
                <button
                  key={tag}
                  onClick={() => handleTagFilterToggle(tag)}
                  className={cn(
                    "inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium",
                    "transition-all duration-200 active:scale-95",
                    "hover:ring-2 hover:ring-offset-1 hover:scale-105",
                    filters.tags.includes(tag)
                      ? getTagStyle(tag) + " ring-2 ring-current animate-scale-in"
                      : "bg-muted text-muted-foreground hover:bg-muted/80"
                  )}
                  aria-pressed={filters.tags.includes(tag)}
                  aria-label={`Filter by ${tag}`}
                >
                  {tag}
                  {filters.tags.includes(tag) && (
                    <X className="w-3 h-3" />
                  )}
                </button>
              ))}
              {filters.tags.length > 0 && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={clearTagFilters}
                  className="text-xs"
                >
                  Clear {filters.tags.length} filter{filters.tags.length > 1 ? 's' : ''}
                </Button>
              )}
            </div>
          )}
        </CardContent>
      </Card>
      
      {/* Selection Actions */}
      {selectedIds.size > 0 && (
        <Card className="bg-muted/50 border-primary/30 animate-scale-in">
          <CardContent className="py-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <CheckSquare className="w-5 h-5 text-primary" />
                <span className="font-medium">
                  {selectedIds.size} calculation{selectedIds.size !== 1 ? 's' : ''} selected
                </span>
              </div>
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  disabled
                  title="Comparison feature coming soon"
                  className="interactive-scale"
                >
                  Compare Selected
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleClearSelection}
                  className="interactive-scale"
                >
                  Clear Selection
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
      
      {/* Results table */}
      <Card>
        <CardContent className="p-0">
          {isLoading ? (
            <div className="p-6">
              {/* Skeleton loading state */}
              <div className="space-y-4">
                <div className="grid grid-cols-6 gap-4 pb-3 border-b">
                  {[...Array(6)].map((_, i) => (
                    <div key={i} className="h-4 skeleton" />
                  ))}
                </div>
                {[...Array(5)].map((_, rowIdx) => (
                  <div 
                    key={rowIdx} 
                    className={`grid grid-cols-6 gap-4 py-3 animate-slide-up stagger-${rowIdx + 1}`}
                  >
                    {[...Array(6)].map((_, colIdx) => (
                      <div key={colIdx} className="h-4 skeleton" />
                    ))}
                  </div>
                ))}
              </div>
              <p className="mt-6 text-center text-sm text-muted-foreground animate-pulse-slow">
                Loading history...
              </p>
            </div>
          ) : !history?.calculations?.length ? (
            <div className="p-12 text-center animate-fade-in">
              <Search className="w-16 h-16 text-muted-foreground/30 mx-auto mb-4" />
              <p className="text-lg font-semibold mb-2">No calculations found</p>
              <p className="text-sm text-muted-foreground max-w-md mx-auto">
                {filters.nucleus || filters.force || filters.status || filters.tags.length > 0
                  ? 'Try adjusting your filters or clearing them to see more results.'
                  : 'Start a new calculation to see it appear here.'}
              </p>
              {(filters.nucleus || filters.force || filters.status || filters.tags.length > 0) && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    setFilters({ nucleus: '', force: '', status: '', tags: [] })
                    setPage(1)
                  }}
                  className="mt-4 interactive-scale"
                >
                  Clear All Filters
                </Button>
              )}
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full min-w-[1000px]">
                <thead>
                  <tr className="border-b bg-muted/50">
                    <th className="text-left p-4 font-semibold w-10 sticky left-0 bg-muted/50 z-10">
                      <button
                        onClick={handleToggleAllSelection}
                        className="hover:scale-110 transition-transform"
                        title={selectedIds.size === sortedCalculations.length ? "Deselect all" : "Select all"}
                        aria-label={selectedIds.size === sortedCalculations.length ? "Deselect all" : "Select all"}
                      >
                        {selectedIds.size === sortedCalculations.length ? (
                          <CheckSquare className="w-5 h-5 text-primary" />
                        ) : selectedIds.size > 0 ? (
                          <CheckSquare className="w-5 h-5 text-primary/50" />
                        ) : (
                          <Square className="w-5 h-5 text-muted-foreground" />
                        )}
                      </button>
                    </th>
                    <th className="text-left p-4 font-semibold sticky left-10 bg-muted/50 z-10">
                      <button
                        onClick={() => handleSort('mass')}
                        className="flex items-center gap-1 hover:text-primary transition-colors"
                        title="Sort by mass number"
                      >
                        Nucleus
                        <SortIcon field="mass" />
                      </button>
                    </th>
                    <th className="text-left p-4 font-semibold">
                      <button
                        onClick={() => handleSort('force')}
                        className="flex items-center gap-1 hover:text-primary transition-colors"
                        title="Sort by force"
                      >
                        Force
                        <SortIcon field="force" />
                      </button>
                    </th>
                    <th className="text-left p-4 font-semibold">Type</th>
                    <th className="text-left p-4 font-semibold">Tags</th>
                    <th className="text-left p-4 font-semibold">
                      <button
                        onClick={() => handleSort('status')}
                        className="flex items-center gap-1 hover:text-primary transition-colors"
                        title="Sort by status"
                      >
                        Status
                        <SortIcon field="status" />
                      </button>
                    </th>
                    <th className="text-left p-4 font-semibold">
                      <button
                        onClick={() => handleSort('energy')}
                        className="flex items-center gap-1 hover:text-primary transition-colors"
                        title="Sort by total energy"
                      >
                        Energy (MeV)
                        <SortIcon field="energy" />
                      </button>
                    </th>
                    <th className="text-left p-4 font-semibold" title="Binding energy per nucleon">
                      E/A (MeV)
                    </th>
                    <th className="text-left p-4 font-semibold" title="Calculation duration">
                      Duration
                    </th>
                    <th className="text-left p-4 font-semibold">
                      <button
                        onClick={() => handleSort('date')}
                        className="flex items-center gap-1 hover:text-primary transition-colors"
                        title="Sort by date"
                      >
                        Date
                        <SortIcon field="date" />
                      </button>
                    </th>
                    <th className="text-left p-4 font-semibold">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {sortedCalculations.map((calc, idx) => {
                    const phaseInfo = getPhaseInfo(calc.phase)
                    // Backend returns nucleus_symbol and nucleus_a from CalculationSummary
                    const nucleusName = calc.nucleus_symbol && calc.nucleus_a > 0 
                      ? `${calc.nucleus_symbol}-${calc.nucleus_a}`
                      : 'Unknown'
                    const massNumber = calc.nucleus_a ?? 0
                    const runType = calc.run_type ?? 'calculation'
                    const energyPerNucleon = calc.energy && massNumber > 0
                      ? calc.energy / massNumber
                      : null
                    const duration = calc.started_at && calc.completed_at
                      ? new Date(calc.completed_at).getTime() - new Date(calc.started_at).getTime()
                      : null
                    
                    return (
                      <tr 
                        key={calc.id} 
                        className={cn(
                          "border-b transition-all duration-200",
                          "hover:bg-muted/50 hover:scale-[1.005]",
                          "animate-slide-up",
                          idx < 10 && `stagger-${idx + 1}`,
                          selectedIds.has(calc.id) && "bg-primary/5 border-primary/20"
                        )}
                      >
                        <td className="p-4 sticky left-0 bg-background z-10">
                          <button
                            onClick={() => handleToggleSelection(calc.id)}
                            className="hover:scale-110 transition-transform"
                            aria-label={selectedIds.has(calc.id) ? "Deselect calculation" : "Select calculation"}
                          >
                            {selectedIds.has(calc.id) ? (
                              <CheckSquare className="w-5 h-5 text-primary" />
                            ) : (
                              <Square className="w-5 h-5 text-muted-foreground" />
                            )}
                          </button>
                        </td>
                        <td className="p-4 sticky left-10 bg-background z-10">
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
                          {calc.tags && calc.tags.length > 0 ? (
                            <div className="flex flex-wrap gap-1">
                              {calc.tags.map((tag, idx) => (
                                <span
                                  key={idx}
                                  className={cn(
                                    "inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium",
                                    getTagStyle(tag)
                                  )}
                                  title={tag}
                                >
                                  {tag}
                                </span>
                              ))}
                            </div>
                          ) : (
                            <span className="text-muted-foreground">—</span>
                          )}
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
                        <td className="p-4 font-mono" title="Binding energy per nucleon">
                          {energyPerNucleon != null
                            ? formatNumber(energyPerNucleon, 3)
                            : '—'
                          }
                        </td>
                        <td className="p-4 text-sm font-mono" title="Calculation duration">
                          {duration != null
                            ? formatDuration(duration)
                            : (calc.phase === 'iterating' ? (
                              <span className="text-blue-600 animate-pulse-slow">Running...</span>
                            ) : '—')
                          }
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
                              className={cn(
                                "interactive-scale",
                                deleteConfirm === calc.id && "animate-pulse-slow"
                              )}
                              aria-label={deleteConfirm === calc.id ? "Confirm delete" : "Delete calculation"}
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
