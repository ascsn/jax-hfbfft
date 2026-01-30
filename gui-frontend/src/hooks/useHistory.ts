import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '@/services/api'
import { HistoryResponse, CalculationResults, CalculationStatus, BetaSurfaceResult } from '@/types'

interface HistoryParams {
  nucleus?: string
  element?: string
  min_a?: number
  max_a?: number
  force?: string
  force_name?: string
  status?: string
  page?: number
  page_size?: number
  sort_by?: string
  sort_desc?: boolean
}

export function useHistory(params?: HistoryParams) {
  return useQuery({
    queryKey: ['history', params],
    queryFn: () => api.getHistory(params as Record<string, string | number>) as Promise<HistoryResponse>,
  })
}

export function useHistoricalCalculation(id: string | null) {
  return useQuery({
    queryKey: ['history', 'calculation', id],
    queryFn: () => id ? api.getHistoricalCalculation(id) as Promise<CalculationStatus> : null,
    enabled: !!id,
  })
}

export function useHistoricalResults(id: string | null) {
  return useQuery({
    queryKey: ['history', 'results', id],
    queryFn: () => id ? api.getHistoricalResults(id) as Promise<CalculationResults> : null,
    enabled: !!id,
  })
}

export function useHistoricalSurface(id: string | null) {
  return useQuery({
    queryKey: ['history', 'surface', id],
    queryFn: () => id ? api.getHistoricalSurface(id) as Promise<BetaSurfaceResult> : null,
    enabled: !!id,
  })
}

export function useDeleteFromHistory() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (id: string) => api.deleteFromHistory(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['history'] })
    },
  })
}

export function useHistoryStats() {
  return useQuery({
    queryKey: ['history', 'stats'],
    queryFn: () => api.getHistoryStats(),
  })
}
