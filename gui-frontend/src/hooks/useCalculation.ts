import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '@/services/api'
import { useStore } from '@/store'
import { CalculationRequest, CalculationStatus } from '@/types'
import { useWebSocket } from './useWebSocket'

export function useStartCalculation() {
  const queryClient = useQueryClient()
  const { addCalculation, getCalculationRequest } = useStore()
  const { subscribe } = useWebSocket()
  
  return useMutation({
    mutationFn: async (request?: CalculationRequest) => {
      const calcRequest = request || getCalculationRequest()
      const response = await api.startCalculation(calcRequest)
      return { ...response, request: calcRequest }
    },
    onSuccess: (data) => {
      // Subscribe to updates
      subscribe(data.calculation_id)
      
      // Optimistically add to active calculations
      addCalculation({
        id: data.calculation_id,
        nucleus: data.request.nucleus,
        force_name: data.request.force_name,
        phase: 'pending',
        progress: {
          calculation_id: data.calculation_id,
          phase: 'pending',
          iteration: 0,
          max_iterations: data.request.iteration?.max_iterations ?? 200,
          fluctuation: 0,
          energy: 0,
          message: 'Starting...',
        },
        started_at: new Date().toISOString(),
      })
      
      // Invalidate active calculations list
      queryClient.invalidateQueries({ queryKey: ['calculations'] })
    },
  })
}

export function useCalculation(id: string | null) {
  return useQuery({
    queryKey: ['calculation', id],
    queryFn: () => id ? api.getCalculation(id) as Promise<CalculationStatus> : null,
    enabled: !!id,
    refetchInterval: (data) => {
      // Only refetch if not in a terminal state
      const status = data?.state?.data as CalculationStatus | undefined
      if (status?.phase && ['converged', 'failed', 'cancelled'].includes(status.phase)) {
        return false
      }
      return 2000 // Refetch every 2 seconds
    },
  })
}

export function useCancelCalculation() {
  const queryClient = useQueryClient()
  const { removeCalculation } = useStore()
  
  return useMutation({
    mutationFn: (id: string) => api.cancelCalculation(id),
    onSuccess: (_, id) => {
      removeCalculation(id)
      queryClient.invalidateQueries({ queryKey: ['calculations'] })
    },
  })
}
