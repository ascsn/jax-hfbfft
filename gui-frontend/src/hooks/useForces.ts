import { useQuery } from '@tanstack/react-query'
import { api } from '@/services/api'
import { ForceInfo } from '@/types'

export function useForces() {
  return useQuery({
    queryKey: ['forces'],
    queryFn: () => api.listForces() as Promise<ForceInfo[]>,
    staleTime: Infinity, // Forces don't change
  })
}
