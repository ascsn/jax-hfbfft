import { useQuery } from '@tanstack/react-query'
import type { DensityType } from '@/components/DensityVisualizer'

export interface DensityData {
  density: number[][][]
  grid: {
    nx: number
    ny: number
    nz: number
    dx: number
    dy: number
    dz: number
  }
  type: DensityType
  metadata: {
    min_value: number
    max_value: number
    units: string
  }
}

export function useDensityData(
  calculationId: string | undefined,
  densityType: string = 'total',
  options?: {
    downsample?: number
    enabled?: boolean
  }
) {
  return useQuery({
    queryKey: ['density', calculationId, densityType, options?.downsample],
    queryFn: async () => {
      if (!calculationId) throw new Error('No calculation ID')
      
      const params = new URLSearchParams()
      params.append('type', densityType)
      if (options?.downsample) {
        params.append('downsample', options.downsample.toString())
      }
      
      const response = await fetch(
        `/api/calculations/${calculationId}/densities?${params}`,
        {
          headers: { 'Content-Type': 'application/json' },
        }
      )
      
      if (!response.ok) {
        throw new Error(`Failed to fetch density data: ${response.statusText}`)
      }
      
      return response.json() as Promise<DensityData>
    },
    enabled: options?.enabled !== false && !!calculationId,
    staleTime: 5 * 60 * 1000, // 5 minutes
    gcTime: 10 * 60 * 1000, // 10 minutes (renamed from cacheTime)
  })
}
