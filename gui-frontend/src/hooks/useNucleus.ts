import { useQuery } from '@tanstack/react-query'
import { api } from '@/services/api'
import { Element, NucleusPreset } from '@/types'

export function useElements() {
  return useQuery({
    queryKey: ['elements'],
    queryFn: () => api.listElements() as Promise<Element[]>,
    staleTime: Infinity,
  })
}

export function usePresets() {
  return useQuery({
    queryKey: ['presets'],
    queryFn: () => api.listPresets() as Promise<NucleusPreset[]>,
    staleTime: Infinity,
  })
}
