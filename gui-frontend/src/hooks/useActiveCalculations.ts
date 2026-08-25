import { useEffect, useRef } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '@/services/api'
import { useStore } from '@/store'
import { useWebSocket } from './useWebSocket'
import { CalculationStatus } from '@/types'

/**
 * Hook to fetch and subscribe to all active calculations on app startup.
 * This ensures that if the page is refreshed, we restore the active calculations
 * from the server and subscribe to their WebSocket updates.
 */
export function useActiveCalculations() {
  const { subscribe } = useWebSocket()
  const subscribedRef = useRef<Set<string>>(new Set())
  const processedRef = useRef<Set<string>>(new Set())
  
  // Fetch active calculations from the server
  const query = useQuery({
    queryKey: ['calculations'],
    queryFn: () => api.listCalculations() as Promise<CalculationStatus[]>,
    refetchInterval: 5000, // Poll every 5 seconds as a fallback
  })
  
  // When we get data, sync it to the store and subscribe to updates
  // Note: We use getState() inside to avoid infinite loops from depending on activeCalculations
  useEffect(() => {
    if (query.data && Array.isArray(query.data)) {
      const { addCalculation, updateCalculation, activeCalculations } = useStore.getState()
      
      query.data.forEach((calc: CalculationStatus) => {
        // Check if we already have this calculation in the store
        const existing = activeCalculations.get(calc.id)
        
        if (!existing) {
          // Add new calculation to store
          addCalculation(calc)
        } else if (!processedRef.current.has(calc.id)) {
          // Only update once per calculation to avoid churn
          updateCalculation(calc.id, calc)
          processedRef.current.add(calc.id)
        }
        
        // Subscribe to WebSocket updates if not already subscribed
        if (!subscribedRef.current.has(calc.id)) {
          subscribe(calc.id)
          subscribedRef.current.add(calc.id)
        }
      })
    }
  }, [query.data, subscribe])
  
  return query
}
