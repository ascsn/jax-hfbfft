import { useQuery } from '@tanstack/react-query'
import { useStore } from '@/store'
import { api } from '@/services/api'
import { SystemStatus } from '@/types'
import { useEffect } from 'react'

export function useSystemStatus() {
  const { setSystemStatus } = useStore()
  
  const query = useQuery({
    queryKey: ['systemStatus'],
    queryFn: () => api.getSystemStatus() as Promise<SystemStatus>,
    refetchInterval: 10000, // Refresh every 10 seconds
  })
  
  useEffect(() => {
    if (query.data) {
      setSystemStatus(query.data)
    }
  }, [query.data, setSystemStatus])
  
  return query
}
