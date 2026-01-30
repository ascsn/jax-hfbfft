const API_BASE = '/api'

async function fetchApi<T>(
  endpoint: string, 
  options?: RequestInit
): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
    ...options,
  })
  
  if (!response.ok) {
    const error = await response.json().catch(() => ({}))
    throw new Error(error.detail || `API error: ${response.status}`)
  }
  
  return response.json()
}

// Calculations
export const api = {
  // Start a new calculation
  startCalculation: (request: unknown) =>
    fetchApi<{ calculation_id: string; message: string }>('/calculations', {
      method: 'POST',
      body: JSON.stringify(request),
    }),
  
  // Get calculation status
  getCalculation: (id: string) =>
    fetchApi<unknown>(`/calculations/${id}`),
  
  // List active calculations
  listCalculations: () =>
    fetchApi<unknown[]>('/calculations'),
  
  // Cancel a calculation
  cancelCalculation: (id: string) =>
    fetchApi<{ message: string }>(`/calculations/${id}`, {
      method: 'DELETE',
    }),
  
  // History
  getHistory: (params?: Record<string, string | number>) => {
    const searchParams = new URLSearchParams()
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined && value !== null && value !== '') {
          searchParams.append(key, String(value))
        }
      })
    }
    const query = searchParams.toString()
    return fetchApi<unknown>(`/history${query ? `?${query}` : ''}`)
  },
  
  getHistoricalCalculation: (id: string) =>
    fetchApi<unknown>(`/history/${id}`),
  
  getHistoricalResults: (id: string) =>
    fetchApi<unknown>(`/history/${id}/results`),

  getHistoricalSurface: (id: string) =>
    fetchApi<unknown>(`/history/${id}/surface`),
  
  deleteFromHistory: (id: string) =>
    fetchApi<{ message: string }>(`/history/${id}`, {
      method: 'DELETE',
    }),
  
  getHistoryStats: () =>
    fetchApi<unknown>('/history/stats'),
  
  // System
  getSystemStatus: () =>
    fetchApi<unknown>('/status'),
  
  getDiagnostics: () =>
    fetchApi<unknown>('/diagnostics'),
  
  triggerWarmup: (force?: boolean) =>
    fetchApi<unknown>('/warmup' + (force ? '?force=true' : ''), {
      method: 'POST',
    }),
  
  getWarmupStatus: () =>
    fetchApi<unknown>('/warmup/status'),
  
  // Forces
  listForces: () =>
    fetchApi<unknown[]>('/forces'),
  
  getForce: (name: string) =>
    fetchApi<unknown>('/forces/' + name),
  
  // Elements and presets
  listElements: () =>
    fetchApi<unknown[]>('/elements'),
  
  listPresets: () =>
    fetchApi<unknown[]>('/presets'),

  // Surfaces
  runBetaSurface: (request: unknown) =>
    fetchApi<unknown>('/surfaces/beta', {
      method: 'POST',
      body: JSON.stringify(request),
    }),

  startBetaSurface: (request: unknown) =>
    fetchApi<{ calculation_id: string; message: string }>('/surfaces/beta/start', {
      method: 'POST',
      body: JSON.stringify(request),
    }),
}
