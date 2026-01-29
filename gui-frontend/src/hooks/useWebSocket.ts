import { useEffect, useRef, useCallback } from 'react'
import { useStore } from '@/store'
import { WSMessage, CalculationProgress, WarmupStatus, CalculationStatus } from '@/types'

const WS_URL = `ws://${window.location.host}/ws`

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<number | null>(null)
  const { 
    setWsConnected, 
    updateCalculationProgress, 
    setWarmupStatus,
    updateCalculation,
  } = useStore()

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return
    }

    try {
      const ws = new WebSocket(WS_URL)
      wsRef.current = ws

      ws.onopen = () => {
        console.log('WebSocket connected')
        setWsConnected(true)
        
        // Request warmup status
        ws.send(JSON.stringify({ type: 'get_warmup_status' }))
      }

      ws.onclose = () => {
        console.log('WebSocket disconnected')
        setWsConnected(false)
        
        // Reconnect after 3 seconds
        reconnectTimeoutRef.current = window.setTimeout(() => {
          connect()
        }, 3000)
      }

      ws.onerror = (error) => {
        console.error('WebSocket error:', error)
      }

      ws.onmessage = (event) => {
        try {
          const message: WSMessage = JSON.parse(event.data)
          handleMessage(message)
        } catch (e) {
          console.error('Failed to parse WebSocket message:', e)
        }
      }
    } catch (e) {
      console.error('Failed to create WebSocket:', e)
    }
  }, [setWsConnected])

  const handleMessage = useCallback((message: WSMessage) => {
    switch (message.type) {
      case 'progress':
        updateCalculationProgress(message.data as CalculationProgress)
        break
      
      case 'status':
        const status = message.data as CalculationStatus
        updateCalculation(status.id, status)
        break
      
      case 'warmup_status':
        setWarmupStatus(message.data as WarmupStatus)
        break
      
      case 'subscribed':
      case 'unsubscribed':
        // Acknowledgement messages
        break
      
      case 'pong':
        // Heartbeat response
        break
      
      case 'error':
        console.error('WebSocket error message:', message.message)
        break
      
      default:
        console.log('Unknown WebSocket message type:', message.type)
    }
  }, [updateCalculationProgress, setWarmupStatus, updateCalculation])

  const subscribe = useCallback((calculationId: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'subscribe',
        calculation_id: calculationId,
      }))
    }
  }, [])

  const unsubscribe = useCallback((calculationId: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'unsubscribe',
        calculation_id: calculationId,
      }))
    }
  }, [])

  useEffect(() => {
    connect()

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
      wsRef.current?.close()
    }
  }, [connect])

  return { subscribe, unsubscribe }
}
