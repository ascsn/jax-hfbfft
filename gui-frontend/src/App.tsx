import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { Layout } from './components/Layout'
import { CalculatorPage } from './pages/CalculatorPage'
import { HistoryPage } from './pages/HistoryPage'
import { ResultsPage } from './pages/ResultsPage'
import { SurfacePage } from './pages/SurfacePage'
import { useSystemStatus, useWebSocket, useForces, useActiveCalculations } from './hooks'

function App() {
  // Initialize WebSocket connection
  useWebSocket()
  
  // Prefetch system status and forces
  useSystemStatus()
  useForces()
  
  // Fetch and subscribe to active calculations (handles page refresh)
  useActiveCalculations()

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<CalculatorPage />} />
          <Route path="surface" element={<SurfacePage />} />
          <Route path="surface/:id" element={<SurfacePage />} />
          <Route path="history" element={<HistoryPage />} />
          <Route path="results/:id" element={<ResultsPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}

export default App
