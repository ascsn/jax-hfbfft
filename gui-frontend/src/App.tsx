import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { Layout } from './components/Layout'
import { CalculatorPage } from './pages/CalculatorPage'
import { HistoryPage } from './pages/HistoryPage'
import { ResultsPage } from './pages/ResultsPage'
import { useSystemStatus, useWebSocket, useForces } from './hooks'

function App() {
  // Initialize WebSocket connection
  useWebSocket()
  
  // Prefetch system status and forces
  useSystemStatus()
  useForces()

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<CalculatorPage />} />
          <Route path="history" element={<HistoryPage />} />
          <Route path="results/:id" element={<ResultsPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}

export default App
