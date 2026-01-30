import { useState } from 'react'
import { Outlet, Link, useLocation } from 'react-router-dom'
import { useStore } from '@/store'
import { Button } from '@/components/ui'
import { Atom, History, Cpu, Wifi, WifiOff, Info, ChevronDown, ChevronUp } from 'lucide-react'
import { cn } from '@/lib/utils'
import { DiagnosticsPanel } from './DiagnosticsPanel'

export function Layout() {
  const location = useLocation()
  const { systemStatus, wsConnected } = useStore()
  const [showDiagnostics, setShowDiagnostics] = useState(false)
  
  const navItems = [
    { path: '/', label: 'Calculator', icon: Atom },
    { path: '/surface', label: 'Surfaces', icon: Atom },
    { path: '/history', label: 'History', icon: History },
  ]
  
  return (
    <div className="min-h-screen flex flex-col bg-background">
      {/* Header */}
      <header className="border-b bg-card">
        <div className="container mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-6">
            {/* Logo */}
            <Link to="/" className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg nucleus-gradient flex items-center justify-center">
                <Atom className="w-5 h-5 text-white" />
              </div>
              <span className="font-bold text-xl">HFBFFT</span>
            </Link>
            
            {/* Navigation */}
            <nav className="flex items-center gap-1">
              {navItems.map(({ path, label, icon: Icon }) => (
                <Link
                  key={path}
                  to={path}
                  className={cn(
                    "flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-colors",
                    location.pathname === path
                      ? "bg-primary text-primary-foreground"
                      : "text-muted-foreground hover:text-foreground hover:bg-muted"
                  )}
                >
                  <Icon className="w-4 h-4" />
                  {label}
                </Link>
              ))}
            </nav>
          </div>
          
          {/* Status indicators */}
          <div className="flex items-center gap-4">
            {/* Backend indicator */}
            {systemStatus && (
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Cpu className="w-4 h-4" />
                <span className="capitalize">{systemStatus.backend}</span>
              </div>
            )}
            
            {/* Diagnostics toggle */}
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowDiagnostics(!showDiagnostics)}
              className="gap-2"
            >
              <Info className="w-4 h-4" />
              {showDiagnostics ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
            </Button>
            
            {/* WebSocket connection */}
            <div className="flex items-center gap-1">
              {wsConnected ? (
                <Wifi className="w-4 h-4 text-green-500" />
              ) : (
                <WifiOff className="w-4 h-4 text-destructive" />
              )}
            </div>
          </div>
        </div>
      </header>
      
      {/* Diagnostics Panel (expandable) */}
      {showDiagnostics && (
        <div className="border-b bg-muted/30">
          <div className="container mx-auto px-4 py-4">
            <DiagnosticsPanel />
          </div>
        </div>
      )}
      
      {/* Main content */}
      <main className="flex-1 container mx-auto px-4 py-6">
        <Outlet />
      </main>
      
      {/* Footer */}
      <footer className="border-t bg-card py-4">
        <div className="container mx-auto px-4 text-center text-sm text-muted-foreground">
          <p>HFBFFT - Hartree-Fock-Bogoliubov Nuclear Structure Calculator</p>
          <p className="mt-1">
            <a 
              href="https://github.com/ascsn/jax-hfbfft" 
              target="_blank" 
              rel="noopener noreferrer"
              className="hover:text-foreground"
            >
              GitHub
            </a>
            {' • '}
            <a 
              href="/docs" 
              target="_blank"
              className="hover:text-foreground"
            >
              API Docs
            </a>
          </p>
        </div>
      </footer>
    </div>
  )
}
