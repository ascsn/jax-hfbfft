import { useState } from 'react'
import { Outlet, Link, useLocation } from 'react-router-dom'
import { useStore } from '@/store'
import { Button } from '@/components/ui'
import { Atom, History, Cpu, Wifi, WifiOff, Info, ChevronDown, ChevronUp, Zap, Radio, Menu, X, type LucideIcon } from 'lucide-react'
import { cn } from '@/lib/utils'
import { DiagnosticsPanel } from './DiagnosticsPanel'
import { ThemeToggle } from './ThemeToggle'

/**
 * Navigation item configuration interface
 * 
 * @property path - Route path for the navigation item
 * @property label - Display label shown in the nav bar
 * @property icon - Lucide icon component to display
 * @property category - Optional category for grouping ('core' for main features, 'research' for advanced modules, 'admin' for settings)
 * @property enabled - Whether the item should be visible and clickable (default: true)
 * @property description - Optional tooltip or description for the feature
 */
interface NavigationItem {
  path: string
  label: string
  icon: LucideIcon
  category?: 'core' | 'research' | 'admin'
  enabled?: boolean
  description?: string
}

export function Layout() {
  const location = useLocation()
  const { systemStatus, wsConnected } = useStore()
  const [showDiagnostics, setShowDiagnostics] = useState(false)
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  
  /**
   * Navigation configuration
   * 
   * To add a new navigation item:
   * 1. Add a new object to this array with path, label, icon, and category
   * 2. Set enabled: true when the feature is ready
   * 3. Add the corresponding route in your router configuration
   * 4. (Optional) Add a description for tooltips or documentation
   * 
   * Categories:
   * - 'core': Main calculation and data visualization features
   * - 'research': Advanced physics modules (reactions, excitations, etc.)
   * - 'admin': System settings and administration
   */
  const navItems: NavigationItem[] = [
    // Core features (always enabled)
    { 
      path: '/', 
      label: 'Calculator', 
      icon: Atom,
      category: 'core',
      enabled: true,
      description: 'Perform HFB calculations'
    },
    { 
      path: '/surface', 
      label: 'Surfaces', 
      icon: Atom,
      category: 'core',
      enabled: true,
      description: 'Visualize energy surfaces'
    },
    { 
      path: '/history', 
      label: 'History', 
      icon: History,
      category: 'core',
      enabled: true,
      description: 'View calculation history'
    },
    
    // Research modules (coming soon - currently disabled)
    { 
      path: '/reactions', 
      label: 'Reactions', 
      icon: Zap,
      category: 'research',
      enabled: false,
      description: 'Nuclear reaction calculations'
    },
    { 
      path: '/excitations', 
      label: 'Collective Excitations', 
      icon: Radio,
      category: 'research',
      enabled: false,
      description: 'Collective excitation analysis'
    },
  ]
  
  // Filter to show only enabled navigation items
  const enabledNavItems = navItems.filter(item => item.enabled !== false)
  
  return (
    <div className="min-h-screen flex flex-col bg-background">
      {/* Header */}
      <header className="sticky top-0 z-50 border-b bg-card/95 backdrop-blur supports-[backdrop-filter]:bg-card/60 transition-all duration-300">
        <div className="container mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-6">
            {/* Logo */}
            <Link to="/" className="flex items-center gap-2 group">
              <div className="w-8 h-8 rounded-lg nucleus-gradient flex items-center justify-center transition-transform duration-300 group-hover:scale-110">
                <Atom className="w-5 h-5 text-white" />
              </div>
              <span className="font-bold text-xl">HFBFFT</span>
            </Link>
            
            {/* Navigation */}
            <nav className="hidden md:flex items-center gap-1" role="navigation" aria-label="Main navigation">
              {enabledNavItems.map(({ path, label, icon: Icon, description }) => (
                <Link
                  key={path}
                  to={path}
                  title={description}
                  className={cn(
                    "relative flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-all duration-200",
                    "hover:scale-105 active:scale-95",
                    location.pathname === path
                      ? "bg-primary text-primary-foreground shadow-md"
                      : "text-muted-foreground hover:text-foreground hover:bg-muted/80"
                  )}
                >
                  <Icon className="w-4 h-4" />
                  {label}
                  {/* Animated underline for active state */}
                  {location.pathname === path && (
                    <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary-foreground animate-slide-up" />
                  )}
                </Link>
              ))}
            </nav>
            
            {/* Mobile menu button */}
            <Button
              variant="ghost"
              size="sm"
              className="md:hidden interactive-scale"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              aria-label="Toggle mobile menu"
              aria-expanded={mobileMenuOpen}
            >
              {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </Button>
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
            {/* Theme toggle */}
            <ThemeToggle />
            
            {/* Diagnostics toggle */}
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowDiagnostics(!showDiagnostics)}
              className="gap-2 interactive-scale"
              aria-label={showDiagnostics ? "Hide diagnostics" : "Show diagnostics"}
              aria-expanded={showDiagnostics}
            >
              <Info className="w-4 h-4" />
              {showDiagnostics ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
            </Button>
            
            {/* WebSocket connection */}
            <div 
              className="flex items-center gap-1" 
              role="status" 
              aria-live="polite"
              aria-label={wsConnected ? "Connected to server" : "Disconnected from server"}
            >
              {wsConnected ? (
                <Wifi className="w-4 h-4 text-green-500 status-pulse" />
              ) : (
                <WifiOff className="w-4 h-4 text-destructive animate-pulse-slow" />
              )}
            </div>
          </div>
        </div>
      </header>
      
      {/* Mobile Navigation Menu */}
      {mobileMenuOpen && (
        <div className="md:hidden border-b bg-card animate-slide-down">
          <nav className="container mx-auto px-4 py-2 flex flex-col gap-1" role="navigation" aria-label="Mobile navigation">
            {enabledNavItems.map(({ path, label, icon: Icon, description }) => (
              <Link
                key={path}
                to={path}
                title={description}
                onClick={() => setMobileMenuOpen(false)}
                className={cn(
                  "flex items-center gap-3 px-4 py-3 rounded-md text-sm font-medium transition-all duration-200",
                  "hover:scale-[1.02] active:scale-95",
                  location.pathname === path
                    ? "bg-primary text-primary-foreground shadow-md"
                    : "text-muted-foreground hover:text-foreground hover:bg-muted/80"
                )}
              >
                <Icon className="w-5 h-5" />
                {label}
              </Link>
            ))}
          </nav>
        </div>
      )}
      
      {/* Diagnostics Panel (expandable) */}
      {showDiagnostics && (
        <div className="border-b bg-muted/30 animate-slide-down">
          <div className="container mx-auto px-4 py-4">
            <DiagnosticsPanel />
          </div>
        </div>
      )}
      
      {/* Main content */}
      <main className="flex-1 container mx-auto px-4 py-6 animate-fade-in">
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
