import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

// Format number with appropriate precision
export function formatNumber(value: number, precision: number = 3): string {
  if (Math.abs(value) < 0.001 && value !== 0) {
    return value.toExponential(precision)
  }
  return value.toFixed(precision)
}

// Format energy value
export function formatEnergy(value: number): string {
  return `${formatNumber(value, 3)} MeV`
}

// Format radius value  
export function formatRadius(value: number): string {
  return `${formatNumber(value, 3)} fm`
}

// Get nucleus symbol from Z
export function getElementSymbol(z: number): string {
  const symbols: Record<number, string> = {
    1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O',
    9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',
    16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti',
    23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu',
    30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr',
    37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo', 43: 'Tc',
    44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn',
    51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba', 57: 'La',
    58: 'Ce', 59: 'Pr', 60: 'Nd', 61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd',
    65: 'Tb', 66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb', 71: 'Lu',
    72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt',
    79: 'Au', 80: 'Hg', 81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At',
    86: 'Rn', 87: 'Fr', 88: 'Ra', 89: 'Ac', 90: 'Th', 91: 'Pa', 92: 'U',
    93: 'Np', 94: 'Pu',
  }
  return symbols[z] || `Z${z}`
}

// Format nucleus name
export function formatNucleus(z: number, n: number): string {
  const symbol = getElementSymbol(z)
  const a = z + n
  return `${symbol}-${a}`
}

// Get phase display info
export function getPhaseInfo(phase: string): { label: string; color: string } {
  switch (phase) {
    case 'pending':
      return { label: 'Pending', color: 'text-muted-foreground' }
    case 'warmup':
      return { label: 'Warming Up', color: 'text-yellow-500' }
    case 'initializing':
      return { label: 'Initializing', color: 'text-blue-500' }
    case 'iterating':
      return { label: 'Iterating', color: 'text-primary' }
    case 'converged':
      return { label: 'Converged', color: 'text-green-500' }
    case 'failed':
      return { label: 'Failed', color: 'text-destructive' }
    case 'cancelled':
      return { label: 'Cancelled', color: 'text-muted-foreground' }
    default:
      return { label: phase, color: 'text-foreground' }
  }
}

// Format time duration
export function formatDuration(seconds: number): string {
  if (seconds < 60) {
    return `${seconds.toFixed(1)}s`
  } else if (seconds < 3600) {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}m ${secs.toFixed(0)}s`
  } else {
    const hours = Math.floor(seconds / 3600)
    const mins = Math.floor((seconds % 3600) / 60)
    return `${hours}h ${mins}m`
  }
}

// Format date
export function formatDate(dateStr: string): string {
  const date = new Date(dateStr)
  return date.toLocaleString()
}

// Get tag styling based on semantic meaning
export function getTagStyle(tag: string): string {
  const normalizedTag = tag.toLowerCase()
  
  // Semantic color mapping
  if (normalizedTag === 'constrained') {
    return 'bg-amber-500/10 text-amber-700 dark:text-amber-400'
  }
  if (normalizedTag === 'multipole') {
    return 'bg-yellow-500/10 text-yellow-700 dark:text-yellow-400'
  }
  if (normalizedTag === 'beta-gamma' || normalizedTag === 'beta_gamma') {
    return 'bg-rose-500/10 text-rose-700 dark:text-rose-400'
  }
  if (normalizedTag === 'pairing') {
    return 'bg-purple-500/10 text-purple-700 dark:text-purple-400'
  }
  if (normalizedTag === 'vdi' || normalizedTag === 'dddi') {
    return 'bg-indigo-500/10 text-indigo-700 dark:text-indigo-400'
  }
  if (normalizedTag === 'surface-scan' || normalizedTag === 'surface_scan') {
    return 'bg-blue-500/10 text-blue-700 dark:text-blue-400'
  }
  
  // Default gray for unknown tags
  return 'bg-muted text-muted-foreground'
}
