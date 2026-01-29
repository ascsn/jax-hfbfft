import { useQuery } from '@tanstack/react-query'

interface NucleusPreset {
  symbol: string
  name: string
  z: number
  n: number
  a: number
}

export function usePresets() {
  return useQuery({
    queryKey: ['presets'],
    queryFn: async (): Promise<NucleusPreset[]> => {
      // Common nuclei presets
      const presets: NucleusPreset[] = [
        { symbol: 'O', name: 'Oxygen-16', z: 8, n: 8, a: 16 },
        { symbol: 'Ca', name: 'Calcium-40', z: 20, n: 20, a: 40 },
        { symbol: 'Ca', name: 'Calcium-48', z: 20, n: 28, a: 48 },
        { symbol: 'Ni', name: 'Nickel-56', z: 28, n: 28, a: 56 },
        { symbol: 'Ni', name: 'Nickel-78', z: 28, n: 50, a: 78 },
        { symbol: 'Zr', name: 'Zirconium-90', z: 40, n: 50, a: 90 },
        { symbol: 'Sn', name: 'Tin-100', z: 50, n: 50, a: 100 },
        { symbol: 'Sn', name: 'Tin-132', z: 50, n: 82, a: 132 },
        { symbol: 'Pb', name: 'Lead-208', z: 82, n: 126, a: 208 },
        { symbol: 'He', name: 'Helium-4', z: 2, n: 2, a: 4 },
        { symbol: 'C', name: 'Carbon-12', z: 6, n: 6, a: 12 },
        { symbol: 'Mg', name: 'Magnesium-24', z: 12, n: 12, a: 24 },
        { symbol: 'Si', name: 'Silicon-28', z: 14, n: 14, a: 28 },
        { symbol: 'Fe', name: 'Iron-56', z: 26, n: 30, a: 56 },
        { symbol: 'Kr', name: 'Krypton-84', z: 36, n: 48, a: 84 },
        { symbol: 'Xe', name: 'Xenon-132', z: 54, n: 78, a: 132 },
      ]
      return presets
    },
    staleTime: Infinity, // Presets never change
  })
}
