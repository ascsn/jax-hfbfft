import { useRef, useMemo, useState, useEffect } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, PerspectiveCamera, Html } from '@react-three/drei'
import { Card, CardHeader, CardTitle, CardContent, Button, Select } from '@/components/ui'
import { Play, Pause, Eye, Settings, Loader2 } from 'lucide-react'
import * as THREE from 'three'

type VisualizationDimension = '1D' | '2D' | '3D'
type VisualizationMode3D = 'contours' | 'points'
export type DensityType = 'total' | 'neutron' | 'proton' | 'tau_n' | 'tau_p' | 'tau_total'

interface DensityData {
  density: number[][][]  // 3D density array [nx][ny][nz]
  grid: {
    nx: number
    ny: number
    nz: number
    dx: number
    dy: number
    dz: number
  }
  type: DensityType
  metadata?: {
    min_value: number
    max_value: number
    units: string
  }
}

interface DensityVisualizerProps {
  densityUrl?: string  // URL to fetch density data
  initialData?: DensityData
  isLoading?: boolean
  onDensityTypeChange?: (type: DensityType) => void
}

// Generate volumetric points for visualization
function generateVolumePoints(
  density: number[][][],
  threshold: number,
  grid: { nx: number; ny: number; nz: number; dx: number; dy: number; dz: number },
  maxPoints: number = 50000
): { positions: Float32Array; colors: Float32Array } {
  const positions: number[] = []
  const colors: number[] = []
  const { nx, ny, nz, dx, dy, dz } = grid
  
  // Find max density for normalization
  let maxDensity = 0
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      for (let k = 0; k < nz; k++) {
        maxDensity = Math.max(maxDensity, density[i][j][k])
      }
    }
  }
  
  // Center offset
  const cx = (nx * dx) / 2
  const cy = (ny * dy) / 2
  const cz = (nz * dz) / 2
  
  // Collect points above threshold
  const candidatePoints: { x: number; y: number; z: number; val: number }[] = []
  
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      for (let k = 0; k < nz; k++) {
        const val = density[i][j][k]
        if (val > threshold * maxDensity) {
          candidatePoints.push({
            x: i * dx - cx,
            y: j * dy - cy,
            z: k * dz - cz,
            val: val / maxDensity
          })
        }
      }
    }
  }
  
  // Sample if too many points
  const points = candidatePoints.length > maxPoints
    ? candidatePoints.sort(() => Math.random() - 0.5).slice(0, maxPoints)
    : candidatePoints
  
  // Create position and color arrays
  for (const p of points) {
    positions.push(p.x, p.y, p.z)
    
    // Color based on density (blue = low, red = high)
    const t = p.val
    colors.push(
      t,           // R
      0.2,         // G
      1 - t,       // B
    )
  }
  
  return {
    positions: new Float32Array(positions),
    colors: new Float32Array(colors),
  }
}

// Demo density generator (nuclear-like distribution)
function generateDemoDensity(nx: number, ny: number, nz: number): number[][][] {
  const density: number[][][] = []
  const cx = nx / 2
  const cy = ny / 2
  const cz = nz / 2
  const R = Math.min(nx, ny, nz) * 0.35  // Nuclear radius
  const a = 0.5  // Surface diffuseness
  
  for (let i = 0; i < nx; i++) {
    density[i] = []
    for (let j = 0; j < ny; j++) {
      density[i][j] = []
      for (let k = 0; k < nz; k++) {
        const r = Math.sqrt(
          (i - cx) ** 2 + (j - cy) ** 2 + (k - cz) ** 2
        )
        // Woods-Saxon distribution
        density[i][j][k] = 0.16 / (1 + Math.exp((r - R) / a))
      }
    }
  }
  
  return density
}

function DensityCloud({ 
  data, 
  threshold = 0.1,
  rotating = false,
}: { 
  data: DensityData
  threshold: number
  rotating: boolean
}) {
  const pointsRef = useRef<THREE.Points>(null)
  
  const { positions, colors } = useMemo(() => {
    return generateVolumePoints(data.density, threshold, data.grid)
  }, [data, threshold])
  
  useFrame((_state, delta) => {
    if (rotating && pointsRef.current) {
      pointsRef.current.rotation.y += delta * 0.5
    }
  })
  
  if (positions.length === 0) {
    return null
  }
  
  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          args={[positions, 3]}
        />
        <bufferAttribute
          attach="attributes-color"
          args={[colors, 3]}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.15}
        vertexColors
        transparent
        opacity={0.8}
        sizeAttenuation
      />
    </points>
  )
}

function GridHelper({ size = 10 }: { size?: number }) {
  return (
    <>
      <gridHelper args={[size, 10, '#444', '#222']} rotation={[0, 0, 0]} />
      <axesHelper args={[size / 2]} />
      
      {/* Axis labels */}
      <Html position={[size / 2 + 0.5, 0, 0]} center>
        <div className="text-xs text-red-500 font-mono">X</div>
      </Html>
      <Html position={[0, size / 2 + 0.5, 0]} center>
        <div className="text-xs text-green-500 font-mono">Y</div>
      </Html>
      <Html position={[0, 0, size / 2 + 0.5]} center>
        <div className="text-xs text-blue-500 font-mono">Z</div>
      </Html>
    </>
  )
}

// Marching cubes lookup tables (standard 256-entry tables)
const edgeTable: number[] = [
  0x0, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
  0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
  0x190, 0x99, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
  0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
  0x230, 0x339, 0x33, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
  0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
  0x3a0, 0x2a9, 0x1a3, 0xaa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
  0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
  0x460, 0x569, 0x663, 0x76a, 0x66, 0x16f, 0x265, 0x36c,
  0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
  0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff, 0x3f5, 0x2fc,
  0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
  0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55, 0x15c,
  0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
  0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc,
  0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
  0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
  0xcc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
  0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
  0x15c, 0x55, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
  0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
  0x2fc, 0x3f5, 0xff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
  0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
  0x36c, 0x265, 0x16f, 0x66, 0x76a, 0x663, 0x569, 0x460,
  0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
  0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa, 0x1a3, 0x2a9, 0x3a0,
  0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
  0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33, 0x339, 0x230,
  0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
  0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99, 0x190,
  0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
  0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0,
]

const triTable: number[][] = [
  [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1],
  [3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1],
  [3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1],
  [3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1],
  [9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1],
  [1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1],
  [9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1],
  [2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1],
  [8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1],
  [9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1],
  [4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1],
  [3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1],
  [1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1],
  [4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1],
  [4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1],
  [9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1],
  [1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1],
  [5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1],
  [2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1],
  [9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1],
  [0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1],
  [2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1],
  [10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1],
  [4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1],
  [5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1],
  [5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1],
  [9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1],
  [0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1],
  [1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1],
  [10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1],
  [8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1],
  [2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1],
  [7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1],
  [9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1],
  [2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1],
  [11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1],
  [9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1],
  [5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1],
  [11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1],
  [11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1],
  [1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1],
  [9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1],
  [5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1],
  [2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1],
  [0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1],
  [5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1],
  [6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1],
  [0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1],
  [3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1],
  [6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1],
  [5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1],
  [1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1],
  [10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1],
  [6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1],
  [1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1],
  [8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1],
  [7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1],
  [3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1],
  [5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1],
  [0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1],
  [9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1],
  [8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1],
  [5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1],
  [0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1],
  [6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1],
  [10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1],
  [10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1],
  [8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1],
  [1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1],
  [3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1],
  [0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1],
  [10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1],
  [0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1],
  [3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1],
  [6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1],
  [9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1],
  [8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1],
  [3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1],
  [6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1],
  [0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1],
  [10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1],
  [10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1],
  [1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1],
  [2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1],
  [7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1],
  [7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1],
  [2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1],
  [1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1],
  [11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1],
  [8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1],
  [0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1],
  [7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1],
  [10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1],
  [2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1],
  [6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1],
  [7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1],
  [2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1],
  [1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1],
  [10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1],
  [10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1],
  [0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1],
  [7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1],
  [6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1],
  [8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1],
  [9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1],
  [6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1],
  [1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1],
  [4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1],
  [10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1],
  [8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1],
  [0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1],
  [1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1],
  [8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1],
  [10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1],
  [4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1],
  [10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1],
  [5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1],
  [11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1],
  [9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1],
  [6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1],
  [7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1],
  [3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1],
  [7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1],
  [9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1],
  [3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1],
  [6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1],
  [9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1],
  [1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1],
  [4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1],
  [7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1],
  [6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1],
  [3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1],
  [0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1],
  [6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1],
  [1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1],
  [0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1],
  [11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1],
  [6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1],
  [5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1],
  [9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1],
  [1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1],
  [1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1],
  [10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1],
  [0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1],
  [5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1],
  [10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1],
  [11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1],
  [0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1],
  [9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1],
  [7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1],
  [2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1],
  [8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1],
  [9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1],
  [9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1],
  [1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1],
  [9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1],
  [9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1],
  [5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1],
  [0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1],
  [10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1],
  [2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1],
  [0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1],
  [0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1],
  [9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1],
  [5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1],
  [3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1],
  [5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1],
  [8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1],
  [0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1],
  [9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1],
  [0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1],
  [1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1],
  [3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1],
  [4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1],
  [9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1],
  [11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1],
  [11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1],
  [2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1],
  [9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1],
  [3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1],
  [1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1],
  [4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1],
  [4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1],
  [0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1],
  [3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1],
  [3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1],
  [0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1],
  [9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1],
  [1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
]

const edgeIndex: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 0],
  [4, 5], [5, 6], [6, 7], [7, 4],
  [0, 4], [1, 5], [2, 6], [3, 7],
]

// Generate isosurface mesh using marching cubes algorithm (full table)
function generateIsosurface(
  density: number[][][],
  isoValue: number,
  grid: { nx: number; ny: number; nz: number; dx: number; dy: number; dz: number }
): THREE.BufferGeometry {
  const { nx, ny, nz, dx, dy, dz } = grid
  const geometry = new THREE.BufferGeometry()
  
  const vertices: number[] = []
  const normals: number[] = []
  
  // Center offset
  const cx = (nx * dx) / 2
  const cy = (ny * dy) / 2
  const cz = (nz * dz) / 2
  
  // Get density value with bounds checking
  const getDensity = (i: number, j: number, k: number): number => {
    if (i < 0 || i >= nx || j < 0 || j >= ny || k < 0 || k >= nz) return 0
    return density[i][j][k]
  }
  
  const getGradient = (i: number, j: number, k: number): [number, number, number] => {
    const gx = (getDensity(i + 1, j, k) - getDensity(i - 1, j, k)) / (2 * dx)
    const gy = (getDensity(i, j + 1, k) - getDensity(i, j - 1, k)) / (2 * dy)
    const gz = (getDensity(i, j, k + 1) - getDensity(i, j, k - 1)) / (2 * dz)
    const len = Math.sqrt(gx * gx + gy * gy + gz * gz) || 1
    return [gx / len, gy / len, gz / len]
  }
  
  const interpolate = (
    p1: [number, number, number],
    p2: [number, number, number],
    v1: number,
    v2: number
  ): [number, number, number] => {
    if (Math.abs(isoValue - v1) < 1e-6) return p1
    if (Math.abs(isoValue - v2) < 1e-6) return p2
    if (Math.abs(v1 - v2) < 1e-6) return p1
    const t = (isoValue - v1) / (v2 - v1)
    return [
      p1[0] + t * (p2[0] - p1[0]),
      p1[1] + t * (p2[1] - p1[1]),
      p1[2] + t * (p2[2] - p1[2])
    ]
  }
  
  const interpolateNormal = (
    n1: [number, number, number],
    n2: [number, number, number],
    v1: number,
    v2: number
  ): [number, number, number] => {
    if (Math.abs(isoValue - v1) < 1e-6) return n1
    if (Math.abs(isoValue - v2) < 1e-6) return n2
    if (Math.abs(v1 - v2) < 1e-6) return n1
    const t = (isoValue - v1) / (v2 - v1)
    const nxv = n1[0] + t * (n2[0] - n1[0])
    const nyv = n1[1] + t * (n2[1] - n1[1])
    const nzv = n1[2] + t * (n2[2] - n1[2])
    const len = Math.sqrt(nxv * nxv + nyv * nyv + nzv * nzv) || 1
    return [nxv / len, nyv / len, nzv / len]
  }
  
  for (let i = 0; i < nx - 1; i++) {
    for (let j = 0; j < ny - 1; j++) {
      for (let k = 0; k < nz - 1; k++) {
        const v = [
          getDensity(i, j, k),
          getDensity(i + 1, j, k),
          getDensity(i + 1, j + 1, k),
          getDensity(i, j + 1, k),
          getDensity(i, j, k + 1),
          getDensity(i + 1, j, k + 1),
          getDensity(i + 1, j + 1, k + 1),
          getDensity(i, j + 1, k + 1)
        ]
        
        let cubeIndex = 0
        for (let n = 0; n < 8; n++) {
          if (v[n] > isoValue) cubeIndex |= (1 << n)
        }
        
        const edges = edgeTable[cubeIndex]
        if (edges === 0) continue
        
        const p: [number, number, number][] = [
          [i * dx - cx, j * dy - cy, k * dz - cz],
          [(i + 1) * dx - cx, j * dy - cy, k * dz - cz],
          [(i + 1) * dx - cx, (j + 1) * dy - cy, k * dz - cz],
          [i * dx - cx, (j + 1) * dy - cy, k * dz - cz],
          [i * dx - cx, j * dy - cy, (k + 1) * dz - cz],
          [(i + 1) * dx - cx, j * dy - cy, (k + 1) * dz - cz],
          [(i + 1) * dx - cx, (j + 1) * dy - cy, (k + 1) * dz - cz],
          [i * dx - cx, (j + 1) * dy - cy, (k + 1) * dz - cz]
        ]
        
        const g: [number, number, number][] = [
          getGradient(i, j, k),
          getGradient(i + 1, j, k),
          getGradient(i + 1, j + 1, k),
          getGradient(i, j + 1, k),
          getGradient(i, j, k + 1),
          getGradient(i + 1, j, k + 1),
          getGradient(i + 1, j + 1, k + 1),
          getGradient(i, j + 1, k + 1)
        ]
        
        const vertList: ([number, number, number] | null)[] = new Array(12).fill(null)
        const normList: ([number, number, number] | null)[] = new Array(12).fill(null)
        
        for (let e = 0; e < 12; e++) {
          if (edges & (1 << e)) {
            const [a, b] = edgeIndex[e]
            vertList[e] = interpolate(p[a], p[b], v[a], v[b])
            normList[e] = interpolateNormal(g[a], g[b], v[a], v[b])
          }
        }
        
        const tri = triTable[cubeIndex]
        for (let t = 0; t < 16; t += 3) {
          const a = tri[t]
          const b = tri[t + 1]
          const c = tri[t + 2]
          if (a === -1 || b === -1 || c === -1) break
          const va = vertList[a]
          const vb = vertList[b]
          const vc = vertList[c]
          const na = normList[a]
          const nb = normList[b]
          const nc = normList[c]
          if (!va || !vb || !vc || !na || !nb || !nc) continue
          vertices.push(...va, ...vb, ...vc)
          normals.push(...na, ...nb, ...nc)
        }
      }
    }
  }
  
  if (vertices.length > 0) {
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3))
    geometry.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3))
    geometry.computeBoundingSphere()
  }
  
  return geometry
}

function DensityIsosurface({ 
  data, 
  isoValue = 0.08,
}: { 
  data: DensityData
  isoValue: number
}) {
  const meshRef = useRef<THREE.Mesh>(null)
  
  const geometry = useMemo(() => {
    return generateIsosurface(data.density, isoValue, data.grid)
  }, [data, isoValue])
  
  if (!geometry.attributes.position || geometry.attributes.position.count === 0) {
    // Fall back to Woods-Saxon sphere approximation if marching cubes produces no geometry
    // Calculate approximate nuclear radius based on density distribution
    const { nx, ny, nz, dx, dy, dz } = data.grid
    
    // Find the extent of the density above threshold
    let maxRadius = 0
    const cx = nx / 2
    const cy = ny / 2
    const cz = nz / 2
    
    for (let i = 0; i < nx; i++) {
      for (let j = 0; j < ny; j++) {
        for (let k = 0; k < nz; k++) {
          if (data.density[i][j][k] > isoValue) {
            const r = Math.sqrt(
              ((i - cx) * dx) ** 2 +
              ((j - cy) * dy) ** 2 +
              ((k - cz) * dz) ** 2
            )
            maxRadius = Math.max(maxRadius, r)
          }
        }
      }
    }
    
    const radius = maxRadius || (Math.min(nx, ny, nz) * dx * 0.35)
    
    return (
      <mesh ref={meshRef}>
        <sphereGeometry args={[radius, 64, 64]} />
        <meshStandardMaterial
          color="#2dd4bf"
          transparent
          opacity={0.7}
          side={THREE.DoubleSide}
          roughness={0.3}
          metalness={0.2}
          emissive="#0d9488"
          emissiveIntensity={0.2}
        />
      </mesh>
    )
  }
  
  return (
    <mesh ref={meshRef} geometry={geometry}>
      <meshStandardMaterial
        color="#2dd4bf"
        transparent
        opacity={0.7}
        side={THREE.DoubleSide}
        roughness={0.3}
        metalness={0.2}
        flatShading={false}
        emissive="#0d9488"
        emissiveIntensity={0.2}
      />
    </mesh>
  )
}

// 2D Density Slice Visualizer
function DensitySlice2D({ 
  data, 
  sliceAxis = 'z',
  sliceIndex,
}: { 
  data: DensityData
  sliceAxis: 'x' | 'y' | 'z'
  sliceIndex: number
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  
  useEffect(() => {
    if (!canvasRef.current) return
    
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    const { density, grid } = data
    const { nx, ny, nz } = grid
    
    // Get the 2D slice data
    let sliceData: number[][] = []
    let width = 0, height = 0
    
    if (sliceAxis === 'z') {
      width = nx
      height = ny
      const k = Math.min(Math.max(0, sliceIndex), nz - 1)
      for (let i = 0; i < nx; i++) {
        sliceData[i] = []
        for (let j = 0; j < ny; j++) {
          sliceData[i][j] = density[i][j][k]
        }
      }
    } else if (sliceAxis === 'y') {
      width = nx
      height = nz
      const j = Math.min(Math.max(0, sliceIndex), ny - 1)
      for (let i = 0; i < nx; i++) {
        sliceData[i] = []
        for (let k = 0; k < nz; k++) {
          sliceData[i][k] = density[i][j][k]
        }
      }
    } else { // x
      width = ny
      height = nz
      const i = Math.min(Math.max(0, sliceIndex), nx - 1)
      for (let j = 0; j < ny; j++) {
        sliceData[j] = []
        for (let k = 0; k < nz; k++) {
          sliceData[j][k] = density[i][j][k]
        }
      }
    }
    
    // Find min/max for color scaling
    let minVal = Infinity, maxVal = -Infinity
    for (let i = 0; i < width; i++) {
      for (let j = 0; j < height; j++) {
        minVal = Math.min(minVal, sliceData[i][j])
        maxVal = Math.max(maxVal, sliceData[i][j])
      }
    }
    
    // Handle case where all values are the same
    const range = maxVal - minVal
    const effectiveRange = range > 0 ? range : 1
    
    // Calculate scale to fit the container (target ~350px for each dimension)
    const targetSize = 350
    const scale = Math.max(1, Math.floor(targetSize / Math.max(width, height)))
    
    canvas.width = width * scale
    canvas.height = height * scale
    
    // Draw the colormap
    const imageData = ctx.createImageData(canvas.width, canvas.height)
    
    for (let i = 0; i < width; i++) {
      for (let j = 0; j < height; j++) {
        const val = (sliceData[i][j] - minVal) / effectiveRange
        
        // Improved Viridis-like colormap
        const r = Math.floor(255 * Math.max(0, Math.min(1, -0.27 + 2.78 * val - 2.02 * val * val + 0.57 * val * val * val)))
        const g = Math.floor(255 * Math.max(0, Math.min(1, 0.01 + 1.34 * val - 0.38 * val * val)))
        const b = Math.floor(255 * Math.max(0, Math.min(1, 0.33 + 1.04 * val - 1.92 * val * val + 0.70 * val * val * val)))
        
        // Fill the scaled pixels
        for (let si = 0; si < scale; si++) {
          for (let sj = 0; sj < scale; sj++) {
            const idx = ((j * scale + sj) * canvas.width + (i * scale + si)) * 4
            imageData.data[idx] = r
            imageData.data[idx + 1] = g
            imageData.data[idx + 2] = b
            imageData.data[idx + 3] = 255
          }
        }
      }
    }
    
    ctx.putImageData(imageData, 0, 0)
  }, [data, sliceAxis, sliceIndex])
  
  return (
    <div className="relative bg-gray-900 flex items-center justify-center min-h-[500px] p-4">
      <canvas 
        ref={canvasRef} 
        className="max-w-full max-h-[480px] object-contain border border-gray-700 rounded"
        style={{ imageRendering: 'pixelated' }}
      />
      <div className="absolute top-4 left-4 bg-black/70 p-2 rounded text-xs text-white">
        <p>{sliceAxis.toUpperCase()}-axis slice at index {sliceIndex}</p>
        <p className="text-muted-foreground mt-1">Density: {data.type}</p>
      </div>
    </div>
  )
}

// 1D Density Lineout
function DensityLineout({ 
  data, 
  axis = 'z',
}: { 
  data: DensityData
  axis: 'x' | 'y' | 'z'
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  
  useEffect(() => {
    if (!canvasRef.current) return
    
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    const { density, grid } = data
    const { nx, ny, nz, dx, dy, dz } = grid
    
    // Calculate radial average along the specified axis
    const centerX = nx / 2
    const centerY = ny / 2
    const centerZ = nz / 2
    
    let lineData: { r: number; rho: number }[] = []
    
    if (axis === 'z') {
      for (let k = 0; k < nz; k++) {
        const z = (k - centerZ) * dz
        let sum = 0
        let count = 0
        // Average over xy plane at this z
        for (let i = 0; i < nx; i++) {
          for (let j = 0; j < ny; j++) {
            const r = Math.sqrt(((i - centerX) * dx) ** 2 + ((j - centerY) * dy) ** 2)
            if (r < 2) { // Only include points near axis
              sum += density[i][j][k]
              count++
            }
          }
        }
        if (count > 0) {
          lineData.push({ r: z, rho: sum / count })
        }
      }
    } else {
      // Radial profile from center
      const maxR = Math.min(nx * dx, ny * dy, nz * dz) / 2
      const numPoints = 100
      for (let i = 0; i <= numPoints; i++) {
        const r = (i / numPoints) * maxR
        let sum = 0
        let count = 0
        
        for (let ix = 0; ix < nx; ix++) {
          for (let iy = 0; iy < ny; iy++) {
            for (let iz = 0; iz < nz; iz++) {
              const dist = Math.sqrt(
                ((ix - centerX) * dx) ** 2 +
                ((iy - centerY) * dy) ** 2 +
                ((iz - centerZ) * dz) ** 2
              )
              if (Math.abs(dist - r) < 0.5) {
                sum += density[ix][iy][iz]
                count++
              }
            }
          }
        }
        
        if (count > 0) {
          lineData.push({ r, rho: sum / count })
        }
      }
    }
    
    if (lineData.length === 0) return
    
    // Set canvas size - larger for better visibility
    canvas.width = 700
    canvas.height = 350
    
    // Clear canvas
    ctx.fillStyle = '#1a1a2e'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    
    // Find data range
    const minR = Math.min(...lineData.map(d => d.r))
    const maxR = Math.max(...lineData.map(d => d.r))
    const maxRho = Math.max(...lineData.map(d => d.rho)) || 1
    
    const margin = 50
    const plotWidth = canvas.width - 2 * margin
    const plotHeight = canvas.height - 2 * margin
    
    // Draw axes
    ctx.strokeStyle = '#555'
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(margin, margin)
    ctx.lineTo(margin, canvas.height - margin)
    ctx.lineTo(canvas.width - margin, canvas.height - margin)
    ctx.stroke()
    
    // Draw grid
    ctx.strokeStyle = '#333'
    ctx.lineWidth = 0.5
    for (let i = 0; i <= 5; i++) {
      const y = margin + (plotHeight / 5) * i
      ctx.beginPath()
      ctx.moveTo(margin, y)
      ctx.lineTo(canvas.width - margin, y)
      ctx.stroke()
    }
    
    // Draw data
    ctx.strokeStyle = '#4a90d9'
    ctx.lineWidth = 2.5
    ctx.beginPath()
    
    const rRange = maxR - minR || 1
    for (let i = 0; i < lineData.length; i++) {
      const { r, rho } = lineData[i]
      const x = margin + ((r - minR) / rRange) * plotWidth
      const y = canvas.height - margin - (rho / maxRho) * plotHeight
      
      if (i === 0) {
        ctx.moveTo(x, y)
      } else {
        ctx.lineTo(x, y)
      }
    }
    ctx.stroke()
    
    // Draw labels
    ctx.fillStyle = '#fff'
    ctx.font = '14px monospace'
    ctx.textAlign = 'center'
    ctx.fillText(axis === 'z' ? 'z (fm)' : 'r (fm)', canvas.width / 2, canvas.height - 12)
    
    ctx.save()
    ctx.translate(18, canvas.height / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.fillText('ρ (fm⁻³)', 0, 0)
    ctx.restore()
    
    // Draw max value label
    ctx.textAlign = 'left'
    ctx.font = '12px monospace'
    ctx.fillText(`Max: ${maxRho.toFixed(4)} fm⁻³`, margin + 10, margin + 20)
    
  }, [data, axis])
  
  return (
    <div className="relative bg-gray-900 flex items-center justify-center min-h-[500px] p-4">
      <canvas 
        ref={canvasRef} 
        className="max-w-full max-h-[480px] object-contain"
      />
      <div className="absolute top-4 right-4 bg-black/70 p-2 rounded text-xs text-white">
        <p>{axis === 'z' ? 'Axial profile' : 'Radial profile'}</p>
        <p className="text-muted-foreground mt-1">Density: {data.type}</p>
      </div>
    </div>
  )
}

export function DensityVisualizer({ densityUrl: _densityUrl, initialData, isLoading, onDensityTypeChange }: DensityVisualizerProps) {
  const [dimension, setDimension] = useState<VisualizationDimension>('3D')
  const [mode3D, setMode3D] = useState<VisualizationMode3D>('contours')
  const [densityType, setDensityType] = useState<DensityType>('total')
  const [isoValue, setIsoValue] = useState(0.02) // Default 0.02 fm^-3 for contours (half saturation)
  const [threshold, setThreshold] = useState(0.1) // For point cloud mode
  const [rotating, setRotating] = useState(false)
  const [sliceAxis, setSliceAxis] = useState<'x' | 'y' | 'z'>('z')
  const [sliceIndex, setSliceIndex] = useState(12)
  const [lineoutAxis, setLineoutAxis] = useState<'x' | 'y' | 'z'>('z')
  const [showAdvanced, setShowAdvanced] = useState(false)
  
  // Notify parent when density type changes
  useEffect(() => {
    if (onDensityTypeChange) {
      onDensityTypeChange(densityType)
    }
  }, [densityType, onDensityTypeChange])
  
  // Use real data if provided, otherwise fall back to demo data
  const data: DensityData = useMemo(() => {
    if (initialData && initialData.density && initialData.density.length > 0) {
      return initialData
    }
    
    // Generate demo nuclear density (only if no real data)
    const nx = 24, ny = 24, nz = 24
    const dx = 1.0, dy = 1.0, dz = 1.0
    
    return {
      density: generateDemoDensity(nx, ny, nz),
      grid: { nx, ny, nz, dx, dy, dz },
      type: 'total'
    }
  }, [initialData])
  
  // Contour level options (in fm^-3)
  // Nuclear saturation density is ~0.16 fm^-3
  const isoValueOptions = [
    { value: '0.01', label: '0.01 fm⁻³' },
    { value: '0.02', label: '0.02 fm⁻³ (recommended)' },
    { value: '0.04', label: '0.04 fm⁻³' },
    { value: '0.06', label: '0.06 fm⁻³' },
    { value: '0.08', label: '0.08 fm⁻³ (half saturation)' },
    { value: '0.10', label: '0.10 fm⁻³' },
    { value: '0.12', label: '0.12 fm⁻³' },
  ]
  
  const thresholdOptions = [
    { value: '0.01', label: '1% max' },
    { value: '0.05', label: '5% max' },
    { value: '0.1', label: '10% max' },
    { value: '0.2', label: '20% max' },
    { value: '0.5', label: '50% max' },
  ]
  
  const mode3DOptions = [
    { value: 'contours', label: 'Contour Surface' },
    { value: 'points', label: 'Point Cloud' },
  ]
  
  const dimensionOptions = [
    { value: '1D', label: '1D Profile' },
    { value: '2D', label: '2D Slice' },
    { value: '3D', label: '3D Volume' },
  ]
  
  const densityTypeOptions = [
    { value: 'total', label: 'Total ρ' },
    { value: 'neutron', label: 'Neutron ρₙ' },
    { value: 'proton', label: 'Proton ρₚ' },
    { value: 'tau_total', label: 'Kinetic τ' },
    { value: 'tau_n', label: 'Kinetic τₙ' },
    { value: 'tau_p', label: 'Kinetic τₚ' },
  ]
  
  const axisOptions = [
    { value: 'x', label: 'X axis' },
    { value: 'y', label: 'Y axis' },
    { value: 'z', label: 'Z axis' },
  ]
  
  const maxSliceIndex = sliceAxis === 'x' ? data.grid.nx - 1 : 
                        sliceAxis === 'y' ? data.grid.ny - 1 : 
                        data.grid.nz - 1
  
  const hasRealData = initialData && initialData.density && initialData.density.length > 0
  
  return (
    <Card className="overflow-hidden">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Eye className="w-5 h-5" />
              Density Visualization
              {isLoading && <Loader2 className="w-4 h-4 animate-spin text-muted-foreground" />}
            </CardTitle>
            {!hasRealData && !isLoading && (
              <p className="text-xs text-muted-foreground mt-1">
                Showing demo data (calculation data not available)
              </p>
            )}
          </div>
          
          <div className="flex items-center gap-2 flex-wrap">
            {/* Dimension selector */}
            <Select
              className="w-32"
              value={dimension}
              onChange={(e) => setDimension(e.target.value as VisualizationDimension)}
              options={dimensionOptions}
            />
            
            {/* Advanced settings toggle */}
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowAdvanced(!showAdvanced)}
              title="Advanced options"
            >
              <Settings className={`w-4 h-4 ${showAdvanced ? 'text-primary' : ''}`} />
            </Button>
          </div>
        </div>
        
        {/* Advanced controls */}
        {showAdvanced && (
          <div className="mt-3 pt-3 border-t space-y-2">
            <div className="flex items-center gap-2 flex-wrap text-sm">
              <span className="text-muted-foreground">Density:</span>
              <Select
                className="w-36"
                value={densityType}
                onChange={(e) => setDensityType(e.target.value as DensityType)}
                options={densityTypeOptions}
              />
              
              {dimension === '3D' && (
                <>
                  <span className="text-muted-foreground ml-2">Mode:</span>
                  <Select
                    className="w-36"
                    value={mode3D}
                    onChange={(e) => setMode3D(e.target.value as VisualizationMode3D)}
                    options={mode3DOptions}
                  />
                  
                  {mode3D === 'contours' ? (
                    <>
                      <span className="text-muted-foreground ml-2">Level:</span>
                      <Select
                        className="w-40"
                        value={isoValue.toString()}
                        onChange={(e) => setIsoValue(parseFloat(e.target.value))}
                        options={isoValueOptions}
                      />
                    </>
                  ) : (
                    <>
                      <span className="text-muted-foreground ml-2">Threshold:</span>
                      <Select
                        className="w-28"
                        value={threshold.toString()}
                        onChange={(e) => setThreshold(parseFloat(e.target.value))}
                        options={thresholdOptions}
                      />
                    </>
                  )}
                  
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setRotating(!rotating)}
                    title={rotating ? 'Stop rotation' : 'Start rotation'}
                    className="ml-2"
                  >
                    {rotating ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                  </Button>
                </>
              )}
              
              {dimension === '2D' && (
                <>
                  <span className="text-muted-foreground ml-2">Slice:</span>
                  <Select
                    className="w-28"
                    value={sliceAxis}
                    onChange={(e) => {
                      const newAxis = e.target.value as 'x' | 'y' | 'z'
                      setSliceAxis(newAxis)
                      const maxIdx = newAxis === 'x' ? data.grid.nx - 1 : 
                                     newAxis === 'y' ? data.grid.ny - 1 : 
                                     data.grid.nz - 1
                      setSliceIndex(Math.floor(maxIdx / 2))
                    }}
                    options={axisOptions}
                  />
                  <input
                    type="range"
                    min="0"
                    max={maxSliceIndex}
                    value={sliceIndex}
                    onChange={(e) => setSliceIndex(parseInt(e.target.value))}
                    className="w-32"
                  />
                  <span className="text-xs text-muted-foreground">{sliceIndex}</span>
                </>
              )}
              
              {dimension === '1D' && (
                <>
                  <span className="text-muted-foreground ml-2">Profile:</span>
                  <Select
                    className="w-28"
                    value={lineoutAxis}
                    onChange={(e) => setLineoutAxis(e.target.value as 'x' | 'y' | 'z')}
                    options={[
                      { value: 'z', label: 'Axial (z)' },
                      { value: 'radial', label: 'Radial (r)' },
                    ]}
                  />
                </>
              )}
            </div>
          </div>
        )}
      </CardHeader>
      
      <CardContent className="p-0">
        {dimension === '3D' && (
          <div className="h-[500px] bg-gray-900 relative">
            <Canvas>
              <PerspectiveCamera makeDefault position={[15, 15, 15]} />
              <OrbitControls 
                enablePan={true}
                enableZoom={true}
                enableRotate={true}
                autoRotate={rotating}
                autoRotateSpeed={1}
              />
              
              <ambientLight intensity={0.4} />
              <directionalLight position={[10, 10, 10]} intensity={0.8} castShadow />
              <pointLight position={[-10, 10, 5]} intensity={0.5} />
              <pointLight position={[5, -10, -10]} intensity={0.3} />
              <hemisphereLight args={['#87ceeb', '#1a1a2e', 0.3]} />
              
              {mode3D === 'contours' ? (
                <DensityIsosurface 
                  data={data} 
                  isoValue={isoValue}
                />
              ) : (
                <DensityCloud 
                  data={data} 
                  threshold={threshold}
                  rotating={false}  // Rotation handled by OrbitControls
                />
              )}
              
              <GridHelper size={data.grid.nx * data.grid.dx} />
            </Canvas>
            
            {/* Legend */}
            <div className="absolute bottom-4 left-4 bg-black/50 p-2 rounded text-xs text-white">
              {mode3D === 'contours' ? (
                <>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 bg-teal-400 rounded opacity-70" />
                    <span>ρ = {isoValue} fm⁻³ isosurface</span>
                  </div>
                </>
              ) : (
                <>
                  <div className="flex items-center gap-2 mb-1">
                    <div className="w-3 h-3 bg-blue-500 rounded" />
                    <span>Low density</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 bg-red-500 rounded" />
                    <span>High density</span>
                  </div>
                </>
              )}
            </div>
            
            {/* Controls hint */}
            <div className="absolute bottom-4 right-4 bg-black/50 p-2 rounded text-xs text-white">
              <p>🖱️ Drag to rotate</p>
              <p>⚲ Scroll to zoom</p>
            </div>
          </div>
        )}
        
        {dimension === '2D' && (
          <DensitySlice2D 
            data={data}
            sliceAxis={sliceAxis}
            sliceIndex={sliceIndex}
          />
        )}
        
        {dimension === '1D' && (
          <DensityLineout 
            data={data}
            axis={lineoutAxis}
          />
        )}
      </CardContent>
    </Card>
  )
}
