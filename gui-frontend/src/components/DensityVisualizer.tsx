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

// Generate isosurface mesh using marching cubes algorithm
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
  
  // Compute gradient for normal calculation
  const computeGradient = (i: number, j: number, k: number): [number, number, number] => {
    const gx = (getDensity(i+1, j, k) - getDensity(i-1, j, k)) / (2 * dx)
    const gy = (getDensity(i, j+1, k) - getDensity(i, j-1, k)) / (2 * dy)
    const gz = (getDensity(i, j, k+1) - getDensity(i, j, k-1)) / (2 * dz)
    const len = Math.sqrt(gx*gx + gy*gy + gz*gz) || 1
    return [gx/len, gy/len, gz/len]
  }
  
  // Interpolate vertex position
  const interpolate = (
    p1: [number, number, number],
    p2: [number, number, number],
    v1: number,
    v2: number
  ): [number, number, number] => {
    if (Math.abs(isoValue - v1) < 0.00001) return p1
    if (Math.abs(isoValue - v2) < 0.00001) return p2
    if (Math.abs(v1 - v2) < 0.00001) return p1
    const t = (isoValue - v1) / (v2 - v1)
    return [
      p1[0] + t * (p2[0] - p1[0]),
      p1[1] + t * (p2[1] - p1[1]),
      p1[2] + t * (p2[2] - p1[2])
    ]
  }
  
  // Simplified marching cubes: only handle common cases
  // Process each voxel
  for (let i = 0; i < nx - 1; i++) {
    for (let j = 0; j < ny - 1; j++) {
      for (let k = 0; k < nz - 1; k++) {
        // Get corner values
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
        
        // Calculate cube index
        let cubeIndex = 0
        for (let n = 0; n < 8; n++) {
          if (v[n] > isoValue) cubeIndex |= (1 << n)
        }
        
        // Skip if entirely inside or outside
        if (cubeIndex === 0 || cubeIndex === 255) continue
        
        // Get corner positions in world space
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
        
        // Check which edges cross the isosurface
        const edgeVertices: ([number, number, number] | null)[] = []
        
        // Bottom face edges (0-1, 1-2, 2-3, 3-0)
        if ((v[0] < isoValue) !== (v[1] < isoValue)) edgeVertices[0] = interpolate(p[0], p[1], v[0], v[1])
        else edgeVertices[0] = null
        
        if ((v[1] < isoValue) !== (v[2] < isoValue)) edgeVertices[1] = interpolate(p[1], p[2], v[1], v[2])
        else edgeVertices[1] = null
        
        if ((v[2] < isoValue) !== (v[3] < isoValue)) edgeVertices[2] = interpolate(p[2], p[3], v[2], v[3])
        else edgeVertices[2] = null
        
        if ((v[3] < isoValue) !== (v[0] < isoValue)) edgeVertices[3] = interpolate(p[3], p[0], v[3], v[0])
        else edgeVertices[3] = null
        
        // Top face edges (4-5, 5-6, 6-7, 7-4)
        if ((v[4] < isoValue) !== (v[5] < isoValue)) edgeVertices[4] = interpolate(p[4], p[5], v[4], v[5])
        else edgeVertices[4] = null
        
        if ((v[5] < isoValue) !== (v[6] < isoValue)) edgeVertices[5] = interpolate(p[5], p[6], v[5], v[6])
        else edgeVertices[5] = null
        
        if ((v[6] < isoValue) !== (v[7] < isoValue)) edgeVertices[6] = interpolate(p[6], p[7], v[6], v[7])
        else edgeVertices[6] = null
        
        if ((v[7] < isoValue) !== (v[4] < isoValue)) edgeVertices[7] = interpolate(p[7], p[4], v[7], v[4])
        else edgeVertices[7] = null
        
        // Vertical edges (0-4, 1-5, 2-6, 3-7)
        if ((v[0] < isoValue) !== (v[4] < isoValue)) edgeVertices[8] = interpolate(p[0], p[4], v[0], v[4])
        else edgeVertices[8] = null
        
        if ((v[1] < isoValue) !== (v[5] < isoValue)) edgeVertices[9] = interpolate(p[1], p[5], v[1], v[5])
        else edgeVertices[9] = null
        
        if ((v[2] < isoValue) !== (v[6] < isoValue)) edgeVertices[10] = interpolate(p[2], p[6], v[2], v[6])
        else edgeVertices[10] = null
        
        if ((v[3] < isoValue) !== (v[7] < isoValue)) edgeVertices[11] = interpolate(p[3], p[7], v[3], v[7])
        else edgeVertices[11] = null
        
        // Create triangles connecting the edge crossings
        const validEdges = edgeVertices.filter(v => v !== null) as [number, number, number][]
        
        if (validEdges.length >= 3) {
          // Compute normal at cell center
          const [nx_val, ny_val, nz_val] = computeGradient(i, j, k)
          
          // Simple triangle fan from center
          for (let t = 0; t < validEdges.length - 2; t++) {
            vertices.push(...validEdges[0], ...validEdges[t + 1], ...validEdges[t + 2])
            normals.push(nx_val, ny_val, nz_val, nx_val, ny_val, nz_val, nx_val, ny_val, nz_val)
          }
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
        <sphereGeometry args={[radius, 48, 48]} />
        <meshStandardMaterial
          color="#4a90d9"
          transparent
          opacity={0.65}
          side={THREE.DoubleSide}
          roughness={0.4}
          metalness={0.1}
        />
      </mesh>
    )
  }
  
  return (
    <mesh ref={meshRef} geometry={geometry}>
      <meshStandardMaterial
        color="#4a90d9"
        transparent
        opacity={0.65}
        side={THREE.DoubleSide}
        roughness={0.4}
        metalness={0.1}
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
              
              <ambientLight intensity={0.5} />
              <pointLight position={[10, 10, 10]} intensity={1} />
              <pointLight position={[-10, -10, -10]} intensity={0.3} />
              
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
                    <div className="w-3 h-3 bg-blue-400 rounded opacity-70" />
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
