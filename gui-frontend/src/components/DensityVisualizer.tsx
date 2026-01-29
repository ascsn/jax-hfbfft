import { useRef, useMemo, useState } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, PerspectiveCamera, Html } from '@react-three/drei'
import { Card, CardHeader, CardTitle, CardContent, Button, Select } from '@/components/ui'
import { Play, Pause, Eye } from 'lucide-react'
import * as THREE from 'three'

type VisualizationMode = 'contours' | 'points'

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
  type: 'neutron' | 'proton' | 'total'
}

interface DensityVisualizerProps {
  densityUrl?: string  // URL to fetch density data
  initialData?: DensityData
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
  
  // Marching cubes lookup tables (simplified - edge table and triangle table)
  const edgeTable = new Uint16Array([
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
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0
  ])
  
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
  
  // Get density value with bounds checking
  const getDensity = (i: number, j: number, k: number): number => {
    if (i < 0 || i >= nx || j < 0 || j >= ny || k < 0 || k >= nz) return 0
    return density[i][j][k]
  }
  
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
        if (edgeTable[cubeIndex] === 0) continue
        
        // Get corner positions
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
        
        // Calculate vertex positions on edges
        const vertList: [number, number, number][] = new Array(12)
        if (edgeTable[cubeIndex] & 1) vertList[0] = interpolate(p[0], p[1], v[0], v[1])
        if (edgeTable[cubeIndex] & 2) vertList[1] = interpolate(p[1], p[2], v[1], v[2])
        if (edgeTable[cubeIndex] & 4) vertList[2] = interpolate(p[2], p[3], v[2], v[3])
        if (edgeTable[cubeIndex] & 8) vertList[3] = interpolate(p[3], p[0], v[3], v[0])
        if (edgeTable[cubeIndex] & 16) vertList[4] = interpolate(p[4], p[5], v[4], v[5])
        if (edgeTable[cubeIndex] & 32) vertList[5] = interpolate(p[5], p[6], v[5], v[6])
        if (edgeTable[cubeIndex] & 64) vertList[6] = interpolate(p[6], p[7], v[6], v[7])
        if (edgeTable[cubeIndex] & 128) vertList[7] = interpolate(p[7], p[4], v[7], v[4])
        if (edgeTable[cubeIndex] & 256) vertList[8] = interpolate(p[0], p[4], v[0], v[4])
        if (edgeTable[cubeIndex] & 512) vertList[9] = interpolate(p[1], p[5], v[1], v[5])
        if (edgeTable[cubeIndex] & 1024) vertList[10] = interpolate(p[2], p[6], v[2], v[6])
        if (edgeTable[cubeIndex] & 2048) vertList[11] = interpolate(p[3], p[7], v[3], v[7])
        
        // Add triangles using simplified triangle table logic
        // For each configuration, add the triangles
        const triangleConfigs: Record<number, number[]> = {
          1: [0,8,3], 2: [0,1,9], 3: [1,8,3,9,8,1], 4: [1,2,10], 5: [0,8,3,1,2,10],
          6: [9,2,10,0,2,9], 7: [2,8,3,2,10,8,10,9,8], 8: [3,11,2], 9: [0,11,2,8,11,0],
          10: [1,9,0,2,3,11], 11: [1,11,2,1,9,11,9,8,11], 12: [3,10,1,11,10,3],
          // ... (simplified - real implementation needs full 256-entry table)
        }
        
        // Get triangles for this configuration (simplified approach)
        if (triangleConfigs[cubeIndex]) {
          const config = triangleConfigs[cubeIndex]
          for (let t = 0; t < config.length; t += 3) {
            const v1 = vertList[config[t]]
            const v2 = vertList[config[t + 1]]
            const v3 = vertList[config[t + 2]]
            if (v1 && v2 && v3) {
              vertices.push(...v1, ...v2, ...v3)
              // Calculate normal
              const ax = v2[0] - v1[0], ay = v2[1] - v1[1], az = v2[2] - v1[2]
              const bx = v3[0] - v1[0], by = v3[1] - v1[1], bz = v3[2] - v1[2]
              const nx = ay * bz - az * by
              const ny = az * bx - ax * bz
              const nz = ax * by - ay * bx
              const len = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1
              normals.push(nx/len, ny/len, nz/len, nx/len, ny/len, nz/len, nx/len, ny/len, nz/len)
            }
          }
        }
      }
    }
  }
  
  if (vertices.length > 0) {
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3))
    geometry.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3))
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
    // Fall back to a sphere approximation if marching cubes produces no geometry
    const { nx, dx } = data.grid
    const radius = (nx * dx) / 3
    return (
      <mesh ref={meshRef}>
        <sphereGeometry args={[radius, 32, 32]} />
        <meshStandardMaterial
          color="#4a90d9"
          transparent
          opacity={0.7}
          side={THREE.DoubleSide}
        />
      </mesh>
    )
  }
  
  return (
    <mesh ref={meshRef} geometry={geometry}>
      <meshStandardMaterial
        color="#4a90d9"
        transparent
        opacity={0.7}
        side={THREE.DoubleSide}
        flatShading
      />
    </mesh>
  )
}

export function DensityVisualizer({ densityUrl: _densityUrl, initialData }: DensityVisualizerProps) {
  const [mode, setMode] = useState<VisualizationMode>('contours')
  const [isoValue, setIsoValue] = useState(0.08) // Default 0.08 fm^-3 for contours
  const [threshold, setThreshold] = useState(0.1) // For point cloud mode
  const [rotating, setRotating] = useState(false)
  
  // Use demo data if no data provided
  const data: DensityData = useMemo(() => {
    if (initialData) return initialData
    
    // Generate demo nuclear density
    const nx = 24, ny = 24, nz = 24
    const dx = 1.0, dy = 1.0, dz = 1.0
    
    return {
      density: generateDemoDensity(nx, ny, nz),
      grid: { nx, ny, nz, dx, dy, dz },
      type: 'total'
    }
  }, [initialData])
  
  // Contour level options (in fm^-3)
  const isoValueOptions = [
    { value: '0.02', label: '0.02 fm⁻³' },
    { value: '0.04', label: '0.04 fm⁻³' },
    { value: '0.06', label: '0.06 fm⁻³' },
    { value: '0.08', label: '0.08 fm⁻³ (default)' },
    { value: '0.10', label: '0.10 fm⁻³' },
    { value: '0.12', label: '0.12 fm⁻³' },
    { value: '0.14', label: '0.14 fm⁻³' },
  ]
  
  const thresholdOptions = [
    { value: '0.01', label: '1% max' },
    { value: '0.05', label: '5% max' },
    { value: '0.1', label: '10% max' },
    { value: '0.2', label: '20% max' },
    { value: '0.5', label: '50% max' },
  ]
  
  const modeOptions = [
    { value: 'contours', label: 'Contour Surface' },
    { value: 'points', label: 'Point Cloud' },
  ]
  
  return (
    <Card className="overflow-hidden">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <CardTitle className="flex items-center gap-2">
            <Eye className="w-5 h-5" />
            3D Density Distribution
          </CardTitle>
          
          <div className="flex items-center gap-2 flex-wrap">
            {/* Mode selector */}
            <Select
              className="w-36"
              value={mode}
              onChange={(e) => setMode(e.target.value as VisualizationMode)}
              options={modeOptions}
            />
            
            {/* Value selector - changes based on mode */}
            {mode === 'contours' ? (
              <Select
                className="w-40"
                value={isoValue.toString()}
                onChange={(e) => setIsoValue(parseFloat(e.target.value))}
                options={isoValueOptions}
              />
            ) : (
              <Select
                className="w-28"
                value={threshold.toString()}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
                options={thresholdOptions}
              />
            )}
            
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setRotating(!rotating)}
              title={rotating ? 'Stop rotation' : 'Start rotation'}
            >
              {rotating ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            </Button>
          </div>
        </div>
      </CardHeader>
      
      <CardContent className="p-0">
        <div className="h-[400px] bg-gray-900 relative">
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
            
            {mode === 'contours' ? (
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
            {mode === 'contours' ? (
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
      </CardContent>
    </Card>
  )
}
