/**
 * HFBFFT Desktop - Electron Main Process
 * 
 * This wraps the web GUI in a native desktop window.
 * The Python backend is started as a child process.
 */

const { app, BrowserWindow, Menu, shell, dialog } = require('electron')
const path = require('path')
const { spawn } = require('child_process')
const http = require('http')

// Configuration
const DEV_MODE = process.argv.includes('--dev')
const DEFAULT_PORT = 8765
const BACKEND_START_TIMEOUT = 30000  // 30 seconds

let mainWindow = null
let backendProcess = null
let backendPort = DEFAULT_PORT

/**
 * Wait for the backend server to become available
 */
function waitForBackend(port, timeout = BACKEND_START_TIMEOUT) {
  return new Promise((resolve, reject) => {
    const startTime = Date.now()
    
    const checkServer = () => {
      const req = http.get(`http://localhost:${port}/api/status`, (res) => {
        if (res.statusCode === 200) {
          resolve(true)
        } else {
          retry()
        }
      })
      
      req.on('error', () => retry())
      req.setTimeout(1000, () => {
        req.destroy()
        retry()
      })
    }
    
    const retry = () => {
      if (Date.now() - startTime > timeout) {
        reject(new Error('Backend startup timeout'))
      } else {
        setTimeout(checkServer, 500)
      }
    }
    
    checkServer()
  })
}

/**
 * Start the Python backend server
 */
async function startBackend() {
  if (DEV_MODE) {
    console.log('Dev mode: expecting backend at http://localhost:' + backendPort)
    return
  }
  
  // Find Python executable
  const pythonPath = process.platform === 'win32' 
    ? 'python' 
    : 'python3'
  
  // Start the backend
  console.log('Starting HFBFFT backend...')
  
  backendProcess = spawn(pythonPath, [
    '-m', 'jax_hfbfft.gui.server',
    '--port', backendPort.toString(),
    '--host', 'localhost',
    '--no-browser'
  ], {
    env: { ...process.env },
    stdio: ['ignore', 'pipe', 'pipe']
  })
  
  backendProcess.stdout.on('data', (data) => {
    console.log(`Backend: ${data}`)
  })
  
  backendProcess.stderr.on('data', (data) => {
    console.error(`Backend error: ${data}`)
  })
  
  backendProcess.on('exit', (code) => {
    console.log(`Backend exited with code ${code}`)
    if (mainWindow && !mainWindow.isDestroyed()) {
      dialog.showErrorBox(
        'Backend Error',
        'The calculation backend has stopped unexpectedly.'
      )
    }
  })
  
  // Wait for backend to be ready
  try {
    await waitForBackend(backendPort)
    console.log('Backend is ready')
  } catch (error) {
    console.error('Failed to start backend:', error)
    throw error
  }
}

/**
 * Create the main application window
 */
function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1000,
    minHeight: 700,
    title: 'HFBFFT - Nuclear Structure Calculator',
    icon: path.join(__dirname, 'icons', 'icon.png'),
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js')
    },
    show: false  // Don't show until ready
  })
  
  // Load the app
  const url = `http://localhost:${backendPort}`
  mainWindow.loadURL(url)
  
  // Show when ready
  mainWindow.once('ready-to-show', () => {
    mainWindow.show()
  })
  
  // Handle external links
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith('http://localhost') || url.startsWith('file://')) {
      return { action: 'allow' }
    }
    shell.openExternal(url)
    return { action: 'deny' }
  })
  
  // Clean up on close
  mainWindow.on('closed', () => {
    mainWindow = null
  })
  
  // Create menu
  createMenu()
}

/**
 * Create application menu
 */
function createMenu() {
  const template = [
    {
      label: 'File',
      submenu: [
        {
          label: 'New Calculation',
          accelerator: 'CmdOrCtrl+N',
          click: () => {
            mainWindow.loadURL(`http://localhost:${backendPort}/`)
          }
        },
        { type: 'separator' },
        { role: 'quit' }
      ]
    },
    {
      label: 'Edit',
      submenu: [
        { role: 'undo' },
        { role: 'redo' },
        { type: 'separator' },
        { role: 'cut' },
        { role: 'copy' },
        { role: 'paste' }
      ]
    },
    {
      label: 'View',
      submenu: [
        { role: 'reload' },
        { role: 'forceReload' },
        { role: 'toggleDevTools' },
        { type: 'separator' },
        { role: 'resetZoom' },
        { role: 'zoomIn' },
        { role: 'zoomOut' },
        { type: 'separator' },
        { role: 'togglefullscreen' }
      ]
    },
    {
      label: 'Go',
      submenu: [
        {
          label: 'Calculator',
          click: () => mainWindow.loadURL(`http://localhost:${backendPort}/`)
        },
        {
          label: 'History',
          click: () => mainWindow.loadURL(`http://localhost:${backendPort}/history`)
        }
      ]
    },
    {
      label: 'Help',
      submenu: [
        {
          label: 'Documentation',
          click: () => shell.openExternal('https://github.com/ascsn/jax-hfbfft')
        },
        {
          label: 'About HFBFFT',
          click: () => {
            dialog.showMessageBox(mainWindow, {
              type: 'info',
              title: 'About HFBFFT',
              message: 'HFBFFT',
              detail: 'Hartree-Fock-Bogoliubov Nuclear Structure Calculator\n\n' +
                      'Powered by JAX for GPU-accelerated calculations.\n\n' +
                      'Version 0.1.0'
            })
          }
        }
      ]
    }
  ]
  
  const menu = Menu.buildFromTemplate(template)
  Menu.setApplicationMenu(menu)
}

/**
 * Stop the backend process
 */
function stopBackend() {
  if (backendProcess) {
    console.log('Stopping backend...')
    backendProcess.kill('SIGTERM')
    backendProcess = null
  }
}

// App event handlers
app.whenReady().then(async () => {
  try {
    await startBackend()
    createWindow()
  } catch (error) {
    dialog.showErrorBox(
      'Startup Error',
      `Failed to start HFBFFT: ${error.message}`
    )
    app.quit()
  }
  
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow()
    }
  })
})

app.on('window-all-closed', () => {
  stopBackend()
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

app.on('before-quit', () => {
  stopBackend()
})
