/**
 * Preload script for Electron
 * 
 * This runs in the renderer process context but has access to Node.js APIs.
 * Used to expose safe APIs to the web content.
 */

const { contextBridge, ipcRenderer } = require('electron')

// Expose protected methods to the renderer process
contextBridge.exposeInMainWorld('electronAPI', {
  // Platform info
  platform: process.platform,
  isElectron: true,
  
  // Window controls
  minimize: () => ipcRenderer.send('window:minimize'),
  maximize: () => ipcRenderer.send('window:maximize'),
  close: () => ipcRenderer.send('window:close'),
  
  // File dialogs
  showSaveDialog: (options) => ipcRenderer.invoke('dialog:save', options),
  showOpenDialog: (options) => ipcRenderer.invoke('dialog:open', options),
  
  // App info
  getVersion: () => ipcRenderer.invoke('app:version'),
})

console.log('HFBFFT Desktop preload script loaded')
