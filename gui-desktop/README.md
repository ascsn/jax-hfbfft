# HFBFFT Desktop Application

Electron wrapper for the HFBFFT web GUI, providing a native desktop experience.

## Features

- Native desktop window with system integration
- Automatic backend startup
- Application menu with keyboard shortcuts
- Cross-platform support (Linux, macOS, Windows)

## Development

### Prerequisites

- Node.js 18+
- Python with jax-hfbfft installed

### Running in Development Mode

1. Start the Python backend separately:
   ```bash
   hfbfft gui --no-browser
   ```

2. Run Electron in dev mode:
   ```bash
   cd gui-desktop
   npm install
   npm run dev
   ```

### Building for Distribution

```bash
# Install dependencies
npm install

# Build for current platform
npm run build

# Build for specific platform
npm run build:linux
npm run build:mac
npm run build:win
```

Built packages will be in the `release/` directory.

## Architecture

```
gui-desktop/
├── main.js        # Electron main process
├── preload.js     # Preload script for secure IPC
├── package.json   # Dependencies and build config
└── icons/         # Application icons
    ├── icon.png   # Linux/generic
    ├── icon.icns  # macOS
    └── icon.ico   # Windows
```

The desktop app embeds the web GUI by:
1. Starting the Python FastAPI backend as a child process
2. Opening a BrowserWindow pointed at localhost
3. Managing the backend lifecycle

## Configuration

The following environment variables can be used:

- `HFBFFT_PORT`: Backend port (default: 8765)
- `HFBFFT_PYTHON`: Python executable path (default: `python3`)
