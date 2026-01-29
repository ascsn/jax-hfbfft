# Frontend Build Directory

This directory contains the built React frontend.

## Building the Frontend

From the project root:

```bash
cd gui-frontend
npm install
npm run build
```

The build script will copy files here automatically using:

```bash
npm run build:copy
```

## Manual Copy

If the automatic copy fails, manually copy:

```bash
cp -r gui-frontend/dist/* src/jax_hfbfft/gui/frontend/
```

## Development

For development with hot reload:

1. Start the backend:
   ```bash
   hfbfft gui --no-browser
   ```

2. Start Vite dev server:
   ```bash
   cd gui-frontend
   npm run dev
   ```

3. Open http://localhost:5173 (Vite will proxy API calls to the backend)
