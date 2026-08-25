"""
FastAPI server for HFBFFT GUI.

This module provides the main FastAPI application and server startup
for the web-based GUI.
"""

import asyncio
import webbrowser
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from jax_hfbfft.gui.routes import calculations, history, websocket, system, surfaces
from jax_hfbfft.gui.services.warmup import get_warmup_manager
from jax_hfbfft.gui.services.storage import get_storage


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    print("Starting HFBFFT GUI server...")
    
    # Initialize storage
    storage = await get_storage()
    
    # Start JIT warmup in background (don't block startup)
    warmup_manager = get_warmup_manager()
    warmup_manager.start_background_warmup()
    
    yield
    
    # Shutdown
    print("Shutting down HFBFFT GUI server...")
    await storage.close()


def create_app(
    title: str = "HFBFFT GUI",
    debug: bool = False,
) -> FastAPI:
    """
    Create the FastAPI application.
    
    Args:
        title: Application title.
        debug: Enable debug mode.
        
    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(
        title=title,
        description="Web-based GUI for Hartree-Fock-Bogoliubov calculations",
        version="0.1.0",
        lifespan=lifespan,
        debug=debug,
    )
    
    # CORS middleware for development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, restrict this
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API routes
    app.include_router(calculations.router, prefix="/api", tags=["calculations"])
    app.include_router(history.router, prefix="/api", tags=["history"])
    app.include_router(system.router, prefix="/api", tags=["system"])
    app.include_router(surfaces.router, prefix="/api", tags=["surfaces"])
    app.include_router(websocket.router, tags=["websocket"])
    
    # Serve static frontend files
    frontend_dir = Path(__file__).parent / "frontend"
    assets_dir = frontend_dir / "assets"
    index_path = frontend_dir / "index.html"
    
    # Only mount static files if frontend is actually built
    if assets_dir.exists() and index_path.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")
        
        @app.get("/")
        async def serve_index():
            """Serve the frontend index.html."""
            return FileResponse(str(index_path))
        
        @app.get("/{path:path}")
        async def serve_spa(path: str):
            """Serve SPA routes (fallback to index.html).
            
            Note: API routes (/api/*) are handled by their routers,
            not this catch-all.
            """
            # Don't catch API routes - they should 404 if not found
            if path.startswith("api/") or path.startswith("ws") or path == "docs" or path == "openapi.json":
                from fastapi import HTTPException
                raise HTTPException(status_code=404, detail="Not found")
            
            file_path = frontend_dir / path
            if file_path.exists() and file_path.is_file():
                return FileResponse(str(file_path))
            return FileResponse(str(index_path))
    else:
        @app.get("/")
        async def no_frontend():
            """Placeholder when frontend is not built."""
            return {
                "message": "HFBFFT GUI API is running",
                "frontend": "Not built. Run: cd gui-frontend && npm install && npm run build:copy",
                "api_docs": "/docs",
                "note": "Visit /docs to explore the API directly",
            }
    
    return app


def start_server(
    host: str = "127.0.0.1",
    port: int = 8080,
    open_browser: bool = True,
    debug: bool = False,
):
    """
    Start the GUI server.
    
    Args:
        host: Host to bind to.
        port: Port to bind to.
        open_browser: Whether to open browser automatically.
        debug: Enable debug mode.
    """
    import uvicorn
    
    app = create_app(debug=debug)
    
    if open_browser:
        # Open browser after a short delay
        async def open_browser_delayed():
            await asyncio.sleep(1.5)
            url = f"http://{host}:{port}"
            print(f"\nOpening browser at {url}")
            webbrowser.open(url)
        
        @app.on_event("startup")
        async def startup_open_browser():
            asyncio.create_task(open_browser_delayed())
    
    print(f"\nStarting HFBFFT GUI server at http://{host}:{port}")
    print("API documentation available at /docs")
    print("\nPress Ctrl+C to stop the server\n")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info" if debug else "warning",
    )


def get_app() -> FastAPI:
    """Get or create the FastAPI app (lazy initialization)."""
    global _app
    if _app is None:
        _app = create_app()
    return _app


_app: Optional[FastAPI] = None


# Factory function for uvicorn (avoids import-time app creation)
def app_factory() -> FastAPI:
    """Factory function for creating the app (used by uvicorn)."""
    return create_app()
