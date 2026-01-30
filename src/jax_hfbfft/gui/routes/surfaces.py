"""
Surface scan API routes.

Endpoints for constrained surface calculations (e.g., beta deformation scans).
"""

from fastapi import APIRouter, HTTPException

from jax_hfbfft.gui.models import BetaSurfaceRequest, BetaSurfaceResult
from jax_hfbfft.gui.services.hfb_service import get_hfb_service

router = APIRouter()


@router.post("/surfaces/beta", response_model=BetaSurfaceResult)
async def beta_surface(request: BetaSurfaceRequest):
    """Run a 1D beta deformation surface scan (blocking)."""
    service = get_hfb_service()
    try:
        result = await service.run_beta_surface(request)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result


@router.post("/surfaces/beta/start", response_model=dict)
async def start_beta_surface(request: BetaSurfaceRequest):
    """Start a 1D beta deformation surface scan (async)."""
    service = get_hfb_service()
    try:
        calc_id = await service.start_beta_surface(request)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "calculation_id": calc_id,
        "message": "Surface scan started",
        "status_url": f"/api/calculations/{calc_id}",
    }
