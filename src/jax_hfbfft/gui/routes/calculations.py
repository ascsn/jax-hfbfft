"""
Calculation API routes.

This module provides REST API endpoints for starting, monitoring,
and cancelling HFB calculations.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Optional

from jax_hfbfft.gui.models import (
    CalculationRequest,
    CalculationStatus,
    CalculationSummary,
)
from jax_hfbfft.gui.services.hfb_service import get_hfb_service
from jax_hfbfft.gui.services.storage import get_storage

router = APIRouter()


@router.post("/calculations", response_model=dict)
async def start_calculation(
    request: CalculationRequest,
    background_tasks: BackgroundTasks,
):
    """
    Start a new HFB calculation.
    
    This endpoint queues a new calculation and returns immediately
    with a calculation ID. Use the WebSocket or polling to track progress.
    
    Args:
        request: Calculation parameters.
        
    Returns:
        Dictionary with calculation ID.
    """
    service = get_hfb_service()
    
    # Define callback to save completed calculations
    async def save_on_complete(progress):
        if progress.phase.value in ("converged", "failed", "cancelled"):
            status = await service.get_calculation(progress.calculation_id)
            if status:
                storage = await get_storage()
                await storage.save_calculation(status)
    
    calc_id = await service.start_calculation(request, progress_callback=save_on_complete)
    
    return {
        "calculation_id": calc_id,
        "message": "Calculation started",
        "status_url": f"/api/calculations/{calc_id}",
    }


@router.get("/calculations", response_model=List[CalculationStatus])
async def list_calculations():
    """
    List all active calculations.
    
    Returns only currently running or recently completed calculations.
    Use /api/history for historical calculations.
    """
    service = get_hfb_service()
    statuses = await service.list_calculations()
    
    return statuses


@router.get("/calculations/{calc_id}", response_model=CalculationStatus)
async def get_calculation(calc_id: str):
    """
    Get the status and results of a calculation.
    
    Args:
        calc_id: The calculation ID.
        
    Returns:
        Full calculation status including results if complete.
    """
    service = get_hfb_service()
    status = await service.get_calculation(calc_id)
    
    if not status:
        # Check historical storage
        storage = await get_storage()
        status = await storage.get_calculation(calc_id)
    
    if not status:
        raise HTTPException(status_code=404, detail="Calculation not found")
    
    return status


@router.delete("/calculations/{calc_id}")
async def cancel_calculation(calc_id: str):
    """
    Cancel a running calculation.
    
    Args:
        calc_id: The calculation ID.
        
    Returns:
        Confirmation message.
    """
    service = get_hfb_service()
    success = await service.cancel_calculation(calc_id)
    
    if not success:
        raise HTTPException(
            status_code=404,
            detail="Calculation not found or already completed"
        )
    
    return {"message": "Calculation cancelled", "calculation_id": calc_id}


@router.get("/calculations/{calc_id}/densities")
async def get_densities(
    calc_id: str,
    type: str = "total",
    downsample: Optional[int] = None
):
    """
    Get density data for 3D visualization.
    
    Args:
        calc_id: The calculation ID.
        type: Which density to return: "neutron", "proton", "total",
              "tau_n", "tau_p", "tau_total"
        downsample: Optional grid size to downsample to (e.g., 16 for faster transfer)
        
    Returns:
        3D density array data for visualization.
    """
    service = get_hfb_service()
    
    # Try to get density data
    density_data = await service.get_density_data(calc_id, type, downsample)
    
    if density_data is None:
        # Check if calculation exists
        status = await service.get_calculation(calc_id)
        if not status:
            raise HTTPException(status_code=404, detail="Calculation not found")
        if not status.results:
            raise HTTPException(status_code=400, detail="Calculation not complete")
        raise HTTPException(
            status_code=404,
            detail="Density data not available (calculation may have been cleared)"
        )
    
    return density_data
