"""
History API routes.

This module provides REST API endpoints for querying and managing
calculation history.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from datetime import datetime

from jax_hfbfft.gui.models import (
    CalculationStatus,
    CalculationResults,
    CalculationPhase,
    HistoryFilter,
    HistoryResponse,
)
from jax_hfbfft.gui.services.storage import get_storage

router = APIRouter()


@router.get("/history", response_model=HistoryResponse)
async def get_history(
    nucleus: Optional[str] = Query(None, description="Filter by nucleus (e.g., 'O-16', 'Ca40')"),
    element: Optional[str] = Query(None, description="Filter by element symbol"),
    min_a: Optional[int] = Query(None, description="Minimum mass number"),
    max_a: Optional[int] = Query(None, description="Maximum mass number"),
    force_name: Optional[str] = Query(None, description="Filter by force name"),
    force: Optional[str] = Query(None, description="Alias for force_name"),
    status: Optional[str] = Query(None, description="Filter by status"),
    from_date: Optional[datetime] = Query(None, description="From date"),
    to_date: Optional[datetime] = Query(None, description="To date"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Results per page"),
    sort_by: str = Query("started_at", description="Sort column"),
    sort_desc: bool = Query(True, description="Sort descending"),
):
    """
    Get calculation history with filtering and pagination.
    
    Returns a paginated list of historical calculations with optional
    filtering by nucleus, force, status, and date range.
    """
    storage = await get_storage()
    
    # Parse nucleus string if provided (e.g., "O-16", "Ca40", "O16")
    parsed_element = element
    parsed_a = None
    if nucleus:
        import re
        # Match patterns like "O-16", "Ca-40", "O16", "Ca40"
        match = re.match(r'^([A-Za-z]+)-?(\d+)$', nucleus.strip())
        if match:
            parsed_element = match.group(1).capitalize()
            parsed_a = int(match.group(2))
    
    # Use force alias if force_name not provided
    effective_force = force_name or force
    
    # Build filter
    filter_obj = None
    if any([parsed_element, min_a, max_a, parsed_a, effective_force, status, from_date, to_date]):
        phase = None
        if status:
            try:
                phase = CalculationPhase(status)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid status: {status}"
                )
        
        # If we parsed a specific mass number, use it as both min and max
        eff_min_a = parsed_a if parsed_a else min_a
        eff_max_a = parsed_a if parsed_a else max_a
        
        filter_obj = HistoryFilter(
            element=parsed_element,
            min_a=eff_min_a,
            max_a=eff_max_a,
            force_name=effective_force,
            status=phase,
            from_date=from_date,
            to_date=to_date,
        )
    
    return await storage.get_history(
        filter=filter_obj,
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        sort_desc=sort_desc,
    )


@router.get("/history/{calc_id}", response_model=CalculationStatus)
async def get_historical_calculation(calc_id: str):
    """
    Get a historical calculation by ID.
    
    Args:
        calc_id: The calculation ID.
        
    Returns:
        Full calculation status and results.
    """
    storage = await get_storage()
    status = await storage.get_calculation(calc_id)
    
    if not status:
        raise HTTPException(status_code=404, detail="Calculation not found in history")
    
    return status


@router.get("/history/{calc_id}/results", response_model=CalculationResults)
async def get_historical_results(calc_id: str):
    """
    Get the results of a historical calculation.
    
    Args:
        calc_id: The calculation ID.
        
    Returns:
        Calculation results.
    """
    storage = await get_storage()
    results = await storage.get_results(calc_id)
    
    if not results:
        raise HTTPException(
            status_code=404,
            detail="Results not found for this calculation"
        )
    
    return results


@router.delete("/history/{calc_id}")
async def delete_from_history(calc_id: str):
    """
    Delete a calculation from history.
    
    Args:
        calc_id: The calculation ID.
        
    Returns:
        Confirmation message.
    """
    storage = await get_storage()
    success = await storage.delete_calculation(calc_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Calculation not found in history")
    
    return {"message": "Calculation deleted from history", "calculation_id": calc_id}


@router.get("/history/stats")
async def get_history_stats():
    """
    Get summary statistics about calculation history.
    
    Returns counts by status, common nuclei, etc.
    """
    storage = await get_storage()
    await storage.initialize()
    
    # Query for stats
    stats = {
        "total_calculations": 0,
        "by_status": {},
        "by_element": {},
        "by_force": {},
    }
    
    # Get counts by status
    async with storage._db.execute(
        "SELECT phase, COUNT(*) as count FROM calculations GROUP BY phase"
    ) as cursor:
        async for row in cursor:
            stats["by_status"][row["phase"]] = row["count"]
            stats["total_calculations"] += row["count"]
    
    # Get counts by element (top 10)
    async with storage._db.execute(
        """SELECT element_symbol, COUNT(*) as count 
           FROM calculations 
           GROUP BY element_symbol 
           ORDER BY count DESC 
           LIMIT 10"""
    ) as cursor:
        async for row in cursor:
            stats["by_element"][row["element_symbol"]] = row["count"]
    
    # Get counts by force
    async with storage._db.execute(
        "SELECT force_name, COUNT(*) as count FROM calculations GROUP BY force_name"
    ) as cursor:
        async for row in cursor:
            stats["by_force"][row["force_name"]] = row["count"]
    
    return stats
