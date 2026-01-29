"""
WebSocket routes for live updates.

This module provides WebSocket endpoints for streaming calculation
progress updates to connected clients.
"""

import asyncio
import json
from typing import Dict, Set
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from jax_hfbfft.gui.models import CalculationProgress, CalculationPhase
from jax_hfbfft.gui.services.hfb_service import get_hfb_service
from jax_hfbfft.gui.services.warmup import get_warmup_manager

router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections and subscriptions."""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.subscriptions: Dict[str, Set[WebSocket]] = {}  # calc_id -> connections
    
    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.add(websocket)
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        self.active_connections.discard(websocket)
        # Remove from all subscriptions
        for subscribers in self.subscriptions.values():
            subscribers.discard(websocket)
    
    def subscribe(self, websocket: WebSocket, calc_id: str):
        """Subscribe a connection to calculation updates."""
        if calc_id not in self.subscriptions:
            self.subscriptions[calc_id] = set()
        self.subscriptions[calc_id].add(websocket)
    
    def unsubscribe(self, websocket: WebSocket, calc_id: str):
        """Unsubscribe from calculation updates."""
        if calc_id in self.subscriptions:
            self.subscriptions[calc_id].discard(websocket)
    
    async def broadcast_progress(self, calc_id: str, progress: CalculationProgress):
        """Broadcast progress update to all subscribers."""
        if calc_id not in self.subscriptions:
            return
        
        message = {
            "type": "progress",
            "data": progress.model_dump(),
        }
        
        dead_connections = []
        for websocket in self.subscriptions[calc_id]:
            try:
                await websocket.send_json(message)
            except Exception:
                dead_connections.append(websocket)
        
        # Clean up dead connections
        for ws in dead_connections:
            self.disconnect(ws)
    
    async def send_to_all(self, message: dict):
        """Send a message to all connected clients."""
        dead_connections = []
        for websocket in self.active_connections:
            try:
                await websocket.send_json(message)
            except Exception:
                dead_connections.append(websocket)
        
        for ws in dead_connections:
            self.disconnect(ws)


# Global connection manager
manager = ConnectionManager()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Main WebSocket endpoint for GUI communication.
    
    Messages from client:
        {"type": "subscribe", "calculation_id": "uuid"}
        {"type": "unsubscribe", "calculation_id": "uuid"}
        {"type": "get_warmup_status"}
    
    Messages to client:
        {"type": "progress", "data": {...}}
        {"type": "warmup_status", "data": {...}}
        {"type": "error", "message": "..."}
    """
    await manager.connect(websocket)
    
    # Start background task to send warmup status updates
    warmup_task = asyncio.create_task(send_warmup_updates(websocket))
    
    try:
        while True:
            try:
                data = await websocket.receive_json()
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON message",
                })
                continue
            
            msg_type = data.get("type")
            
            if msg_type == "subscribe":
                calc_id = data.get("calculation_id")
                if calc_id:
                    manager.subscribe(websocket, calc_id)
                    
                    # Register callback for this calculation
                    service = get_hfb_service()
                    
                    async def progress_callback(progress: CalculationProgress):
                        await manager.broadcast_progress(progress.calculation_id, progress)
                    
                    await service.add_progress_callback(calc_id, progress_callback)
                    
                    # Send current status immediately
                    status = await service.get_calculation(calc_id)
                    if status:
                        await websocket.send_json({
                            "type": "status",
                            "data": status.model_dump(mode="json"),
                        })
                    
                    await websocket.send_json({
                        "type": "subscribed",
                        "calculation_id": calc_id,
                    })
            
            elif msg_type == "unsubscribe":
                calc_id = data.get("calculation_id")
                if calc_id:
                    manager.unsubscribe(websocket, calc_id)
                    await websocket.send_json({
                        "type": "unsubscribed",
                        "calculation_id": calc_id,
                    })
            
            elif msg_type == "get_warmup_status":
                warmup_manager = get_warmup_manager()
                await websocket.send_json({
                    "type": "warmup_status",
                    "data": warmup_manager.get_status(),
                })
            
            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})
            
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}",
                })
    
    except WebSocketDisconnect:
        pass
    finally:
        warmup_task.cancel()
        manager.disconnect(websocket)


async def send_warmup_updates(websocket: WebSocket):
    """Send periodic warmup status updates during warmup."""
    warmup_manager = get_warmup_manager()
    
    try:
        while not warmup_manager.is_ready:
            await asyncio.sleep(1.0)
            
            try:
                await websocket.send_json({
                    "type": "warmup_status",
                    "data": warmup_manager.get_status(),
                })
            except Exception:
                break
        
        # Send final status
        try:
            await websocket.send_json({
                "type": "warmup_status",
                "data": warmup_manager.get_status(),
            })
        except Exception:
            pass
    
    except asyncio.CancelledError:
        pass
