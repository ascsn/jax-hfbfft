"""
WebSocket message models for type-safe communication.

This module defines Pydantic models for all WebSocket messages
to ensure type safety and schema validation.
"""

from typing import Union, Literal
from pydantic import BaseModel, Field

from jax_hfbfft.gui.models import CalculationProgress, CalculationStatus


#  Client -> Server Messages

class SubscribeMessage(BaseModel):
    """Subscribe to calculation progress updates."""
    type: Literal["subscribe"] = "subscribe"
    calculation_id: str = Field(..., description="Calculation ID to subscribe to")


class UnsubscribeMessage(BaseModel):
    """Unsubscribe from calculation updates."""
    type: Literal["unsubscribe"] = "unsubscribe"
    calculation_id: str


class GetWarmupStatusMessage(BaseModel):
    """Request current warmup status."""
    type: Literal["get_warmup_status"] = "get_warmup_status"


class PingMessage(BaseModel):
    """Ping message for connection keepalive."""
    type: Literal["ping"] = "ping"


ClientMessage = Union[
    SubscribeMessage,
    UnsubscribeMessage,
    GetWarmupStatusMessage,
    PingMessage,
]


# Server -> Client Messages

class ProgressMessage(BaseModel):
    """Progress update for a calculation."""
    type: Literal["progress"] = "progress"
    data: CalculationProgress


class StatusMessage(BaseModel):
    """Current status of a calculation."""
    type: Literal["status"] = "status"
    data: CalculationStatus


class SubscribedMessage(BaseModel):
    """Confirmation of subscription."""
    type: Literal["subscribed"] = "subscribed"
    calculation_id: str


class UnsubscribedMessage(BaseModel):
    """Confirmation of unsubscription."""
    type: Literal["unsubscribed"] = "unsubscribed"
    calculation_id: str


class WarmupStatusMessage(BaseModel):
    """JAX warmup/compilation status."""
    type: Literal["warmup_status"] = "warmup_status"
    data: dict = Field(..., description="Warmup status dict with is_ready and message")


class PongMessage(BaseModel):
    """Pong response to ping."""
    type: Literal["pong"] = "pong"


class ErrorMessage(BaseModel):
    """Error message."""
    type: Literal["error"] = "error"
    message: str = Field(..., description="Error description")
    code: str | None = Field(None, description="Error code")


ServerMessage = Union[
    ProgressMessage,
    StatusMessage,
    SubscribedMessage,
    UnsubscribedMessage,
    WarmupStatusMessage,
    PongMessage,
    ErrorMessage,
]


# Event types for frontend
class WebSocketEvent:
    """WebSocket event type constants."""
    
    # Client events
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    GET_WARMUP_STATUS = "get_warmup_status"
    PING = "ping"
    
    # Server events
    PROGRESS = "progress"
    STATUS = "status"
    SUBSCRIBED = "subscribed"
    UNSUBSCRIBED = "unsubscribed"
    WARMUP_STATUS = "warmup_status"
    PONG = "pong"
    ERROR = "error"
