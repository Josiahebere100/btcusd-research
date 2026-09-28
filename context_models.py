"""Pydantic models for collector payloads."""
from typing import List, Optional

from pydantic import BaseModel, Field


# ---- Tick -----------------------------------------------------------------


class Tick(BaseModel):
    source: str
    symbol: str
    stream: Optional[str] = None
    price: float
    tick_timestamp: str
    received_at: str
    latency_ms: Optional[int] = None
    session_id: str
    mode: Optional[str] = "LIVE"
    raw_change: Optional[str] = None


class TickBatch(BaseModel):
    ticks: List[Tick] = Field(default_factory=list)


# ---- Round ----------------------------------------------------------------


class PreviousRound(BaseModel):
    round_id: str
    start_price: Optional[float] = None
    end_price: Optional[float] = None
    win_side: Optional[int] = None


class Round(BaseModel):
    source: str
    symbol: str
    stream: Optional[str] = None
    round_id: str
    round_start_timestamp: Optional[str] = None
    trade_cutoff_timestamp: Optional[str] = None
    price_start_timestamp: Optional[str] = None
    price_end_timestamp: Optional[str] = None
    start_price: Optional[float] = None
    end_price: Optional[float] = None
    win_side: Optional[int] = None
    raw_direction: Optional[str] = None
    status_code: Optional[int] = None
    status_change_at: Optional[str] = None
    current_time: Optional[str] = None
    session_id: str
    mode: Optional[str] = "LIVE"
    previous_rounds: List[PreviousRound] = Field(default_factory=list)


class RoundBatch(BaseModel):
    rounds: List[Round] = Field(default_factory=list)


# ---- Session --------------------------------------------------------------


class Session(BaseModel):
    session_id: str
    start_timestamp: str
    end_timestamp: Optional[str] = None
    gap_before_ms: Optional[int] = None
    status: str
    reason: Optional[str] = None


class SessionBatch(BaseModel):
    sessions: List[Session] = Field(default_factory=list)


# ---- Health beacon (from collector) ---------------------------------------


class HealthBeacon(BaseModel):
    collector_started_at: Optional[str] = None
    collector_uptime_ms: Optional[int] = None
    ws_state: Optional[str] = None
    last_tick_at: Optional[str] = None
    last_round_at: Optional[str] = None
    last_tick_price: Optional[float] = None
    last_tick_latency_ms: Optional[int] = None
    last_round_id: Optional[str] = None
    ticks_received: Optional[int] = 0
    rounds_received: Optional[int] = 0
    ticks_rejected: Optional[int] = 0
    rounds_rejected: Optional[int] = 0
    ticks_forwarded: Optional[int] = 0
    rounds_forwarded: Optional[int] = 0
    ticks_forward_failed: Optional[int] = 0
    rounds_forward_failed: Optional[int] = 0
    reconnect_count: Optional[int] = 0
    disconnect_count: Optional[int] = 0
    last_disconnect_reason: Optional[str] = None
    current_session_id: Optional[str] = None
    mode: Optional[str] = "LIVE"
