"""In-memory state tracker for the feed.

Populated as data arrives; used by GET /api/collector/health to
determine whether the feed is VERIFIED, DEGRADED, UNVERIFIED, or OFFLINE.

Single-instance assumption. If we scale to multiple instances later,
this state should move to Redis.
"""
import time
from dataclasses import dataclass, field
from typing import Optional

from .config import (
    FEED_MAX_LATENCY_MS,
    FEED_ROUND_MAX_AGE_MS,
    FEED_TICK_MAX_AGE_MS,
)


@dataclass
class FeedState:
    # Last received
    last_tick_at: Optional[float] = None          # unix seconds
    last_round_at: Optional[float] = None
    last_tick_timestamp: Optional[str] = None     # ISO from source
    last_round_id: Optional[str] = None
    last_tick_price: Optional[float] = None
    last_tick_latency_ms: Optional[int] = None
    last_tick_source: Optional[str] = None
    last_tick_symbol: Optional[str] = None
    last_round_source: Optional[str] = None
    last_round_symbol: Optional[str] = None
    current_session_id: Optional[str] = None

    # Counters
    ticks_received: int = 0
    rounds_received: int = 0
    ticks_rejected: int = 0
    rounds_rejected: int = 0
    sessions_received: int = 0

    # Collector beacon
    last_beacon_at: Optional[float] = None
    last_beacon: Optional[dict] = field(default_factory=None)


_state = FeedState()


def get_state() -> FeedState:
    return _state


def note_tick(tick: dict) -> None:
    now = time.time()
    _state.last_tick_at = now
    _state.last_tick_timestamp = tick.get("tick_timestamp")
    _state.last_tick_price = tick.get("price")
    _state.last_tick_latency_ms = tick.get("latency_ms")
    _state.last_tick_source = tick.get("source")
    _state.last_tick_symbol = tick.get("symbol")
    _state.current_session_id = tick.get("session_id")
    _state.ticks_received += 1


def note_tick_rejected() -> None:
    _state.ticks_rejected += 1


def note_round(round_obj: dict) -> None:
    now = time.time()
    _state.last_round_at = now
    _state.last_round_id = round_obj.get("round_id")
    _state.last_round_source = round_obj.get("source")
    _state.last_round_symbol = round_obj.get("symbol")
    _state.rounds_received += 1


def note_round_rejected() -> None:
    _state.rounds_rejected += 1


def note_session() -> None:
    _state.sessions_received += 1


def note_beacon(beacon: dict) -> None:
    _state.last_beacon_at = time.time()
    _state.last_beacon = beacon


# ---- Feed verification ----------------------------------------------------


def compute_feed_status() -> dict:
    """Return the current feed status and the reason for it.

    Status values:
      VERIFIED   — real recent tick AND real recent round, both validated
      DEGRADED   — data flowing but one or more conditions failing
      UNVERIFIED — no data yet, or conditions never satisfied
      OFFLINE    — no data received for a long time
    """
    now = time.time()
    reasons = []

    # Age checks
    tick_age_ms = (
        int((now - _state.last_tick_at) * 1000)
        if _state.last_tick_at else None
    )
    round_age_ms = (
        int((now - _state.last_round_at) * 1000)
        if _state.last_round_at else None
    )

    if tick_age_ms is None:
        reasons.append("no_tick_received")
    elif tick_age_ms > FEED_TICK_MAX_AGE_MS:
        reasons.append(f"tick_stale:{tick_age_ms}ms")

    if round_age_ms is None:
        reasons.append("no_round_received")
    elif round_age_ms > FEED_ROUND_MAX_AGE_MS:
        reasons.append(f"round_stale:{round_age_ms}ms")

    # Source/symbol validation
    if _state.last_tick_source and _state.last_tick_source != "BC.GAME":
        reasons.append(f"wrong_tick_source:{_state.last_tick_source}")
    if _state.last_tick_symbol and _state.last_tick_symbol != "BTC-USD":
        reasons.append(f"wrong_tick_symbol:{_state.last_tick_symbol}")
    if _state.last_round_source and _state.last_round_source != "BC.GAME":
        reasons.append(f"wrong_round_source:{_state.last_round_source}")
    if _state.last_round_symbol and _state.last_round_symbol != "BTC/USD":
        reasons.append(f"wrong_round_symbol:{_state.last_round_symbol}")

    # Latency check
    if (
        _state.last_tick_latency_ms is not None
        and _state.last_tick_latency_ms > FEED_MAX_LATENCY_MS
    ):
        reasons.append(f"high_latency:{_state.last_tick_latency_ms}ms")

    # Session check
    if not _state.current_session_id:
        reasons.append("no_session")

    # Determine status
    if tick_age_ms is None and round_age_ms is None:
        status = "UNVERIFIED"
    elif tick_age_ms is not None and tick_age_ms > 60_000:
        status = "OFFLINE"
    elif reasons:
        status = "DEGRADED"
    else:
        status = "VERIFIED"

    return {
        "status": status,
        "reason": ",".join(reasons) if reasons else "ok",
        "tick_age_ms": tick_age_ms,
        "round_age_ms": round_age_ms,
    }
