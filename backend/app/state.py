"""In-memory state tracker for the feed."""
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

from .config import (
    FEED_MAX_LATENCY_MS,
    FEED_ROUND_MAX_AGE_MS,
    FEED_TICK_MAX_AGE_MS,
)

# Keep the last ~10 minutes of ticks in memory for engine consumption.
_TICK_HISTORY_MAX_AGE_S = 600


@dataclass
class FeedState:
    # Last received
    last_tick_at: Optional[float] = None
    last_round_at: Optional[float] = None
    last_tick_timestamp: Optional[str] = None
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
    predictions_created: int = 0
    outcomes_evaluated: int = 0

    # Round tracking for prediction triggering
    seen_round_ids: set = field(default_factory=set)  # (session_id, round_id) keys

    # Rolling tick history: (unix_seconds_float, price), oldest first
    recent_ticks: Deque[Tuple[float, float]] = field(
        default_factory=lambda: deque()
    )

    # Collector beacon
    last_beacon_at: Optional[float] = None
    last_beacon: Optional[dict] = None


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

    price = tick.get("price")
    if price is not None:
        _state.recent_ticks.append((now, float(price)))
        cutoff = now - _TICK_HISTORY_MAX_AGE_S
        while _state.recent_ticks and _state.recent_ticks[0][0] < cutoff:
            _state.recent_ticks.popleft()


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


def note_prediction_created() -> None:
    _state.predictions_created += 1


def note_outcome_evaluated() -> None:
    _state.outcomes_evaluated += 1


def note_beacon(beacon: dict) -> None:
    _state.last_beacon_at = time.time()
    _state.last_beacon = beacon


def recent_ticks_snapshot() -> List[Tuple[float, float]]:
    return list(_state.recent_ticks)


# ---- Feed verification ----------------------------------------------------


def compute_feed_status() -> dict:
    now = time.time()
    reasons = []

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

    if _state.last_tick_source and _state.last_tick_source != "BC.GAME":
        reasons.append(f"wrong_tick_source:{_state.last_tick_source}")
    if _state.last_tick_symbol and _state.last_tick_symbol != "BTC-USD":
        reasons.append(f"wrong_tick_symbol:{_state.last_tick_symbol}")
    if _state.last_round_source and _state.last_round_source != "BC.GAME":
        reasons.append(f"wrong_round_source:{_state.last_round_source}")
    if _state.last_round_symbol and _state.last_round_symbol != "BTC/USD":
        reasons.append(f"wrong_round_symbol:{_state.last_round_symbol}")

    if (
        _state.last_tick_latency_ms is not None
        and _state.last_tick_latency_ms > FEED_MAX_LATENCY_MS
    ):
        reasons.append(f"high_latency:{_state.last_tick_latency_ms}ms")

    if not _state.current_session_id:
        reasons.append("no_session")

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
