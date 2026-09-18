"""Engine interface.

Every prediction engine (CRT, Labouchere, Trig/Euler, Combination, and later
the fruit-fly connectome) implements this interface. The orchestrator treats
them interchangeably.

An engine receives an EngineContext with the current round and recent price
history, and returns an EngineOutput.

Engines are allowed to return NO_SIGNAL. In fact, NO_SIGNAL should be the
default output for any engine that isn't confident.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class EngineContext:
    session_id: str
    round_id: str
    round_start_timestamp: datetime
    trade_cutoff_timestamp: datetime
    price_start_timestamp: Optional[datetime]
    price_end_timestamp: Optional[datetime]
    current_time: datetime
    current_price: float
    # Rolling history of (timestamp_seconds_float, price), oldest first
    recent_ticks: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class EngineOutput:
    engine: str
    direction: str  # "UP" | "DOWN" | "NO_SIGNAL"
    confidence: Optional[float] = None
    pattern_signature: Optional[str] = None
    raw_state: Optional[Dict[str, Any]] = None

    def validate(self) -> None:
        if self.direction not in ("UP", "DOWN", "NO_SIGNAL"):
            raise ValueError(f"invalid direction: {self.direction}")
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence out of range: {self.confidence}")


class Engine:
    """Abstract engine."""

    name: str = "base"

    def run(self, ctx: EngineContext) -> EngineOutput:
        raise NotImplementedError
