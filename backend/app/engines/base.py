"""Engine interface."""
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
    recent_ticks: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class EngineOutput:
    engine: str
    direction: str
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
    is_async: bool = False

    def run(self, ctx: EngineContext) -> EngineOutput:
        raise NotImplementedError
