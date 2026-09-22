"""Trail Tracer engine.

Treats the recent tick path as a SHAPE and matches it against learned
shape families. Extracts turning points, segment slopes, overall trend,
and terrain context. Signature is categorical; direction is learned from
history via shared pattern memory.

Signature: trail:nt{n}_pe{first_type}_{trend}{last_move}_t{terrain}_dw{dwell}
State space ~300-400 states. Learnable within hours.
"""
import math
import os
from typing import Any, Dict, List, Optional, Tuple

from ._lookup import lookup_direction
from ._temporal import dwell_bin
from .base import Engine, EngineContext, EngineOutput


WINDOW_TICKS = 30
MIN_TICKS = 12
TURN_THRESHOLD_PCT = 0.00015
MAX_TURNS = 6
DWELL_MAX_BACK = 8

# Default active. Override via TRAIL_MODE env var if needed.
TRAIL_MODE = os.environ.get("TRAIL_MODE", "active").strip().lower()


def _normalize_path(
    ticks: List[Tuple[float, float]],
) -> Optional[Tuple[List[float], List[float]]]:
    if len(ticks) < MIN_TICKS:
        return None
    window = ticks[-WINDOW_TICKS:]
    times = [t for t, _ in window]
    prices = [p for _, p in window]
    lo = min(prices)
    hi = max(prices)
    span = hi - lo
    if span <= 0:
        return times, [0.5] * len(prices)
    return times, [(p - lo) / span for p in prices]


def _detect_turns(prices: List[float]) -> List[Tuple[int, str]]:
    n = len(prices)
    if n < 3:
        return []
    turns: List[Tuple[int, str]] = []
    for i in range(1, n - 1):
        prev = prices[i - 1]
        curr = prices[i]
        nxt = prices[i + 1]
        if curr > prev and curr > nxt:
            if curr - min(prev, nxt) >= TURN_THRESHOLD_PCT:
                turns.append((i, "peak"))
        elif curr < prev and curr < nxt:
            if min(prev, nxt) - curr >= TURN_THRESHOLD_PCT:
                turns.append((i, "valley"))
    if len(turns) > MAX_TURNS:
        def swing(t):
            i = t[0]
            lo = min(prices[max(0, i - 3):i + 4])
            hi = max(prices[max(0, i - 3):i + 4])
            return hi - lo
        turns = sorted(turns, key=swing, reverse=True)[:MAX_TURNS]
        turns = sorted(turns, key=lambda x: x[0])
    return turns


def _last_move(prices: List[float]) -> str:
    if len(prices) < 2:
        return "F"
    if prices[-1] > prices[-2]:
        return "U"
    if prices[-1] < prices[-2]:
        return "D"
    return "F"


def _overall_trend(prices: List[float]) -> str:
    if len(prices) < 2:
        return "F"
    delta = prices[-1] - prices[0]
    if delta > 0.10:
        return "U"
    if delta < -0.10:
        return "D"
    return "F"


def _terrain_bin(ticks: List[Tuple[float, float]]) -> int:
    if len(ticks) < 8:
        return 1
    prices = [p for _, p in ticks[-20:]]
    returns = []
    for i in range(1, len(prices)):
        if prices[i - 1] > 0:
            returns.append((prices[i] - prices[i - 1]) / prices[i - 1])
    if not returns:
        return 1
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / max(len(returns) - 1, 1)
    vol = math.sqrt(var)
    if vol < 0.00005:
        return 0
    if vol < 0.00020:
        return 1
    return 2


def _core_signature(prices: List[float], terrain: int) -> Tuple[str, Dict[str, Any]]:
    turns = _detect_turns(prices)
    n_turns = min(len(turns), 9)

    if not turns:
        first_type = "n"
    else:
        first_type = turns[0][1][0]

    last_dir = _last_move(prices)
    trend = _overall_trend(prices)

    sig = f"trail:nt{n_turns}_pe{first_type}_{trend}{last_dir}_t{terrain}"

    features = {
        "n_turns": n_turns,
        "turn_positions": [t[0] for t in turns],
        "turn_types": [t[1] for t in turns],
        "first_turn_type": first_type,
        "trend": trend,
        "last_move": last_dir,
        "terrain": terrain,
    }
    return sig, features


class TrailTracerEngine(Engine):
    name = "trail_tracer"

    def run(self, ctx: EngineContext) -> EngineOutput:
        raise NotImplementedError("TrailTracer is async; use run_async()")

    async def run_async(self, ctx: EngineContext) -> EngineOutput:
        if TRAIL_MODE != "active":
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature=None,
                raw_state={"reason": f"TRAIL_MODE={TRAIL_MODE}"},
            )

        if not ctx.recent_ticks or len(ctx.recent_ticks) < MIN_TICKS:
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature=None,
                raw_state={"reason": "insufficient ticks"},
            )

        window = ctx.recent_ticks[-WINDOW_TICKS:]
        prices = [p for _, p in window]

        normalized = _normalize_path(window)
        if normalized is None:
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature=None,
                raw_state={"reason": "normalization failed"},
            )
        times, norm_prices = normalized

        terrain = _terrain_bin(window)
        core_sig, features = _core_signature(norm_prices, terrain)

        dwell_s = self._compute_dwell(ctx.recent_ticks, core_sig)
        full_sig = f"{core_sig}_dw{dwell_bin(dwell_s)}"

        raw: Dict[str, Any] = {
            **features,
            "dwell_s": dwell_s,
            "path_times": [round(t - times[0], 3) for t in times],
            "path_prices_normalized": [round(p, 4) for p in norm_prices],
            "path_prices_raw": [round(p, 2) for p in prices],
        }

        try:
            lookup = await lookup_direction(full_sig)
        except Exception as e:
            raw["lookup_error"] = str(e)
            lookup = None

        if lookup is None:
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature=full_sig,
                raw_state={**raw, "lookup": "insufficient_history"},
            )

        direction, occurrences, confidence = lookup
        return EngineOutput(
            engine=self.name,
            direction=direction,
            confidence=confidence,
            pattern_signature=full_sig,
            raw_state={
                **raw,
                "lookup": {
                    "occurrences": occurrences,
                    "confidence": confidence,
                    "chosen_direction": direction,
                },
            },
        )

    def _compute_dwell(
        self,
        ticks: List[Tuple[float, float]],
        current_core_sig: str,
    ) -> float:
        if len(ticks) < MIN_TICKS + 2:
            return 0.0
        now_ts = ticks[-1][0]
        for back in range(1, DWELL_MAX_BACK + 1):
            sub = ticks[:-back]
            if len(sub) < MIN_TICKS:
                return now_ts - ticks[len(ticks) - back - 1][0] if back < len(ticks) else 0.0
            window = sub[-WINDOW_TICKS:]
            norm = _normalize_path(window)
            if norm is None:
                return now_ts - sub[-1][0]
            _, norm_prices = norm
            terrain = _terrain_bin(window)
            sig, _ = _core_signature(norm_prices, terrain)
            if sig != current_core_sig:
                return now_ts - ticks[len(ticks) - back][0]
        return now_ts - ticks[-DWELL_MAX_BACK][0] if len(ticks) >= DWELL_MAX_BACK else 0.0
