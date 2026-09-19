"""Candlestick Engine.

Aggregates recent ticks into OHLC candles at a fixed interval, classifies
each candle's geometry (body, wicks, position in range), and produces a
compact shape signature. Direction learned from history via shared
pattern memory.

Signature: cs:{candle_type}_{position}_{prev_dir}
Candle types: doji, marubozu_bull, marubozu_bear, hammer, star,
              body_bull, body_bear, inside
Position bins: lo, mid, hi  (close relative to recent range)
Prev direction: U, D, F

Total states: 8 x 3 x 3 = 72. Fast to learn.
"""
from typing import Any, Dict, List, Optional, Tuple

from ._lookup import lookup_direction
from .base import Engine, EngineContext, EngineOutput

CANDLE_SECONDS = 5.0
LOOKBACK_SECONDS = 40.0     # window of ticks to consider
POSITION_LOOKBACK = 5       # candles used for the recent range


def _build_candles(
    ticks: List[Tuple[float, float]], candle_seconds: float
) -> List[Dict[str, float]]:
    """Aggregate (timestamp, price) ticks into completed OHLC candles."""
    if not ticks:
        return []
    candles: List[Dict[str, float]] = []
    current_bucket = None
    o = h = l = c = None
    for ts, p in ticks:
        bucket = int(ts // candle_seconds)
        if current_bucket is None:
            current_bucket = bucket
            o = h = l = c = p
        elif bucket == current_bucket:
            if p > h:
                h = p
            if p < l:
                l = p
            c = p
        else:
            candles.append({
                "o": o, "h": h, "l": l, "c": c,
                "start": current_bucket * candle_seconds,
            })
            current_bucket = bucket
            o = h = l = c = p
    # Do not include the current, possibly-incomplete candle
    return candles


def _classify_candle(
    candle: Dict[str, float],
    prev_candle: Optional[Dict[str, float]],
) -> str:
    """Return a compact type label for one candle."""
    o, h, l, c = candle["o"], candle["h"], candle["l"], candle["c"]
    rng = h - l
    if rng < 1e-9:
        return "doji"

    body = abs(c - o)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    body_ratio = body / rng
    upper_ratio = upper_wick / rng
    lower_ratio = lower_wick / rng

    if c > o:
        direction = "bull"
    elif c < o:
        direction = "bear"
    else:
        direction = "flat"

    if body_ratio < 0.10:
        return "doji"
    if body_ratio > 0.85:
        return "marubozu_bull" if direction == "bull" else "marubozu_bear"
    if lower_ratio > 0.55 and body_ratio < 0.45:
        return "hammer"
    if upper_ratio > 0.55 and body_ratio < 0.45:
        return "star"

    if prev_candle is not None:
        prev_o, prev_c = prev_candle["o"], prev_candle["c"]
        if (
            min(o, c) > min(prev_o, prev_c)
            and max(o, c) < max(prev_o, prev_c)
        ):
            return "inside"

    return "body_bull" if direction == "bull" else "body_bear"


def _position_bin(
    close: float,
    recent_candles: List[Dict[str, float]],
) -> str:
    if not recent_candles:
        return "mid"
    highs = [c["h"] for c in recent_candles]
    lows = [c["l"] for c in recent_candles]
    hi = max(highs)
    lo = min(lows)
    if hi <= lo:
        return "mid"
    frac = (close - lo) / (hi - lo)
    if frac <= 0.33:
        return "lo"
    if frac >= 0.67:
        return "hi"
    return "mid"


def _prev_direction(prev_candle: Optional[Dict[str, float]]) -> str:
    if prev_candle is None:
        return "F"
    if prev_candle["c"] > prev_candle["o"]:
        return "U"
    if prev_candle["c"] < prev_candle["o"]:
        return "D"
    return "F"


class CandlestickEngine(Engine):
    name = "candlestick"

    def run(self, ctx: EngineContext) -> EngineOutput:
        raise NotImplementedError(
            "Candlestick engine is async; use run_async()"
        )

    async def run_async(self, ctx: EngineContext) -> EngineOutput:
        if not ctx.recent_ticks or len(ctx.recent_ticks) < 10:
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature=None,
                raw_state={"reason": "insufficient ticks"},
            )

        now_ts = ctx.recent_ticks[-1][0]
        cutoff = now_ts - LOOKBACK_SECONDS
        ticks = [(ts, p) for ts, p in ctx.recent_ticks if ts >= cutoff]

        candles = _build_candles(ticks, CANDLE_SECONDS)
        if len(candles) < 2:
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature=None,
                raw_state={
                    "reason": f"only {len(candles)} completed candles"
                },
            )

        prev_candle = candles[-2]
        curr_candle = candles[-1]
        context_for_inside = candles[-3] if len(candles) >= 3 else None

        curr_type = _classify_candle(curr_candle, prev_candle)
        recent_for_position = candles[-POSITION_LOOKBACK:]
        pos = _position_bin(curr_candle["c"], recent_for_position)
        prev_dir = _prev_direction(prev_candle)

        signature = f"cs:{curr_type}_{pos}_{prev_dir}"

        raw: Dict[str, Any] = {
            "candle_type": curr_type,
            "position_bin": pos,
            "prev_dir": prev_dir,
            "num_candles": len(candles),
            "curr_o": curr_candle["o"],
            "curr_h": curr_candle["h"],
            "curr_l": curr_candle["l"],
            "curr_c": curr_candle["c"],
            "curr_range": curr_candle["h"] - curr_candle["l"],
        }

        try:
            lookup = await lookup_direction(signature)
        except Exception as e:
            raw["lookup_error"] = str(e)
            lookup = None

        if lookup is None:
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature=signature,
                raw_state={**raw, "lookup": "insufficient_history"},
            )

        direction, occurrences, confidence = lookup
        return EngineOutput(
            engine=self.name,
            direction=direction,
            confidence=confidence,
            pattern_signature=signature,
            raw_state={
                **raw,
                "lookup": {
                    "occurrences": occurrences,
                    "confidence": confidence,
                    "chosen_direction": direction,
                },
            },
        )
