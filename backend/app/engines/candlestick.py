"""Candlestick engine with streak (dwell equivalent) enrichment."""
from typing import Any, Dict, List, Optional, Tuple

from ._lookup import lookup_direction
from .base import Engine, EngineContext, EngineOutput

CANDLE_SECONDS = 5.0
LOOKBACK_SECONDS = 60.0
POSITION_LOOKBACK = 5


def _build_candles(ticks, sec):
    if not ticks:
        return []
    out = []
    bucket = None
    o = h = l = c = None
    for ts, p in ticks:
        b = int(ts // sec)
        if bucket is None:
            bucket, o, h, l, c = b, p, p, p, p
        elif b == bucket:
            if p > h:
                h = p
            if p < l:
                l = p
            c = p
        else:
            out.append({"o": o, "h": h, "l": l, "c": c, "start": bucket * sec})
            bucket, o, h, l, c = b, p, p, p, p
    return out


def _classify(candle, prev):
    o, h, l, c = candle["o"], candle["h"], candle["l"], candle["c"]
    rng = h - l
    if rng < 1e-9:
        return "doji"
    body = abs(c - o)
    upw = h - max(o, c)
    low = min(o, c) - l
    br = body / rng
    ur = upw / rng
    lr = low / rng
    if c > o:
        d = "bull"
    elif c < o:
        d = "bear"
    else:
        d = "flat"
    if br < 0.10:
        return "doji"
    if br > 0.85:
        return "marubozu_bull" if d == "bull" else "marubozu_bear"
    if lr > 0.55 and br < 0.45:
        return "hammer"
    if ur > 0.55 and br < 0.45:
        return "star"
    if prev is not None:
        if min(o, c) > min(prev["o"], prev["c"]) and max(o, c) < max(prev["o"], prev["c"]):
            return "inside"
    return "body_bull" if d == "bull" else "body_bear"


def _position(close, candles):
    if not candles:
        return "mid"
    hi = max(c["h"] for c in candles)
    lo = min(c["l"] for c in candles)
    if hi <= lo:
        return "mid"
    f = (close - lo) / (hi - lo)
    if f <= 0.33:
        return "lo"
    if f >= 0.67:
        return "hi"
    return "mid"


def _prev_dir(prev):
    if prev is None:
        return "F"
    if prev["c"] > prev["o"]:
        return "U"
    if prev["c"] < prev["o"]:
        return "D"
    return "F"


class CandlestickEngine(Engine):
    name = "candlestick"

    def run(self, ctx):
        raise NotImplementedError("Candlestick is async; use run_async()")

    async def run_async(self, ctx: EngineContext) -> EngineOutput:
        if not ctx.recent_ticks or len(ctx.recent_ticks) < 10:
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature=None,
                raw_state={"reason": "insufficient ticks"},
            )
        now_ts = ctx.recent_ticks[-1][0]
        ticks = [(t, p) for t, p in ctx.recent_ticks if t >= now_ts - LOOKBACK_SECONDS]
        candles = _build_candles(ticks, CANDLE_SECONDS)
        if len(candles) < 3:
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature=None,
                raw_state={"reason": "not enough candles"},
            )
        prev = candles[-2]
        curr = candles[-1]
        ct = _classify(curr, prev)
        pos = _position(curr["c"], candles[-POSITION_LOOKBACK:])
        pd = _prev_dir(prev)

        # Streak: how many preceding candles share the current candle's type
        streak = 1
        for k in range(len(candles) - 2, max(0, len(candles) - 8), -1):
            earlier = candles[k]
            earlier_prev = candles[k - 1] if k - 1 >= 0 else None
            if _classify(earlier, earlier_prev) == ct:
                streak += 1
            else:
                break
        streak_bin = min(streak - 1, 3)

        sig = f"cs:{ct}_{pos}_{pd}_st{streak_bin}"

        raw: Dict[str, Any] = {
            "candle_type": ct,
            "position": pos,
            "prev_dir": pd,
            "streak": streak,
        }

        try:
            lookup = await lookup_direction(sig)
        except Exception as e:
            raw["lookup_error"] = str(e)
            lookup = None

        if lookup is None:
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature=sig,
                raw_state={**raw, "lookup": "insufficient_history"},
            )
        direction, occ, conf = lookup
        return EngineOutput(
            engine=self.name,
            direction=direction,
            confidence=conf,
            pattern_signature=sig,
            raw_state={
                **raw,
                "lookup": {
                    "occurrences": occ,
                    "confidence": conf,
                    "chosen_direction": direction,
                },
            },
        )
