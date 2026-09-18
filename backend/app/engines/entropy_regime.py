"""Entropy / Regime Engine.

Two information-theoretic measures over the recent tick window:
  - Permutation entropy (order patterns of consecutive ticks)
  - Return entropy (Shannon entropy of log-return distribution)

Signature: 48 states. Direction learned from history.
"""
import math
from collections import Counter
from typing import Any, Dict, List, Optional

from ._lookup import lookup_direction
from .base import Engine, EngineContext, EngineOutput

LOOKBACK_TICKS = 32
PERM_ORDER = 3
RETURN_BINS = 8
ENTROPY_BINS = 4


def _permutation_entropy(prices: List[float], m: int) -> Optional[float]:
    n = len(prices)
    if n < m + 1:
        return None
    counts: Counter = Counter()
    for i in range(n - m + 1):
        window = prices[i:i + m]
        order = tuple(sorted(range(m), key=lambda k: window[k]))
        counts[order] += 1
    total = sum(counts.values())
    if total == 0:
        return None
    h = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            h -= p * math.log(p)
    max_h = math.log(math.factorial(m))
    return h / max_h if max_h > 0 else None


def _return_entropy(prices: List[float], num_bins: int) -> Optional[float]:
    n = len(prices)
    if n < 4:
        return None
    log_p = [math.log(max(p, 1e-9)) for p in prices]
    returns = [log_p[i] - log_p[i - 1] for i in range(1, n)]
    if not returns:
        return None
    lo, hi = min(returns), max(returns)
    if hi <= lo:
        return 0.0
    span = hi - lo
    counts = [0] * num_bins
    for r in returns:
        idx = int((r - lo) / span * num_bins)
        if idx >= num_bins:
            idx = num_bins - 1
        counts[idx] += 1
    total = sum(counts)
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log(p)
    max_h = math.log(num_bins)
    return h / max_h if max_h > 0 else None


def _last_direction(prices: List[float]) -> str:
    if len(prices) < 2:
        return "F"
    if prices[-1] > prices[-2]:
        return "U"
    if prices[-1] < prices[-2]:
        return "D"
    return "F"


def _bin(v: float, n_bins: int) -> int:
    if v <= 0.0:
        return 0
    if v >= 1.0:
        return n_bins - 1
    return int(v * n_bins)


def _signature(pe: float, re: float, last_dir: str) -> str:
    return f"ent:p{_bin(pe, ENTROPY_BINS)}_r{_bin(re, ENTROPY_BINS)}_d{last_dir}"


class EntropyRegimeEngine(Engine):
    name = "entropy_regime"

    def run(self, ctx: EngineContext) -> EngineOutput:
        raise NotImplementedError("Entropy engine is async; use run_async()")

    async def run_async(self, ctx: EngineContext) -> EngineOutput:
        if not ctx.recent_ticks or len(ctx.recent_ticks) < 8:
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature=None,
                raw_state={"reason": "insufficient ticks"},
            )

        window = ctx.recent_ticks[-LOOKBACK_TICKS:]
        prices = [p for _, p in window]

        pe = _permutation_entropy(prices, PERM_ORDER)
        re = _return_entropy(prices, RETURN_BINS)
        last_dir = _last_direction(prices)

        raw: Dict[str, Any] = {
            "permutation_entropy": pe,
            "return_entropy": re,
            "last_direction": last_dir,
            "window_size": len(prices),
        }

        if pe is None or re is None:
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature=None,
                raw_state={**raw, "reason": "entropy computation failed"},
            )

        signature = _signature(pe, re, last_dir)

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
