"""Entropy/Regime engine with dwell + velocity."""
import math
from collections import Counter
from typing import Any, Dict, List, Optional

from ._lookup import lookup_direction
from ._temporal import (
    EngineSnapshot,
    compute_dwell_s,
    compute_velocity,
    dwell_bin,
    velocity_bin,
)
from .base import Engine, EngineContext, EngineOutput

LOOKBACK_TICKS = 32
PERM_ORDER = 3
RETURN_BINS = 8
ENTROPY_BINS = 4


def _permutation_entropy(prices, m):
    n = len(prices)
    if n < m + 1:
        return None
    counts = Counter()
    for i in range(n - m + 1):
        w = prices[i:i + m]
        order = tuple(sorted(range(m), key=lambda k: w[k]))
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


def _return_entropy(prices, bins):
    n = len(prices)
    if n < 4:
        return None
    log_p = [math.log(max(p, 1e-9)) for p in prices]
    rets = [log_p[i] - log_p[i - 1] for i in range(1, n)]
    if not rets:
        return None
    lo, hi = min(rets), max(rets)
    if hi <= lo:
        return 0.0
    span = hi - lo
    counts = [0] * bins
    for r in rets:
        i = int((r - lo) / span * bins)
        if i >= bins:
            i = bins - 1
        counts[i] += 1
    total = sum(counts)
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log(p)
    max_h = math.log(bins)
    return h / max_h if max_h > 0 else None


def _last_direction(prices):
    if len(prices) < 2:
        return "F"
    if prices[-1] > prices[-2]:
        return "U"
    if prices[-1] < prices[-2]:
        return "D"
    return "F"


def _bin(v, n):
    if v <= 0.0:
        return 0
    if v >= 1.0:
        return n - 1
    return int(v * n)


class EntropyRegimeEngine(Engine):
    name = "entropy_regime"

    def _core_sig(self, window):
        if len(window) < 8:
            return None
        prices = [p for _, p in window[-LOOKBACK_TICKS:]]
        pe = _permutation_entropy(prices, PERM_ORDER)
        re = _return_entropy(prices, RETURN_BINS)
        if pe is None or re is None:
            return None
        return f"ent:p{_bin(pe, ENTROPY_BINS)}_r{_bin(re, ENTROPY_BINS)}_d{_last_direction(prices)}"

    def _snapshot_for_window(self, window):
        sig = self._core_sig(window)
        if sig is None:
            return None
        prices = [p for _, p in window[-LOOKBACK_TICKS:]]
        pe = _permutation_entropy(prices, PERM_ORDER) or 0.0
        re = _return_entropy(prices, RETURN_BINS) or 0.0
        return EngineSnapshot(signature=sig, features={"pe": pe, "re": re})

    def run(self, ctx):
        raise NotImplementedError("Entropy is async; use run_async()")

    async def run_async(self, ctx: EngineContext) -> EngineOutput:
        if not ctx.recent_ticks or len(ctx.recent_ticks) < 8:
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature=None,
                raw_state={"reason": "insufficient ticks"},
            )

        snap = self._snapshot_for_window(ctx.recent_ticks)
        if snap is None:
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature=None,
                raw_state={"reason": "snapshot failed"},
            )

        dwell_s = compute_dwell_s(
            self._snapshot_for_window, ctx.recent_ticks, snap.signature, max_back=10
        )
        vel = compute_velocity(
            self._snapshot_for_window, ctx.recent_ticks, snap, offset_ticks=4
        )
        full_sig = f"{snap.signature}_dw{dwell_bin(dwell_s)}_v{velocity_bin(vel)}"

        raw: Dict[str, Any] = {
            "pe": snap.features["pe"],
            "re": snap.features["re"],
            "dwell_s": dwell_s,
            "velocity": vel,
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

        direction, occ, conf = lookup
        return EngineOutput(
            engine=self.name,
            direction=direction,
            confidence=conf,
            pattern_signature=full_sig,
            raw_state={
                **raw,
                "lookup": {
                    "occurrences": occ,
                    "confidence": conf,
                    "chosen_direction": direction,
                },
            },
        )
