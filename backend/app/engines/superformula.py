"""Superformula (Gielis) engine with dwell enrichment."""
import math
from typing import Any, Dict, List, Optional, Tuple

from ._lookup import lookup_direction
from ._temporal import EngineSnapshot, compute_dwell_s, dwell_bin
from .base import Engine, EngineContext, EngineOutput

WINDOW_TICKS = 24
NUM_SAMPLES = 32
M_CANDIDATES = [1, 2, 3, 4, 5, 6, 8, 10, 12]
N1_CANDIDATES = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
N2_CANDIDATES = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
M_BINS = 5
N_BINS = 4


def _sf_r(phi, m, n1, n2, n3):
    t1 = abs(math.cos(m * phi / 4.0)) ** n2
    t2 = abs(math.sin(m * phi / 4.0)) ** n3
    base = t1 + t2
    if base <= 0:
        return 0.0
    return base ** (-1.0 / n1)


def _normalize(prices):
    n = len(prices)
    if n < 4:
        return None
    lo, hi = min(prices), max(prices)
    if hi <= lo:
        return None
    norm = [0.2 + 0.8 * (p - lo) / (hi - lo) for p in prices]
    return [(norm[i], 2.0 * math.pi * i / n) for i in range(n)]


def _fit(points):
    best_rmse = float("inf")
    best = None
    step = max(1, len(points) // NUM_SAMPLES)
    sampled = points[::step]
    for m in M_CANDIDATES:
        for n1 in N1_CANDIDATES:
            for n2 in N2_CANDIDATES:
                n3 = n2
                err = 0.0
                cnt = 0
                for r_obs, phi in sampled:
                    r_pred = _sf_r(phi, m, n1, n2, n3)
                    if not math.isfinite(r_pred):
                        err += 1.0
                        cnt += 1
                        continue
                    err += (r_obs - r_pred) ** 2
                    cnt += 1
                if cnt == 0:
                    continue
                rmse = math.sqrt(err / cnt)
                if rmse < best_rmse:
                    best_rmse = rmse
                    best = (m, n1, n2, n3, rmse)
    return best


def _shape_dir(prices):
    if len(prices) < 2:
        return "F"
    if prices[-1] > prices[0]:
        return "U"
    if prices[-1] < prices[0]:
        return "D"
    return "F"


def _quant(v, candidates, n_bins):
    if not candidates:
        return 0
    sc = sorted(candidates)
    idx = 0
    for i, c in enumerate(sc):
        if v >= c:
            idx = i
    return min(int(idx / max(len(sc) - 1, 1) * n_bins), n_bins - 1)


class SuperformulaEngine(Engine):
    name = "superformula"

    def _core(self, window):
        if len(window) < 8:
            return None
        prices = [p for _, p in window[-WINDOW_TICKS:]]
        points = _normalize(prices)
        if points is None:
            return None
        fit = _fit(points)
        if fit is None:
            return None
        m, n1, n2, n3, rmse = fit
        d = _shape_dir(prices)
        sig = f"sf:m{_quant(m, M_CANDIDATES, M_BINS)}_n1_{_quant(n1, N1_CANDIDATES, N_BINS)}_n2_{_quant(n2, N2_CANDIDATES, N_BINS)}_d{d}"
        return EngineSnapshot(
            signature=sig,
            features={"m": float(m), "n1": float(n1), "n2": float(n2)},
        )

    def _snapshot_for_window(self, window):
        return self._core(window)

    def run(self, ctx):
        raise NotImplementedError("Superformula is async; use run_async()")

    async def run_async(self, ctx: EngineContext) -> EngineOutput:
        if not ctx.recent_ticks or len(ctx.recent_ticks) < 8:
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature=None,
                raw_state={"reason": "insufficient ticks"},
            )
        snap = self._core(ctx.recent_ticks)
        if snap is None:
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature=None,
                raw_state={"reason": "snapshot failed"},
            )

        dwell_s = compute_dwell_s(
            self._snapshot_for_window, ctx.recent_ticks, snap.signature, max_back=8
        )
        full_sig = f"{snap.signature}_dw{dwell_bin(dwell_s)}"

        raw: Dict[str, Any] = {"dwell_s": dwell_s, **snap.features}

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
