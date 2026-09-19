"""Trig/Euler engine with full 2D geometric relationship detection.

Relationships detected between market trajectory and each trig curve:
  parallel, perpendicular, coincident, intersecting,
  converging, diverging
Plus graph-level flags: concurrent, transversal.

Skew and oblique are 3D-only and correctly excluded.

Signature: trig:rel{X}_cx{Y}_tr{Z}_dw{W}_dir{D}_sg{S}
State space: 6 x 2 x 2 x 4 x 3 x 2 = 576.
"""
import math
from typing import Any, Dict, List, Optional, Tuple

from ._lookup import lookup_direction
from ._temporal import (
    EngineSnapshot,
    compute_dwell_s,
    dwell_bin,
)
from .base import Engine, EngineContext, EngineOutput

WINDOW_TICKS = 24
REL_WINDOW = 10
SINGULARITY_EPS = 0.02

REL_PARALLEL = 0
REL_PERPENDICULAR = 1
REL_COINCIDENT = 2
REL_INTERSECTING = 3
REL_CONVERGING = 4
REL_DIVERGING = 5

REL_NAMES = {
    0: "parallel",
    1: "perpendicular",
    2: "coincident",
    3: "intersecting",
    4: "converging",
    5: "diverging",
}


def _safe_tan(t):
    c = math.cos(t)
    if abs(c) < SINGULARITY_EPS:
        return None
    return math.sin(t) / c


def _safe_cot(t):
    s = math.sin(t)
    if abs(s) < SINGULARITY_EPS:
        return None
    return math.cos(t) / s


def _safe_sec(t):
    c = math.cos(t)
    if abs(c) < SINGULARITY_EPS:
        return None
    return 1.0 / c


def _safe_csc(t):
    s = math.sin(t)
    if abs(s) < SINGULARITY_EPS:
        return None
    return 1.0 / s


def _theta_series(prices):
    if not prices:
        return []
    p_min = min(prices)
    p_max = max(prices)
    span = p_max - p_min
    if span <= 0:
        return [0.0] * len(prices)
    return [2.0 * math.pi * (p - p_min) / span for p in prices]


def _slope(series):
    if len(series) < 2:
        return 0.0
    return (series[-1] - series[0]) / (len(series) - 1)


def _classify_relationship(market, trig):
    """Return one of the 6 relationship codes for two series of equal length."""
    n = len(market)
    if n < 3 or len(trig) != n:
        return REL_DIVERGING

    diff = [market[i] - trig[i] for i in range(n)]
    mslope = _slope(market)
    tslope = _slope(trig)
    scale = max(abs(mslope), abs(tslope), 1e-6)

    # Parallel: slopes roughly equal
    if abs(mslope - tslope) < 0.1 * scale:
        return REL_PARALLEL

    # Perpendicular: dot product of tangent vectors (1, mslope) · (1, tslope) ≈ 0
    if abs(1.0 + mslope * tslope) < 0.3:
        return REL_PERPENDICULAR

    # Coincident: values nearly equal throughout
    if all(abs(d) < SINGULARITY_EPS for d in diff):
        return REL_COINCIDENT

    # Intersecting: sign of difference changes
    signs = [1 if d > 0 else (-1 if d < 0 else 0) for d in diff]
    signs = [s for s in signs if s != 0]
    if len(signs) >= 2 and signs[0] != signs[-1]:
        return REL_INTERSECTING

    # Converging / diverging: absolute distance trend
    dist_start = abs(diff[0])
    dist_end = abs(diff[-1])
    if dist_end < dist_start:
        return REL_CONVERGING
    return REL_DIVERGING


def _dominant_relationship(market, curves):
    """Return the most common relationship across market-vs-curve pairs."""
    counts = {}
    for trig in curves:
        r = _classify_relationship(market, trig)
        counts[r] = counts.get(r, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _detect_concurrent(curves, tol=0.05):
    """True if 3+ curves are within tolerance at any common index."""
    if not curves or len(curves[0]) < 1:
        return False
    n = len(curves[0])
    for i in range(n):
        vals = [c[i] for c in curves if i < len(c)]
        for a in range(len(vals)):
            for b in range(a + 1, len(vals)):
                if abs(vals[a] - vals[b]) > tol:
                    continue
                for c in range(b + 1, len(vals)):
                    if abs(vals[a] - vals[c]) <= tol and abs(vals[b] - vals[c]) <= tol:
                        return True
    return False


def _detect_transversal(curves, tol=0.05):
    """True if any curve crosses 2+ others near the same index."""
    if len(curves) < 3:
        return False
    n = len(curves[0])
    for t in range(1, n):
        crossers = 0
        for a in range(len(curves)):
            crosses = 0
            for b in range(len(curves)):
                if a == b:
                    continue
                d_prev = curves[a][t - 1] - curves[b][t - 1]
                d_now = curves[a][t] - curves[b][t]
                if d_prev == 0 or d_now == 0:
                    continue
                if (d_prev > 0) != (d_now > 0) and abs(d_now) < 10 * tol:
                    crosses += 1
            if crosses >= 2:
                crossers += 1
        if crossers >= 1:
            return True
    return False


def _direction_of(prices):
    if len(prices) < 2:
        return 1
    if prices[-1] > prices[0]:
        return 2
    if prices[-1] < prices[0]:
        return 0
    return 1


class TrigEulerEngine(Engine):
    name = "trig_euler"

    def _compute_core(self, window):
        if len(window) < 6:
            return None
        prices = [p for _, p in window[-WINDOW_TICKS:]]
        thetas = _theta_series(prices)
        theta = thetas[-1]

        recent = prices[-REL_WINDOW:]
        recent_thetas = thetas[-REL_WINDOW:]
        curves = [
            [math.sin(t) for t in recent_thetas],
            [math.cos(t) for t in recent_thetas],
            [(_safe_tan(t) or 0.0) for t in recent_thetas],
            [(_safe_cot(t) or 0.0) for t in recent_thetas],
            [(_safe_sec(t) or 0.0) for t in recent_thetas],
            [(_safe_csc(t) or 0.0) for t in recent_thetas],
        ]
        market_norm = recent  # raw prices, but relationship detection is scale-invariant

        rel = _dominant_relationship(market_norm, curves)
        cx = 1 if _detect_concurrent(curves) else 0
        tr = 1 if _detect_transversal(curves) else 0
        direction = _direction_of(recent)

        singular_count = sum(
            1 for v in [
                _safe_tan(theta),
                _safe_cot(theta),
                _safe_sec(theta),
                _safe_csc(theta),
            ]
            if v is None
        )
        sg = 1 if singular_count >= 2 else 0

        sig = f"trig:rel{rel}_cx{cx}_tr{tr}_dir{direction}_sg{sg}"
        return EngineSnapshot(
            signature=sig,
            features={
                "theta": theta,
                "n_parallel": float(rel == REL_PARALLEL),
                "n_perp": float(rel == REL_PERPENDICULAR),
                "cx": float(cx),
                "tr": float(tr),
            },
        )

    def _snapshot_for_window(self, window):
        return self._compute_core(window)

    def run(self, ctx):
        raise NotImplementedError("TrigEuler is async; use run_async()")

    async def run_async(self, ctx: EngineContext) -> EngineOutput:
        if not ctx.recent_ticks or len(ctx.recent_ticks) < 6:
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature=None,
                raw_state={"reason": "insufficient ticks"},
            )

        snap = self._compute_core(ctx.recent_ticks)
        if snap is None:
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature=None,
                raw_state={"reason": "snapshot failed"},
            )

        dwell_s = compute_dwell_s(
            self._snapshot_for_window, ctx.recent_ticks, snap.signature, max_back=12
        )
        full_sig = f"{snap.signature}_dw{dwell_bin(dwell_s)}"

        raw: Dict[str, Any] = {
            "theta": snap.features.get("theta"),
            "dwell_s": dwell_s,
            "concurrent": int(snap.features.get("cx", 0)),
            "transversal": int(snap.features.get("tr", 0)),
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
