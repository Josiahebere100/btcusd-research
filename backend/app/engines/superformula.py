"""Superformula (Gielis) Engine.

Fits Johan Gielis's Superformula to the recent normalized price curve:

    r(phi) = ( |cos(m*phi/4)/a|^n2 + |sin(m*phi/4)/b|^n3 )^(-1/n1)

The shape parameters (m, n1, n2) become a coarse geometric signature.
Direction learned from history.

Signature space: ~180 states.
"""
import math
from typing import Any, Dict, List, Optional, Tuple

from ._lookup import lookup_direction
from .base import Engine, EngineContext, EngineOutput

WINDOW_TICKS = 24
NUM_SAMPLES = 32
M_CANDIDATES = [1, 2, 3, 4, 5, 6, 8, 10, 12]
N1_CANDIDATES = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
N2_CANDIDATES = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
M_BINS = 5
N_BINS = 4


def _superformula_r(phi: float, m: float, n1: float,
                    n2: float, n3: float,
                    a: float = 1.0, b: float = 1.0) -> float:
    t1 = abs(math.cos(m * phi / 4.0) / a) ** n2
    t2 = abs(math.sin(m * phi / 4.0) / b) ** n3
    base = t1 + t2
    if base <= 0:
        return 0.0
    return base ** (-1.0 / n1)


def _normalize_to_unit_circle(prices: List[float]) -> Optional[List[Tuple[float, float]]]:
    n = len(prices)
    if n < 4:
        return None
    p_min = min(prices)
    p_max = max(prices)
    span = p_max - p_min
    if span <= 0:
        return None
    normalized = [0.2 + 0.8 * (p - p_min) / span for p in prices]
    points = []
    for i, r in enumerate(normalized):
        phi = 2.0 * math.pi * i / n
        points.append((r, phi))
    return points


def _fit_superformula(
    points: List[Tuple[float, float]],
) -> Optional[Tuple[float, float, float, float, float]]:
    best_rmse = float("inf")
    best_params = None
    step = max(1, len(points) // NUM_SAMPLES)
    sampled = points[::step]
    for m in M_CANDIDATES:
        for n1 in N1_CANDIDATES:
            for n2 in N2_CANDIDATES:
                n3 = n2
                err = 0.0
                count = 0
                for r_obs, phi in sampled:
                    r_pred = _superformula_r(phi, m, n1, n2, n3)
                    if not math.isfinite(r_pred):
                        err += 1.0
                        count += 1
                        continue
                    err += (r_obs - r_pred) ** 2
                    count += 1
                if count == 0:
                    continue
                rmse = math.sqrt(err / count)
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params = (m, n1, n2, n3, rmse)
    return best_params


def _shape_direction(prices: List[float]) -> str:
    if len(prices) < 2:
        return "F"
    if prices[-1] > prices[0]:
        return "U"
    if prices[-1] < prices[0]:
        return "D"
    return "F"


def _quantize(value: float, candidates: List[float], n_bins: int) -> int:
    if not candidates:
        return 0
    sorted_c = sorted(candidates)
    idx = 0
    for i, c in enumerate(sorted_c):
        if value >= c:
            idx = i
    return min(int(idx / max(len(sorted_c) - 1, 1) * n_bins), n_bins - 1)


def _signature(m: float, n1: float, n2: float, shape_dir: str) -> str:
    m_bin = _quantize(m, [float(x) for x in M_CANDIDATES], M_BINS)
    n1_bin = _quantize(n1, N1_CANDIDATES, N_BINS)
    n2_bin = _quantize(n2, N2_CANDIDATES, N_BINS)
    return f"sf:m{m_bin}_n1_{n1_bin}_n2_{n2_bin}_d{shape_dir}"


class SuperformulaEngine(Engine):
    name = "superformula"

    def run(self, ctx: EngineContext) -> EngineOutput:
        raise NotImplementedError("Superformula engine is async; use run_async()")

    async def run_async(self, ctx: EngineContext) -> EngineOutput:
        if not ctx.recent_ticks or len(ctx.recent_ticks) < 8:
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature=None,
                raw_state={"reason": "insufficient ticks"},
            )

        window = ctx.recent_ticks[-WINDOW_TICKS:]
        prices = [p for _, p in window]

        points = _normalize_to_unit_circle(prices)
        if points is None:
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature=None,
                raw_state={"reason": "normalization failed"},
            )

        fit = _fit_superformula(points)
        if fit is None:
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature=None,
                raw_state={"reason": "fit failed"},
            )

        m, n1, n2, n3, rmse = fit
        shape_dir = _shape_direction(prices)
        signature = _signature(m, n1, n2, shape_dir)

        raw: Dict[str, Any] = {
            "m": m,
            "n1": n1,
            "n2": n2,
            "n3": n3,
            "rmse": rmse,
            "shape_direction": shape_dir,
            "window_size": len(prices),
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
