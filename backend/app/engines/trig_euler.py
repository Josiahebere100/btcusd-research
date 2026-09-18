"""Trig/Euler engine."""
import math
from typing import Any, Dict, List, Optional, Tuple

from ._lookup import lookup_direction
from .base import Engine, EngineContext, EngineOutput

WINDOW_TICKS = 16
SINGULARITY_EPS = 0.02
THETA_BINS = 12
VELOCITY_BINS = 5
ACCEL_BINS = 5


def _safe_tan(theta: float) -> Optional[float]:
    c = math.cos(theta)
    if abs(c) < SINGULARITY_EPS: return None
    return math.sin(theta) / c

def _safe_cot(theta: float) -> Optional[float]:
    s = math.sin(theta)
    if abs(s) < SINGULARITY_EPS: return None
    return math.cos(theta) / s

def _safe_sec(theta: float) -> Optional[float]:
    c = math.cos(theta)
    if abs(c) < SINGULARITY_EPS: return None
    return 1.0 / c

def _safe_csc(theta: float) -> Optional[float]:
    s = math.sin(theta)
    if abs(s) < SINGULARITY_EPS: return None
    return 1.0 / s


def _theta_from_prices(prices: List[float]) -> Optional[List[float]]:
    if not prices: return None
    p_min, p_max = min(prices), max(prices)
    span = p_max - p_min
    if span <= 0: return [0.0 for _ in prices]
    return [2.0 * math.pi * (p - p_min) / span for p in prices]


def _phase_velocity_acceleration(thetas, dt_s):
    if len(thetas) < 3 or dt_s <= 0: return None, None
    unwrapped = [thetas[0]]
    for t in thetas[1:]:
        prev = unwrapped[-1]
        delta = t - prev
        while delta > math.pi: delta -= 2 * math.pi
        while delta < -math.pi: delta += 2 * math.pi
        unwrapped.append(prev + delta)
    v1 = (unwrapped[-1] - unwrapped[-2]) / dt_s
    v0 = (unwrapped[-2] - unwrapped[-3]) / dt_s
    return v1, (v1 - v0) / dt_s


def _bin(v, n, lo, hi):
    if hi <= lo or v < lo: return 0
    if v >= hi: return n - 1
    return int((v - lo) / (hi - lo) * n)


def _signature(theta, velocity, acceleration, singular_flags):
    tb = _bin(theta, THETA_BINS, 0.0, 2.0 * math.pi)
    vb = _bin(velocity, VELOCITY_BINS, -2.0, 2.0) if velocity is not None else 0
    ab = _bin(acceleration, ACCEL_BINS, -4.0, 4.0) if acceleration is not None else 0
    return f"trig:t{tb}_v{vb}_a{ab}_s{singular_flags}"


class TrigEulerEngine(Engine):
    name = "trig_euler"

    def run(self, ctx: EngineContext) -> EngineOutput:
        raise NotImplementedError("TrigEuler is async; use run_async()")

    async def run_async(self, ctx: EngineContext) -> EngineOutput:
        if not ctx.recent_ticks or len(ctx.recent_ticks) < 4:
            return EngineOutput(
                engine=self.name, direction="NO_SIGNAL",
                pattern_signature=None,
                raw_state={"reason": "insufficient ticks"},
            )

        window = ctx.recent_ticks[-WINDOW_TICKS:]
        prices = [p for _, p in window]
        times = [t for t, _ in window]
        thetas = _theta_from_prices(prices)
        if thetas is None:
            return EngineOutput(
                engine=self.name, direction="NO_SIGNAL",
                pattern_signature=None,
                raw_state={"reason": "theta failed"},
            )

        current_theta = thetas[-1]
        cos_t, sin_t = math.cos(current_theta), math.sin(current_theta)
        tan_t = _safe_tan(current_theta)
        cot_t = _safe_cot(current_theta)
        sec_t = _safe_sec(current_theta)
        csc_t = _safe_csc(current_theta)
        singular_flags = sum(1 for v in (tan_t, cot_t, sec_t, csc_t) if v is None)

        avg_dt_s = (times[-1] - times[0]) / max(len(times) - 1, 1)
        velocity, acceleration = _phase_velocity_acceleration(thetas, avg_dt_s)

        signature = _signature(current_theta, velocity, acceleration, singular_flags)

        raw: Dict[str, Any] = {
            "theta": current_theta,
            "sin": sin_t, "cos": cos_t,
            "tan": tan_t, "cot": cot_t, "sec": sec_t, "csc": csc_t,
            "phase_velocity": velocity,
            "phase_acceleration": acceleration,
            "singularity_flags": singular_flags,
        }

        try:
            lookup = await lookup_direction(signature)
        except Exception as e:
            raw["lookup_error"] = str(e)
            lookup = None

        if lookup is None:
            return EngineOutput(
                engine=self.name, direction="NO_SIGNAL",
                pattern_signature=signature,
                raw_state={**raw, "lookup": "insufficient_history"},
            )

        direction, occurrences, confidence = lookup
        return EngineOutput(
            engine=self.name, direction=direction, confidence=confidence,
            pattern_signature=signature,
            raw_state={**raw, "lookup": {
                "occurrences": occurrences, "confidence": confidence,
                "chosen_direction": direction,
            }},
        )
