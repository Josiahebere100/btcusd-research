"""Trig/Euler engine.

Tracks six trig trajectories (sin, cos, tan, cot, sec, csc) derived from
a normalized price angle θ, and maintains the Euler complex representation
e^(iθ) = cos(θ) + i·sin(θ).

Tracks phase, phase velocity, phase acceleration, and the complex
position/velocity/acceleration.

Detects and handles singularities (θ near π/2 + kπ for tan/sec, θ near kπ
for cot/csc) so that Infinity/NaN do not corrupt downstream state.

Phase 15 scope: trajectory + Euler + singularity handling + signature.
NOT YET implemented (future phases):
  - Geometric relationships (parallel, perpendicular, concurrent, etc.)
  - Temporal event stream (market ↔ SIN intersection, etc.)
"""
import math
from typing import Any, Dict, List, Optional, Tuple

from ..db import get_pool
from .base import Engine, EngineContext, EngineOutput

MIN_OCCURRENCES = 10
MIN_ACCURACY = 0.55

# How many recent ticks to use for the trajectory window.
WINDOW_TICKS = 16

# Singularity tolerance: if |cos θ| < this, sec/tan are considered at a
# singularity. Same for |sin θ| for csc/cot.
SINGULARITY_EPS = 0.02

# Quantization bins for the signature.
THETA_BINS = 12
VELOCITY_BINS = 5
ACCEL_BINS = 5


# ---- Safe trig helpers ----------------------------------------------------


def _safe_tan(theta: float) -> Optional[float]:
    c = math.cos(theta)
    if abs(c) < SINGULARITY_EPS:
        return None
    return math.sin(theta) / c


def _safe_cot(theta: float) -> Optional[float]:
    s = math.sin(theta)
    if abs(s) < SINGULARITY_EPS:
        return None
    return math.cos(theta) / s


def _safe_sec(theta: float) -> Optional[float]:
    c = math.cos(theta)
    if abs(c) < SINGULARITY_EPS:
        return None
    return 1.0 / c


def _safe_csc(theta: float) -> Optional[float]:
    s = math.sin(theta)
    if abs(s) < SINGULARITY_EPS:
        return None
    return 1.0 / s


# ---- Angle normalization --------------------------------------------------


def _theta_from_prices(prices: List[float]) -> Optional[List[float]]:
    """Map the recent price range monotonically onto [0, 2π).

    The mapping is deterministic: theta_i = 2π * (p_i - p_min) / (p_max - p_min).
    If all prices are equal, returns a list of zeros.
    """
    if not prices:
        return None
    p_min = min(prices)
    p_max = max(prices)
    span = p_max - p_min
    if span <= 0:
        return [0.0 for _ in prices]
    return [2.0 * math.pi * (p - p_min) / span for p in prices]


# ---- Euler / phase analysis ----------------------------------------------


def _phase_velocity_acceleration(
    thetas: List[float], dt_s: float
) -> Tuple[Optional[float], Optional[float]]:
    """Compute phase velocity (dθ/dt) and acceleration (d²θ/dt²).

    Uses unwrapped angles to avoid 2π wrap artifacts.
    """
    if len(thetas) < 3 or dt_s <= 0:
        return None, None

    # Unwrap so consecutive differences are small.
    unwrapped = [thetas[0]]
    for t in thetas[1:]:
        prev = unwrapped[-1]
        delta = t - prev
        while delta > math.pi:
            delta -= 2 * math.pi
        while delta < -math.pi:
            delta += 2 * math.pi
        unwrapped.append(prev + delta)

    # Finite differences.
    v1 = (unwrapped[-1] - unwrapped[-2]) / dt_s
    v0 = (unwrapped[-2] - unwrapped[-3]) / dt_s
    velocity = v1
    acceleration = (v1 - v0) / dt_s
    return velocity, acceleration


# ---- Signature ------------------------------------------------------------


def _bin(value: float, n_bins: int, lo: float, hi: float) -> int:
    if hi <= lo:
        return 0
    if value < lo:
        return 0
    if value >= hi:
        return n_bins - 1
    return int((value - lo) / (hi - lo) * n_bins)


def _signature(
    theta: float,
    velocity: Optional[float],
    acceleration: Optional[float],
    singular_flags: int,
) -> str:
    tb = _bin(theta, THETA_BINS, 0.0, 2.0 * math.pi)
    vb = _bin(velocity, VELOCITY_BINS, -2.0, 2.0) if velocity is not None else 0
    ab = _bin(acceleration, ACCEL_BINS, -4.0, 4.0) if acceleration is not None else 0
    return f"trig:t{tb}_v{vb}_a{ab}_s{singular_flags}"


# ---- Engine ---------------------------------------------------------------


class TrigEulerEngine(Engine):
    name = "trig_euler"

    async def _lookup_pattern(
        self, signature: str
    ) -> Optional[Tuple[str, int, float]]:
        pool = await get_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT id, occurrence_count, accuracy
                    FROM patterns
                    WHERE pattern_signature = %s
                    """,
                    (signature,),
                )
                row = await cur.fetchone()
                if not row:
                    return None

                pattern_id = row[0]
                occurrence_count = int(row[1] or 0)
                accuracy = float(row[2]) if row[2] is not None else None

                if occurrence_count < MIN_OCCURRENCES or accuracy is None:
                    return None
                if accuracy < MIN_ACCURACY:
                    return None

                await cur.execute(
                    """
                    SELECT COUNT(*) FROM pattern_success_memory
                    WHERE pattern_id = %s AND result = 'UP'
                    """,
                    (pattern_id,),
                )
                up_row = await cur.fetchone()
                up_correct = int(up_row[0]) if up_row else 0

                await cur.execute(
                    """
                    SELECT COUNT(*) FROM pattern_success_memory
                    WHERE pattern_id = %s AND result = 'DOWN'
                    """,
                    (pattern_id,),
                )
                down_row = await cur.fetchone()
                down_correct = int(down_row[0]) if down_row else 0

                if up_correct == down_correct:
                    return None
                direction = "UP" if up_correct > down_correct else "DOWN"
                return direction, occurrence_count, accuracy

    def run(self, ctx: EngineContext) -> EngineOutput:
        raise NotImplementedError("TrigEuler engine is async; use run_async()")

    async def run_async(self, ctx: EngineContext) -> EngineOutput:
        if not ctx.recent_ticks or len(ctx.recent_ticks) < 4:
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                confidence=None,
                pattern_signature=None,
                raw_state={"reason": "insufficient ticks"},
            )

        window = ctx.recent_ticks[-WINDOW_TICKS:]
        prices = [p for _, p in window]
        times = [t for t, _ in window]

        thetas = _theta_from_prices(prices)
        if thetas is None:
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                confidence=None,
                pattern_signature=None,
                raw_state={"reason": "theta computation failed"},
            )

        current_theta = thetas[-1]

        # Compute trig values at the current theta.
        cos_t = math.cos(current_theta)
        sin_t = math.sin(current_theta)
        tan_t = _safe_tan(current_theta)
        cot_t = _safe_cot(current_theta)
        sec_t = _safe_sec(current_theta)
        csc_t = _safe_csc(current_theta)

        singular_flags = sum(
            1 for v in (tan_t, cot_t, sec_t, csc_t) if v is None
        )

        # Euler complex position at current theta.
        complex_position = {"real": cos_t, "imag": sin_t}

        # Phase velocity / acceleration from the theta trajectory.
        avg_dt_s = (times[-1] - times[0]) / max(len(times) - 1, 1)
        velocity, acceleration = _phase_velocity_acceleration(thetas, avg_dt_s)

        # Complex velocity: derivative of e^(iθ) = i·e^(iθ)·dθ/dt
        complex_velocity = None
        if velocity is not None:
            complex_velocity = {
                "real": -sin_t * velocity,
                "imag": cos_t * velocity,
            }

        # Complex acceleration: d²/dt² of e^(iθ) = -(e^(iθ))·(dθ/dt)² + i·e^(iθ)·d²θ/dt²
        complex_acceleration = None
        if velocity is not None and acceleration is not None:
            complex_acceleration = {
                "real": -cos_t * velocity * velocity - sin_t * acceleration,
                "imag": -sin_t * velocity * velocity + cos_t * acceleration,
            }

        signature = _signature(
            current_theta, velocity, acceleration, singular_flags
        )

        raw_state: Dict[str, Any] = {
            "theta": current_theta,
            "sin": sin_t,
            "cos": cos_t,
            "tan": tan_t,
            "cot": cot_t,
            "sec": sec_t,
            "csc": csc_t,
            "phase_velocity": velocity,
            "phase_acceleration": acceleration,
            "complex_position": complex_position,
            "complex_velocity": complex_velocity,
            "complex_acceleration": complex_acceleration,
            "singularity_flags": singular_flags,
            "window_size": len(window),
        }

        try:
            lookup = await self._lookup_pattern(signature)
        except Exception as e:
            raw_state["lookup_error"] = str(e)
            lookup = None

        if lookup is None:
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                confidence=None,
                pattern_signature=signature,
                raw_state={**raw_state, "lookup": "insufficient_history"},
            )

        direction, occurrences, accuracy = lookup
        return EngineOutput(
            engine=self.name,
            direction=direction,
            confidence=accuracy,
            pattern_signature=signature,
            raw_state={
                **raw_state,
                "lookup": {
                    "occurrences": occurrences,
                    "accuracy": accuracy,
                    "chosen_direction": direction,
                },
            },
        )
