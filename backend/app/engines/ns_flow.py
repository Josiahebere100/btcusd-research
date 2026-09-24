"""NS Flow Lab engine (live).

Live version of the NS Flow Lab backtester. Reads from ctx.recent_ticks,
applies the trained logistic regression weights, emits calibrated
probabilities. Weights loaded from FLOW_WEIGHTS_JSON env var.
"""
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

from .base import Engine, EngineContext, EngineOutput


WINDOW = 40
SIGNAL_THRESHOLD = 0.58
MIN_TICKS = 45

FEATURE_NAMES = [
    "velocity", "acceleration", "jerk", "curvature", "diffusion",
    "terrain_volatility", "path_signed", "path_absolute",
    "flow_inertia", "flow_viscous", "pressure_proxy", "advection",
    "trend_fast", "trend_slow", "velocity_consistency",
]


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-min(x, 60.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(x, -60.0))
    return z / (1.0 + z)


def _clamp(x: float, lo: float = -10.0, hi: float = 10.0) -> float:
    return max(lo, min(hi, x))


def _safe_div(a: float, b: float, d: float = 0.0) -> float:
    if abs(b) < 1e-12:
        return d
    return a / b


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _stdev(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(max(var, 0.0))


def _build_features(ticks: List[Tuple[float, float]], window: int = WINDOW):
    if len(ticks) < 4:
        return None
    section = ticks[-window:]
    prices = [p for _, p in section]
    times = [t for t, _ in section]

    velocities = []
    for k in range(1, len(section)):
        dt = times[k] - times[k - 1]
        if dt <= 0:
            continue
        velocities.append(_safe_div(prices[k] - prices[k - 1], dt))
    if len(velocities) < 3:
        return None
    velocity = velocities[-1]

    accelerations = []
    for k in range(1, len(velocities)):
        dt = times[k] - times[k - 1]
        if dt <= 0:
            continue
        accelerations.append(_safe_div(velocities[k] - velocities[k - 1], dt))
    acceleration = accelerations[-1] if accelerations else 0.0

    if len(accelerations) >= 2:
        dt = times[-1] - times[-2]
        jerk = _safe_div(accelerations[-1] - accelerations[-2], dt)
    else:
        jerk = 0.0

    curvature = _safe_div(acceleration, (1.0 + velocity * velocity) ** 1.5)
    p0, p1, p2 = prices[-3], prices[-2], prices[-1]
    diffusion = p2 - 2.0 * p1 + p0

    returns = []
    for k in range(1, len(prices)):
        if prices[k - 1] != 0:
            returns.append((prices[k] - prices[k - 1]) / prices[k - 1])
    terrain_volatility = _stdev(returns)

    path_signed = prices[-1] - prices[0]
    path_absolute = sum(abs(prices[k] - prices[k - 1]) for k in range(1, len(prices)))

    flow_inertia = velocity + acceleration
    advection = velocity * acceleration
    local_mean = _mean(prices)
    pressure_proxy = local_mean - prices[-1]

    fast_n = min(8, len(prices))
    slow_n = min(len(prices), max(16, fast_n * 2))
    fast_mean = _mean(prices[-fast_n:])
    slow_mean = _mean(prices[-slow_n:])
    current_price = prices[-1]
    trend_fast = _safe_div(current_price - fast_mean, current_price)
    trend_slow = _safe_div(current_price - slow_mean, current_price)

    signs = []
    for v in velocities[-12:]:
        if v > 0:
            signs.append(1)
        elif v < 0:
            signs.append(-1)
        else:
            signs.append(0)
    velocity_consistency = _safe_div(sum(signs), len(signs))

    scale = max(abs(current_price), 1e-12)
    return {
        "velocity": _clamp(velocity / scale),
        "acceleration": _clamp(acceleration / scale),
        "jerk": _clamp(jerk / scale),
        "curvature": _clamp(curvature / scale),
        "diffusion": _clamp(diffusion / scale),
        "terrain_volatility": _clamp(terrain_volatility, -1, 1),
        "path_signed": _clamp(path_signed / scale),
        "path_absolute": _clamp(path_absolute / scale),
        "flow_inertia": _clamp(flow_inertia / scale),
        "flow_viscous": _clamp(diffusion / scale),
        "pressure_proxy": _clamp(pressure_proxy / scale),
        "advection": _clamp(advection / scale),
        "trend_fast": _clamp(trend_fast),
        "trend_slow": _clamp(trend_slow),
        "velocity_consistency": _clamp(velocity_consistency),
    }


def _prob_bucket(p_up: float) -> int:
    if p_up >= 0.85:
        return 4
    if p_up >= 0.70:
        return 3
    if p_up >= 0.60:
        return 2
    if p_up >= 0.50:
        return 1
    return 0


class NSFlowEngine(Engine):
    name = "ns_flow"

    def __init__(self):
        self.weights: Dict[str, float] = {}
        self.bias: float = 0.0
        self.loaded = False
        self.load_error: Optional[str] = None
        self._load_weights()

    def _load_weights(self):
        raw = os.environ.get("FLOW_WEIGHTS_JSON", "").strip()
        if not raw:
            self.load_error = "FLOW_WEIGHTS_JSON not set"
            return
        try:
            data = json.loads(raw)
            weights = data.get("weights", {})
            bias = float(data.get("bias", 0.0))
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            self.load_error = f"invalid JSON: {e}"
            return
        missing = [n for n in FEATURE_NAMES if n not in weights]
        if missing:
            self.load_error = f"missing weights: {missing}"
            return
        self.weights = {n: float(weights[n]) for n in FEATURE_NAMES}
        self.bias = bias
        self.loaded = True

    def _score(self, features):
        z = self.bias
        for n in FEATURE_NAMES:
            z += self.weights[n] * features.get(n, 0.0)
        return z

    def run(self, ctx):
        raise NotImplementedError("NSFlowEngine is async; use run_async()")

    async def run_async(self, ctx: EngineContext) -> EngineOutput:
        if not self.loaded:
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature=None,
                raw_state={"reason": "weights not loaded", "error": self.load_error},
            )
        if not ctx.recent_ticks or len(ctx.recent_ticks) < MIN_TICKS:
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature=None,
                raw_state={"reason": "insufficient ticks"},
            )

        features = _build_features(ctx.recent_ticks, WINDOW)
        if features is None:
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature=None,
                raw_state={"reason": "feature extraction failed"},
            )

        z = self._score(features)
        p_up = _sigmoid(z)
        p_down = 1.0 - p_up

        if p_up >= 0.5:
            direction = "UP"
            selected_prob = p_up
        else:
            direction = "DOWN"
            selected_prob = p_down

        bucket = _prob_bucket(selected_prob)
        sig = f"nsflow:{direction.lower()}:p{bucket}"

        raw = {
            "p_up": p_up,
            "p_down": p_down,
            "z": z,
            "selected_probability": selected_prob,
            "probability_bucket": bucket,
            "velocity_consistency": features.get("velocity_consistency"),
            "features": features,
        }

        if selected_prob < SIGNAL_THRESHOLD:
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                confidence=None,
                pattern_signature=sig,
                raw_state=raw,
            )

        return EngineOutput(
            engine=self.name,
            direction=direction,
            confidence=selected_prob,
            pattern_signature=sig,
            raw_state=raw,
        )
