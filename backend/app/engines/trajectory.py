"""Trajectory Engine.

Trajectory-centric multi-hypothesis engine.

The primitive object is the tick path, not the candle. Each round:
1. Reconstruct the trajectory as a sequence of (t, p) points
2. Compute geometric state (velocity, acceleration, curvature, turning)
3. Fit 12 mathematical model families
4. Determine which family best explains the path
5. Classify terrain (calm / normal / volatile)
6. Classify shape (LINEAR / CURVE / WIGGLE / ZIGZAG / LOOP)
7. Signature: traj:{shape}_{best_model}_{terrain}_dw{dwell}

State space: 5 x 12 x 3 x 4 = 720.
Direction learned from history via shared pattern memory.

The full hypothesis vector (rms of all 12 families) is stored in
raw_state for future meta-learning.
"""
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._lookup import lookup_direction
from ._temporal import dwell_bin
from .base import Engine, EngineContext, EngineOutput


WINDOW_TICKS = 30
MIN_TICKS = 12
DWELL_MAX_BACK = 10
TERRAIN_CALM = 0.00005
TERRAIN_VOLATILE = 0.00020
DIRECTION_CHANGE_THRESHOLD = 0.005


# ============================================================
# Trajectory construction
# ============================================================

def _build_curve(ticks: List[Tuple[float, float]]) -> Optional[Dict[str, np.ndarray]]:
    if len(ticks) < MIN_TICKS:
        return None
    window = ticks[-WINDOW_TICKS:]
    times = np.array([t for t, _ in window], dtype=np.float64)
    prices = np.array([p for _, p in window], dtype=np.float64)

    times_norm = times - times[0]
    span_t = max(times_norm[-1], 1e-9)
    x = times_norm / span_t

    p0 = prices[0]
    p_span = max(abs(prices - p0).max(), 1e-9)
    y = (prices - p0) / p_span

    velocities = np.zeros_like(prices)
    for i in range(1, len(prices)):
        dt = times[i] - times[i - 1]
        if dt > 1e-9:
            velocities[i] = (prices[i] - prices[i - 1]) / dt

    accelerations = np.zeros_like(prices)
    for i in range(2, len(prices)):
        dt = times[i] - times[i - 1]
        if dt > 1e-9:
            accelerations[i] = (velocities[i] - velocities[i - 1]) / dt

    return {
        "x": x,
        "y": y,
        "prices": prices,
        "times": times,
        "velocities": velocities,
        "accelerations": accelerations,
    }


# ============================================================
# Geometric features
# ============================================================

def _arc_length(xs, ys):
    dx = np.diff(xs)
    dy = np.diff(ys)
    return float(np.sqrt(dx * dx + dy * dy).sum())


def _endpoint_distance(xs, ys):
    dx = xs[-1] - xs[0]
    dy = ys[-1] - ys[0]
    return float(math.sqrt(dx * dx + dy * dy))


def _curvature_profile(xs, ys):
    if len(xs) < 3:
        return np.array([])
    d1x = xs[1:-1] - xs[:-2]
    d1y = ys[1:-1] - ys[:-2]
    d2x = xs[2:] - xs[1:-1]
    d2y = ys[2:] - ys[1:-1]
    cross = d1x * d2y - d1y * d2x
    m1 = np.sqrt(d1x * d1x + d1y * d1y)
    m2 = np.sqrt(d2x * d2x + d2y * d2y)
    denom = m1 * m2 * (m1 + m2)
    denom = np.where(denom < 1e-12, 1e-12, denom)
    return 2.0 * cross / denom


def _direction_changes(ys, threshold=DIRECTION_CHANGE_THRESHOLD):
    count = 0
    prev = 0
    for i in range(1, len(ys)):
        d = ys[i] - ys[i - 1]
        if abs(d) < threshold:
            continue
        s = 1 if d > 0 else -1
        if prev != 0 and s != prev:
            count += 1
        prev = s
    return count


def _classify_shape(tortuosity, turns):
    if tortuosity > 2.0:
        return "LOOP"
    if turns >= 6:
        return "ZIGZAG"
    if turns >= 4 and tortuosity > 1.4:
        return "WIGGLE"
    if tortuosity < 1.15:
        return "LINEAR"
    return "CURVE"


# ============================================================
# Formula / hypothesis registry
# ============================================================
#
# Each function takes (xs, ys) normalized arrays and returns
# (rms_residual, fitted_values) or (inf, None) if unfit.
# Model names are stable - they are part of the signature.
# Add new families here; the state space grows accordingly.

def _fit_linear(xs, ys):
    if len(xs) < 3:
        return float("inf"), None
    try:
        c = np.polyfit(xs, ys, 1)
        pred = np.polyval(c, xs)
        return float(np.sqrt(np.mean((ys - pred) ** 2))), pred
    except Exception:
        return float("inf"), None


def _fit_quadratic(xs, ys):
    if len(xs) < 4:
        return float("inf"), None
    try:
        c = np.polyfit(xs, ys, 2)
        pred = np.polyval(c, xs)
        return float(np.sqrt(np.mean((ys - pred) ** 2))), pred
    except Exception:
        return float("inf"), None


def _fit_cubic(xs, ys):
    if len(xs) < 5:
        return float("inf"), None
    try:
        c = np.polyfit(xs, ys, 3)
        pred = np.polyval(c, xs)
        return float(np.sqrt(np.mean((ys - pred) ** 2))), pred
    except Exception:
        return float("inf"), None


def _fit_quartic(xs, ys):
    if len(xs) < 6:
        return float("inf"), None
    try:
        c = np.polyfit(xs, ys, 4)
        pred = np.polyval(c, xs)
        return float(np.sqrt(np.mean((ys - pred) ** 2))), pred
    except Exception:
        return float("inf"), None


def _fit_spline(xs, ys):
    """Simple 3-knot cubic spline over thirds of the window."""
    if len(xs) < 6:
        return float("inf"), None
    try:
        n = len(xs)
        i1 = n // 3
        i2 = 2 * n // 3
        segments = [(xs[: i1 + 1], ys[: i1 + 1]),
                    (xs[i1: i2 + 1], ys[i1: i2 + 1]),
                    (xs[i2:], ys[i2:])]
        pred = np.zeros_like(ys)
        for seg_x, seg_y in segments:
            if len(seg_x) < 2:
                return float("inf"), None
            c = np.polyfit(seg_x, seg_y, 2)
            pred[seg_x.index.values if hasattr(seg_x, "index") else 0:0] = 0
        # simpler: fit each segment and place predictions by index
        pred = np.zeros_like(ys)
        for start, (seg_x, seg_y) in zip([0, i1, i2], segments):
            if len(seg_x) < 2:
                return float("inf"), None
            c = np.polyfit(seg_x, seg_y, 2)
            pred[start:start + len(seg_x)] = np.polyval(c, seg_x)
        return float(np.sqrt(np.mean((ys - pred) ** 2))), pred
    except Exception:
        return float("inf"), None


def _fit_fourier_1(xs, ys):
    """Single harmonic: a + b*sin(2*pi*f*x + phi). Grid-search f."""
    if len(xs) < 6:
        return float("inf"), None
    try:
        best_rms = float("inf")
        best_pred = None
        for k in range(1, 6):
            f = float(k)
            omega = 2.0 * math.pi * f
            design = np.stack([np.ones_like(xs),
                               np.sin(omega * xs),
                               np.cos(omega * xs)], axis=1)
            coeffs, *_ = np.linalg.lstsq(design, ys, rcond=None)
            pred = design @ coeffs
            rms = float(np.sqrt(np.mean((ys - pred) ** 2)))
            if rms < best_rms:
                best_rms = rms
                best_pred = pred
        return best_rms, best_pred
    except Exception:
        return float("inf"), None


def _fit_fourier_2(xs, ys):
    """Two harmonics."""
    if len(xs) < 8:
        return float("inf"), None
    try:
        best_rms = float("inf")
        best_pred = None
        for k1 in range(1, 4):
            for k2 in range(k1 + 1, 6):
                w1 = 2.0 * math.pi * float(k1)
                w2 = 2.0 * math.pi * float(k2)
                design = np.stack([
                    np.ones_like(xs),
                    np.sin(w1 * xs), np.cos(w1 * xs),
                    np.sin(w2 * xs), np.cos(w2 * xs),
                ], axis=1)
                coeffs, *_ = np.linalg.lstsq(design, ys, rcond=None)
                pred = design @ coeffs
                rms = float(np.sqrt(np.mean((ys - pred) ** 2)))
                if rms < best_rms:
                    best_rms = rms
                    best_pred = pred
        return best_rms, best_pred
    except Exception:
        return float("inf"), None


def _fit_ar3(xs, ys):
    """AR(3): y_t = c + phi_1*y_{t-1} + phi_2*y_{t-2} + phi_3*y_{t-3}."""
    if len(ys) < 6:
        return float("inf"), None
    try:
        order = 3
        n = len(ys) - order
        design = np.zeros((n, order + 1))
        for i in range(n):
            design[i, 0] = 1.0
            for j in range(order):
                design[i, j + 1] = ys[i + order - 1 - j]
        target = ys[order:]
        coeffs, *_ = np.linalg.lstsq(design, target, rcond=None)
        pred_tail = design @ coeffs
        pred = np.concatenate([ys[:order], pred_tail])
        return float(np.sqrt(np.mean((ys - pred) ** 2))), pred
    except Exception:
        return float("inf"), None


def _fit_ar5(xs, ys):
    """AR(5)."""
    if len(ys) < 8:
        return float("inf"), None
    try:
        order = 5
        n = len(ys) - order
        design = np.zeros((n, order + 1))
        for i in range(n):
            design[i, 0] = 1.0
            for j in range(order):
                design[i, j + 1] = ys[i + order - 1 - j]
        target = ys[order:]
        coeffs, *_ = np.linalg.lstsq(design, target, rcond=None)
        pred_tail = design @ coeffs
        pred = np.concatenate([ys[:order], pred_tail])
        return float(np.sqrt(np.mean((ys - pred) ** 2))), pred
    except Exception:
        return float("inf"), None


def _fit_exponential(xs, ys):
    if len(xs) < 4:
        return float("inf"), None
    if not (np.all(ys > 0) or np.all(ys < 0)):
        return float("inf"), None
    try:
        sgn = 1.0 if ys[0] > 0 else -1.0
        ly = np.log(np.abs(ys))
        b, log_a = np.polyfit(xs, ly, 1)
        a = sgn * float(np.exp(log_a))
        pred = a * np.exp(b * xs)
        return float(np.sqrt(np.mean((ys - pred) ** 2))), pred
    except Exception:
        return float("inf"), None


def _fit_logarithmic(xs, ys):
    mask = xs > 1e-4
    if mask.sum() < 4:
        return float("inf"), None
    try:
        lx = np.log(xs[mask])
        c = np.polyfit(lx, ys[mask], 1)
        pred_masked = np.polyval(c, lx)
        pred = np.zeros_like(ys)
        pred[mask] = pred_masked
        return float(np.sqrt(np.mean((ys[mask] - pred_masked) ** 2))), pred
    except Exception:
        return float("inf"), None


def _fit_power(xs, ys):
    mask = (np.abs(xs) > 1e-4) & (np.abs(ys) > 1e-6)
    if mask.sum() < 4:
        return float("inf"), None
    try:
        sign_match = np.sign(xs[mask]) == np.sign(ys[mask])
        if sign_match.sum() < 4:
            return float("inf"), None
        lx = np.log(np.abs(xs[mask])[sign_match])
        ly = np.log(np.abs(ys[mask])[sign_match])
        b, log_a = np.polyfit(lx, ly, 1)
        a = float(np.exp(log_a))
        pred = a * np.sign(xs) * np.abs(xs) ** b
        return float(np.sqrt(np.mean((ys - pred) ** 2))), pred
    except Exception:
        return float("inf"), None


# The registry. Adding entries here grows the state space.
# Keep model names short (they appear in the signature).
MODEL_REGISTRY = [
    ("lin", _fit_linear),
    ("quad", _fit_quadratic),
    ("cubic", _fit_cubic),
    ("quart", _fit_quartic),
    ("spline", _fit_spline),
    ("four1", _fit_fourier_1),
    ("four2", _fit_fourier_2),
    ("ar3", _fit_ar3),
    ("ar5", _fit_ar5),
    ("exp", _fit_exponential),
    ("log", _fit_logarithmic),
    ("pow", _fit_power),
]


def _fit_all(xs, ys):
    results = {}
    for name, fn in MODEL_REGISTRY:
        try:
            rms, _ = fn(xs, ys)
        except Exception:
            rms = float("inf")
        results[name] = rms
    best = min(results.items(), key=lambda kv: kv[1])
    return best[0], best[1], results


# ============================================================
# Terrain classification
# ============================================================

def _classify_terrain(ticks):
    prices = [p for _, p in ticks[-WINDOW_TICKS:]]
    returns = []
    for i in range(1, len(prices)):
        if prices[i - 1] > 0:
            returns.append((prices[i] - prices[i - 1]) / prices[i - 1])
    if not returns:
        return "normal", 0.0
    m = sum(returns) / len(returns)
    var = sum((r - m) ** 2 for r in returns) / max(len(returns) - 1, 1)
    vol = math.sqrt(var)
    if vol < TERRAIN_CALM:
        return "calm", vol
    if vol < TERRAIN_VOLATILE:
        return "normal", vol
    return "volatile", vol


# ============================================================
# Signature
# ============================================================

def _core_signature(ticks):
    curve = _build_curve(ticks)
    if curve is None:
        return None
    xs = curve["x"]
    ys = curve["y"]

    arc = _arc_length(xs, ys)
    end = _endpoint_distance(xs, ys)
    tort = arc / end if end > 1e-9 else 1.0
    turns = _direction_changes(ys)
    shape = _classify_shape(tort, turns)
    terrain, vol = _classify_terrain(ticks)
    best_model, best_rms, all_rms = _fit_all(xs, ys)

    sig = "traj:" + shape + "_" + best_model + "_" + terrain

    features = {
        "shape": shape,
        "best_model": best_model,
        "best_rms": best_rms,
        "terrain": terrain,
        "volatility": vol,
        "tortuosity": tort,
        "turns": turns,
        "arc_length": arc,
        "endpoint_distance": end,
        "hypothesis_vector": all_rms,
    }
    return sig, features


# ============================================================
# Engine
# ============================================================

class TrajectoryEngine(Engine):
    name = "trajectory"

    def run(self, ctx):
        raise NotImplementedError("TrajectoryEngine is async; use run_async()")

    async def run_async(self, ctx: EngineContext) -> EngineOutput:
        if not ctx.recent_ticks or len(ctx.recent_ticks) < MIN_TICKS:
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature=None,
                raw_state={"reason": "insufficient ticks"},
            )

        result = _core_signature(ctx.recent_ticks)
        if result is None:
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature=None,
                raw_state={"reason": "curve build failed"},
            )
        core_sig, features = result

        dwell_s = 0.0
        now_ts = ctx.recent_ticks[-1][0]
        for back in range(1, DWELL_MAX_BACK + 1):
            sub = ctx.recent_ticks[:-back]
            if len(sub) < MIN_TICKS:
                break
            sub_result = _core_signature(sub)
            if sub_result is None:
                break
            sub_sig, _ = sub_result
            if sub_sig != core_sig:
                dwell_s = now_ts - sub[-1][0]
                break

        full_sig = core_sig + "_dw" + str(dwell_bin(dwell_s))

        raw = dict(features)
        raw["dwell_s"] = dwell_s
        raw["n_points"] = min(len(ctx.recent_ticks), WINDOW_TICKS)

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
