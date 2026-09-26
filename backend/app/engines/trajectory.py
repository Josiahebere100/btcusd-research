"""Trajectory Engine v6.1 (FROZEN).

Multi-hypothesis trajectory engine.

Freeze date: 2026-09-25
Evaluation end: 2026-10-25

Architecture:
- Trajectory is the primitive object: sequence of (t, p) ticks
- Two representations: geometric (normalized) and physical (raw)
- 12 mathematical model families
- Leak-free rolling-origin validation (each origin rebuilds coordinates)
- Normalized complexity penalty (comparable to RMS)
- Weighted ensemble produces a 3-point future trajectory
- Direction derived from trajectory endpoint vs current price
- Structural signature only (no direction leakage)
- Architecture B decision layer:
    * Ensemble is primary
    * Pattern memory fallback when ensemble silent
    * Disagreement -> NO_SIGNAL
    * Pattern confidence GATES/SHRINKS ensemble strength:
        C_combined = C_E * (0.5 + 0.5 * C_P)

Stored per-prediction in raw_state:
- forecast_p0, forecast_span_p, forecast_span_t, forecast_x
- forecast_current_price (for internal direction label)
- per_model_forecast with values + direction + weight
- hypothesis_vector with fit_rms, forecast_rms, combined, direction, transform
- ensemble_agreement (vote concentration, NOT a probability)
- ensemble_strength (signal-to-disagreement, NOT a probability)

Deferred to v7+ (needs data):
- Persistent per-hypothesis weights
- Contextual weights per regime
- Meta-learner on hypothesis_vector
- Trajectory renderer
"""
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ._lookup import lookup_direction
from ._temporal import dwell_bin
from .base import Engine, EngineContext, EngineOutput


# ============================================================
# Configuration
# ============================================================

WINDOW_TICKS = 30
MIN_TICKS = 12
DWELL_MAX_BACK = 10
TERRAIN_CALM = 0.00005
TERRAIN_VOLATILE = 0.00020
DIRECTION_CHANGE_THRESHOLD = 0.005
TAIL_HOLDOUT_K = 3
ROLLING_ORIGIN_MIN_TRAIN = 12
ROLLING_ORIGIN_STEP = 3
FORECAST_HORIZON = 3
ENSEMBLE_MIN_MARGIN = 0.05
PATTERN_LAMBDA = 0.5
MODEL_SCORE_W_FORECAST = 0.65
MODEL_SCORE_W_FIT = 0.25
MODEL_SCORE_W_COMPLEXITY = 0.10
INF = float("inf")


# ============================================================
# Trajectory construction
# ============================================================

def _build_trajectory(ticks):
    if len(ticks) < MIN_TICKS:
        return None
    window = ticks[-WINDOW_TICKS:]
    times = np.array([t for t, _ in window], dtype=np.float64)
    prices = np.array([p for _, p in window], dtype=np.float64)

    t_rel = times - times[0]
    span_t = max(t_rel[-1], 1e-9)
    x_geo = t_rel / span_t

    p_rel = prices - prices[0]
    span_p = max(abs(p_rel).max(), 1e-9)
    y_geo = p_rel / span_p

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
        "x_geo": x_geo, "y_geo": y_geo,
        "times": times, "prices": prices,
        "velocities": velocities, "accelerations": accelerations,
        "span_t": span_t, "span_p": span_p,
    }


# ============================================================
# Geometric features
# ============================================================

def _menger_curvature(xs, ys):
    if len(xs) < 3:
        return np.array([])
    ax, ay = xs[:-2], ys[:-2]
    bx, by = xs[1:-1], ys[1:-1]
    cx, cy = xs[2:], ys[2:]
    ab_x, ab_y = bx - ax, by - ay
    bc_x, bc_y = cx - bx, cy - by
    ca_x, ca_y = cx - ax, cy - ay
    cross = ab_x * bc_y - ab_y * bc_x
    denom = np.sqrt(ab_x**2 + ab_y**2) * np.sqrt(bc_x**2 + bc_y**2) * np.sqrt(ca_x**2 + ca_y**2)
    denom = np.where(denom < 1e-12, 1e-12, denom)
    return 2.0 * cross / denom


def _arc_length(xs, ys):
    dx, dy = np.diff(xs), np.diff(ys)
    return float(np.sqrt(dx * dx + dy * dy).sum())


def _endpoint_distance(xs, ys):
    dx, dy = xs[-1] - xs[0], ys[-1] - ys[0]
    return float(math.sqrt(dx * dx + dy * dy))


def _direction_changes(ys, threshold=DIRECTION_CHANGE_THRESHOLD):
    count, prev = 0, 0
    for i in range(1, len(ys)):
        d = ys[i] - ys[i - 1]
        if abs(d) < threshold:
            continue
        s = 1 if d > 0 else -1
        if prev != 0 and s != prev:
            count += 1
        prev = s
    return count


def _classify_shape(tortuosity, turns, mean_abs_kappa):
    if turns >= 6:
        return "ZIGZAG"
    if turns >= 4 and tortuosity > 1.4:
        return "WIGGLE"
    if mean_abs_kappa > 0.4:
        return "OSCILLATORY"
    if tortuosity < 1.15:
        return "LINEAR"
    if tortuosity > 1.8:
        return "REVERSAL"
    return "CURVE"


def _classify_terrain(prices):
    rets = []
    for i in range(1, len(prices)):
        if prices[i - 1] > 0:
            rets.append((prices[i] - prices[i - 1]) / prices[i - 1])
    if not rets:
        return "normal", 0.0
    m = sum(rets) / len(rets)
    var = sum((r - m) ** 2 for r in rets) / max(len(rets) - 1, 1)
    vol = math.sqrt(var)
    if vol < TERRAIN_CALM:
        return "calm", vol
    if vol < TERRAIN_VOLATILE:
        return "normal", vol
    return "volatile", vol


# ============================================================
# Model fitters
# ============================================================

def _fit_linear(xs, ys):
    if len(xs) < 3:
        return None, INF
    c = np.polyfit(xs, ys, 1)
    return lambda x: np.polyval(c, x), float(np.sqrt(np.mean((ys - np.polyval(c, xs)) ** 2)))


def _fit_quadratic(xs, ys):
    if len(xs) < 4:
        return None, INF
    c = np.polyfit(xs, ys, 2)
    return lambda x: np.polyval(c, x), float(np.sqrt(np.mean((ys - np.polyval(c, xs)) ** 2)))


def _fit_cubic(xs, ys):
    if len(xs) < 5:
        return None, INF
    c = np.polyfit(xs, ys, 3)
    return lambda x: np.polyval(c, x), float(np.sqrt(np.mean((ys - np.polyval(c, xs)) ** 2)))


def _fit_quartic(xs, ys):
    if len(xs) < 6:
        return None, INF
    c = np.polyfit(xs, ys, 4)
    return lambda x: np.polyval(c, x), float(np.sqrt(np.mean((ys - np.polyval(c, xs)) ** 2)))


def _fit_piecewise_quad(xs, ys):
    if len(xs) < 6:
        return None, INF
    n = len(xs)
    cuts = [0, n // 3, 2 * n // 3, n]
    segs = []
    for k in range(3):
        a, b = cuts[k], cuts[k + 1]
        sx, sy = xs[a:b], ys[a:b]
        if len(sx) < 3:
            return None, INF
        coeffs = np.polyfit(sx, sy, 2)
        segs.append((sx[0], sx[-1], coeffs))

    def predict(x):
        x = np.atleast_1d(x)
        out = np.zeros_like(x, dtype=np.float64)
        for i in range(len(x)):
            xi = x[i]
            placed = False
            for lo, hi, coeffs in segs:
                if lo <= xi <= hi:
                    out[i] = np.polyval(coeffs, xi)
                    placed = True
                    break
            if not placed:
                if xi < segs[0][0]:
                    out[i] = np.polyval(segs[0][2], xi)
                else:
                    out[i] = np.polyval(segs[-1][2], xi)
        return out
    return predict, float(np.sqrt(np.mean((ys - predict(xs)) ** 2)))


def _fit_fourier_1(xs, ys):
    if len(xs) < 6:
        return None, INF
    best_rms, best_pred = INF, None
    for k in range(1, 6):
        omega = 2.0 * math.pi * float(k)
        design = np.stack([np.ones_like(xs), np.sin(omega * xs), np.cos(omega * xs)], axis=1)
        coeffs, *_ = np.linalg.lstsq(design, ys, rcond=None)
        pred = design @ coeffs
        rms = float(np.sqrt(np.mean((ys - pred) ** 2)))
        if rms < best_rms:
            best_rms, best_pred = rms, (omega, coeffs)
    if best_pred is None:
        return None, INF
    omega, coeffs = best_pred
    return lambda x: np.stack([np.ones_like(x), np.sin(omega * x), np.cos(omega * x)], axis=1) @ coeffs, best_rms


def _fit_fourier_2(xs, ys):
    if len(xs) < 8:
        return None, INF
    best_rms, best_pred = INF, None
    for k1 in range(1, 4):
        for k2 in range(k1 + 1, 6):
            w1, w2 = 2.0 * math.pi * float(k1), 2.0 * math.pi * float(k2)
            design = np.stack([np.ones_like(xs),
                               np.sin(w1*xs), np.cos(w1*xs),
                               np.sin(w2*xs), np.cos(w2*xs)], axis=1)
            coeffs, *_ = np.linalg.lstsq(design, ys, rcond=None)
            pred = design @ coeffs
            rms = float(np.sqrt(np.mean((ys - pred) ** 2)))
            if rms < best_rms:
                best_rms, best_pred = rms, (w1, w2, coeffs)
    if best_pred is None:
        return None, INF
    w1, w2, coeffs = best_pred
    def predict(x):
        return np.stack([np.ones_like(x),
                         np.sin(w1*x), np.cos(w1*x),
                         np.sin(w2*x), np.cos(w2*x)], axis=1) @ coeffs
    return predict, best_rms


def _fit_ar(xs, ys, order):
    if len(ys) < order + 3:
        return None, INF
    n = len(ys) - order
    design = np.zeros((n, order + 1))
    for i in range(n):
        design[i, 0] = 1.0
        for j in range(order):
            design[i, j + 1] = ys[i + order - 1 - j]
    target = ys[order:]
    coeffs, *_ = np.linalg.lstsq(design, target, rcond=None)
    pred_tail = design @ coeffs
    pred_full = np.concatenate([ys[:order], pred_tail])
    rms = float(np.sqrt(np.mean((ys - pred_full) ** 2)))
    def predict(x):
        history = list(ys[-order:])
        out = []
        for _ in range(len(x)):
            row = np.concatenate([[1.0], list(reversed(history[-order:]))])
            y_next = float(row @ coeffs)
            out.append(y_next)
            history.append(y_next)
        return np.array(out)
    return predict, rms


def _fit_exponential(xs, ys):
    if len(xs) < 4:
        return None, INF
    try:
        y_shift = ys - ys.min() + 0.1
        b, log_a = np.polyfit(xs, np.log(y_shift), 1)
        a = float(np.exp(log_a))
        def predict(x):
            return a * np.exp(b * np.asarray(x)) + ys.min() - 0.1
        return predict, float(np.sqrt(np.mean((ys - predict(xs)) ** 2)))
    except Exception:
        return None, INF


def _fit_logarithmic(xs, ys):
    if len(xs) < 4:
        return None, INF
    x_shift = xs + 0.01
    c = np.polyfit(np.log(x_shift), ys, 1)
    def predict(x):
        return np.polyval(c, np.log(np.asarray(x) + 0.01))
    return predict, float(np.sqrt(np.mean((ys - predict(xs)) ** 2)))


def _fit_power(xs, ys):
    if len(xs) < 4:
        return None, INF
    try:
        x_shift = xs + 0.01
        y_shift = ys - ys.min() + 0.1
        b, log_a = np.polyfit(np.log(x_shift), np.log(y_shift), 1)
        a = float(np.exp(log_a))
        def predict(x):
            return a * ((np.asarray(x) + 0.01) ** b) + ys.min() - 0.1
        return predict, float(np.sqrt(np.mean((ys - predict(xs)) ** 2)))
    except Exception:
        return None, INF


# (short_name, fitter, complexity, transform_description)
MODEL_REGISTRY = [
    ("lin", _fit_linear, 1.0, "y on x"),
    ("quad", _fit_quadratic, 1.2, "y on x"),
    ("cubic", _fit_cubic, 1.4, "y on x"),
    ("quart", _fit_quartic, 1.6, "y on x"),
    ("pieceq", _fit_piecewise_quad, 1.5, "3 x quadratic segments"),
    ("four1", _fit_fourier_1, 1.8, "single harmonic, best k in 1..5"),
    ("four2", _fit_fourier_2, 2.0, "dual harmonic, best pair in 1..5"),
    ("ar3", lambda x, y: _fit_ar(x, y, 3), 1.7, "AR(3) on y sequence"),
    ("ar5", lambda x, y: _fit_ar(x, y, 5), 1.9, "AR(5) on y sequence"),
    ("exp", _fit_exponential, 1.3, "y' = y - min(y) + 0.1"),
    ("log", _fit_logarithmic, 1.3, "x' = x + 0.01"),
    ("pow", _fit_power, 1.3, "x' = x + 0.01, y' = y - min(y) + 0.1"),
]


def _registry_iter():
    for entry in MODEL_REGISTRY:
        yield (entry[0], entry[1], entry[2])


# ============================================================
# Leak-free rolling-origin validation
# ============================================================

def _single_origin_error(times, prices, fitter, origin, k):
    """Fit on [0, origin). Forecast [origin, origin+k). No leakage.

    For origin `o`, coordinate system uses only times[:o] and prices[:o].
    Validation points are transformed with the same training-derived scales.
    """
    t_train = times[:origin]
    p_train = prices[:origin]
    if len(t_train) < 3:
        return INF

    span_t = max(t_train[-1] - t_train[0], 1e-9)
    p0 = p_train[0]
    span_p = max(np.abs(p_train - p0).max(), 1e-9)

    xs_train = (t_train - t_train[0]) / span_t
    ys_train = (p_train - p0) / span_p

    t_val = times[origin:origin+k]
    p_val = prices[origin:origin+k]
    if len(t_val) != k:
        return INF

    xs_val = (t_val - t_train[0]) / span_t
    ys_val = (p_val - p0) / span_p

    try:
        pred, _ = fitter(xs_train, ys_train)
        if pred is None:
            return INF
        y_hat = pred(xs_val)
        if len(y_hat) != k:
            return INF
        return float(np.sqrt(np.mean((ys_val - y_hat) ** 2)))
    except Exception:
        return INF


def _forecast_error(times, prices, fitter, k, min_train, step):
    n = len(times)
    errors = []
    if n >= min_train + k:
        origin = min_train
        while origin + k <= n:
            err = _single_origin_error(times, prices, fitter, origin, k)
            if err < INF:
                errors.append(err)
            origin += step
    if errors:
        return float(np.mean(errors))
    if n >= k + 3:
        return _single_origin_error(times, prices, fitter, n - k, k)
    return INF


# ============================================================
# Hypothesis evaluation
# ============================================================

@dataclass
class Hypothesis:
    name: str
    fit_rms: float
    forecast_rms: float
    complexity: float
    combined: float
    predict_fn: Optional[Callable] = None
    forecast_values: List[float] = field(default_factory=list)
    forecast_direction: Optional[str] = None


def _evaluate_one(name, fitter, complexity, times, prices):
    n = len(times)
    if n < 3:
        return Hypothesis(name, INF, INF, complexity, INF)

    span_t = max(times[-1] - times[0], 1e-9)
    p0 = prices[0]
    span_p = max(np.abs(prices - p0).max(), 1e-9)
    xs_full = (times - times[0]) / span_t
    ys_full = (prices - p0) / span_p

    try:
        predict_fn, fit_rms = fitter(xs_full, ys_full)
    except Exception:
        return Hypothesis(name, INF, INF, complexity, INF)
    if predict_fn is None:
        return Hypothesis(name, INF, INF, complexity, INF)

    forecast_rms = _forecast_error(times, prices, fitter, TAIL_HOLDOUT_K,
                                    ROLLING_ORIGIN_MIN_TRAIN, ROLLING_ORIGIN_STEP)

    fr = fit_rms if fit_rms < INF else 1.0
    pr = forecast_rms if forecast_rms < INF else 1.0

    c_values = [c for (_, _, c) in _registry_iter()]
    c_min, c_max = min(c_values), max(c_values)
    c_range = max(c_max - c_min, 1e-9)
    c_norm = (complexity - c_min) / c_range

    combined = (MODEL_SCORE_W_FORECAST * pr
                + MODEL_SCORE_W_FIT * fr
                + MODEL_SCORE_W_COMPLEXITY * c_norm)
    return Hypothesis(name, fit_rms, forecast_rms, complexity, combined, predict_fn)


def _evaluate_all(times, prices):
    return [_evaluate_one(n, f, c, times, prices) for (n, f, c) in _registry_iter()]


# ============================================================
# Ensemble trajectory
# ============================================================

def _future_x_values(xs, horizon):
    if len(xs) < 2:
        return np.array([xs[-1] + 0.01 * (i + 1) for i in range(horizon)])
    step = xs[-1] - xs[-2]
    if step <= 0:
        step = 0.01
    return np.array([xs[-1] + step * (i + 1) for i in range(horizon)])


def _ensemble_trajectory(hypotheses, xs, ys, horizon):
    valid = [h for h in hypotheses if h.predict_fn is not None and h.combined < INF]
    if not valid:
        return [], None, 0.0, 0.0, {}, {}

    weights = {h.name: 1.0 / (h.combined + 1e-6) for h in valid}
    total_w = sum(weights.values())
    if total_w <= 0:
        return [], None, 0.0, 0.0, {}, {}
    weight_map = {k: v / total_w for k, v in weights.items()}

    future_x = _future_x_values(xs, horizon)
    y_current = float(ys[-1])

    weighted_sum = np.zeros(horizon, dtype=np.float64)
    weight_applied = 0.0
    up_weight = 0.0
    down_weight = 0.0
    per_model = {}
    endpoints = []

    for h in valid:
        try:
            y_hat = np.asarray(h.predict_fn(future_x), dtype=np.float64).flatten()
            if len(y_hat) < horizon:
                continue
            y_hat = y_hat[:horizon]
            w = weight_map[h.name]
            weighted_sum += w * y_hat
            weight_applied += w

            delta_h = float(y_hat[-1]) - y_current
            if delta_h > 0:
                h.forecast_direction = "UP"
                up_weight += w
            elif delta_h < 0:
                h.forecast_direction = "DOWN"
                down_weight += w
            else:
                h.forecast_direction = "FLAT"

            h.forecast_values = [float(v) for v in y_hat]
            endpoints.append((float(y_hat[-1]), w))
            per_model[h.name] = {
                "values": h.forecast_values,
                "delta": delta_h,
                "direction": h.forecast_direction,
                "weight": w,
                "horizon": horizon,
            }
        except Exception:
            continue

    if weight_applied <= 0:
        return [], None, 0.0, 0.0, weight_map, per_model

    trajectory = list(weighted_sum / weight_applied)
    delta_ensemble = trajectory[-1] - y_current

    if delta_ensemble > 0:
        direction = "UP"
    elif delta_ensemble < 0:
        direction = "DOWN"
    else:
        direction = None

    total_vote = up_weight + down_weight
    agreement = (max(up_weight, down_weight) / total_vote) if total_vote > 0 else 0.0

    ep_total_w = sum(w for _, w in endpoints)
    if ep_total_w > 0:
        y_bar = sum(v * w for v, w in endpoints) / ep_total_w
        dispersion = math.sqrt(sum(w * (v - y_bar) ** 2 for v, w in endpoints) / ep_total_w)
    else:
        y_bar = y_current
        dispersion = 0.0

    if dispersion < 1e-6:
        strength = 1.0 if abs(y_bar - y_current) > 1e-4 else 0.0
    else:
        strength = min(1.0, abs(y_bar - y_current) / (3.0 * dispersion))

    return trajectory, direction, agreement, strength, weight_map, per_model


# ============================================================
# Signature
# ============================================================

def _core_signature(ticks):
    traj = _build_trajectory(ticks)
    if traj is None:
        return None

    xs, ys = traj["x_geo"], traj["y_geo"]
    prices = traj["prices"]
    times = traj["times"]

    arc = _arc_length(xs, ys)
    end = _endpoint_distance(xs, ys)
    tort = arc / end if end > 1e-9 else 1.0

    curvature = _menger_curvature(xs, ys)
    mean_abs_kappa = float(np.mean(np.abs(curvature))) if len(curvature) else 0.0
    max_abs_kappa = float(np.max(np.abs(curvature))) if len(curvature) else 0.0

    turns = _direction_changes(ys)
    shape = _classify_shape(tort, turns, mean_abs_kappa)
    terrain, vol = _classify_terrain(prices)

    hypotheses = _evaluate_all(times, prices)
    valid = [h for h in hypotheses if h.combined < INF]
    best = min(valid, key=lambda h: h.combined) if valid else None
    best_name = best.name if best else "none"

    (trajectory, ensemble_dir, agreement, strength,
     weight_map, per_model) = _ensemble_trajectory(hypotheses, xs, ys, FORECAST_HORIZON)

    sig = "traj:" + shape + "_" + best_name + "_" + terrain

    transform_map = {e[0]: e[3] for e in MODEL_REGISTRY}
    hypothesis_vector = {
        h.name: {
            "fit_rms": (None if h.fit_rms == INF else float(h.fit_rms)),
            "forecast_rms": (None if h.forecast_rms == INF else float(h.forecast_rms)),
            "combined": (None if h.combined == INF else float(h.combined)),
            "direction": h.forecast_direction,
            "transform": transform_map.get(h.name, ""),
        }
        for h in hypotheses
    }

    future_x = _future_x_values(xs, FORECAST_HORIZON)

    features = {
        "shape": shape,
        "best_model": best_name,
        "terrain": terrain,
        "volatility": vol,
        "tortuosity": tort,
        "turns": turns,
        "arc_length": arc,
        "endpoint_distance": end,
        "mean_abs_curvature": mean_abs_kappa,
        "max_abs_curvature": max_abs_kappa,
        "ensemble_trajectory": trajectory,
        "ensemble_direction": ensemble_dir,
        "ensemble_agreement": agreement,
        "ensemble_strength": strength,
        "hypothesis_vector": hypothesis_vector,
        "ensemble_weights": weight_map,
        "per_model_forecast": per_model,
        "n_points": len(xs),
        "span_t_seconds": float(traj["span_t"]),
        "span_p_dollars": float(traj["span_p"]),
        "forecast_p0": float(traj["prices"][0]),
        "forecast_span_p": float(traj["span_p"]),
        "forecast_span_t": float(traj["span_t"]),
        "forecast_x": [float(v) for v in future_x],
        "forecast_current_price": float(traj["prices"][-1]),
    }
    return sig, features, ensemble_dir, strength


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
                raw_state={"reason": "trajectory build failed"},
            )
        core_sig, features, ensemble_dir, strength = result

        dwell_s = 0.0
        now_ts = ctx.recent_ticks[-1][0]
        for back in range(1, DWELL_MAX_BACK + 1):
            sub = ctx.recent_ticks[:-back]
            if len(sub) < MIN_TICKS:
                break
            sub_result = _core_signature(sub)
            if sub_result is None:
                break
            if sub_result[0] != core_sig:
                dwell_s = now_ts - sub[-1][0]
                break

        full_sig = core_sig + "_dw" + str(dwell_bin(dwell_s))

        raw = dict(features)
        raw["dwell_s"] = dwell_s

        pattern_lookup = None
        try:
            pattern_lookup = await lookup_direction(full_sig)
        except Exception as e:
            raw["lookup_error"] = str(e)

        if ensemble_dir is None:
            if pattern_lookup is not None:
                p_dir, p_occ, p_conf = pattern_lookup
                return EngineOutput(
                    engine=self.name,
                    direction=p_dir,
                    confidence=p_conf,
                    pattern_signature=full_sig,
                    raw_state={
                        **raw,
                        "decision_source": "pattern_fallback",
                        "lookup": {"occurrences": p_occ,
                                   "confidence": p_conf,
                                   "chosen_direction": p_dir},
                    },
                )
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature=full_sig,
                raw_state={**raw, "decision_source": "both_silent"},
            )

        if pattern_lookup is None:
            return EngineOutput(
                engine=self.name,
                direction=ensemble_dir,
                confidence=strength,
                pattern_signature=full_sig,
                raw_state={
                    **raw,
                    "decision_source": "ensemble_only",
                    "lookup": "insufficient_history",
                },
            )

        p_dir, p_occ, p_conf = pattern_lookup

        if p_dir == ensemble_dir:
            combined_conf = strength * (PATTERN_LAMBDA + (1.0 - PATTERN_LAMBDA) * p_conf)
            combined_conf = min(1.0, combined_conf)
            return EngineOutput(
                engine=self.name,
                direction=ensemble_dir,
                confidence=combined_conf,
                pattern_signature=full_sig,
                raw_state={
                    **raw,
                    "decision_source": "ensemble_and_pattern_agree",
                    "combined_confidence_formula": "C_E * (0.5 + 0.5 * C_P)",
                    "ensemble_strength": strength,
                    "pattern_confidence": p_conf,
                    "lookup": {"occurrences": p_occ,
                               "confidence": p_conf,
                               "chosen_direction": p_dir},
                },
            )

        return EngineOutput(
            engine=self.name,
            direction="NO_SIGNAL",
            pattern_signature=full_sig,
            raw_state={
                **raw,
                "decision_source": "ensemble_pattern_disagree",
                "ensemble_direction": ensemble_dir,
                "pattern_direction": p_dir,
                "ensemble_strength": strength,
                "pattern_confidence": p_conf,
                "lookup": {"occurrences": p_occ,
                           "confidence": p_conf,
                           "chosen_direction": p_dir},
            },
        )
