"""Curve Geometry Engine.

Maps every tick within the play period to a centered, aspect-preserving
graph, extracts both geometric and numeric features, and attempts to fit
multiple formula families to the resulting curve.

Graph construction:
    - Path starts at first tick, translated so path-start is at (0, 0)
    - Bounding box midpoint becomes origin (0, 0)
    - Path is scaled proportionally so the widest half-dimension is 1
    - Every point now has an (x, y) in the range [-1, 1]
    - Both axes cross through the movement, so the shape's natural
      structure determines how it distributes across the four quadrants

Numeric features:
    centroid, std, skew of x and y
    quadrant distribution (fraction in Q1..Q4)
    axis crossings
    step size distribution
    autocorrelation of the y-series

Geometric features:
    arc length, endpoint distance, tortuosity
    turning number, inflection count, direction changes
    convex hull area, self-intersections, signed area

Formula fitting:
    Attempts to fit y(x) with:
        linear, quadratic, cubic, quartic,
        power law, exponential, logarithmic
    Records the best-fitting family, its parameters, RMS residual, R^2.

Signature:
    cg:f{family}_s{shape}_q{quadrant}_dw{dwell}
State space: ~576. Learnable within hours.

Excludes trigonometry (handled by trig_euler).
"""
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._lookup import lookup_direction
from ._temporal import dwell_bin
from .base import Engine, EngineContext, EngineOutput


MIN_PLAY_TICKS = 8
MAX_PLAY_TICKS = 60
TURN_THRESHOLD = 0.002
INF_THRESHOLD = 0.05

CURVE_GEOMETRY_MODE = os.environ.get(
    "CURVE_GEOMETRY_MODE", "active"
).strip().lower()


# ============================================================
# Play period
# ============================================================

def _ticks_in_play_period(
    ctx: EngineContext,
) -> List[Tuple[float, float]]:
    if ctx.round_start_timestamp is None or ctx.trade_cutoff_timestamp is None:
        return ctx.recent_ticks[-MAX_PLAY_TICKS:] if ctx.recent_ticks else []

    start_s = ctx.round_start_timestamp.timestamp()
    cutoff_s = ctx.trade_cutoff_timestamp.timestamp()

    filtered = [
        (t, p) for t, p in ctx.recent_ticks
        if start_s <= t <= cutoff_s
    ]
    if len(filtered) < MIN_PLAY_TICKS:
        return ctx.recent_ticks[-MAX_PLAY_TICKS:]
    if len(filtered) > MAX_PLAY_TICKS:
        step = len(filtered) / MAX_PLAY_TICKS
        filtered = [filtered[int(i * step)] for i in range(MAX_PLAY_TICKS)]
    return filtered


# ============================================================
# Graph construction — centered, aspect-preserving
# ============================================================

def _build_graph_points(
    ticks: List[Tuple[float, float]],
) -> Optional[np.ndarray]:
    """Map ticks to centered, scaled (x, y) points."""
    if len(ticks) < MIN_PLAY_TICKS:
        return None

    t0 = ticks[0][0]
    xs_raw = np.array([t - t0 for t, _ in ticks], dtype=np.float64)
    ys_raw = np.array([p for _, p in ticks], dtype=np.float64)

    ys_raw = ys_raw - ys_raw[0]

    x_mid = (xs_raw.min() + xs_raw.max()) / 2.0
    y_mid = (ys_raw.min() + ys_raw.max()) / 2.0
    xs_c = xs_raw - x_mid
    ys_c = ys_raw - y_mid

    max_half = max(
        abs(xs_c).max(),
        abs(ys_c).max(),
        1e-9,
    )
    xs = xs_c / max_half
    ys = ys_c / max_half

    return np.stack([xs, ys], axis=1)


# ============================================================
# Numeric features
# ============================================================

def _quadrant_distribution(pts: np.ndarray) -> Tuple[float, float, float, float, int]:
    n = len(pts)
    q = [0, 0, 0, 0]
    for x, y in pts:
        if x >= 0 and y >= 0:
            q[0] += 1
        elif x < 0 and y >= 0:
            q[1] += 1
        elif x < 0 and y < 0:
            q[2] += 1
        else:
            q[3] += 1
    fracs = [c / n for c in q]
    dominant = int(np.argmax(q))
    return fracs[0], fracs[1], fracs[2], fracs[3], dominant


def _axis_crossings(ys: np.ndarray) -> int:
    count = 0
    for i in range(1, len(ys)):
        if (ys[i - 1] < 0 and ys[i] >= 0) or (ys[i - 1] >= 0 and ys[i] < 0):
            count += 1
    return count


def _numeric_features(pts: np.ndarray) -> Dict[str, float]:
    xs = pts[:, 0]
    ys = pts[:, 1]

    q1f, q2f, q3f, q4f, dom = _quadrant_distribution(pts)
    y_cross = _axis_crossings(ys)

    step_sizes = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)

    if len(ys) > 2:
        y_c = ys - ys.mean()
        denom = float((y_c * y_c).sum())
        ac1 = float((y_c[:-1] * y_c[1:]).sum() / denom) if denom > 1e-12 else 0.0
    else:
        ac1 = 0.0

    def safe_skew(arr):
        if len(arr) < 3:
            return 0.0
        m = arr.mean()
        s = arr.std()
        if s < 1e-9:
            return 0.0
        return float(((arr - m) ** 3).mean() / (s ** 3))

    return {
        "x_mean": float(xs.mean()),
        "y_mean": float(ys.mean()),
        "x_std": float(xs.std()),
        "y_std": float(ys.std()),
        "x_skew": safe_skew(xs),
        "y_skew": safe_skew(ys),
        "q1_frac": q1f,
        "q2_frac": q2f,
        "q3_frac": q3f,
        "q4_frac": q4f,
        "dominant_quad": float(dom),
        "y_crossings": float(y_cross),
        "mean_step": float(step_sizes.mean()) if len(step_sizes) else 0.0,
        "std_step": float(step_sizes.std()) if len(step_sizes) else 0.0,
        "autocorr_y1": ac1,
    }


# ============================================================
# Geometric features
# ============================================================

def _arc_length(pts: np.ndarray) -> float:
    d = np.diff(pts, axis=0)
    return float(np.sqrt((d ** 2).sum(axis=1)).sum())


def _endpoint_distance(pts: np.ndarray) -> float:
    dx = pts[-1, 0] - pts[0, 0]
    dy = pts[-1, 1] - pts[0, 1]
    return float(math.sqrt(dx * dx + dy * dy))


def _curvature_profile(pts: np.ndarray) -> np.ndarray:
    d1 = pts[1:-1] - pts[:-2]
    d2 = pts[2:] - pts[1:-1]
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    m1 = np.sqrt((d1 ** 2).sum(axis=1))
    m2 = np.sqrt((d2 ** 2).sum(axis=1))
    denom = m1 * m2 * (m1 + m2)
    denom = np.where(denom < 1e-12, 1e-12, denom)
    return 2.0 * cross / denom


def _turning_number(pts: np.ndarray) -> float:
    curv = _curvature_profile(pts)
    if len(curv) == 0:
        return 0.0
    return float(curv.sum() / (2.0 * math.pi))


def _inflection_count(curv: np.ndarray) -> int:
    count = 0
    prev_sign = 0
    for c in curv:
        if abs(c) < INF_THRESHOLD:
            continue
        s = 1 if c > 0 else -1
        if prev_sign != 0 and s != prev_sign:
            count += 1
        prev_sign = s
    return count


def _direction_changes(ys: np.ndarray) -> int:
    count = 0
    prev_dir = 0
    for i in range(1, len(ys)):
        delta = ys[i] - ys[i - 1]
        if abs(delta) < TURN_THRESHOLD:
            continue
        d = 1 if delta > 0 else -1
        if prev_dir != 0 and d != prev_dir:
            count += 1
        prev_dir = d
    return count


def _convex_hull_area(points: np.ndarray) -> float:
    pts = [tuple(p) for p in points]
    pts = sorted(set(pts))
    if len(pts) < 3:
        return 0.0

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return 0.0
    area = 0.0
    n = len(hull)
    for i in range(n):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _self_intersections(pts: np.ndarray, max_check: int = 30) -> int:
    n = min(len(pts), max_check)

    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    def intersect(p1, p2, p3, p4):
        return (
            ccw(p1, p3, p4) != ccw(p2, p3, p4)
            and ccw(p1, p2, p3) != ccw(p1, p2, p4)
        )

    count = 0
    for i in range(n - 1):
        for j in range(i + 2, n - 1):
            if i == 0 and j == n - 2:
                continue
            if intersect(pts[i], pts[i + 1], pts[j], pts[j + 1]):
                count += 1
    return count


def _signed_area(pts: np.ndarray) -> float:
    n = len(pts)
    total = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return float(total / 2.0)


def _geometric_features(pts: np.ndarray) -> Dict[str, float]:
    ys = pts[:, 1]
    arc = _arc_length(pts)
    end = _endpoint_distance(pts)
    tort = arc / end if end > 1e-9 else 1.0

    curv = _curvature_profile(pts)
    turn_num = _turning_number(pts)
    inflections = _inflection_count(curv)
    turns = _direction_changes(ys)

    hull_area = _convex_hull_area(pts)
    self_int = _self_intersections(pts)
    signed_area = _signed_area(pts)

    return {
        "arc_length": arc,
        "endpoint_distance": end,
        "tortuosity": tort,
        "turning_number": turn_num,
        "inflections": float(inflections),
        "n_turns": float(turns),
        "hull_area": hull_area,
        "self_intersections": float(self_int),
        "signed_area": signed_area,
    }


# ============================================================
# Formula fitting
# ============================================================

def _rms(ys: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((ys - pred) ** 2)))


def _fit_linear(xs, ys):
    try:
        coeffs = np.polyfit(xs, ys, 1)
        return coeffs.tolist(), _rms(ys, np.polyval(coeffs, xs))
    except Exception:
        return None, float("inf")


def _fit_poly(xs, ys, degree):
    if len(xs) < degree + 2:
        return None, float("inf")
    try:
        coeffs = np.polyfit(xs, ys, degree)
        return coeffs.tolist(), _rms(ys, np.polyval(coeffs, xs))
    except Exception:
        return None, float("inf")


def _fit_power(xs, ys):
    mask = (np.abs(xs) > 1e-4) & (np.abs(ys) > 1e-6)
    if mask.sum() < 3:
        return None, float("inf")
    try:
        sign_match = np.sign(xs[mask]) == np.sign(ys[mask])
        if sign_match.sum() < 3:
            return None, float("inf")
        lx = np.log(np.abs(xs[mask]))[sign_match]
        ly = np.log(np.abs(ys[mask]))[sign_match]
        b, log_a = np.polyfit(lx, ly, 1)
        a = float(np.exp(log_a))
        pred = a * np.sign(xs) * np.abs(xs) ** b
        return [a, float(b)], _rms(ys, pred)
    except Exception:
        return None, float("inf")


def _fit_exponential(xs, ys):
    if len(xs) < 3:
        return None, float("inf")
    try:
        if not (np.all(ys > 0) or np.all(ys < 0)):
            return None, float("inf")
        sgn = 1.0 if np.all(ys > 0) else -1.0
        ly = np.log(np.abs(ys))
        b, log_a = np.polyfit(xs, ly, 1)
        a = sgn * float(np.exp(log_a))
        pred = a * np.exp(b * xs)
        return [a, float(b)], _rms(ys, pred)
    except Exception:
        return None, float("inf")


def _fit_logarithmic(xs, ys):
    if len(xs) < 3:
        return None, float("inf")
    try:
        lx = np.log(np.abs(xs) + 1e-3)
        coeffs = np.polyfit(lx, ys, 1)
        return coeffs.tolist(), _rms(ys, np.polyval(coeffs, lx))
    except Exception:
        return None, float("inf")


def _try_all_formulas(xs: np.ndarray, ys: np.ndarray) -> Dict[str, Any]:
    results = {}

    p, r = _fit_linear(xs, ys);      results["linear"] = (p, r)
    p, r = _fit_poly(xs, ys, 2);     results["quadratic"] = (p, r)
    p, r = _fit_poly(xs, ys, 3);     results["cubic"] = (p, r)
    p, r = _fit_poly(xs, ys, 4);     results["quartic"] = (p, r)
    p, r = _fit_power(xs, ys);       results["power"] = (p, r)
    p, r = _fit_exponential(xs, ys); results["exponential"] = (p, r)
    p, r = _fit_logarithmic(xs, ys); results["logarithmic"] = (p, r)

    best_family = min(results, key=lambda k: results[k][1])
    best_params, best_rms = results[best_family]

    y_var = float(np.var(ys))
    best_r2 = max(0.0, 1.0 - (best_rms ** 2) / y_var) if y_var > 1e-9 else 0.0

    return {
        "best_family": best_family,
        "best_rms": best_rms,
        "best_r2": best_r2,
        "best_params": best_params,
        "all_results": {k: float(v[1]) for k, v in results.items()},
    }


# ============================================================
# Shape classification
# ============================================================

def _classify_shape(g: Dict[str, float]) -> str:
    if g["self_intersections"] >= 1:
        return "LOOP"
    if g["n_turns"] >= 6:
        return "ZIG"
    if g["n_turns"] >= 4 and g["tortuosity"] > 1.5:
        return "WIG"
    if g["inflections"] >= 2:
        return "S"
    if g["tortuosity"] < 1.15:
        return "LIN"
    return "CUR"


FAMILY_CODES = {
    "linear": 0,
    "quadratic": 1,
    "cubic": 2,
    "quartic": 3,
    "power": 4,
    "exponential": 5,
    "logarithmic": 6,
}

SHAPE_CODES = {
    "LIN": 0, "CUR": 1, "S": 2, "WIG": 3, "ZIG": 4, "LOOP": 5,
}


def _core_signature(fit: Dict[str, Any], shape: str, dominant_quad: int) -> str:
    family_code = FAMILY_CODES.get(fit["best_family"], 7)
    shape_code = SHAPE_CODES.get(shape, 9)
    return f"cg:f{family_code}_s{shape_code}_q{dominant_quad}"


# ============================================================
# Engine
# ============================================================

class CurveGeometryEngine(Engine):
    name = "curve_geometry"

    def run(self, ctx: EngineContext) -> EngineOutput:
        raise NotImplementedError("CurveGeometry is async; use run_async()")

    async def run_async(self, ctx: EngineContext) -> EngineOutput:
        if CURVE_GEOMETRY_MODE != "active":
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature=None,
                raw_state={"reason": f"CURVE_GEOMETRY_MODE={CURVE_GEOMETRY_MODE}"},
            )

        ticks = _ticks_in_play_period(ctx)
        if len(ticks) < MIN_PLAY_TICKS:
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature=None,
                raw_state={"reason": "insufficient play-period ticks", "n_ticks": len(ticks)},
            )

        pts = _build_graph_points(ticks)
        if pts is None or len(pts) < MIN_PLAY_TICKS:
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature=None,
                raw_state={"reason": "graph construction failed"},
            )

        try:
            numeric = _numeric_features(pts)
            geometric = _geometric_features(pts)
            fit = _try_all_formulas(pts[:, 0], pts[:, 1])
        except Exception as e:
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature=None,
                raw_state={"reason": f"feature extraction failed: {e}"},
            )

        shape = _classify_shape(geometric)
        dominant_quad = int(numeric["dominant_quad"])
        core_sig = _core_signature(fit, shape, dominant_quad)
        dwell_s = self._compute_dwell(ctx.recent_ticks, shape, dominant_quad)
        full_sig = f"{core_sig}_dw{dwell_bin(dwell_s)}"

        raw: Dict[str, Any] = {
            "numeric": numeric,
            "geometric": geometric,
            "formula_fit": {
                "family": fit["best_family"],
                "rms": fit["best_rms"],
                "r2": fit["best_r2"],
                "params": fit["best_params"],
                "all_families": fit["all_results"],
            },
            "shape": shape,
            "dominant_quadrant": dominant_quad,
            "dwell_s": dwell_s,
            "graph_x": [round(float(x), 4) for x in pts[:, 0]],
            "graph_y": [round(float(y), 4) for y in pts[:, 1]],
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

        direction, occurrences, confidence = lookup
        return EngineOutput(
            engine=self.name,
            direction=direction,
            confidence=confidence,
            pattern_signature=full_sig,
            raw_state={
                **raw,
                "lookup": {
                    "occurrences": occurrences,
                    "confidence": confidence,
                    "chosen_direction": direction,
                },
            },
        )

    def _compute_dwell(self, ticks, current_shape, current_quad) -> float:
        if len(ticks) < 40:
            return 0.0
        now_ts = ticks[-1][0]
        for back_seconds in (5, 15, 30, 60):
            sub = [(t, p) for t, p in ticks if t >= now_ts - back_seconds]
            if len(sub) < MIN_PLAY_TICKS:
                continue
            sub_pts = _build_graph_points(sub)
            if sub_pts is None:
                continue
            try:
                sub_g = _geometric_features(sub_pts)
                sub_n = _numeric_features(sub_pts)
            except Exception:
                continue
            if _classify_shape(sub_g) != current_shape or int(sub_n["dominant_quad"]) != current_quad:
                return now_ts - sub[0][0]
        return float(60)
