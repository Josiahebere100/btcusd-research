"""Ensemble Kalman — post-engine weighted vote.

Runs after all base engines have produced their outputs for the
current round. Reads the base engines' CURRENT directions (not the
previous round's), applies Kalman-smoothed weights derived from
recent rolling accuracy, and produces a single weighted vote.

Called from prediction.py as a post-engine pass, not as an engine in
the _engines list.
"""
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

from ..db import get_pool
from .base import EngineContext, EngineOutput


LOOKBACK_PREDICTIONS = 300
ACCURACY_WINDOW = 100
MIN_SAMPLES = 30
MIN_AGREE = 0.55
MIN_WEIGHT = 0.05
MIN_FIRING_ENGINES = 2
BOLLINGER_K = 2.0
HOLT_ALPHA = 0.3
HOLT_BETA = 0.1

BASE_ENGINES = [
    "crt",
    "labouchere",
    "trig_euler",
    "entropy_regime",
    "superformula",
    "navier_stokes",
    "candlestick",
    "momentum",
    "trail_tracer",
    "curve_geometry",
]

EK_MODE = os.environ.get("EK_MODE", "active").strip().lower()


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _holt(series, alpha, beta):
    if not series:
        return 0.5, 0.0
    if len(series) == 1:
        return series[0], 0.0
    l = series[0]
    b = series[1] - series[0]
    for y in series[1:]:
        prev = l
        l = alpha * y + (1 - alpha) * (l + b)
        b = beta * (l - prev) + (1 - beta) * b
    return l, b


async def _load_recent_outcomes(limit):
    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT p.pattern_signature, o.actual_direction
                FROM prediction_snapshots p
                JOIN outcomes o ON o.prediction_id = p.id
                WHERE LEFT(p.pattern_signature, 1) = '{'
                  AND o.actual_direction IN ('UP', 'DOWN')
                ORDER BY p.prediction_timestamp DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = await cur.fetchall()
    return [{"sig": r[0], "actual": r[1]} for r in rows]


def _engine_history(outcomes, engine):
    hist = []
    dir_key = f"{engine}_dir"
    for row in outcomes:
        try:
            sig = json.loads(row["sig"])
        except Exception:
            continue
        actual = row["actual"]
        pred = sig.get(dir_key)
        if pred not in ("UP", "DOWN"):
            continue
        hist.append(pred == actual)
    return hist


def _kalman_weight(hist):
    if len(hist) < MIN_SAMPLES:
        return None, {"reason": "insufficient history", "n": len(hist)}

    recent = hist[:ACCURACY_WINDOW]
    current_acc = sum(recent) / len(recent)

    acc_series = []
    for i in range(ACCURACY_WINDOW, len(hist) + 1):
        window = hist[i - ACCURACY_WINDOW:i]
        acc_series.append(sum(window) / ACCURACY_WINDOW)
    acc_series.reverse()
    if len(acc_series) < 5:
        return None, {"reason": "accuracy series too short", "n": len(acc_series)}

    mean_acc = _mean(acc_series)
    std_acc = _std(acc_series)
    lower_band = mean_acc - BOLLINGER_K * std_acc

    if current_acc < lower_band:
        return None, {
            "reason": "below Bollinger lower band",
            "current": current_acc,
            "lower": lower_band,
        }

    _, slope = _holt(acc_series, HOLT_ALPHA, HOLT_BETA)
    trend_factor = 1.0 if slope >= 0 else max(0.3, 1.0 + slope * 10.0)

    base = math.exp((current_acc - 0.5) * 4.0)
    weight = base * trend_factor

    return weight, {
        "current_acc": current_acc,
        "lower_band": lower_band,
        "slope": slope,
        "trend_factor": trend_factor,
        "weight": weight,
    }


async def run_ensemble_kalman(
    base_outputs: List[EngineOutput],
    ctx: EngineContext,
) -> Optional[EngineOutput]:
    """Post-engine pass. Reads current engine outputs, weighted vote."""
    if EK_MODE != "active":
        return None

    # Collect current directions from THIS round's outputs
    current_dirs: Dict[str, str] = {}
    for o in base_outputs:
        if o.engine in BASE_ENGINES and o.direction in ("UP", "DOWN"):
            current_dirs[o.engine] = o.direction

    if len(current_dirs) < MIN_FIRING_ENGINES:
        return EngineOutput(
            engine="ensemble_kalman",
            direction="NO_SIGNAL",
            confidence=None,
            pattern_signature=None,
            raw_state={
                "reason": "fewer than MIN_FIRING_ENGINES active this round",
                "n_firing": len(current_dirs),
                "required": MIN_FIRING_ENGINES,
                "firing": list(current_dirs.keys()),
            },
        )

    try:
        outcomes = await _load_recent_outcomes(LOOKBACK_PREDICTIONS)
    except Exception as e:
        return EngineOutput(
            engine="ensemble_kalman",
            direction="NO_SIGNAL",
            confidence=None,
            pattern_signature=None,
            raw_state={"reason": "db read failed", "error": str(e)},
        )

    if not outcomes:
        return EngineOutput(
            engine="ensemble_kalman",
            direction="NO_SIGNAL",
            confidence=None,
            pattern_signature=None,
            raw_state={"reason": "no recent outcomes"},
        )

    weights: Dict[str, float] = {}
    details: Dict[str, Dict[str, Any]] = {}
    for engine in current_dirs.keys():
        hist = _engine_history(outcomes, engine)
        w, meta = _kalman_weight(hist)
        if w is not None and w >= MIN_WEIGHT:
            weights[engine] = w
            details[engine] = meta

    if len(weights) < MIN_FIRING_ENGINES:
        return EngineOutput(
            engine="ensemble_kalman",
            direction="NO_SIGNAL",
            confidence=None,
            pattern_signature=None,
            raw_state={
                "reason": "fewer than MIN_FIRING_ENGINES with weights",
                "n_weighted": len(weights),
                "details": details,
            },
        )

    up_weight = 0.0
    down_weight = 0.0
    agreeing_up = []
    agreeing_down = []
    for engine, w in weights.items():
        d = current_dirs[engine]
        if d == "UP":
            up_weight += w
            agreeing_up.append(engine)
        else:
            down_weight += w
            agreeing_down.append(engine)

    total = up_weight + down_weight
    if total <= 0:
        return EngineOutput(
            engine="ensemble_kalman",
            direction="NO_SIGNAL",
            confidence=None,
            pattern_signature=None,
            raw_state={"reason": "no weighted votes"},
        )

    up_frac = up_weight / total

    if up_frac >= MIN_AGREE:
        direction = "UP"
        selected_conf = up_frac
        agreeing = agreeing_up
    elif (1.0 - up_frac) >= MIN_AGREE:
        direction = "DOWN"
        selected_conf = 1.0 - up_frac
        agreeing = agreeing_down
    else:
        return EngineOutput(
            engine="ensemble_kalman",
            direction="NO_SIGNAL",
            confidence=None,
            pattern_signature=None,
            raw_state={
                "reason": "disagreement below threshold",
                "up_frac": up_frac,
                "weights": weights,
            },
        )

    bucket = int(selected_conf * 10)
    signature = f"ek:{direction.lower()}:{bucket}"

    raw = {
        "up_weight": up_weight,
        "down_weight": down_weight,
        "up_frac": up_frac,
        "weights": weights,
        "agreeing": agreeing,
        "details": details,
        "n_outcomes": len(outcomes),
    }

    return EngineOutput(
        engine="ensemble_kalman",
        direction=direction,
        confidence=selected_conf,
        pattern_signature=signature,
        raw_state=raw,
    )
