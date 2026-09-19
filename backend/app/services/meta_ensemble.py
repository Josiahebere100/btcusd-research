"""Meta-ensemble that combines base engine outputs with statistical weighting."""
import json
import math
from typing import Dict, List, Optional

from ..db import get_pool
from ..engines.base import EngineContext, EngineOutput

LOOKBACK_ROUNDS = 300
ROLLING_WINDOW = 50
MIN_SAMPLES = 30
BOLLINGER_STD = 2.0
HOLT_ALPHA = 0.3
HOLT_BETA = 0.1
MIN_WEIGHT = 0.05
MIN_AGREE = 0.55


def _holt(series: List[float], alpha: float, beta: float):
    if not series:
        return 0.5, 0.0
    if len(series) == 1:
        return series[0], 0.0
    l = series[0]
    b = series[1] - series[0]
    for y in series[1:]:
        prev_l = l
        l = alpha * y + (1 - alpha) * (l + b)
        b = beta * (l - prev_l) + (1 - beta) * b
    return l, b


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


async def _load_recent_outcomes(limit: int):
    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT p.pattern_signature, o.actual_direction
                FROM prediction_snapshots p
                JOIN outcomes o ON o.prediction_id = p.id
                WHERE LEFT(p.pattern_signature, 1) = '{'
                ORDER BY p.prediction_timestamp DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = await cur.fetchall()
    return [{"sig": r[0], "actual": r[1]} for r in rows]


def _engine_history(outcomes: List[dict], engine: str) -> List[bool]:
    hist = []
    dir_key = f"{engine}_dir"
    for row in outcomes:
        try:
            sig_json = json.loads(row["sig"])
        except Exception:
            continue
        actual = row["actual"]
        if actual not in ("UP", "DOWN"):
            continue
        pred = sig_json.get(dir_key)
        if pred not in ("UP", "DOWN"):
            continue
        hist.append(pred == actual)
    return hist


def _weight_for_engine(hist: List[bool]) -> Optional[float]:
    if len(hist) < MIN_SAMPLES:
        return None

    recent = hist[:ROLLING_WINDOW]
    current_acc = sum(recent) / len(recent)

    acc_series = []
    for i in range(ROLLING_WINDOW, len(hist) + 1):
        window = hist[i - ROLLING_WINDOW:i]
        acc_series.append(sum(window) / ROLLING_WINDOW)
    acc_series.reverse()
    if len(acc_series) < 5:
        return None

    mean_acc = _mean(acc_series)
    std_acc = _std(acc_series)
    lower_band = mean_acc - BOLLINGER_STD * std_acc

    if current_acc < lower_band:
        return None

    _, slope = _holt(acc_series, HOLT_ALPHA, HOLT_BETA)
    trend_factor = 1.0 if slope >= 0 else max(0.3, 1.0 + slope * 10)

    base = math.exp((current_acc - 0.5) * 4.0)
    weight = base * trend_factor
    return weight if weight >= MIN_WEIGHT else None


async def run_meta_ensemble(
    base_outputs: List[EngineOutput],
    ctx: EngineContext,
) -> Optional[EngineOutput]:
    current: Dict[str, str] = {}
    for o in base_outputs:
        if o.direction in ("UP", "DOWN"):
            current[o.engine] = o.direction
    if not current:
        return None

    outcomes = await _load_recent_outcomes(LOOKBACK_ROUNDS)

    weights: Dict[str, float] = {}
    for engine in current.keys():
        hist = _engine_history(outcomes, engine)
        w = _weight_for_engine(hist)
        if w is not None:
            weights[engine] = w

    if not weights:
        return None

    up_votes = sum(w for e, w in weights.items() if current.get(e) == "UP")
    down_votes = sum(w for e, w in weights.items() if current.get(e) == "DOWN")
    total = up_votes + down_votes
    if total <= 0:
        return None

    up_frac = up_votes / total

    if up_frac >= MIN_AGREE:
        direction = "UP"
        confidence = up_frac
    elif up_frac <= (1 - MIN_AGREE):
        direction = "DOWN"
        confidence = 1.0 - up_frac
    else:
        return None

    agreeing = sorted(
        e for e, w in weights.items() if current.get(e) == direction
    )
    signature = f"meta:{direction}:{'+'.join(agreeing)}"

    return EngineOutput(
        engine="meta_ensemble",
        direction=direction,
        confidence=confidence,
        pattern_signature=signature,
        raw_state={
            "weights": weights,
            "up_votes": up_votes,
            "down_votes": down_votes,
            "up_frac": up_frac,
            "agreeing": agreeing,
        },
    )
