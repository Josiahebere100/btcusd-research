"""Ensemble Kalman engine.

Combines the base engines' directional calls via Kalman-smoothed weights
derived from their recent rolling accuracy. Applies a Bollinger gate to
silence engines whose accuracy has dropped below their historical band,
and a Holt trend decay to reduce weight on engines that are degrading.

Reads pattern_signature JSON from recent predictions to reconstruct what
each engine said.

Signature: ek:up:0.62  or  ek:down:0.58  (direction + rounded weight agree)
State space: small (direction x 10 probability bins).
"""
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

from ..db import get_pool
from .base import Engine, EngineContext, EngineOutput
from ._lookup import lookup_direction


# Configuration
LOOKBACK_PREDICTIONS = 300
ACCURACY_WINDOW = 100
MIN_SAMPLES = 30
MIN_AGREE = 0.55
MIN_WEIGHT = 0.05
BOLLINGER_K = 2.0
HOLT_ALPHA = 0.3
HOLT_BETA = 0.1

# Engines to combine. Excludes:
#   - combination (aggregator, not a base engine)
#   - ensemble_kalman (self)
#   - ns_flow (currently bugged, insufficient signals)
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


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _holt(series: List[float], alpha: float, beta: float) -> Tuple[float, float]:
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


class EnsembleKalmanEngine(Engine):
    name = "ensemble_kalman"

    def run(self, ctx):
        raise NotImplementedError("EnsembleKalman is async; use run_async()")

    async def run_async(self, ctx: EngineContext) -> EngineOutput:
        if EK_MODE != "active":
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature=None,
                raw_state={"reason": f"EK_MODE={EK_MODE}"},
            )

        try:
            outcomes = await self._load_recent_outcomes()
        except Exception as e:
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature=None,
                raw_state={"reason": f"db read failed: {e}"},
            )

        if not outcomes:
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature=None,
                raw_state={"reason": "no recent outcomes"},
            )

        weights = {}
        details = {}
        for engine in BASE_ENGINES:
            hist = self._engine_history(outcomes, engine)
            w, meta = self._kalman_weight(hist)
            if w is not None and w >= MIN_WEIGHT:
                weights[engine] = w
                details[engine] = meta

        if not weights:
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature=None,
                raw_state={"reason": "all engines silenced", "n_outcomes": len(outcomes)},
            )

        # Weighted vote across engines that are firing THIS round.
        # We read the current engine outputs from the most recent prediction's JSON
        # is not how it works here — the ctx doesn't carry other engines' outputs.
        # Instead: we use the LAST prediction's _dir fields to infer what each engine
        # would say. This is a simplification; v2 will read from ctx directly.
        # For now: we issue a vote based on the ensemble's own bias.
        # Alternative: read the ensemble's own confidence trend.

        # Simplest viable: compute the weighted majority over the last N outcomes.
        # Actually no — the ensemble should produce a CURRENT direction.
        # We need each engine's CURRENT call. Since ctx doesn't expose them,
        # we approximate using the most recent prediction snapshot.

        current_dirs = await self._get_current_engine_dirs()
        if not current_dirs:
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature=None,
                raw_state={"reason": "no current engine state available"},
            )

        up_weight = 0.0
        down_weight = 0.0
        agreeing_up = []
        agreeing_down = []
        for engine, w in weights.items():
            d = current_dirs.get(engine)
            if d == "UP":
                up_weight += w
                agreeing_up.append(engine)
            elif d == "DOWN":
                down_weight += w
                agreeing_down.append(engine)

        total = up_weight + down_weight
        if total <= 0:
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature=None,
                raw_state={"reason": "no weighted votes", "weights": weights},
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
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature=None,
                raw_state={
                    "reason": "disagreement below threshold",
                    "up_frac": up_frac,
                    "weights": weights,
                },
            )

        bucket = int(selected_conf * 10)
        signature = f"ek:{direction.lower()}:{bucket}"

        raw: Dict[str, Any] = {
            "up_weight": up_weight,
            "down_weight": down_weight,
            "up_frac": up_frac,
            "weights": weights,
            "agreeing": agreeing,
            "details": details,
            "n_outcomes": len(outcomes),
        }

        return EngineOutput(
            engine=self.name,
            direction=direction,
            confidence=selected_conf,
            pattern_signature=signature,
            raw_state=raw,
        )

    async def _load_recent_outcomes(self) -> List[Dict[str, Any]]:
        """Load recent predictions with settled outcomes."""
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
                    (LOOKBACK_PREDICTIONS,),
                )
                rows = await cur.fetchall()
        return [{"sig": r[0], "actual": r[1]} for r in rows]

    async def _get_current_engine_dirs(self) -> Dict[str, str]:
        """Read the most recent prediction's engine directions."""
        pool = await get_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT pattern_signature
                    FROM prediction_snapshots
                    WHERE LEFT(pattern_signature, 1) = '{'
                    ORDER BY prediction_timestamp DESC
                    LIMIT 1
                    """
                )
                row = await cur.fetchone()
        if not row or not row[0]:
            return {}
        try:
            sig = json.loads(row[0])
        except Exception:
            return {}
        out = {}
        for engine in BASE_ENGINES:
            d = sig.get(f"{engine}_dir")
            if d in ("UP", "DOWN"):
                out[engine] = d
        return out

    def _engine_history(self, outcomes: List[Dict[str, Any]], engine: str) -> List[bool]:
        """Return list of True/False (correct/incorrect) per occurrence, newest first."""
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

    def _kalman_weight(self, hist: List[bool]) -> Tuple[Optional[float], Dict[str, Any]]:
        """Compute Kalman-smoothed weight for one engine."""
        if len(hist) < MIN_SAMPLES:
            return None, {"reason": "insufficient history", "n": len(hist)}

        recent = hist[:ACCURACY_WINDOW]
        current_acc = sum(recent) / len(recent)

        # Build rolling accuracy series for Bollinger + Holt
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

        # Bollinger gate
        if current_acc < lower_band:
            return None, {
                "reason": "below lower Bollinger band",
                "current": current_acc,
                "lower": lower_band,
                "mean": mean_acc,
                "std": std_acc,
            }

        # Holt trend
        _, slope = _holt(acc_series, HOLT_ALPHA, HOLT_BETA)
        trend_factor = 1.0 if slope >= 0 else max(0.3, 1.0 + slope * 10.0)

        # Base weight: exponential emphasis above 50%
        base = math.exp((current_acc - 0.5) * 4.0)
        weight = base * trend_factor

        return weight, {
            "current_acc": current_acc,
            "mean": mean_acc,
            "std": std_acc,
            "lower_band": lower_band,
            "slope": slope,
            "trend_factor": trend_factor,
            "weight": weight,
        }
