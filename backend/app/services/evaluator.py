"""Outcome evaluator + pattern memory + combination memory + per-model scoring.

Called by prediction.py's _schedule_evaluation after the target time
has passed. For each prediction:

  1. Look up actual price at target time
  2. Write outcome row
  3. Update pattern memory per engine
  4. Update combination memory
  5. Score trajectory models (if this prediction used the trajectory engine)
  6. Score projectile kinematic forecast (if this prediction used the projectile engine)
"""
import json
import math
from datetime import datetime
from typing import Optional

from ..db import get_pool
from ..state import note_outcome_evaluated
from .combination import record_combination_outcome
from .pattern_memory import record_signature_outcome


# ============================================================
# Price lookup
# ============================================================

async def _price_at(conn, target: datetime) -> Optional[float]:
    """Nearest tick price at or before target, else first tick after."""
    async with conn.cursor() as cur:
        await cur.execute(
            """
            SELECT price FROM market_ticks
            WHERE tick_timestamp <= %s
            ORDER BY tick_timestamp DESC LIMIT 1
            """,
            (target,),
        )
        row = await cur.fetchone()
        if row:
            return float(row[0])
        await cur.execute(
            """
            SELECT price FROM market_ticks
            WHERE tick_timestamp > %s
            ORDER BY tick_timestamp ASC LIMIT 1
            """,
            (target,),
        )
        row = await cur.fetchone()
        return float(row[0]) if row else None


# ============================================================
# Signature parsing
# ============================================================

def _parse_signatures(raw: Optional[str]):
    """Split pattern_signature JSON into (engine_sigs, combination_key)."""
    if not raw:
        return {}, None
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            comb = parsed.pop("_combination", None)
            return parsed, comb
        return {"unknown": str(parsed)}, None
    except (json.JSONDecodeError, ValueError):
        return {"crt": raw}, None


# ============================================================
# Main entry point
# ============================================================

async def evaluate_prediction(
    prediction_id: int,
    predicted_direction: str,
) -> None:
    """Resolve a prediction's outcome and update all memory layers."""
    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT id, session_id, round_id, target_timestamp,
                       price_at_prediction, direction, pattern_signature
                FROM prediction_snapshots
                WHERE id = %s
                """,
                (prediction_id,),
            )
            pred = await cur.fetchone()
            if pred is None:
                print(f"[evaluator] prediction {prediction_id} not found")
                return

            pred_id = pred[0]
            session_id = pred[1]
            round_id = pred[2]
            target = pred[3]
            price_at_pred = float(pred[4])
            pred_direction = pred[5]
            sig_raw = pred[6]

            actual_price = await _price_at(conn, target)
            if actual_price is None:
                print(f"[evaluator] no price near {target} for {prediction_id}")
                return

            diff = actual_price - price_at_pred
            pct = (diff / price_at_pred) * 100.0 if price_at_pred else 0.0

            if diff > 0:
                actual_direction = "UP"
            elif diff < 0:
                actual_direction = "DOWN"
            else:
                actual_direction = "FLAT"

            correct = (
                pred_direction in ("UP", "DOWN")
                and actual_direction == pred_direction
            )

            # Write outcome row (only for directional predictions)
            if pred_direction in ("UP", "DOWN"):
                await cur.execute(
                    """
                    INSERT INTO outcomes
                      (prediction_id, session_id, round_id, target_timestamp,
                       actual_price, actual_direction, price_difference,
                       percentage_difference, correct)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (prediction_id) DO NOTHING
                    """,
                    (
                        prediction_id, session_id, round_id, target,
                        actual_price, actual_direction, diff, pct, correct,
                    ),
                )

            await cur.execute(
                "UPDATE prediction_snapshots SET status = 'EVALUATED' WHERE id = %s",
                (prediction_id,),
            )
        await conn.commit()

    # ---- Pattern memory per engine -----------------------------------
    # Baselines are excluded (they're read-only controls and should not
    # pollute pattern memory). Nested objects like "baseline" are skipped
    # because they aren't strings.
    engine_sigs, comb_key = _parse_signatures(sig_raw)
    for eng_name, sig in engine_sigs.items():
        if eng_name.startswith("baseline_"):
            continue
        if eng_name == "baseline":
            continue
        if eng_name.endswith("_state"):
            continue
        if not isinstance(sig, str):
            continue
        try:
            await record_signature_outcome(
                signature=sig,
                session_id=session_id,
                prediction_id=pred_id,
                engine=eng_name,
                actual_direction=actual_direction,
                prediction_direction=pred_direction,
            )
        except Exception as e:
            print(f"[pattern_memory] failed sig={sig}: {e}")

    # ---- Combination memory ------------------------------------------
    if comb_key:
        try:
            await record_combination_outcome(
                key=comb_key,
                actual_direction=actual_direction,
                session_id=session_id,
                prediction_id=pred_id,
            )
        except Exception as e:
            print(f"[combination_memory] failed key={comb_key}: {e}")

    # ---- Trajectory per-model scoring --------------------------------
    try:
        await _score_trajectory_models(
            prediction_id=pred_id,
            sig_raw=sig_raw,
            actual_direction=actual_direction,
            target=target,
        )
    except Exception as e:
        print(f"[trajectory_score] failed: {e}")

    # ---- Projectile kinematic scoring --------------------------------
    try:
        await _score_projectile(
            prediction_id=pred_id,
            sig_raw=sig_raw,
            actual_direction=actual_direction,
            target=target,
        )
    except Exception as e:
        print(f"[projectile_score] failed: {e}")

    note_outcome_evaluated()

    print(
        f"[evaluator] #{prediction_id} "
        f"pred={pred_direction} actual={actual_direction} "
        f"correct={correct} pct={pct:.4f}"
    )


# ============================================================
# Trajectory per-model scoring
# ============================================================

async def _score_trajectory_models(prediction_id, sig_raw, actual_direction, target):
    """Score each trajectory hypothesis model against actual future ticks.

    Produces two labels per model:
      internal_direction_correct — sign(p_{t+3} - p_t) using the
                                    engine's stored current price
      target_direction_correct   — 15-second Up/Down outcome

    Uses strictly-future ticks (> prediction_timestamp), scoped to
    BC.GAME BTC-USD. Idempotent via ON CONFLICT.
    """
    if not sig_raw:
        return
    try:
        data = json.loads(sig_raw)
    except Exception:
        return
    if not isinstance(data, dict):
        return
    tstate = data.get("trajectory_state")
    if not isinstance(tstate, dict):
        return
    per_model = tstate.get("per_model_forecast")
    if not isinstance(per_model, dict) or not per_model:
        return

    p0 = tstate.get("forecast_p0")
    span_p = tstate.get("forecast_span_p")
    forecast_x = tstate.get("forecast_x")
    current_price = tstate.get("forecast_current_price")
    if p0 is None or span_p is None or not forecast_x or current_price is None:
        return

    horizon = len(forecast_x)

    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT prediction_timestamp FROM prediction_snapshots WHERE id = %s",
                (prediction_id,),
            )
            row = await cur.fetchone()
    if not row:
        return
    pred_ts = row[0]

    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT tick_timestamp, price
                FROM market_ticks
                WHERE source = 'BC.GAME'
                  AND symbol = 'BTC-USD'
                  AND tick_timestamp > %s
                ORDER BY tick_timestamp ASC
                LIMIT %s
                """,
                (pred_ts, horizon),
            )
            rows = await cur.fetchall()

    if len(rows) < horizon:
        return

    actual_prices = [float(r[1]) for r in rows]

    # Internal direction: sign(p_{t+3} - p_t) using stored current price
    internal_delta = actual_prices[-1] - float(current_price)
    if internal_delta > 0:
        internal_actual_dir = "UP"
    elif internal_delta < 0:
        internal_actual_dir = "DOWN"
    else:
        internal_actual_dir = "FLAT"

    target_actual_dir = actual_direction

    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            for model_name, payload in per_model.items():
                if not isinstance(payload, dict):
                    continue
                fcast_values = payload.get("values") or []
                fcast_dir = payload.get("direction")
                weight = payload.get("weight")

                if len(fcast_values) != horizon:
                    continue

                # Reconstruct price-space forecast:
                #   normalized y -> price: p_hat = p0 + span_p * y_hat
                forecast_prices = [float(p0) + float(span_p) * v for v in fcast_values]

                endpoint_error = abs(forecast_prices[-1] - actual_prices[-1])

                sq_sum = 0.0
                for i in range(horizon):
                    diff = forecast_prices[i] - actual_prices[i]
                    sq_sum += diff * diff
                trajectory_rms = float(math.sqrt(sq_sum / horizon))

                internal_correct = None
                target_correct = None
                if fcast_dir in ("UP", "DOWN"):
                    if internal_actual_dir in ("UP", "DOWN"):
                        internal_correct = (fcast_dir == internal_actual_dir)
                    if target_actual_dir in ("UP", "DOWN"):
                        target_correct = (fcast_dir == target_actual_dir)

                await cur.execute(
                    """
                    INSERT INTO trajectory_model_outcomes
                      (prediction_id, model_name, horizon,
                       forecast_values, actual_values,
                       endpoint_error, trajectory_rms,
                       internal_direction_correct,
                       target_direction_correct,
                       weight)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (prediction_id, model_name, horizon) DO NOTHING
                    """,
                    (
                        prediction_id,
                        model_name,
                        horizon,
                        json.dumps(forecast_prices),
                        json.dumps(actual_prices),
                        endpoint_error,
                        trajectory_rms,
                        internal_correct,
                        target_correct,
                        weight,
                    ),
                )
        await conn.commit()


# ============================================================
# Projectile kinematic scoring
# ============================================================

async def _score_projectile(prediction_id, sig_raw, actual_direction, target):
    """Score the projectile kinematic forecast against the actual tick
    at the play-period cutoff.

    Reuses trajectory_model_outcomes with model_name='projectile_kinematic'.
    """
    if not sig_raw:
        return
    try:
        data = json.loads(sig_raw)
    except Exception:
        return
    if not isinstance(data, dict):
        return
    pstate = data.get("projectile_state")
    if not isinstance(pstate, dict):
        return

    coords = pstate.get("coordinates") or {}
    p0 = coords.get("p0")
    y_unit = coords.get("y_unit")
    forecast = pstate.get("forecast") or {}
    t_cutoff = forecast.get("t_cutoff_s")
    y_forecast = forecast.get("y_forecast")
    kin_dir = forecast.get("direction")

    if None in (p0, y_unit, t_cutoff, y_forecast):
        return

    predicted_price = float(p0) + float(y_forecast) * float(y_unit)

    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT prediction_timestamp FROM prediction_snapshots WHERE id = %s",
                (prediction_id,),
            )
            row = await cur.fetchone()
    if not row:
        return
    pred_ts = row[0]

    # Actual price at the end of the play period = first tick after
    # prediction_timestamp is the closest we can get without a full
    # play-period reconstruction here.
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT tick_timestamp, price
                FROM market_ticks
                WHERE source = 'BC.GAME'
                  AND symbol = 'BTC-USD'
                  AND tick_timestamp > %s
                ORDER BY tick_timestamp ASC
                LIMIT 1
                """,
                (pred_ts,),
            )
            future_row = await cur.fetchone()
    if not future_row:
        return
    actual_at_horizon = float(future_row[1])

    err = predicted_price - actual_at_horizon

    correct = None
    if kin_dir in ("UP", "DOWN") and actual_direction in ("UP", "DOWN"):
        correct = (kin_dir == actual_direction)

    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO trajectory_model_outcomes
                  (prediction_id, model_name, horizon,
                   forecast_values, actual_values,
                   endpoint_error, trajectory_rms,
                   internal_direction_correct,
                   target_direction_correct,
                   weight)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (prediction_id, model_name, horizon) DO NOTHING
                """,
                (
                    prediction_id,
                    "projectile_kinematic",
                    1,
                    json.dumps([predicted_price]),
                    json.dumps([actual_at_horizon]),
                    abs(err),
                    abs(err),
                    correct,
                    correct,
                    None,
                ),
            )
        await conn.commit()
