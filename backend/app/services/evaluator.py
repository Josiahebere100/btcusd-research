"""Outcome evaluator + pattern memory hook."""
import json
from datetime import datetime
from typing import Optional

from ..db import get_pool
from .pattern_memory import record_signature_outcome


async def _price_at(conn, target: datetime) -> Optional[float]:
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


def _parse_signatures(raw: Optional[str]) -> dict:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
        return {"unknown": str(parsed)}
    except (json.JSONDecodeError, ValueError):
        # Backwards compat: plain-string signature from before Phase 16.
        return {"crt": raw}


async def evaluate_prediction(
    prediction_id: int,
    predicted_direction: str,
) -> None:
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

            if pred_direction not in ("UP", "DOWN"):
                # No directional call: still mark EVALUATED. Pattern memory
                # will be populated from the actual_direction below.
                pass

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

    # Record pattern memory outside the transaction (separate pools).
    sig_map = _parse_signatures(sig_raw)
    for eng_name, sig in sig_map.items():
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
