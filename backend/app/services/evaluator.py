"""Outcome evaluator."""
from datetime import datetime
from typing import Optional

from ..db import get_pool


async def _price_at(conn, target: datetime) -> Optional[float]:
    async with conn.cursor() as cur:
        await cur.execute(
            """
            SELECT price, tick_timestamp
            FROM market_ticks
            WHERE tick_timestamp <= %s
            ORDER BY tick_timestamp DESC
            LIMIT 1
            """,
            (target,),
        )
        row = await cur.fetchone()
        if row:
            return float(row[0])

        await cur.execute(
            """
            SELECT price, tick_timestamp
            FROM market_ticks
            WHERE tick_timestamp > %s
            ORDER BY tick_timestamp ASC
            LIMIT 1
            """,
            (target,),
        )
        row = await cur.fetchone()
        return float(row[0]) if row else None


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
                       price_at_prediction, direction
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
            pred_session = pred[1]
            pred_round = pred[2]
            target = pred[3]
            price_at_prediction = float(pred[4])
            pred_direction = pred[5]

            if pred_direction not in ("UP", "DOWN"):
                await cur.execute(
                    """
                    UPDATE prediction_snapshots
                    SET status = 'EVALUATED'
                    WHERE id = %s
                    """,
                    (prediction_id,),
                )
                await conn.commit()
                return

            actual_price = await _price_at(conn, target)
            if actual_price is None:
                print(
                    f"[evaluator] no price near {target} "
                    f"for prediction {prediction_id}"
                )
                return

            diff = actual_price - price_at_prediction
            pct = (
                (diff / price_at_prediction) * 100.0
                if price_at_prediction
                else 0.0
            )

            if diff > 0:
                actual_direction = "UP"
            elif diff < 0:
                actual_direction = "DOWN"
            else:
                actual_direction = "FLAT"

            correct = actual_direction == pred_direction

            await cur.execute(
                """
                INSERT INTO outcomes
                  (prediction_id, session_id, round_id, target_timestamp,
                   actual_price, actual_direction, price_difference,
                   percentage_difference, correct)
                VALUES
                  (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (prediction_id) DO NOTHING
                """,
                (
                    prediction_id,
                    pred_session,
                    pred_round,
                    target,
                    actual_price,
                    actual_direction,
                    diff,
                    pct,
                    correct,
                ),
            )

            await cur.execute(
                """
                UPDATE prediction_snapshots
                SET status = 'EVALUATED'
                WHERE id = %s
                """,
                (prediction_id,),
            )

        await conn.commit()

    print(
        f"[evaluator] prediction {prediction_id}: "
        f"predicted={predicted_direction} actual={actual_direction} "
        f"correct={correct} pct={pct:.4f}"
    )
