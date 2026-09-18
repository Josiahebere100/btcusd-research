"""Pattern memory: records each signature's outcome."""
from typing import Optional

from ..db import get_pool


async def record_signature_outcome(
    signature: str,
    session_id: str,
    prediction_id: int,
    engine: str,
    actual_direction: str,       # "UP" or "DOWN"
    prediction_direction: str,   # "UP", "DOWN", or "NO_SIGNAL"
) -> None:
    if actual_direction not in ("UP", "DOWN"):
        return

    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO patterns (pattern_signature, pattern_type,
                                      occurrence_count, last_seen)
                VALUES (%s, 'primary', 1, now())
                ON CONFLICT (pattern_signature) DO UPDATE
                  SET occurrence_count = patterns.occurrence_count + 1,
                      last_seen = now()
                RETURNING id
                """,
                (signature,),
            )
            row = await cur.fetchone()
            if row is None:
                return
            pattern_id = row[0]

            if prediction_direction in ("UP", "DOWN"):
                is_correct = prediction_direction == actual_direction
                if is_correct:
                    await cur.execute(
                        """
                        UPDATE patterns
                        SET correct_count = correct_count + 1,
                            accuracy = (correct_count + 1)::numeric
                                     / NULLIF(correct_count + incorrect_count + 1, 0)
                        WHERE id = %s
                        """,
                        (pattern_id,),
                    )
                else:
                    await cur.execute(
                        """
                        UPDATE patterns
                        SET incorrect_count = incorrect_count + 1,
                            accuracy = correct_count::numeric
                                     / NULLIF(correct_count + incorrect_count + 1, 0)
                        WHERE id = %s
                        """,
                        (pattern_id,),
                    )

            if actual_direction == "UP":
                await cur.execute(
                    """
                    INSERT INTO pattern_success_memory
                      (pattern_id, session_id, prediction_id, result, engine)
                    VALUES (%s, %s, %s, 'UP', %s)
                    """,
                    (pattern_id, session_id, prediction_id, engine),
                )
            else:
                await cur.execute(
                    """
                    INSERT INTO pattern_failure_memory
                      (pattern_id, session_id, prediction_id, result, engine,
                       failure_type)
                    VALUES (%s, %s, %s, 'DOWN', %s, 'market_down')
                    """,
                    (pattern_id, session_id, prediction_id, engine),
                )

        await conn.commit()
