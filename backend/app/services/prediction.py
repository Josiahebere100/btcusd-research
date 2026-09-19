"""Combination engine with Wilson confidence interval.

Fires only when a combination's recent outcome bias is statistically
significant (95% confidence interval strictly above or below 0.50).
Uses a sliding window over `combination_outcomes`.
"""
import math
from typing import Dict, List, Optional, Tuple

from ..db import get_pool
from ..engines.base import EngineOutput

MIN_OCCURRENCES = 20
MIN_RECENT_OCCURRENCES = 40
RECENT_WINDOW = 50
CONFIDENCE_Z = 1.96


def combination_key(outputs: List[EngineOutput]) -> str:
    parts = [f"{o.engine}:{o.direction}" for o in outputs]
    parts.sort()
    return "|".join(parts)


def _wilson_bounds(p: float, n: int, z: float = CONFIDENCE_Z) -> Tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = p + z2 / (2.0 * n)
    margin = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    lower = (center - margin) / denom
    upper = (center + margin) / denom
    return (max(0.0, lower), min(1.0, upper))


async def lookup_combination(key: str) -> Optional[Tuple[str, float, int]]:
    """Return (direction, confidence, occurrences) if statistically
    significant; else None.

    Uses recent_accuracy over the last RECENT_WINDOW outcomes, tested
    via the Wilson 95% confidence interval.
    """
    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            # Lifetime count check
            await cur.execute(
                """
                SELECT occurrences
                FROM combination_signals
                WHERE combination = %s
                """,
                (key,),
            )
            row = await cur.fetchone()
            if not row:
                return None
            occurrences = int(row[0] or 0)
            if occurrences < MIN_OCCURRENCES:
                return None

            # Recent window from combination_outcomes
            await cur.execute(
                """
                SELECT actual_direction
                FROM combination_outcomes
                WHERE combination = %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (key, RECENT_WINDOW),
            )
            recent = await cur.fetchall()

    n_total = len(recent)
    if n_total < MIN_RECENT_OCCURRENCES:
        return None

    n_up = sum(1 for r in recent if r[0] == "UP")
    p_up = n_up / n_total
    lower, upper = _wilson_bounds(p_up, n_total)

    if lower > 0.50:
        return ("UP", lower, n_total)
    if upper < 0.50:
        return ("DOWN", 1.0 - upper, n_total)
    return None


async def record_combination_outcome(
    key: str,
    actual_direction: str,
    session_id: Optional[str] = None,
    prediction_id: Optional[int] = None,
) -> None:
    """Write the outcome to combination_outcomes and refresh aggregate stats."""
    if actual_direction not in ("UP", "DOWN"):
        return

    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            # Insert the per-outcome record.
            await cur.execute(
                """
                INSERT INTO combination_outcomes
                  (combination, session_id, prediction_id, actual_direction)
                VALUES (%s, %s, %s, %s)
                """,
                (key, session_id, prediction_id, actual_direction),
            )

            # Upsert aggregate counters.
            if actual_direction == "UP":
                await cur.execute(
                    """
                    INSERT INTO combination_signals
                      (combination, occurrences, correct, incorrect)
                    VALUES (%s, 1, 1, 0)
                    ON CONFLICT (combination) DO UPDATE SET
                      occurrences = combination_signals.occurrences + 1,
                      correct = combination_signals.correct + 1
                    """,
                    (key,),
                )
            else:
                await cur.execute(
                    """
                    INSERT INTO combination_signals
                      (combination, occurrences, correct, incorrect)
                    VALUES (%s, 1, 0, 1)
                    ON CONFLICT (combination) DO UPDATE SET
                      occurrences = combination_signals.occurrences + 1,
                      incorrect = combination_signals.incorrect + 1
                    """,
                    (key,),
                )

            # Refresh recent_accuracy from the last RECENT_WINDOW outcomes.
            await cur.execute(
                """
                SELECT
                    COUNT(*) FILTER (WHERE actual_direction = 'UP') AS n_up,
                    COUNT(*) AS n_total
                FROM (
                    SELECT actual_direction
                    FROM combination_outcomes
                    WHERE combination = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                ) sub
                """,
                (key, RECENT_WINDOW),
            )
            ra = await cur.fetchone()
            if ra and ra[1] and int(ra[1]) > 0:
                recent_up = int(ra[0] or 0)
                recent_total = int(ra[1])
                recent_acc = recent_up / recent_total
            else:
                recent_acc = None

            # Refresh accuracy and combined_output.
            await cur.execute(
                """
                UPDATE combination_signals
                SET accuracy = correct::numeric / NULLIF(correct + incorrect, 0),
                    recent_accuracy = %s,
                    combined_output = CASE
                      WHEN %s >= 0.55 THEN 'UP'
                      WHEN %s <= 0.45 THEN 'DOWN'
                      ELSE 'NO_SIGNAL'
                    END
                WHERE combination = %s
                """,
                (recent_acc, recent_acc, recent_acc, key),
            )

        await conn.commit()
