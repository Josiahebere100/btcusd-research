"""Combination engine.

Records the tuple of (engine, direction) from every prediction, and
tracks what the market actually did after each combination appeared.

Direction is determined by empirical frequency, NOT by vote. A combination
only emits a signal when it has >=10 samples with >=55% directional bias.
"""
from typing import List, Optional, Tuple

from ..db import get_pool
from ..engines.base import EngineOutput

MIN_OCCURRENCES = 10
MIN_BIAS = 0.55


def combination_key(outputs: List[EngineOutput]) -> str:
    """Build a stable, sorted combination key from all engine outputs."""
    parts = [f"{o.engine}:{o.direction}" for o in outputs]
    parts.sort()
    return "|".join(parts)


def _implied_from_history(correct: int, incorrect: int) -> Optional[str]:
    """Given historical counts, return 'UP' or 'DOWN' if bias is sufficient."""
    total = correct + incorrect
    if total < MIN_OCCURRENCES:
        return None
    p_up = correct / total
    if p_up >= MIN_BIAS:
        return "UP"
    if p_up <= (1 - MIN_BIAS):
        return "DOWN"
    return None


async def lookup_combination(key: str) -> Optional[Tuple[str, float, int]]:
    """Return (direction, confidence, occurrences) if this combination
    has enough history and sufficient bias; else None."""
    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT occurrences, correct, incorrect
                FROM combination_signals
                WHERE combination = %s
                """,
                (key,),
            )
            row = await cur.fetchone()
            if not row:
                return None
            occurrences = int(row[0] or 0)
            correct = int(row[1] or 0)
            incorrect = int(row[2] or 0)

            direction = _implied_from_history(correct, incorrect)
            if direction is None:
                return None

            total = correct + incorrect
            p_up = correct / total
            confidence = p_up if direction == "UP" else (1 - p_up)
            return direction, confidence, total


async def record_combination_outcome(
    key: str,
    actual_direction: str,  # "UP" | "DOWN" | "FLAT"
) -> None:
    """Called by the evaluator after each outcome.
    correct = market went UP; incorrect = market went DOWN (tracking bias)."""
    if actual_direction not in ("UP", "DOWN"):
        return
    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
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
            # Refresh the derived fields.
            await cur.execute(
                """
                UPDATE combination_signals
                SET accuracy = correct::numeric / NULLIF(correct + incorrect, 0),
                    recent_accuracy = correct::numeric / NULLIF(correct + incorrect, 0),
                    combined_output = CASE
                      WHEN correct::numeric / NULLIF(correct + incorrect, 0) >= 0.55 THEN 'UP'
                      WHEN correct::numeric / NULLIF(correct + incorrect, 0) <= 0.45 THEN 'DOWN'
                      ELSE 'NO_SIGNAL'
                    END
                WHERE combination = %s
                """,
                (key,),
            )
        await conn.commit()
