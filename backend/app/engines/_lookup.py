"""Shared pattern lookup used by every engine.

Uses recent_accuracy (last N outcomes) when available, falling back to
lifetime accuracy. This protects against concept drift: a pattern whose
bias held for weeks but stopped holding yesterday will silence itself
rather than keep firing wrong signals.
"""
from typing import Optional, Tuple

from ..db import get_pool

MIN_OCCURRENCES = 10
MIN_BIAS = 0.55


async def lookup_direction(signature: str) -> Optional[Tuple[str, int, float]]:
    """Return (direction, occurrences, confidence) for a signature.

    Direction is determined by the recent (sliding-window) bias toward
    UP vs DOWN. Falls back to lifetime bias if recent is not yet computed.
    Returns None if insufficient evidence exists.
    """
    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT id, occurrence_count, recent_accuracy
                FROM patterns
                WHERE pattern_signature = %s
                """,
                (signature,),
            )
            row = await cur.fetchone()
            if not row:
                return None
            pattern_id = row[0]
            occurrence_count = int(row[1] or 0)
            recent_accuracy = float(row[2]) if row[2] is not None else None

            if occurrence_count < MIN_OCCURRENCES:
                return None

            # Count historical UP vs DOWN for this pattern (for fallback).
            await cur.execute(
                "SELECT COUNT(*) FROM pattern_success_memory WHERE pattern_id = %s",
                (pattern_id,),
            )
            up_row = await cur.fetchone()
            n_up = int(up_row[0]) if up_row else 0

            await cur.execute(
                "SELECT COUNT(*) FROM pattern_failure_memory WHERE pattern_id = %s",
                (pattern_id,),
            )
            down_row = await cur.fetchone()
            n_down = int(down_row[0]) if down_row else 0

            total = n_up + n_down
            if total < MIN_OCCURRENCES:
                return None

            # p_up is our best current estimate of P(market UP | this signature).
            # Prefer recent_accuracy (sliding window). Fall back to lifetime.
            if recent_accuracy is not None:
                p_up = recent_accuracy
            else:
                p_up = n_up / total

            if p_up >= MIN_BIAS:
                return ("UP", total, p_up)
            if p_up <= (1 - MIN_BIAS):
                return ("DOWN", total, 1 - p_up)
            return None
