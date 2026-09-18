"""Shared pattern lookup used by every engine."""
from typing import Optional, Tuple

from ..db import get_pool

MIN_OCCURRENCES = 10
MIN_BIAS = 0.55


async def lookup_direction(signature: str) -> Optional[Tuple[str, int, float]]:
    """Return (direction, occurrences, confidence) for a signature.

    Direction is the historically dominant market move following this
    signature (UP or DOWN). Returns None if insufficient evidence exists.
    """
    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT id, occurrence_count
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
            if occurrence_count < MIN_OCCURRENCES:
                return None

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

            p_up = n_up / total
            if p_up >= MIN_BIAS:
                return ("UP", total, p_up)
            if p_up <= (1 - MIN_BIAS):
                return ("DOWN", total, 1 - p_up)
            return None
