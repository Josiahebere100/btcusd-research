"""Shared pattern lookup used by every engine.

Uses the Wilson score interval (95% confidence) to decide whether a
pattern's recent directional bias is statistically real. Filters out
patterns that crossed the threshold by chance in small samples.

A pattern fires only when:
  - it has >= MIN_OCCURRENCES lifetime occurrences
  - it has >= MIN_RECENT_OCCURRENCES in the recent window
  - the 95% confidence interval of its recent UP fraction is strictly
    above 0.50 (fire UP) or strictly below 0.50 (fire DOWN)
"""
import math
from typing import Optional, Tuple

from ..db import get_pool

MIN_OCCURRENCES = 20
MIN_RECENT_OCCURRENCES = 40
RECENT_WINDOW = 50
CONFIDENCE_Z = 1.96  # 95% two-sided


def _wilson_bounds(p: float, n: int, z: float = CONFIDENCE_Z) -> Tuple[float, float]:
    """Return (lower, upper) Wilson score interval bounds for a binomial
    proportion p observed in n samples."""
    if n <= 0:
        return (0.0, 1.0)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = p + z2 / (2.0 * n)
    margin = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    lower = (center - margin) / denom
    upper = (center + margin) / denom
    return (max(0.0, lower), min(1.0, upper))


async def lookup_direction(signature: str) -> Optional[Tuple[str, int, float]]:
    """Return (direction, occurrences, confidence) for a signature.

    Direction is decided by the Wilson 95% confidence interval of the
    recent UP fraction. Confidence returned is the interval bound we're
    confident about (lower for UP, 1-upper for DOWN).
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
            lifetime_count = int(row[1] or 0)
            if lifetime_count < MIN_OCCURRENCES:
                return None

            # Recent outcomes only.
            await cur.execute(
                """
                SELECT
                    COUNT(*) FILTER (WHERE result = 'UP') AS n_up,
                    COUNT(*) AS n_total
                FROM (
                    SELECT created_at, 'UP' AS result
                    FROM pattern_success_memory
                    WHERE pattern_id = %s
                    UNION ALL
                    SELECT created_at, 'DOWN' AS result
                    FROM pattern_failure_memory
                    WHERE pattern_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                ) combined
                """,
                (pattern_id, pattern_id, RECENT_WINDOW),
            )
            r = await cur.fetchone()
            if not r:
                return None
            n_up = int(r[0] or 0)
            n_total = int(r[1] or 0)
            if n_total < MIN_RECENT_OCCURRENCES:
                return None

            p_up = n_up / n_total
            lower, upper = _wilson_bounds(p_up, n_total)

            # Fire UP only when we're 95% confident P(UP) > 0.50.
            if lower > 0.50:
                return ("UP", n_total, lower)

            # Fire DOWN only when we're 95% confident P(UP) < 0.50.
            if upper < 0.50:
                return ("DOWN", n_total, 1.0 - upper)

            return None
