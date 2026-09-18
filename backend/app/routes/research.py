"""Research routes: read-only endpoints for the visualization frontend."""
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query

from ..auth import require_collector_key
from ..db import get_pool

router = APIRouter()

# Singularity tolerance: |cos| < this -> tan/sec undefined. |sin| < this -> cot/csc undefined.
SINGULARITY_EPS = 0.02

# Reference window used for stable theta mapping.
# We compute the price range over the last N minutes and hold it fixed
# for the returned window. This keeps theta smooth across polls.
REFERENCE_WINDOW_MINUTES = 60


def _safe(v: float, kind: str, theta: float) -> Optional[float]:
    if kind == "tan":
        c = math.cos(theta)
        if abs(c) < SINGULARITY_EPS:
            return None
        return math.sin(theta) / c
    if kind == "cot":
        s = math.sin(theta)
        if abs(s) < SINGULARITY_EPS:
            return None
        return math.cos(theta) / s
    if kind == "sec":
        c = math.cos(theta)
        if abs(c) < SINGULARITY_EPS:
            return None
        return 1.0 / c
    if kind == "csc":
        s = math.sin(theta)
        if abs(s) < SINGULARITY_EPS:
            return None
        return 1.0 / s
    raise ValueError(kind)


@router.get("/trig-state")
async def trig_state(
    seconds: int = Query(120, ge=10, le=1800),
    _auth: str = Depends(require_collector_key),
) -> Dict[str, Any]:
    """Return the last N seconds of ticks enriched with theta and trig values.

    Theta is computed by mapping the price onto [0, 2π) using a reference
    range taken from the last REFERENCE_WINDOW_MINUTES minutes. This gives
    stable, smooth theta values across successive calls.
    """
    pool = await get_pool()
    now = datetime.now(timezone.utc)
    since = now - timedelta(seconds=seconds)
    ref_since = now - timedelta(minutes=REFERENCE_WINDOW_MINUTES)

    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            # Reference range (for stable theta)
            await cur.execute(
                """
                SELECT MIN(price), MAX(price)
                FROM market_ticks
                WHERE tick_timestamp >= %s
                """,
                (ref_since,),
            )
            ref_row = await cur.fetchone()
            ref_low = float(ref_row[0]) if ref_row and ref_row[0] is not None else None
            ref_high = float(ref_row[1]) if ref_row and ref_row[1] is not None else None

            # The ticks we're going to return.
            await cur.execute(
                """
                SELECT tick_timestamp, price
                FROM market_ticks
                WHERE tick_timestamp >= %s
                ORDER BY tick_timestamp ASC
                """,
                (since,),
            )
            rows = await cur.fetchall()

    if ref_low is None or ref_high is None or ref_high <= ref_low:
        # Fall back to the range of the returned window itself.
        if rows:
            prices = [float(r[1]) for r in rows]
            ref_low = min(prices)
            ref_high = max(prices)
        else:
            ref_low = 0.0
            ref_high = 1.0

    span = ref_high - ref_low
    if span <= 0:
        span = 1.0

    points: List[Dict[str, Any]] = []
    for row in rows:
        ts: datetime = row[0]
        price = float(row[1])

        theta = 2.0 * math.pi * (price - ref_low) / span
        sin_t = math.sin(theta)
        cos_t = math.cos(theta)
        tan_t = _safe(price, "tan", theta)
        cot_t = _safe(price, "cot", theta)
        sec_t = _safe(price, "sec", theta)
        csc_t = _safe(price, "csc", theta)

        singular = sum(1 for v in (tan_t, cot_t, sec_t, csc_t) if v is None)

        points.append(
            {
                "t": int(ts.timestamp() * 1000),
                "p": price,
                "th": theta,
                "sin": sin_t,
                "cos": cos_t,
                "tan": tan_t,
                "cot": cot_t,
                "sec": sec_t,
                "csc": csc_t,
                "sg": singular,
            }
        )

    return {
        "symbol": "BTC/USD",
        "source": "BC.GAME",
        "window_seconds": seconds,
        "reference_window_minutes": REFERENCE_WINDOW_MINUTES,
        "reference_low": ref_low,
        "reference_high": ref_high,
        "now": now.isoformat(),
        "count": len(points),
        "points": points,
    }
