"""Prediction snapshot system.

Creates immutable snapshots of engine outputs for each active round,
locks them before trade cutoff, and evaluates them at target time.

Every snapshot is immutable. Historical predictions are never rewritten.
"""
import asyncio
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

from ..config import (
    PREDICTION_HORIZON_SECONDS,
    PREDICTION_SAFETY_BUFFER_MS,
)
from ..db import get_pool
from ..engines.base import Engine, EngineContext, EngineOutput
from ..engines.stub import StubEngine


# ---- Engine registry -------------------------------------------------------

_engines: List[Engine] = [StubEngine()]


def get_engines() -> List[Engine]:
    return _engines


# ---- Helpers --------------------------------------------------------------


def _run_engines(ctx: EngineContext) -> List[EngineOutput]:
    outputs: List[EngineOutput] = []
    for engine in _engines:
        try:
            out = engine.run(ctx)
            out.validate()
            outputs.append(out)
        except Exception as e:
            print(f"[engine] {engine.name} raised: {e}")
    return outputs


def _pick_direction(outputs: List[EngineOutput]) -> Tuple[str, Optional[float]]:
    """For Phase 11, the primary engine is the stub, so we just return
    whatever it says. When multiple engines exist, this function will
    consult the combination statistics table (Phase 17)."""
    for o in outputs:
        if o.direction != "NO_SIGNAL":
            return o.direction, o.confidence
    return "NO_SIGNAL", None


# ---- Snapshot creation ----------------------------------------------------


async def create_prediction_for_round(
    session_id: str,
    round_id: str,
    round_start: datetime,
    trade_cutoff: datetime,
    price_start: Optional[datetime],
    price_end: Optional[datetime],
    current_price: float,
    recent_ticks: List[Tuple[float, float]],
    horizon_seconds: Optional[int] = None,
) -> Optional[int]:
    """Create a prediction snapshot for a round.

    Returns the prediction row id, or None if it could not be created.
    """
    horizon = horizon_seconds or PREDICTION_HORIZON_SECONDS
    now = datetime.now(timezone.utc)

    # Compute target timestamp. Existing predictions retain their original
    # horizon even if the setting later changes.
    target = now + timedelta(seconds=horizon)

    # Compute lock deadline.
    lock_deadline = trade_cutoff - timedelta(
        milliseconds=PREDICTION_SAFETY_BUFFER_MS
    )
    lead_time_ms = int((lock_deadline - now).total_seconds() * 1000)

    # Determine status
    if now >= trade_cutoff:
        status = "INVALID"
    elif lead_time_ms < PREDICTION_SAFETY_BUFFER_MS:
        status = "LATE"
    else:
        status = "LOCKED"

    # Run engines
    ctx = EngineContext(
        session_id=session_id,
        round_id=round_id,
        round_start_timestamp=round_start,
        trade_cutoff_timestamp=trade_cutoff,
        price_start_timestamp=price_start,
        price_end_timestamp=price_end,
        current_time=now,
        current_price=current_price,
        recent_ticks=recent_ticks,
    )
    outputs = _run_engines(ctx)
    direction, confidence = _pick_direction(outputs)

    # Pick the pattern signature from the primary engine if any
    pattern_sig = None
    for o in outputs:
        if o.pattern_signature:
            pattern_sig = o.pattern_signature
            break

    # Store the snapshot.
    pool = await get_pool()
    async with pool.connection() as conn:
        row = await conn.execute(
            """
            INSERT INTO prediction_snapshots
              (session_id, round_id, symbol, source, horizon_seconds,
               prediction_timestamp, information_cutoff_timestamp,
               target_timestamp, price_at_prediction, direction,
               confidence, lead_time_ms, status, pattern_signature, engine)
            VALUES
              (%s, %s, 'BTC/USD', 'BC.GAME', %s,
               %s, %s,
               %s, %s, %s,
               %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                session_id,
                round_id,
                horizon,
                now,
                now,
                target,
                current_price,
                direction,
                confidence,
                lead_time_ms,
                status,
                pattern_sig,
                outputs[0].engine if outputs else "none",
            ),
        )
        # asyncpg returns "INSERT 0 1"; psycopg3 doesn't RETURNING via execute.
        # Use fetchone instead.
        pass

        # Re-do with fetchrow for portability.
    async with pool.connection() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO prediction_snapshots
              (session_id, round_id, symbol, source, horizon_seconds,
               prediction_timestamp, information_cutoff_timestamp,
               target_timestamp, price_at_prediction, direction,
               confidence, lead_time_ms, status, pattern_signature, engine)
            VALUES
              (%s, %s, 'BTC/USD', 'BC.GAME', %s,
               %s, %s,
               %s, %s, %s,
               %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                session_id,
                round_id,
                horizon,
                now,
                now,
                target,
                current_price,
                direction,
                confidence,
                lead_time_ms,
                status,
                pattern_sig,
                outputs[0].engine if outputs else "none",
            ),
        )
        await conn.commit()
        prediction_id = row["id"] if row else None

    if prediction_id is None:
        return None

    # Schedule evaluation at target time (fire and forget).
    if status in ("LOCKED", "EVALUATED"):
        asyncio.create_task(
            _schedule_evaluation(prediction_id, target, direction)
        )

    return prediction_id


async def _schedule_evaluation(
    prediction_id: int,
    target: datetime,
    predicted_direction: str,
) -> None:
    """Sleep until target time, then evaluate."""
    now = datetime.now(timezone.utc)
    delay = (target - now).total_seconds()
    if delay > 0:
        await asyncio.sleep(delay)
    # Import here to avoid circular imports.
    from .evaluator import evaluate_prediction

    try:
        await evaluate_prediction(prediction_id, predicted_direction)
    except Exception as e:
        print(f"[evaluator] failed for prediction {prediction_id}: {e}")
