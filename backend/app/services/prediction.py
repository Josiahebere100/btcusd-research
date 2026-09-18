"""Prediction snapshot system."""
import asyncio
import inspect
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

from ..config import (
    PREDICTION_HORIZON_SECONDS,
    PREDICTION_SAFETY_BUFFER_MS,
)
from ..db import get_pool
from ..engines.base import Engine, EngineContext, EngineOutput
from ..engines.crt import CRTEngine
from ..engines.stub import StubEngine


# ---- Engine registry -------------------------------------------------------

_engines: List[Engine] = [
    CRTEngine(),
    # StubEngine() intentionally disabled once CRT is registered.
]


def get_engines() -> List[Engine]:
    return _engines


# ---- Engine execution -----------------------------------------------------


async def _run_engines(ctx: EngineContext) -> List[EngineOutput]:
    outputs: List[EngineOutput] = []
    for engine in _engines:
        try:
            if engine.is_async or inspect.iscoroutinefunction(
                getattr(engine, "run_async", None)
            ):
                out = await engine.run_async(ctx)
            else:
                out = engine.run(ctx)
            out.validate()
            outputs.append(out)
        except Exception as e:
            print(f"[engine] {engine.name} raised: {e}")
    return outputs


def _pick_direction(
    outputs: List[EngineOutput],
) -> Tuple[str, Optional[float], Optional[str]]:
    """Choose a combined direction and return the (direction, confidence,
    pattern_signature). Priority: first non-NO_SIGNAL engine output."""
    for o in outputs:
        if o.direction != "NO_SIGNAL":
            return o.direction, o.confidence, o.pattern_signature
    # No signal from any engine; still record the first pattern signature
    # if present for research purposes.
    for o in outputs:
        if o.pattern_signature:
            return "NO_SIGNAL", None, o.pattern_signature
    return "NO_SIGNAL", None, None


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
    horizon = horizon_seconds or PREDICTION_HORIZON_SECONDS
    now = datetime.now(timezone.utc)
    target = now + timedelta(seconds=horizon)

    lock_deadline = trade_cutoff - timedelta(
        milliseconds=PREDICTION_SAFETY_BUFFER_MS
    )
    lead_time_ms = int((lock_deadline - now).total_seconds() * 1000)

    if now >= trade_cutoff:
        status = "INVALID"
    elif lead_time_ms < PREDICTION_SAFETY_BUFFER_MS:
        status = "LATE"
    else:
        status = "LOCKED"

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
    outputs = await _run_engines(ctx)
    direction, confidence, pattern_sig = _pick_direction(outputs)

    # Record the engine that produced the signal (or the first engine).
    primary_engine = "none"
    for o in outputs:
        if o.direction != "NO_SIGNAL":
            primary_engine = o.engine
            break
    if primary_engine == "none" and outputs:
        primary_engine = outputs[0].engine

    pool = await get_pool()
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
                primary_engine,
            ),
        )
        await conn.commit()
        prediction_id = row["id"] if row else None

    if prediction_id is None:
        return None

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
    now = datetime.now(timezone.utc)
    delay = (target - now).total_seconds()
    if delay > 0:
        await asyncio.sleep(delay)
    from .evaluator import evaluate_prediction

    try:
        await evaluate_prediction(prediction_id, predicted_direction)
    except Exception as e:
        print(f"[evaluator] failed for prediction {prediction_id}: {e}")
