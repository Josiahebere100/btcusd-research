"""Prediction snapshot system."""
import asyncio
import inspect
import json
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

from ..config import (
    PREDICTION_HORIZON_SECONDS,
    PREDICTION_SAFETY_BUFFER_MS,
)
from ..db import get_pool
from ..engines.base import Engine, EngineContext, EngineOutput
from ..engines.candlestick import CandlestickEngine
from ..engines.crt import CRTEngine
from ..engines.curve_geometry import CurveGeometryEngine
from ..engines.ensemble_kalman import run_ensemble_kalman
from ..engines.entropy_regime import EntropyRegimeEngine
from ..engines.labouchere import LabouchereEngine
from ..engines.momentum import MomentumEngine
from ..engines.navier_stokes import NavierStokesAutomatonEngine
from ..engines.ns_flow import NSFlowEngine
from ..engines.superformula import SuperformulaEngine
from ..engines.trail_tracer import TrailTracerEngine
from ..engines.trig_euler import TrigEulerEngine
from .combination import combination_key, lookup_combination


_engines: List[Engine] = [
    CRTEngine(),
    LabouchereEngine(),
    TrigEulerEngine(),
    EntropyRegimeEngine(),
    SuperformulaEngine(),
    NavierStokesAutomatonEngine(),
    CandlestickEngine(),
    MomentumEngine(),
    TrailTracerEngine(),
    CurveGeometryEngine(),
    NSFlowEngine(),
]


def get_engines() -> List[Engine]:
    return _engines


async def _run_engines(ctx: EngineContext) -> List[EngineOutput]:
    outputs: List[EngineOutput] = []
    for engine in _engines:
        try:
            if inspect.iscoroutinefunction(getattr(engine, "run_async", None)):
                out = await engine.run_async(ctx)
            else:
                out = engine.run(ctx)
            out.validate()
            outputs.append(out)
        except Exception as e:
            print(f"[engine] {engine.name} raised: {e}")
    return outputs


async def _pick_direction(outputs: List[EngineOutput]):
    signatures = {}
    for o in outputs:
        if o.pattern_signature:
            signatures[o.engine] = o.pattern_signature
        if o.direction in ("UP", "DOWN"):
            signatures[f"{o.engine}_dir"] = o.direction

    comb_key = combination_key(outputs)
    all_neutral = all(o.direction == "NO_SIGNAL" for o in outputs)

    if not all_neutral:
        try:
            lookup = await lookup_combination(comb_key)
        except Exception as e:
            print(f"[combination] lookup failed: {e}")
            lookup = None
        if lookup is not None:
            direction, confidence, _occ = lookup
            sig_json = json.dumps({**signatures, "_combination": comb_key})
            return direction, confidence, sig_json, "combination", comb_key

    if not all_neutral:
        for o in outputs:
            if o.direction != "NO_SIGNAL":
                sig_json = json.dumps({**signatures, "_combination": comb_key})
                return o.direction, o.confidence, sig_json, o.engine, comb_key

    sig_json = json.dumps({**signatures, "_combination": comb_key})
    primary = outputs[0].engine if outputs else "none"
    return "NO_SIGNAL", None, sig_json, primary, comb_key


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

    try:
        ek_out = await run_ensemble_kalman(outputs, ctx)
        if ek_out is not None:
            outputs = [ek_out] + outputs
    except Exception as e:
        print(f"[ensemble_kalman] failed: {e}")

    direction, confidence, sig_json, primary_source, _comb_key = (
        await _pick_direction(outputs)
    )

    try:
        from ..engines.baselines import run_baselines
        baseline_data = await run_baselines(ctx)
        data = json.loads(sig_json)
        data["baseline"] = baseline_data
        sig_json = json.dumps(data)
    except Exception as e:
        print(f"[baselines] failed: {e}")

    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
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
                    sig_json,
                    primary_source,
                ),
            )
            row = await cur.fetchone()
            prediction_id = row[0] if row else None
        await conn.commit()

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
