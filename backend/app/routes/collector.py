"""Collector ingestion routes.

Every route requires COLLECTOR_API_KEY. Every response is JSON.
NEVER returns HTML for any /api/* path.
"""
import asyncio
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter, Depends

from ..auth import require_collector_key
from ..db import get_pool
from ..models import (
    HealthBeacon,
    RoundBatch,
    SessionBatch,
    TickBatch,
)
from ..services.prediction import create_prediction_for_round
from ..services.research_memory import ResearchMemory
from ..state import (
    compute_feed_status,
    get_state,
    note_beacon,
    note_prediction_created,
    note_round,
    note_round_rejected,
    note_session,
    note_tick,
    note_tick_rejected,
    recent_ticks_snapshot,
)

router = APIRouter()

_research_memory = ResearchMemory()


# ---- Helpers --------------------------------------------------------------


def _iso_to_dt(value: Any):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        v = value.strip()
        if not v:
            return None
        if v.endswith("Z"):
            v = v[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(v)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            return None
    return None


# ---- GET /api/collector/health -------------------------------------------


@router.get("/health")
async def health_get(
    _auth: str = Depends(require_collector_key),
) -> Dict[str, Any]:
    state = get_state()
    feed = compute_feed_status()

    return {
        "collector_authenticated": True,
        "active_credential": _auth,
        "feed_status": feed["status"],
        "feed_reason": feed["reason"],
        "tick_age_ms": feed["tick_age_ms"],
        "round_age_ms": feed["round_age_ms"],
        "last_tick": state.last_tick_timestamp,
        "last_tick_price": state.last_tick_price,
        "last_tick_latency_ms": state.last_tick_latency_ms,
        "last_round": state.last_round_id,
        "current_session_id": state.current_session_id,
        "ticks_received": state.ticks_received,
        "rounds_received": state.rounds_received,
        "ticks_rejected": state.ticks_rejected,
        "rounds_rejected": state.rounds_rejected,
        "sessions_received": state.sessions_received,
        "predictions_created": state.predictions_created,
        "outcomes_evaluated": state.outcomes_evaluated,
        "last_beacon_at": (
            datetime.fromtimestamp(
                state.last_beacon_at, tz=timezone.utc
            ).isoformat()
            if state.last_beacon_at
            else None
        ),
    }


# ---- POST /api/collector/health (beacon) ---------------------------------


@router.post("/health")
async def health_post(
    beacon: HealthBeacon,
    _auth: str = Depends(require_collector_key),
) -> Dict[str, Any]:
    note_beacon(beacon.model_dump())
    return {"ok": True}


# ---- POST /api/collector/tick --------------------------------------------


@router.post("/tick")
async def ingest_ticks(
    batch: TickBatch,
    _auth: str = Depends(require_collector_key),
) -> Dict[str, Any]:
    if not batch.ticks:
        return {"ok": True, "inserted": 0, "rejected": 0}

    pool = await get_pool()
    inserted = 0
    rejected = 0

    async with pool.connection() as conn:
        for t in batch.ticks:
            if t.source != "BC.GAME" or t.symbol != "BTC-USD":
                note_tick_rejected()
                rejected += 1
                continue

            tick_ts = _iso_to_dt(t.tick_timestamp)
            recv_ts = _iso_to_dt(t.received_at)
            if tick_ts is None or recv_ts is None:
                note_tick_rejected()
                rejected += 1
                continue

            try:
                await conn.execute(
                    """
                    INSERT INTO sessions (session_id, start_timestamp, status, reason)
                    VALUES (%s, %s, 'ACTIVE', 'auto_created')
                    ON CONFLICT (session_id) DO NOTHING
                    """,
                    (t.session_id, tick_ts),
                )
                await conn.execute(
                    """
                    INSERT INTO market_ticks
                      (session_id, source, symbol, price, tick_timestamp,
                       received_at, latency_ms, feed_quality)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        t.session_id,
                        t.source,
                        t.symbol,
                        t.price,
                        tick_ts,
                        recv_ts,
                        t.latency_ms,
                        None,
                    ),
                )
                note_tick(t.model_dump())
                inserted += 1
            except Exception as e:
                print(f"[tick] insert failed: {e}")
                note_tick_rejected()
                rejected += 1

        await conn.commit()

    return {"ok": True, "inserted": inserted, "rejected": rejected}


# ---- POST /api/collector/round -------------------------------------------


@router.post("/round")
async def ingest_rounds(
    batch: RoundBatch,
    _auth: str = Depends(require_collector_key),
) -> Dict[str, Any]:
    if not batch.rounds:
        return {"ok": True, "upserted": 0, "rejected": 0}

    pool = await get_pool()
    upserted = 0
    rejected = 0
    state = get_state()

    async with pool.connection() as conn:
        for r in batch.rounds:
            if r.source != "BC.GAME" or r.symbol != "BTC/USD":
                note_round_rejected()
                rejected += 1
                continue

            round_start = _iso_to_dt(r.round_start_timestamp)
            cutoff = _iso_to_dt(r.trade_cutoff_timestamp)
            price_start = _iso_to_dt(r.price_start_timestamp)
            price_end = _iso_to_dt(r.price_end_timestamp)

            if round_start is None:
                note_round_rejected()
                rejected += 1
                continue

            try:
                await conn.execute(
                    """
                    INSERT INTO sessions (session_id, start_timestamp, status, reason)
                    VALUES (%s, %s, 'ACTIVE', 'auto_created')
                    ON CONFLICT (session_id) DO NOTHING
                    """,
                    (r.session_id, round_start),
                )
                await conn.execute(
                    """
                    INSERT INTO rounds
                      (session_id, round_id, symbol, round_start_timestamp,
                       trade_cutoff_timestamp, price_start_timestamp,
                       price_end_timestamp, start_price, end_price, win_side,
                       raw_direction)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (session_id, round_id) DO UPDATE SET
                      trade_cutoff_timestamp = EXCLUDED.trade_cutoff_timestamp,
                      price_start_timestamp = EXCLUDED.price_start_timestamp,
                      price_end_timestamp = EXCLUDED.price_end_timestamp,
                      start_price = EXCLUDED.start_price,
                      end_price = EXCLUDED.end_price,
                      win_side = EXCLUDED.win_side,
                      raw_direction = EXCLUDED.raw_direction
                    """,
                    (
                        r.session_id,
                        r.round_id,
                        r.symbol,
                        round_start,
                        cutoff,
                        price_start,
                        price_end,
                        r.start_price,
                        r.end_price,
                        r.win_side,
                        r.raw_direction,
                    ),
                )
                note_round(r.model_dump())
                upserted += 1

                # --- Resolve research state for the previous round on EVERY round. ---
                # This must not depend on whether a prediction is created for the
                # current round; the thesis labels state(t) with outcome(t+1).
                if (r.session_id, r.round_id) not in state.seen_round_ids:
                    asyncio.create_task(
                        _schedule_research_resolution(
                            session_id=r.session_id,
                            current_round_id=r.round_id,
                            settle_at=price_end,
                        )
                    )

                # --- Trigger prediction for a new round ---
                if (r.session_id, r.round_id) not in state.seen_round_ids:
                    state.seen_round_ids.add((r.session_id, r.round_id))
                    # Cap set size to avoid unbounded growth.
                    if len(state.seen_round_ids) > 10_000:
                        state.seen_round_ids = set(
                            list(state.seen_round_ids)[5000:]
                        )

                    if (
                        round_start is not None
                        and cutoff is not None
                        and state.last_tick_price is not None
                    ):
                        asyncio.create_task(
                            _maybe_create_prediction(
                                session_id=r.session_id,
                                round_id=r.round_id,
                                round_start=round_start,
                                trade_cutoff=cutoff,
                                price_start=price_start,
                                price_end=price_end,
                                current_price=float(state.last_tick_price),
                            )
                        )
            except Exception as e:
                print(f"[round] upsert failed: {e}")
                note_round_rejected()
                rejected += 1

        await conn.commit()

    return {"ok": True, "upserted": upserted, "rejected": rejected}


# ---- POST /api/collector/session -----------------------------------------


@router.post("/session")
async def ingest_sessions(
    batch: SessionBatch,
    _auth: str = Depends(require_collector_key),
) -> Dict[str, Any]:
    if not batch.sessions:
        return {"ok": True, "upserted": 0}

    pool = await get_pool()
    upserted = 0

    async with pool.connection() as conn:
        for s in batch.sessions:
            start_ts = _iso_to_dt(s.start_timestamp)
            end_ts = _iso_to_dt(s.end_timestamp)
            if start_ts is None:
                continue

            try:
                await conn.execute(
                    """
                    INSERT INTO sessions
                      (session_id, start_timestamp, end_timestamp,
                       gap_before_ms, status, reason)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (session_id) DO UPDATE SET
                      end_timestamp = EXCLUDED.end_timestamp,
                      status = EXCLUDED.status,
                      reason = EXCLUDED.reason
                    """,
                    (
                        s.session_id,
                        start_ts,
                        end_ts,
                        s.gap_before_ms,
                        s.status,
                        s.reason,
                    ),
                )
                note_session()
                upserted += 1
            except Exception as e:
                print(f"[session] upsert failed: {e}")

        await conn.commit()

    return {"ok": True, "upserted": upserted}


async def _schedule_research_resolution(
    session_id: str,
    current_round_id: str,
    settle_at,
) -> None:
    """Resolve state(t-1) from authoritative outcome of every current round.

    This is intentionally attached to round ingestion rather than prediction
    creation: the thesis requires next-round labels even when the current
    round has NO prediction. Session boundaries are enforced by ResearchMemory.
    """
    try:
        from datetime import timedelta
        now = datetime.now(timezone.utc)
        if settle_at is not None:
            delay = (settle_at - now).total_seconds() + 1.0
            if delay > 0:
                await asyncio.sleep(delay)
        await _research_memory.resolve_previous_round(
            session_id=session_id,
            current_round_id=current_round_id,
        )
    except Exception as e:
        print(f"[research_memory] resolution failed for round {current_round_id}: {e}")


# ---- Prediction helper ---------------------------------------------------


async def _maybe_create_prediction(
    session_id,
    round_id,
    round_start,
    trade_cutoff,
    price_start,
    price_end,
    current_price,
):
    try:
        pid = await create_prediction_for_round(
            session_id=session_id,
            round_id=round_id,
            round_start=round_start,
            trade_cutoff=trade_cutoff,
            price_start=price_start,
            price_end=price_end,
            current_price=current_price,
            recent_ticks=recent_ticks_snapshot(),
        )
        if pid is not None:
            note_prediction_created()
    except Exception as e:
        print(f"[prediction] failed for round {round_id}: {e}")
