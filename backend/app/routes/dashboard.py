"""Live dashboard endpoint.

Read-only view of the current round: countdown to trade cutoff, current
price, every engine's last-known direction, and recent outcomes.

Does not affect any engine.
"""
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends

from ..auth import require_collector_key
from ..db import get_pool

router = APIRouter()


# Engines shown in the grid. Excludes baselines (controls, not voters).
DASHBOARD_ENGINES = [
    "trig_euler",
    "trajectory",
    "projectile",
    "candlestick",
    "entropy_regime",
    "momentum",
    "trail_tracer",
    "curve_geometry",
    "navier_stokes",
    "ns_flow",
    "labouchere",
    "superformula",
    "crt",
    "combination",
]


def _iso(dt):
    return dt.isoformat() if dt else None


@router.get("/live")
async def live_dashboard(
    _auth: str = Depends(require_collector_key),
) -> Dict[str, Any]:
    pool = await get_pool()
    now = datetime.now(timezone.utc)

    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            # 1. Latest prediction
            await cur.execute(
                """
                SELECT id, round_id, direction, confidence, engine,
                       pattern_signature, prediction_timestamp,
                       target_timestamp, status
                FROM prediction_snapshots
                ORDER BY prediction_timestamp DESC
                LIMIT 1
                """
            )
            pred = await cur.fetchone()

            # 2. Round details for the latest prediction's round
            round_info = None
            if pred:
                round_id = pred[1]
                await cur.execute(
                    """
                    SELECT round_start_timestamp, trade_cutoff_timestamp,
                           price_start_timestamp, price_end_timestamp,
                           status_code, win_side, start_price, end_price
                    FROM rounds
                    WHERE round_id = %s
                    ORDER BY id DESC
                    LIMIT 1
                    """,
                    (round_id,),
                )
                rr = await cur.fetchone()
                if rr:
                    round_info = {
                        "round_start": _iso(rr[0]),
                        "trade_cutoff": _iso(rr[1]),
                        "price_start": _iso(rr[2]),
                        "price_end": _iso(rr[3]),
                        "status_code": rr[4],
                        "win_side": rr[5],
                        "start_price": float(rr[6]) if rr[6] is not None else None,
                        "end_price": float(rr[7]) if rr[7] is not None else None,
                    }

            # 3. Latest tick
            await cur.execute(
                """
                SELECT tick_timestamp, price
                FROM market_ticks
                ORDER BY tick_timestamp DESC
                LIMIT 1
                """
            )
            tick = await cur.fetchone()

            # 4. Price history (last 60s) for the mini chart
            await cur.execute(
                """
                SELECT tick_timestamp, price
                FROM market_ticks
                WHERE tick_timestamp >= now() - interval '60 seconds'
                ORDER BY tick_timestamp ASC
                """
            )
            price_rows = await cur.fetchall()

            # 5. Recent outcomes (last 20)
            await cur.execute(
                """
                SELECT p.prediction_timestamp, p.direction, p.engine,
                       o.actual_direction, o.correct
                FROM outcomes o
                JOIN prediction_snapshots p ON p.id = o.prediction_id
                ORDER BY p.prediction_timestamp DESC
                LIMIT 20
                """
            )
            outcome_rows = await cur.fetchall()

    # ---- Parse engines from the latest prediction's JSON ----
    engines: List[Dict[str, Any]] = []
    if pred and pred[5]:
        try:
            sig = json.loads(pred[5])
        except Exception:
            sig = {}
        for name in DASHBOARD_ENGINES:
            direction = sig.get(f"{name}_dir")
            engines.append({
                "name": name,
                "direction": direction or "NO_SIGNAL",
                "signature": sig.get(name),
                "is_firing": direction in ("UP", "DOWN"),
            })

    recent_outcomes = []
    for r in outcome_rows:
        recent_outcomes.append({
            "time": _iso(r[0]),
            "predicted": r[1],
            "engine": r[2],
            "actual": r[3],
            "correct": r[4],
        })

    # ---- Time math (client will use these to run its own countdown) ----
    trade_cutoff_iso = round_info["trade_cutoff"] if round_info else None
    price_end_iso = round_info["price_end"] if round_info else None

    seconds_to_cutoff = None
    seconds_to_price_end = None
    if trade_cutoff_iso:
        try:
            tc = datetime.fromisoformat(trade_cutoff_iso)
            seconds_to_cutoff = (tc - now).total_seconds()
        except Exception:
            pass
    if price_end_iso:
        try:
            pe = datetime.fromisoformat(price_end_iso)
            seconds_to_price_end = (pe - now).total_seconds()
        except Exception:
            pass

    return {
        "now": now.isoformat(),
        "round": (
            {
                "round_id": pred[1],
                **round_info,
            } if pred and round_info else None
        ),
        "seconds_to_cutoff": seconds_to_cutoff,
        "seconds_to_price_end": seconds_to_price_end,
        "prediction": (
            {
                "prediction_id": pred[0],
                "prediction_timestamp": _iso(pred[6]),
                "target_timestamp": _iso(pred[7]),
                "direction": pred[2],
                "confidence": float(pred[3]) if pred[3] is not None else None,
                "engine": pred[4],
                "status": pred[8],
            } if pred else None
        ),
        "current_price": float(tick[1]) if tick else None,
        "current_tick_at": _iso(tick[0]) if tick else None,
        "price_history": [
            {"t": _iso(r[0]), "p": float(r[1])} for r in price_rows
        ],
        "engines": engines,
        "recent_outcomes": recent_outcomes,
    }
