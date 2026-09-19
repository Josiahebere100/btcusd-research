"""Signals endpoint: per-engine prediction stats and recent signals."""
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, Query

from ..auth import require_collector_key
from ..db import get_pool

router = APIRouter()


@router.get("/recent")
async def recent_signals(
    limit: int = Query(1000, ge=1, le=10000),
    hours: int = Query(24, ge=1, le=87600),
    _auth: str = Depends(require_collector_key),
) -> Dict[str, Any]:
    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT
                    p.id,
                    p.prediction_timestamp,
                    p.direction,
                    p.confidence,
                    p.engine,
                    p.pattern_signature,
                    p.target_timestamp,
                    p.horizon_seconds,
                    p.price_at_prediction,
                    o.actual_price,
                    o.actual_direction,
                    o.correct,
                    o.evaluated_at
                FROM prediction_snapshots p
                LEFT JOIN outcomes o ON o.prediction_id = p.id
                WHERE p.direction IN ('UP', 'DOWN')
                  AND p.prediction_timestamp >= now() - (%s * interval '1 hour')
                ORDER BY p.prediction_timestamp DESC
                LIMIT %s
                """,
                (hours, limit),
            )
            rows = await cur.fetchall()

            signals: List[Dict[str, Any]] = []
            for r in rows:
                signals.append({
                    "id": r[0],
                    "prediction_timestamp": r[1].isoformat() if r[1] else None,
                    "direction": r[2],
                    "confidence": float(r[3]) if r[3] is not None else None,
                    "engine": r[4],
                    "pattern_signature": r[5],
                    "target_timestamp": r[6].isoformat() if r[6] else None,
                    "horizon_seconds": r[7],
                    "price_at_prediction": float(r[8]) if r[8] is not None else None,
                    "actual_price": float(r[9]) if r[9] is not None else None,
                    "actual_direction": r[10],
                    "correct": r[11],
                    "evaluated_at": r[12].isoformat() if r[12] else None,
                })

            await cur.execute(
                """
                SELECT
                    p.engine,
                    COUNT(*) AS total,
                    COUNT(*) FILTER (WHERE o.correct = true) AS correct_count,
                    COUNT(*) FILTER (WHERE o.correct = false) AS incorrect_count,
                    COUNT(*) FILTER (WHERE o.correct IS NULL) AS pending_count
                FROM prediction_snapshots p
                LEFT JOIN outcomes o ON o.prediction_id = p.id
                WHERE p.direction IN ('UP', 'DOWN')
                  AND p.prediction_timestamp >= now() - (%s * interval '1 hour')
                GROUP BY p.engine
                ORDER BY p.engine
                """,
                (hours,),
            )
            stat_rows = await cur.fetchall()

            engine_stats: List[Dict[str, Any]] = []
            for s in stat_rows:
                engine = s[0]
                total = int(s[1] or 0)
                correct = int(s[2] or 0)
                incorrect = int(s[3] or 0)
                pending = int(s[4] or 0)
                settled = correct + incorrect
                accuracy = (correct / settled) if settled > 0 else None
                engine_stats.append({
                    "engine": engine,
                    "total": total,
                    "correct": correct,
                    "incorrect": incorrect,
                    "pending": pending,
                    "accuracy": accuracy,
                })

            known_engines = [
                "entropy_regime", "labouchere", "superformula",
                "trig_euler", "crt", "navier_stokes", "combination",
            ]
            found = {e["engine"] for e in engine_stats}
            for k in known_engines:
                if k not in found:
                    engine_stats.append({
                        "engine": k, "total": 0, "correct": 0,
                        "incorrect": 0, "pending": 0, "accuracy": None,
                    })

    return {
        "window_hours": hours,
        "signal_count": len(signals),
        "signals": signals,
        "engine_stats": engine_stats,
        "now": datetime.now(timezone.utc).isoformat(),
    }
