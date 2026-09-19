"""Memory endpoint: patterns, combinations, per-engine stats."""
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, Query

from ..auth import require_collector_key
from ..db import get_pool

router = APIRouter()


@router.get("/patterns")
async def list_patterns(
    limit: int = Query(100, ge=1, le=1000),
    min_occurrences: int = Query(1, ge=1, le=100000),
    engine_prefix: str = Query("", max_length=20),
    _auth: str = Depends(require_collector_key),
) -> Dict[str, Any]:
    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            if engine_prefix:
                await cur.execute(
                    """
                    SELECT pattern_signature, occurrence_count,
                           correct_count, incorrect_count, accuracy,
                           recent_accuracy, last_seen, ineligible
                    FROM patterns
                    WHERE occurrence_count >= %s
                      AND pattern_signature LIKE %s
                    ORDER BY occurrence_count DESC
                    LIMIT %s
                    """,
                    (min_occurrences, engine_prefix + "%", limit),
                )
            else:
                await cur.execute(
                    """
                    SELECT pattern_signature, occurrence_count,
                           correct_count, incorrect_count, accuracy,
                           recent_accuracy, last_seen, ineligible
                    FROM patterns
                    WHERE occurrence_count >= %s
                    ORDER BY occurrence_count DESC
                    LIMIT %s
                    """,
                    (min_occurrences, limit),
                )
            rows = await cur.fetchall()

    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append({
            "signature": r[0],
            "occurrences": int(r[1] or 0),
            "correct": int(r[2] or 0),
            "incorrect": int(r[3] or 0),
            "accuracy": float(r[4]) if r[4] is not None else None,
            "recent_accuracy": float(r[5]) if r[5] is not None else None,
            "last_seen": r[6].isoformat() if r[6] else None,
            "ineligible": bool(r[7]),
        })
    return {"count": len(out), "patterns": out}


@router.get("/combinations")
async def list_combinations(
    limit: int = Query(100, ge=1, le=1000),
    min_occurrences: int = Query(1, ge=1, le=100000),
    _auth: str = Depends(require_collector_key),
) -> Dict[str, Any]:
    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT combination, occurrences, correct, incorrect,
                       accuracy, recent_accuracy, combined_output
                FROM combination_signals
                WHERE occurrences >= %s
                ORDER BY occurrences DESC
                LIMIT %s
                """,
                (min_occurrences, limit),
            )
            rows = await cur.fetchall()

    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append({
            "combination": r[0],
            "occurrences": int(r[1] or 0),
            "correct": int(r[2] or 0),
            "incorrect": int(r[3] or 0),
            "accuracy": float(r[4]) if r[4] is not None else None,
            "recent_accuracy": float(r[5]) if r[5] is not None else None,
            "combined_output": r[6],
        })
    return {"count": len(out), "combinations": out}


@router.get("/engine-stats")
async def engine_stats(
    hours: int = Query(24, ge=1, le=87600),
    _auth: str = Depends(require_collector_key),
) -> Dict[str, Any]:
    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT
                    engine,
                    COUNT(*) AS total,
                    COUNT(*) FILTER (WHERE direction = 'UP') AS up_calls,
                    COUNT(*) FILTER (WHERE direction = 'DOWN') AS down_calls,
                    COUNT(*) FILTER (WHERE o.correct = true) AS correct,
                    COUNT(*) FILTER (WHERE o.correct = false) AS incorrect
                FROM prediction_snapshots p
                LEFT JOIN outcomes o ON o.prediction_id = p.id
                WHERE direction IN ('UP', 'DOWN')
                  AND prediction_timestamp >= now() - (%s * interval '1 hour')
                GROUP BY engine
                ORDER BY engine
                """,
                (hours,),
            )
            rows = await cur.fetchall()

    out: List[Dict[str, Any]] = []
    for r in rows:
        total = int(r[1] or 0)
        correct = int(r[4] or 0)
        incorrect = int(r[5] or 0)
        settled = correct + incorrect
        out.append({
            "engine": r[0],
            "total": total,
            "up_calls": int(r[2] or 0),
            "down_calls": int(r[3] or 0),
            "correct": correct,
            "incorrect": incorrect,
            "accuracy": (correct / settled) if settled else None,
        })
    return {"window_hours": hours, "engines": out}
