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

@router.get("/by-engine-signatures")
async def by_engine_signatures(
    engine: str = Query(..., min_length=1, max_length=40),
    hours: int = Query(168, ge=1, le=87600),
    limit: int = Query(200, ge=1, le=2000),
    _auth: str = Depends(require_collector_key),
) -> Dict[str, Any]:
    """Aggregate stats for one engine's signatures within a time window."""
    sig_field = engine
    dir_field = f"{engine}_dir"

    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT
                  pattern_signature::jsonb ->> %s AS sig,
                  pattern_signature::jsonb ->> %s AS dir,
                  COUNT(*) AS n,
                  COUNT(*) FILTER (WHERE o.correct = true) AS correct,
                  COUNT(*) FILTER (WHERE o.correct = false) AS incorrect
                FROM prediction_snapshots p
                LEFT JOIN outcomes o ON o.prediction_id = p.id
                WHERE pattern_signature LIKE '{%'
                  AND pattern_signature::jsonb ? %s
                  AND pattern_signature::jsonb ->> %s IN ('UP','DOWN')
                  AND prediction_timestamp >= now() - (%s * interval '1 hour')
                GROUP BY sig, dir
                ORDER BY n DESC
                LIMIT %s
                """,
                (sig_field, dir_field, dir_field, dir_field, hours, limit),
            )
            rows = await cur.fetchall()

    out = []
    for r in rows:
        n = int(r[2] or 0)
        correct = int(r[3] or 0)
        incorrect = int(r[4] or 0)
        settled = correct + incorrect
        out.append({
            "signature": r[0],
            "direction": r[1],
            "occurrences": n,
            "correct": correct,
            "incorrect": incorrect,
            "accuracy": (correct / settled) if settled else None,
        })
    return {
        "engine": engine,
        "window_hours": hours,
        "count": len(out),
        "signatures": out,
    }


@router.get("/by-engine-recent")
async def by_engine_recent(
    engine: str = Query(..., min_length=1, max_length=40),
    limit: int = Query(200, ge=1, le=2000),
    _auth: str = Depends(require_collector_key),
) -> Dict[str, Any]:
    """Individual predictions for one engine — signature, direction, result."""
    sig_field = engine
    dir_field = f"{engine}_dir"

    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT
                  prediction_timestamp,
                  pattern_signature::jsonb ->> %s AS sig,
                  pattern_signature::jsonb ->> %s AS dir,
                  price_at_prediction,
                  o.actual_price,
                  o.actual_direction,
                  o.correct,
                  confidence
                FROM prediction_snapshots p
                LEFT JOIN outcomes o ON o.prediction_id = p.id
                WHERE pattern_signature LIKE '{%'
                  AND pattern_signature::jsonb ? %s
                  AND pattern_signature::jsonb ->> %s IN ('UP','DOWN')
                ORDER BY prediction_timestamp DESC
                LIMIT %s
                """,
                (sig_field, dir_field, dir_field, dir_field, limit),
            )
            rows = await cur.fetchall()

    out = []
    for r in rows:
        out.append({
            "prediction_timestamp": r[0].isoformat() if r[0] else None,
            "signature": r[1],
            "direction": r[2],
            "price_at_prediction": float(r[3]) if r[3] is not None else None,
            "actual_price": float(r[4]) if r[4] is not None else None,
            "actual_direction": r[5],
            "correct": r[6],
            "confidence": float(r[7]) if r[7] is not None else None,
        })
    return {"engine": engine, "count": len(out), "signals": out}
