"""Memory endpoint: patterns, combinations, per-engine stats, supersignatures."""
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, Query

from ..auth import require_collector_key
from ..db import get_pool

router = APIRouter()

ENGINE_PREFIXES = {
    "crt": "crt:",
    "labouchere": "lab:",
    "trig_euler": "trig:",
    "entropy_regime": "ent:",
    "superformula": "sf:",
    "navier_stokes": "ns:",
    "candlestick": "cs:",
    "meta_ensemble": "meta:",
    "momentum": "mom:",
    "trail_tracer": "trail:",
    "curve_geometry": "cg:",
    "ns_flow": "nsflow:",
}


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
                      AND LEFT(pattern_signature, %s) = %s
                    ORDER BY occurrence_count DESC
                    LIMIT %s
                    """,
                    (min_occurrences, len(engine_prefix), engine_prefix, limit),
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


@router.get("/by-engine")
async def by_engine(
    engine: str = Query(..., min_length=1, max_length=40),
    hours: int = Query(168, ge=1, le=87600),
    limit: int = Query(200, ge=1, le=2000),
    only_fired: bool = Query(True),
    _auth: str = Depends(require_collector_key),
) -> Dict[str, Any]:
    """Predictions where the given engine contributed.

    Returns rows with the full supersignature, the combined direction,
    the actual outcome, and whether the combined call was correct.
    """
    dir_key = f"{engine}_dir"

    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            if only_fired:
                await cur.execute(
                    """
                    SELECT
                      p.prediction_timestamp,
                      p.direction,
                      p.confidence,
                      p.pattern_signature,
                      o.actual_direction,
                      o.correct,
                      p.engine AS deciding_engine
                    FROM prediction_snapshots p
                    LEFT JOIN outcomes o ON o.prediction_id = p.id
                    WHERE LEFT(p.pattern_signature, 1) = '{'
                      AND jsonb_exists(p.pattern_signature::jsonb, %s)
                      AND p.pattern_signature::jsonb ->> %s IN ('UP','DOWN')
                      AND p.prediction_timestamp >= now() - (%s * interval '1 hour')
                    ORDER BY p.prediction_timestamp DESC
                    LIMIT %s
                    """,
                    (dir_key, dir_key, hours, limit),
                )
            else:
                await cur.execute(
                    """
                    SELECT
                      p.prediction_timestamp,
                      p.direction,
                      p.confidence,
                      p.pattern_signature,
                      o.actual_direction,
                      o.correct,
                      p.engine AS deciding_engine
                    FROM prediction_snapshots p
                    LEFT JOIN outcomes o ON o.prediction_id = p.id
                    WHERE LEFT(p.pattern_signature, 1) = '{'
                      AND jsonb_exists(p.pattern_signature::jsonb, %s)
                      AND p.prediction_timestamp >= now() - (%s * interval '1 hour')
                    ORDER BY p.prediction_timestamp DESC
                    LIMIT %s
                    """,
                    (dir_key, hours, limit),
                )
            rows = await cur.fetchall()

    out = []
    for r in rows:
        out.append({
            "at": r[0].isoformat() if r[0] else None,
            "direction": r[1],
            "confidence": float(r[2]) if r[2] is not None else None,
            "signature": r[3],
            "actual_direction": r[4],
            "correct": r[5],
            "deciding_engine": r[6],
        })
    return {
        "engine": engine,
        "window_hours": hours,
        "only_fired": only_fired,
        "count": len(out),
        "signals": out,
    }
