"""Baseline control engines.

Controls, not predictive engines. They fire every round and provide
reference points for evaluating real engines.

They do NOT call lookup_direction, do NOT feed pattern memory, and do
NOT decide the round's primary direction.

Their signatures are written under a `baseline` namespace key so they
cannot be confused with real engine signatures.
"""
from typing import Any, Dict, List

from ..db import get_pool
from .base import EngineContext, EngineOutput


BASELINE_ENGINES = [
    "baseline_always_up",
    "baseline_always_down",
    "baseline_last_move_continue",
    "baseline_last_move_reverse",
    "baseline_recent_majority",
]


async def _load_recent_round_results(n: int = 30) -> List[str]:
    try:
        pool = await get_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT win_side
                    FROM rounds
                    WHERE win_side IN (1, 2)
                      AND price_end_timestamp < now() - interval '20 seconds'
                    ORDER BY price_end_timestamp DESC
                    LIMIT %s
                    """,
                    (n,),
                )
                rows = await cur.fetchall()
        return ["UP" if int(r[0]) == 1 else "DOWN" for r in rows]
    except Exception as e:
        print(f"[baselines] rounds read failed: {e}")
        return []


def _mk(engine, direction, made_prediction, raw):
    payload = {"kind": "control", "made_prediction": made_prediction}
    payload.update(raw)
    return EngineOutput(
        engine=engine,
        direction=direction,
        confidence=None,
        pattern_signature=None,
        raw_state=payload,
    )


def _always_up():
    return _mk("baseline_always_up", "UP", True, {"strategy": "always_up"})


def _always_down():
    return _mk("baseline_always_down", "DOWN", True, {"strategy": "always_down"})


def _last_move_continue(recent):
    if not recent:
        return _mk("baseline_last_move_continue", "NO_SIGNAL", False,
                   {"strategy": "last_move_continue", "reason": "no history"})
    return _mk("baseline_last_move_continue", recent[0], True,
               {"strategy": "last_move_continue", "last_outcome": recent[0]})


def _last_move_reverse(recent):
    if not recent:
        return _mk("baseline_last_move_reverse", "NO_SIGNAL", False,
                   {"strategy": "last_move_reverse", "reason": "no history"})
    d = "DOWN" if recent[0] == "UP" else "UP"
    return _mk("baseline_last_move_reverse", d, True,
               {"strategy": "last_move_reverse", "last_outcome": recent[0]})


def _recent_majority(recent, window=20):
    if not recent:
        return _mk("baseline_recent_majority", "NO_SIGNAL", False,
                   {"strategy": "recent_majority", "reason": "no history"})
    w = recent[:window]
    n_up = sum(1 for d in w if d == "UP")
    n_down = len(w) - n_up
    if n_up == n_down:
        return _mk("baseline_recent_majority", "NO_SIGNAL", False,
                   {"strategy": "recent_majority", "reason": "tied",
                    "n_up": n_up, "n_down": n_down})
    d = "UP" if n_up > n_down else "DOWN"
    return _mk("baseline_recent_majority", d, True,
               {"strategy": "recent_majority", "n_up": n_up,
                "n_down": n_down, "window": len(w)})


async def run_baselines(ctx: EngineContext) -> Dict[str, Any]:
    recent = await _load_recent_round_results(n=30)
    outputs = [
        _always_up(),
        _always_down(),
        _last_move_continue(recent),
        _last_move_reverse(recent),
        _recent_majority(recent, window=20),
    ]
    baseline = {}
    for o in outputs:
        baseline[o.engine] = {
            "dir": o.direction,
            "made_prediction": o.raw_state.get("made_prediction", False),
        }
    return baseline
