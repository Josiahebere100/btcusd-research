"""Shared temporal helper for engines.

Adds dwell and velocity to any engine's signature via a structured
snapshot interface. Each engine supplies a snapshot_fn that returns
an EngineSnapshot for a given tick window.
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

DWELL_SHORT_S = 2.0
DWELL_MED_S = 8.0
DWELL_LONG_S = 20.0

VELOCITY_LOW = 0.05
VELOCITY_HIGH = 0.30


@dataclass
class EngineSnapshot:
    signature: str
    features: Dict[str, float] = field(default_factory=dict)


def dwell_bin(dwell_s: float) -> int:
    if dwell_s < DWELL_SHORT_S:
        return 0
    if dwell_s < DWELL_MED_S:
        return 1
    if dwell_s < DWELL_LONG_S:
        return 2
    return 3


def velocity_bin(v: float) -> int:
    if v < VELOCITY_LOW:
        return 0
    if v < VELOCITY_HIGH:
        return 1
    return 2


def _feature_change(now: Dict[str, float], prev: Dict[str, float]) -> float:
    total = 0.0
    for k, v in now.items():
        if k in prev:
            total += abs(v - prev[k])
    return total


def compute_dwell_s(
    snapshot_fn: Callable[[List[Tuple[float, float]]], Optional[EngineSnapshot]],
    recent_ticks: List[Tuple[float, float]],
    current_sig: str,
    max_back: int = 15,
) -> float:
    if len(recent_ticks) < 3 or not current_sig:
        return 0.0
    now_ts = recent_ticks[-1][0]
    lowest = max(1, len(recent_ticks) - max_back)
    for i in range(len(recent_ticks) - 2, lowest - 1, -1):
        sub = recent_ticks[:i + 1]
        try:
            snap = snapshot_fn(sub)
        except Exception:
            break
        if snap is None or snap.signature != current_sig:
            return now_ts - recent_ticks[i + 1][0]
    return now_ts - recent_ticks[lowest][0]


def compute_velocity(
    snapshot_fn: Callable[[List[Tuple[float, float]]], Optional[EngineSnapshot]],
    recent_ticks: List[Tuple[float, float]],
    current_snapshot: EngineSnapshot,
    offset_ticks: int = 4,
) -> float:
    if len(recent_ticks) <= offset_ticks or not current_snapshot.features:
        return 0.0
    prev_window = recent_ticks[:len(recent_ticks) - offset_ticks]
    try:
        prev_snap = snapshot_fn(prev_window)
    except Exception:
        return 0.0
    if prev_snap is None or not prev_snap.features:
        return 0.0
    dt = recent_ticks[-1][0] - recent_ticks[len(recent_ticks) - offset_ticks - 1][0]
    if dt <= 0:
        return 0.0
    return _feature_change(current_snapshot.features, prev_snap.features) / dt
