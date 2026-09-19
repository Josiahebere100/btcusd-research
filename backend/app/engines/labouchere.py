"""Reverse Labouchere engine with dwell + velocity."""
from typing import Any, Dict, List, Optional, Tuple

from ._lookup import lookup_direction
from ._temporal import (
    EngineSnapshot,
    compute_dwell_s,
    compute_velocity,
    dwell_bin,
    velocity_bin,
)
from .base import Engine, EngineContext, EngineOutput

DEFAULT_SEQUENCE: List[int] = [1, 1, 2, 3]
MAX_SEQUENCE_LEN = 32
LOOKBACK_TICKS = 8


def _transition_win(seq):
    if len(seq) < 2:
        return DEFAULT_SEQUENCE[:]
    new_val = seq[0] + seq[-1]
    new_seq = seq[1:-1] + [new_val]
    return new_seq[-MAX_SEQUENCE_LEN:] if len(new_seq) > MAX_SEQUENCE_LEN else new_seq


def _transition_loss(seq):
    if len(seq) < 2:
        return DEFAULT_SEQUENCE[:]
    new_seq = seq[1:-1]
    return new_seq if len(new_seq) >= 2 else DEFAULT_SEQUENCE[:]


def _sequence_signature(seq, transition):
    return f"lab:{transition}:len{len(seq)}_sum{sum(seq)}"


class LabouchereEngine(Engine):
    name = "labouchere"

    def __init__(self, initial: Optional[List[int]] = None):
        self.initial = list(initial or DEFAULT_SEQUENCE)

    def _evolve(self, ticks):
        seq = self.initial[:]
        transitions = []
        last_transition = "none"
        if len(ticks) < 2:
            return seq, last_transition, transitions
        window = ticks[-LOOKBACK_TICKS:]
        for i in range(1, len(window)):
            if window[i][1] > window[i - 1][1]:
                seq = _transition_win(seq)
                transitions.append("win")
                last_transition = "win"
            elif window[i][1] < window[i - 1][1]:
                seq = _transition_loss(seq)
                transitions.append("loss")
                last_transition = "loss"
            else:
                transitions.append("flat")
                last_transition = "flat"
        return seq, last_transition, transitions

    def _snapshot_for_window(self, window):
        if len(window) < 2:
            return None
        seq, last_transition, _ = self._evolve(window)
        return EngineSnapshot(
            signature=_sequence_signature(seq, last_transition),
            features={
                "seq_len": float(len(seq)),
                "seq_sum": float(sum(seq)),
            },
        )

    def run(self, ctx):
        raise NotImplementedError("Labouchere is async; use run_async()")

    async def run_async(self, ctx: EngineContext) -> EngineOutput:
        if not ctx.recent_ticks or len(ctx.recent_ticks) < 2:
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature=None,
                raw_state={"reason": "insufficient ticks"},
            )

        snap = self._snapshot_for_window(ctx.recent_ticks)
        if snap is None:
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature=None,
                raw_state={"reason": "snapshot failed"},
            )

        dwell_s = compute_dwell_s(
            self._snapshot_for_window, ctx.recent_ticks, snap.signature, max_back=12
        )
        vel = compute_velocity(
            self._snapshot_for_window, ctx.recent_ticks, snap, offset_ticks=4
        )
        full_sig = f"{snap.signature}_dw{dwell_bin(dwell_s)}_v{velocity_bin(vel)}"

        raw: Dict[str, Any] = {
            "dwell_s": dwell_s,
            "velocity": vel,
            "seq_len": int(snap.features["seq_len"]),
            "seq_sum": int(snap.features["seq_sum"]),
        }

        try:
            lookup = await lookup_direction(full_sig)
        except Exception as e:
            raw["lookup_error"] = str(e)
            lookup = None

        if lookup is None:
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature=full_sig,
                raw_state={**raw, "lookup": "insufficient_history"},
            )

        direction, occ, conf = lookup
        return EngineOutput(
            engine=self.name,
            direction=direction,
            confidence=conf,
            pattern_signature=full_sig,
            raw_state={
                **raw,
                "lookup": {
                    "occurrences": occ,
                    "confidence": conf,
                    "chosen_direction": direction,
                },
            },
        )
