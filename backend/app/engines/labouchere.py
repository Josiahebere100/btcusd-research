"""Reverse Labouchere engine."""
from typing import Any, Dict, List, Optional, Tuple

from ._lookup import lookup_direction
from .base import Engine, EngineContext, EngineOutput

DEFAULT_SEQUENCE: List[int] = [1, 1, 2, 3]
MAX_SEQUENCE_LEN = 32
LOOKBACK_TICKS = 8


def _transition_win(seq: List[int]) -> List[int]:
    if len(seq) < 2:
        return DEFAULT_SEQUENCE[:]
    new_val = seq[0] + seq[-1]
    new_seq = seq[1:-1] + [new_val]
    return new_seq[-MAX_SEQUENCE_LEN:] if len(new_seq) > MAX_SEQUENCE_LEN else new_seq


def _transition_loss(seq: List[int]) -> List[int]:
    if len(seq) < 2:
        return DEFAULT_SEQUENCE[:]
    new_seq = seq[1:-1]
    return new_seq if len(new_seq) >= 2 else DEFAULT_SEQUENCE[:]


def _sequence_signature(seq: List[int], transition: str) -> str:
    # Coarser signature: length + sum + last transition.
    return f"lab:{transition}:len{len(seq)}_sum{sum(seq)}"


class LabouchereEngine(Engine):
    name = "labouchere"

    def __init__(self, initial: Optional[List[int]] = None):
        self.initial = list(initial or DEFAULT_SEQUENCE)

    def _evolve(self, recent_ticks: List[Tuple[float, float]]):
        seq = self.initial[:]
        transitions: List[str] = []
        last_transition = "none"
        if len(recent_ticks) < 2:
            return seq, last_transition, transitions

        window = recent_ticks[-LOOKBACK_TICKS:]
        for i in range(1, len(window)):
            prev_p = window[i - 1][1]
            curr_p = window[i][1]
            if curr_p > prev_p:
                seq = _transition_win(seq)
                transitions.append("win")
                last_transition = "win"
            elif curr_p < prev_p:
                seq = _transition_loss(seq)
                transitions.append("loss")
                last_transition = "loss"
            else:
                transitions.append("flat")
                last_transition = "flat"
        return seq, last_transition, transitions

    def run(self, ctx: EngineContext) -> EngineOutput:
        raise NotImplementedError("Labouchere is async; use run_async()")

    async def run_async(self, ctx: EngineContext) -> EngineOutput:
        if not ctx.recent_ticks or len(ctx.recent_ticks) < 2:
            return EngineOutput(
                engine=self.name, direction="NO_SIGNAL",
                pattern_signature=None,
                raw_state={"reason": "insufficient ticks"},
            )

        seq, last_transition, transitions = self._evolve(ctx.recent_ticks)
        signature = _sequence_signature(seq, last_transition)

        raw: Dict[str, Any] = {
            "sequence_after": seq,
            "transition": last_transition,
            "transitions_taken": transitions,
        }

        try:
            lookup = await lookup_direction(signature)
        except Exception as e:
            raw["lookup_error"] = str(e)
            lookup = None

        if lookup is None:
            return EngineOutput(
                engine=self.name, direction="NO_SIGNAL",
                pattern_signature=signature,
                raw_state={**raw, "lookup": "insufficient_history"},
            )

        direction, occurrences, confidence = lookup
        return EngineOutput(
            engine=self.name, direction=direction, confidence=confidence,
            pattern_signature=signature,
            raw_state={**raw, "lookup": {
                "occurrences": occurrences, "confidence": confidence,
                "chosen_direction": direction,
            }},
        )
