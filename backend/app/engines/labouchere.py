"""Reverse Labouchere engine.

Maintains a canonical integer sequence shaped by the recent price
up/down history. The sequence's current state is hashed into a
signature. Direction is learned from historical outcomes, not assumed.
"""
from typing import Any, Dict, List, Optional, Tuple

from ..db import get_pool
from .base import Engine, EngineContext, EngineOutput

# Reverse Labouchere default starting sequence, per spec section 19.
DEFAULT_SEQUENCE: List[int] = [1, 1, 2, 3]

# Cap on sequence length to prevent runaway growth.
MAX_SEQUENCE_LEN = 32

MIN_OCCURRENCES = 10
MIN_ACCURACY = 0.55

# How many recent ticks to use for updating the sequence.
# Too few and the shape is unstable; too many and old moves dominate.
LOOKBACK_TICKS = 8


def _transition_win(seq: List[int]) -> List[int]:
    """Reverse Labouchere 'win': remove ends, append their sum."""
    if len(seq) < 2:
        return [1, 1, 2, 3]
    new_val = seq[0] + seq[-1]
    new_seq = seq[1:-1] + [new_val]
    if len(new_seq) > MAX_SEQUENCE_LEN:
        new_seq = new_seq[-MAX_SEQUENCE_LEN:]
    return new_seq


def _transition_loss(seq: List[int]) -> List[int]:
    """Reverse Labouchere 'loss': remove ends."""
    if len(seq) < 2:
        return [1, 1, 2, 3]
    new_seq = seq[1:-1]
    if len(new_seq) < 2:
        new_seq = DEFAULT_SEQUENCE[:]
    return new_seq


def _sequence_signature(seq: List[int], transition: str) -> str:
    """Compact, deterministic signature of the sequence and last transition."""
    body = "_".join(str(x) for x in seq)
    return f"lab:{transition}:{body}"


class LabouchereEngine(Engine):
    name = "labouchere"

    def __init__(self, initial: Optional[List[int]] = None):
        self.initial = list(initial or DEFAULT_SEQUENCE)

    # ---- Sequence evolution ----------------------------------------------

    def _evolve(self, recent_ticks: List[Tuple[float, float]]) -> Tuple[List[int], str, List[str]]:
        """Walk the recent price history, applying a transition per step.

        Returns (final_sequence, last_transition, transitions_taken).
        """
        seq = self.initial[:]
        transitions: List[str] = []
        last_transition = "none"

        if len(recent_ticks) < 2:
            return seq, last_transition, transitions

        window = recent_ticks[-LOOKBACK_TICKS:]
        for i in range(1, len(window)):
            prev_price = window[i - 1][1]
            curr_price = window[i][1]
            if curr_price > prev_price:
                seq = _transition_win(seq)
                transitions.append("win")
                last_transition = "win"
            elif curr_price < prev_price:
                seq = _transition_loss(seq)
                transitions.append("loss")
                last_transition = "loss"
            else:
                transitions.append("flat")
                last_transition = "flat"

        return seq, last_transition, transitions

    # ---- Historical lookup ----------------------------------------------

    async def _lookup_pattern(
        self, signature: str
    ) -> Optional[Tuple[str, int, float]]:
        pool = await get_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT id, occurrence_count, accuracy
                    FROM patterns
                    WHERE pattern_signature = %s
                    """,
                    (signature,),
                )
                row = await cur.fetchone()
                if not row:
                    return None

                pattern_id = row[0]
                occurrence_count = int(row[1] or 0)
                accuracy = float(row[2]) if row[2] is not None else None

                if occurrence_count < MIN_OCCURRENCES or accuracy is None:
                    return None
                if accuracy < MIN_ACCURACY:
                    return None

                await cur.execute(
                    """
                    SELECT COUNT(*) FROM pattern_success_memory
                    WHERE pattern_id = %s AND result = 'UP'
                    """,
                    (pattern_id,),
                )
                up_row = await cur.fetchone()
                up_correct = int(up_row[0]) if up_row else 0

                await cur.execute(
                    """
                    SELECT COUNT(*) FROM pattern_success_memory
                    WHERE pattern_id = %s AND result = 'DOWN'
                    """,
                    (pattern_id,),
                )
                down_row = await cur.fetchone()
                down_correct = int(down_row[0]) if down_row else 0

                if up_correct == down_correct:
                    return None
                direction = "UP" if up_correct > down_correct else "DOWN"
                return direction, occurrence_count, accuracy

    # ---- Entry points ----------------------------------------------------

    def run(self, ctx: EngineContext) -> EngineOutput:
        raise NotImplementedError("Labouchere engine is async; use run_async()")

    async def run_async(self, ctx: EngineContext) -> EngineOutput:
        if not ctx.recent_ticks or len(ctx.recent_ticks) < 2:
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                confidence=None,
                pattern_signature=None,
                raw_state={"reason": "insufficient ticks"},
            )

        seq, last_transition, transitions = self._evolve(ctx.recent_ticks)
        signature = _sequence_signature(seq, last_transition)

        raw_state: Dict[str, Any] = {
            "sequence_before": self.initial,
            "sequence_after": seq,
            "transition": last_transition,
            "transitions_taken": transitions,
            "state_signature": signature,
        }

        try:
            lookup = await self._lookup_pattern(signature)
        except Exception as e:
            raw_state["lookup_error"] = str(e)
            lookup = None

        if lookup is None:
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                confidence=None,
                pattern_signature=signature,
                raw_state={**raw_state, "lookup": "insufficient_history"},
            )

        direction, occurrences, accuracy = lookup
        return EngineOutput(
            engine=self.name,
            direction=direction,
            confidence=accuracy,
            pattern_signature=signature,
            raw_state={
                **raw_state,
                "lookup": {
                    "occurrences": occurrences,
                    "accuracy": accuracy,
                    "chosen_direction": direction,
                },
            },
        )
