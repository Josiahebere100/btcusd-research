"""CRT (Chinese Remainder Theorem) engine.

Transforms recent BTC/USD price history into CRT residues over a set of
pairwise-coprime moduli, reconstructs the integer, and derives a
deterministic pattern signature.

Direction is NEVER assumed from CRT parity. It is looked up in the
patterns table (accuracy of past CRT states that preceded UP vs DOWN).
Until sufficient evidence accumulates, the engine returns NO_SIGNAL.
"""
from typing import Any, Dict, List, Optional, Tuple

from ..db import get_pool
from .base import Engine, EngineContext, EngineOutput

# Pairwise-coprime moduli (all primes, so coprime by construction).
DEFAULT_MODULI: List[int] = [7, 11, 13, 17, 19, 23]

# Scale factor to preserve price precision.
# BTC/USD typically has 5 decimals; scale by 1e5 then take modulo to keep
# the integer bounded.
SCALE = 100_000

# Lookup thresholds for direction determination.
MIN_OCCURRENCES = 10
MIN_ACCURACY = 0.55


def _product(moduli: List[int]) -> int:
    p = 1
    for m in moduli:
        p *= m
    return p


def _crt_residues(value: int, moduli: List[int]) -> List[int]:
    return [value % m for m in moduli]


def _crt_reconstruct(residues: List[int], moduli: List[int]) -> int:
    """Classic CRT reconstruction.

    Given residues r_i mod m_i, returns the unique x in [0, prod(moduli))
    such that x ≡ r_i (mod m_i).
    """
    M = _product(moduli)
    total = 0
    for r_i, m_i in zip(residues, moduli):
        M_i = M // m_i
        inv = pow(M_i, -1, m_i)  # Python 3.8+: modular inverse
        total += r_i * M_i * inv
    return total % M


def _normalized_state(reconstruction: int, modulus_product: int) -> float:
    """Return a scalar in [0, 1) describing the CRT state."""
    return reconstruction / modulus_product


def _pattern_signature(residues: List[int]) -> str:
    """A deterministic, short, human-readable signature of the residue vector."""
    return "crt:" + "-".join(str(r) for r in residues)


class CRTEngine(Engine):
    name = "crt"

    def __init__(self, moduli: Optional[List[int]] = None):
        self.moduli = moduli or DEFAULT_MODULI
        self.modulus_product = _product(self.moduli)

    # ---- Input transformation --------------------------------------------

    def _quantize(self, price: float) -> int:
        """Turn a price into a bounded integer.

        Multiply by scale, floor to integer, then mod by the modulus product.
        This gives a stable, deterministic integer that varies with price.
        """
        scaled = int(price * SCALE)
        return scaled % self.modulus_product

    # ---- Database lookup -------------------------------------------------

    async def _lookup_pattern(
        self, signature: str
    ) -> Optional[Tuple[str, int, float]]:
        """Return (direction, occurrences, accuracy) if the pattern is
        eligible for live use; else None.

        The 'direction' returned is the historically dominant direction
        for this signature, based on pattern_success_memory and
        pattern_failure_memory.
        """
        pool = await get_pool()
        async with pool.connection() as conn:
            # Fetch overall stats for this signature
            row = await conn.fetchrow(
                """
                SELECT p.id,
                       p.occurrence_count,
                       p.accuracy
                FROM patterns p
                WHERE p.pattern_signature = %s
                """,
                (signature,),
            )
            if not row:
                return None

            occurrence_count = int(row["occurrence_count"] or 0)
            accuracy = float(row["accuracy"]) if row["accuracy"] is not None else None
            pattern_id = row["id"]

            if occurrence_count < MIN_OCCURRENCES or accuracy is None:
                return None
            if accuracy < MIN_ACCURACY:
                return None

            # Determine which direction has been correct more often for this
            # pattern, by counting successes where the direction was UP vs DOWN.
            up_row = await conn.fetchrow(
                """
                SELECT COUNT(*) AS c
                FROM pattern_success_memory
                WHERE pattern_id = %s AND result = 'UP'
                """,
                (pattern_id,),
            )
            down_row = await conn.fetchrow(
                """
                SELECT COUNT(*) AS c
                FROM pattern_success_memory
                WHERE pattern_id = %s AND result = 'DOWN'
                """,
                (pattern_id,),
            )
            up_correct = int(up_row["c"]) if up_row else 0
            down_correct = int(down_row["c"]) if down_row else 0

            if up_correct == down_correct:
                return None
            direction = "UP" if up_correct > down_correct else "DOWN"
            return direction, occurrence_count, accuracy

    # ---- Main entry point -------------------------------------------------

    def run(self, ctx: EngineContext) -> EngineOutput:
        raise NotImplementedError("CRT engine is async; use run_async()")

    async def run_async(self, ctx: EngineContext) -> EngineOutput:
        if not ctx.recent_ticks:
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                confidence=None,
                pattern_signature=None,
                raw_state={"reason": "no recent ticks"},
            )

        # Use the most recent price.
        _, price = ctx.recent_ticks[-1]
        integer_input = self._quantize(price)
        residues = _crt_residues(integer_input, self.moduli)
        reconstruction = _crt_reconstruct(residues, self.moduli)
        normalized = _normalized_state(reconstruction, self.modulus_product)
        signature = _pattern_signature(residues)

        raw_state: Dict[str, Any] = {
            "input_price": price,
            "integer_input": integer_input,
            "moduli": self.moduli,
            "residues": residues,
            "product_modulus": self.modulus_product,
            "reconstruction": reconstruction,
            "normalized_state": normalized,
        }

        # Look up historical evidence.
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
