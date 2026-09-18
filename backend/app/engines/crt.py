"""CRT (Chinese Remainder Theorem) engine."""
from typing import Any, Dict, List, Optional, Tuple

from ..db import get_pool
from .base import Engine, EngineContext, EngineOutput

DEFAULT_MODULI: List[int] = [7, 11, 13, 17, 19, 23]
SCALE = 100_000
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
    M = _product(moduli)
    total = 0
    for r_i, m_i in zip(residues, moduli):
        M_i = M // m_i
        inv = pow(M_i, -1, m_i)
        total += r_i * M_i * inv
    return total % M


def _normalized_state(reconstruction: int, modulus_product: int) -> float:
    return reconstruction / modulus_product


def _pattern_signature(residues: List[int]) -> str:
    return "crt:" + "-".join(str(r) for r in residues)


class CRTEngine(Engine):
    name = "crt"

    def __init__(self, moduli: Optional[List[int]] = None):
        self.moduli = moduli or DEFAULT_MODULI
        self.modulus_product = _product(self.moduli)

    def _quantize(self, price: float) -> int:
        scaled = int(price * SCALE)
        return scaled % self.modulus_product

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
