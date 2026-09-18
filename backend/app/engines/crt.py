"""CRT (Chinese Remainder Theorem) engine."""
from typing import Any, Dict, List, Optional

from ._lookup import lookup_direction
from .base import Engine, EngineContext, EngineOutput

# Reduced to 1001 states total (7*11*13) so signatures recur often enough
# to accumulate 10+ observations in reasonable time.
DEFAULT_MODULI: List[int] = [7, 11, 13]
SCALE = 100_000


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


def _pattern_signature(residues: List[int]) -> str:
    return "crt:" + "-".join(str(r) for r in residues)


class CRTEngine(Engine):
    name = "crt"

    def __init__(self, moduli: Optional[List[int]] = None):
        self.moduli = moduli or DEFAULT_MODULI
        self.modulus_product = _product(self.moduli)

    def _quantize(self, price: float) -> int:
        return int(price * SCALE) % self.modulus_product

    def run(self, ctx: EngineContext) -> EngineOutput:
        raise NotImplementedError("CRT engine is async; use run_async()")

    async def run_async(self, ctx: EngineContext) -> EngineOutput:
        if not ctx.recent_ticks:
            return EngineOutput(
                engine=self.name, direction="NO_SIGNAL",
                pattern_signature=None,
                raw_state={"reason": "no recent ticks"},
            )

        _, price = ctx.recent_ticks[-1]
        integer_input = self._quantize(price)
        residues = _crt_residues(integer_input, self.moduli)
        reconstruction = _crt_reconstruct(residues, self.moduli)
        signature = _pattern_signature(residues)

        raw: Dict[str, Any] = {
            "input_price": price,
            "integer_input": integer_input,
            "moduli": self.moduli,
            "residues": residues,
            "reconstruction": reconstruction,
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
