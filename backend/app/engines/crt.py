"""CRT engine with dwell enrichment."""
from typing import Any, Dict, List, Optional

from ._lookup import lookup_direction
from ._temporal import (
    EngineSnapshot,
    compute_dwell_s,
    dwell_bin,
)
from .base import Engine, EngineContext, EngineOutput

DEFAULT_MODULI: List[int] = [7, 11, 13]
SCALE = 100_000


def _product(moduli):
    p = 1
    for m in moduli:
        p *= m
    return p


def _crt_residues(value, moduli):
    return [value % m for m in moduli]


def _crt_reconstruct(residues, moduli):
    M = _product(moduli)
    total = 0
    for r_i, m_i in zip(residues, moduli):
        M_i = M // m_i
        inv = pow(M_i, -1, m_i)
        total += r_i * M_i * inv
    return total % M


def _pattern_signature(residues):
    return "crt:" + "-".join(str(r) for r in residues)


class CRTEngine(Engine):
    name = "crt"

    def __init__(self, moduli: Optional[List[int]] = None):
        self.moduli = moduli or DEFAULT_MODULI
        self.modulus_product = _product(self.moduli)

    def _quantize(self, price: float) -> int:
        return int(price * SCALE) % self.modulus_product

    def _snapshot_for_window(self, window):
        if not window:
            return None
        _, price = window[-1]
        iv = self._quantize(price)
        res = _crt_residues(iv, self.moduli)
        rec = _crt_reconstruct(res, self.moduli)
        return EngineSnapshot(
            signature=_pattern_signature(res),
            features={
                "reconstruction": float(rec),
                "residue_sum": float(sum(res)),
            },
        )

    def run(self, ctx):
        raise NotImplementedError("CRT engine is async; use run_async()")

    async def run_async(self, ctx: EngineContext) -> EngineOutput:
        if not ctx.recent_ticks:
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature=None,
                raw_state={"reason": "no ticks"},
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
            self._snapshot_for_window, ctx.recent_ticks, snap.signature, max_back=15
        )
        full_sig = f"{snap.signature}_dw{dwell_bin(dwell_s)}"

        raw: Dict[str, Any] = {
            "residues": _crt_residues(
                self._quantize(ctx.recent_ticks[-1][1]), self.moduli
            ),
            "dwell_s": dwell_s,
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
