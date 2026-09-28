"""Evidence-preserving fusion of seed and adaptive interaction research."""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from .base import Engine, EngineContext, EngineOutput


class ResearchEnsemble(Engine):
    name = "research_ensemble"
    is_async = True

    def __init__(self, seed_weight: float = 0.45, adaptive_weight: float = 0.55) -> None:
        total = seed_weight + adaptive_weight
        if total <= 0:
            raise ValueError("research ensemble weights must sum to > 0")
        self.seed_weight = seed_weight / total
        self.adaptive_weight = adaptive_weight / total

    async def evaluate(
        self,
        outputs: Sequence[EngineOutput],
        seed: EngineOutput,
        adaptive: EngineOutput,
        ctx: EngineContext,
    ) -> EngineOutput:
        sr = seed.raw_state or {}
        ar = adaptive.raw_state or {}

        seed_probability_up = sr.get("seed_probability_up")
        seed_probability_down = sr.get("seed_probability_down")

        # Normalize the seed's two-sided probability representation into the
        # single p_up value used by the fusion calculation.
        #
        # A DOWN seed legitimately stores its probability as p_down. Convert
        # that to the mathematically equivalent p_up = 1 - p_down instead of
        # silently dropping the seed evidence from the ensemble.
        sp = seed_probability_up
        if sp is None and isinstance(seed_probability_down, (int, float)):
            sp = 1.0 - float(seed_probability_down)

        ap = ar.get("p_up")

        # Preserve evidence rather than fabricating a value when one side is absent.
        if sp is None and ap is None:
            direction = "NO_SIGNAL"
            p_up = p_down = None
        else:
            pieces = []
            if isinstance(sp, (int, float)):
                pieces.append((float(sp), self.seed_weight))
            if isinstance(ap, (int, float)):
                pieces.append((float(ap), self.adaptive_weight))
            denom = sum(w for _, w in pieces)
            p_up = sum(p * w for p, w in pieces) / denom
            p_down = 1.0 - p_up
            direction = "UP" if p_up >= 0.55 else "DOWN" if p_up <= 0.45 else "NO_SIGNAL"

        agreement = (
            seed.direction != "NO_SIGNAL"
            and adaptive.direction != "NO_SIGNAL"
            and seed.direction == adaptive.direction
        )
        conflict = (
            seed.direction != "NO_SIGNAL"
            and adaptive.direction != "NO_SIGNAL"
            and seed.direction != adaptive.direction
        )

        confidence = max(p_up or 0.0, p_down or 0.0) if direction != "NO_SIGNAL" else None
        sig = f"research:{direction.lower()}:seed{'A' if agreement else 'C' if conflict else 'N'}"

        return EngineOutput(
            engine=self.name,
            direction=direction,
            confidence=confidence,
            pattern_signature=sig,
            raw_state={
                "seed_direction": seed.direction,
                "adaptive_direction": adaptive.direction,
                "seed_probability_up": sp,
                "seed_probability_down": seed_probability_down,
                "adaptive_probability_up": ap,
                "probability_up": p_up,
                "probability_down": p_down,
                "agreement": agreement,
                "conflict": conflict,
                "seed_rules": sr.get("fired_rules", []),
                "adaptive_interactions": ar.get("matched_interactions", []),
                "sway_direction": ar.get("sway_direction"),
                "sway_delta": ar.get("sway_delta"),
            },
        )

    def run(self, ctx: EngineContext) -> EngineOutput:
        raise NotImplementedError("ResearchEnsemble is a post-pass over engine outputs.")

    async def run_async(self, ctx: EngineContext) -> EngineOutput:
        return EngineOutput(
            engine=self.name,
            direction="NO_SIGNAL",
            pattern_signature="research:requires_postpass",
            raw_state={"reason": "requires seed + adaptive outputs"},
        )
