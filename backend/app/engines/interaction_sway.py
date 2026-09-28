"""Adaptive live Interaction/Sway research engine.

The engine consumes the complete current EngineOutput set. Exact signatures
from NO_SIGNAL engines remain contextual state; only UP/DOWN directions count
as directional votes.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import math

from .base import Engine, EngineContext, EngineOutput
from .interaction_features import (
    atom_values,
    canonical_state_json,
    directional_engine_count,
    extract_observed_state,
    select_research_atoms,
)
from ..services.research_memory import ResearchMemory


class InteractionSwayEngine(Engine):
    name = "interaction_sway"
    is_async = True

    def __init__(
        self,
        memory: Optional[ResearchMemory] = None,
        min_support: int = 5,
        min_effect: float = 0.04,
        max_interactions: int = 24,
        max_atoms: int = 28,
        max_candidates: int = 6000,
        persist: bool = True,
    ) -> None:
        self.memory = memory or ResearchMemory(min_support=min_support)
        self.min_support = min_support
        self.min_effect = min_effect
        self.max_interactions = max_interactions
        self.max_atoms = max_atoms
        self.max_candidates = max(1, int(max_candidates))
        self.persist = persist

    @staticmethod
    def _beta_shrunk(up: int, down: int, prior_p: float = 0.5, strength: float = 8.0) -> float:
        n = up + down
        if n <= 0:
            return prior_p
        alpha = max(1e-6, prior_p * strength)
        beta = max(1e-6, (1.0 - prior_p) * strength)
        return (up + alpha) / (n + alpha + beta)

    def _atom_relevance(self, atoms_obj) -> List[str]:
        return select_research_atoms(atoms_obj, max_atoms=self.max_atoms)

    async def evaluate(self, outputs: Sequence[EngineOutput], ctx: EngineContext) -> EngineOutput:
        atoms_obj = extract_observed_state(outputs)
        signature_engines = sorted({a.engine for a in atoms_obj if a.kind == "exact_signature"})
        if len(signature_engines) < 2:
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature="sway:insufficient_context",
                raw_state={"reason": "fewer than two signature-bearing engines"},
            )

        atoms = self._atom_relevance(atoms_obj)
        state_json = canonical_state_json(outputs)
        try:
            await self.memory.ensure_schema()
        except Exception as exc:
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature="sway:memory_unavailable",
                raw_state={
                    "reason": "research_schema_unavailable",
                    "error": str(exc),
                    "active_engine_count": len(signature_engines),
                    "context_atoms": len(atoms),
                },
            )
        # Persist the observation before any historical-stat query. This
        # guarantees that NO_SIGNAL / memory-failure states remain available
        # for later research rather than disappearing from the corpus.
        observation_persist_error = None
        if self.persist:
            try:
                await self.memory.upsert_observation(
                session_id=ctx.session_id,
                round_id=ctx.round_id,
                atoms=atoms,
                state_json=state_json,
                active_engine_count=len(signature_engines),
                    directional_engine_count=directional_engine_count(outputs),
                )
            except Exception as exc:
                observation_persist_error = str(exc)
        direction_atoms = [a for a in atoms_obj if a.kind == "direction" and a.direction in {"UP", "DOWN"}]
        trajectory_direction = next((a.direction for a in direction_atoms if a.engine == "trajectory"), None)

        try:
            stats = await self.memory.stats_for_atoms(atoms, max_candidates=self.max_candidates)
        except Exception as exc:
            # Online live engine fails closed without breaking the base system.
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature="sway:memory_unavailable",
                raw_state={
                    "reason": "research_memory_unavailable",
                    "error": str(exc),
                    "active_engine_count": len(signature_engines),
                    "context_atoms": len(atoms),
                    "observation_persist_error": observation_persist_error,
                },
            )

        candidates: List[Dict[str, Any]] = []
        for s in stats:
            n = int(s["n"])
            p_up = self._beta_shrunk(int(s["up"]), int(s["down"]))
            p_down = 1.0 - p_up
            effect = abs(p_up - 0.5)
            if effect < self.min_effect or n < self.min_support:
                continue
            candidates.append({
                "key": s["key"],
                "n": n,
                "up": int(s["up"]),
                "down": int(s["down"]),
                "p_up": p_up,
                "p_down": p_down,
                "effect": effect,
                "score": effect * math.sqrt(n),
            })

        candidates.sort(key=lambda x: (-x["score"], -x["n"], str(x["key"])))
        selected = candidates[:self.max_interactions]

        if not selected:
            direction = "NO_SIGNAL"
            p_up = p_down = None
        else:
            weights = [max(1.0, math.sqrt(x["n"])) for x in selected]
            z = sum(weights)
            p_up = sum(x["p_up"] * w for x, w in zip(selected, weights)) / z
            p_down = 1.0 - p_up
            direction = "UP" if p_up > 0.55 else "DOWN" if p_down > 0.55 else "NO_SIGNAL"

        reversal_prob: Optional[float] = None
        continuation_prob: Optional[float] = None
        sway_delta: Optional[float] = None
        sway_direction: Optional[str] = None
        sway_mode: Optional[str] = None

        if trajectory_direction and selected:
            reversal_prob = p_down if trajectory_direction == "UP" else p_up
            continuation_prob = p_up if trajectory_direction == "UP" else p_down
            # Baseline reversal for a near-balanced binary outcome is 0.5.
            # Keep it conservative until an explicit baseline memory table is
            # available; this does not use timestamps.
            baseline_reversal = 0.5
            sway_delta = reversal_prob - baseline_reversal
            if sway_delta >= self.min_effect:
                sway_direction = "DOWN" if trajectory_direction == "UP" else "UP"
                sway_mode = "reversal"
            elif sway_delta <= -self.min_effect:
                sway_direction = trajectory_direction
                sway_mode = "continuation"

        raw = {
            "active_engine_count": len(signature_engines),
            "directional_engine_count": directional_engine_count(outputs),
            "context_atom_count": len(atoms),
            "trajectory_direction": trajectory_direction,
            "p_up": p_up,
            "p_down": p_down,
            "reversal_probability": reversal_prob,
            "continuation_probability": continuation_prob,
            "sway_direction": sway_direction,
            "sway_delta": sway_delta,
            "sway_mode": sway_mode,
            "matched_interactions": selected,
            "exact_signature_count": len([a for a in atoms_obj if a.kind == "exact_signature"]),

            "state_json": state_json,
        }

        # Persist the current research prediction separately from the base
        # production prediction. The observation was already persisted above.
        if self.persist:
            try:
                await self.memory.record_prediction(
                session_id=ctx.session_id,
                round_id=ctx.round_id,
                engine=self.name,
                direction=direction,
                p_up=p_up,
                p_down=p_down,
                confidence=(max(p_up or 0.0, p_down or 0.0) if direction != "NO_SIGNAL" else None),
                sway_direction=sway_direction,
                sway_delta=sway_delta,
                matched_interactions=selected,
                    state_json=state_json,
                )
            except Exception as exc:
                raw["persistence_error"] = str(exc)
        if observation_persist_error is not None:
            raw["observation_persistence_error"] = observation_persist_error

        confidence = max(p_up or 0.0, p_down or 0.0) if direction != "NO_SIGNAL" else None
        signature = f"sway:{direction.lower()}:n{len(selected)}"
        return EngineOutput(
            engine=self.name,
            direction=direction,
            confidence=confidence,
            pattern_signature=signature,
            raw_state=raw,
        )

    def run(self, ctx: EngineContext) -> EngineOutput:
        raise NotImplementedError("InteractionSwayEngine requires the complete EngineOutput set.")

    async def run_async(self, ctx: EngineContext) -> EngineOutput:
        return EngineOutput(
            engine=self.name,
            direction="NO_SIGNAL",
            pattern_signature="sway:requires_postpass",
            raw_state={"reason": "requires complete engine output set"},
        )
