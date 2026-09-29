"""Live hard-coded seed hypotheses discovered from the current research corpus.

These rules are explicit hypotheses. They are not guarantees and they are not
allowed to mutate themselves. The engine separately tracks live performance
through the research persistence layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .base import Engine, EngineContext, EngineOutput
from .interaction_features import extract_observed_state, parse_signature_features


@dataclass(frozen=True)
class SeedRule:
    rule_id: str
    description: str
    target: str
    historical_support: int
    historical_probability: float
    evaluator: str
    source: str = "trajectory_all_engines.csv"
    tags: Tuple[str, ...] = ()


# The discovery set is deliberately isolated from the adaptive miner.
SEED_RULES: Tuple[SeedRule, ...] = (
    SeedRule(
        "SEED-001",
        "trajectory DOWN + candlestick D-state + CRT c=4 + trig rel=5",
        "UP",
        45,
        0.8000,
        "r001",
        tags=("reversal", "medium_support"),
    ),
    SeedRule(
        "SEED-002",
        "SEED-001 + navier_stokes p=1",
        "UP",
        24,
        0.9583,
        "r002",
        tags=("reversal", "low_sample_high_amplitude"),
    ),
    SeedRule(
        "SEED-003",
        "trajectory UP + CRT b=6 + labouchere sum=3 + navier_stokes v=1",
        "DOWN",
        32,
        0.8125,
        "r003",
        tags=("reversal", "low_sample_high_amplitude"),
    ),
    SeedRule(
        "SEED-004",
        "SEED-003 + superformula d=D",
        "DOWN",
        24,
        0.9167,
        "r004",
        tags=("reversal", "low_sample_high_amplitude"),
    ),
    SeedRule(
        "SEED-005",
        "trajectory DOWN + candlestick D-state + CRT c=4 + labouchere sum=7 + navier_stokes p=1",
        "UP",
        23,
        0.9130,
        "r005",
        tags=("reversal", "low_sample_high_amplitude"),
    ),
    SeedRule(
        "SEED-006",
        "exact entropy + exact navier_stokes + exact ns_flow contextual triple",
        "UP",
        48,
        0.7292,
        "r006",
        tags=("context_only",),
    ),
    SeedRule(
        "SEED-007",
        "exact trajectory + exact candlestick",
        "DOWN",
        58,
        0.7069,
        "r007",
        tags=("exact_context",),
    ),
    SeedRule(
        "SEED-008",
        "exact entropy + exact superformula",
        "UP",
        81,
        0.6790,
        "r008",
        tags=("exact_context",),
    ),
    SeedRule(
        "SEED-009",
        "exact trail + exact curve geometry",
        "UP",
        103,
        0.6311,
        "r009",
        tags=("exact_context", "higher_support"),
    ),
    SeedRule(
        "SEED-010",
        "CRT b=8 + candlestick body context",
        "UP",
        126,
        0.6667,
        "r010",
        tags=("structural_context",),
    ),
    SeedRule(
        "SEED-011",
        "exact entropy + momentum m=1",
        "UP",
        191,
        0.6492,
        "r011",
        tags=("structural_context", "higher_support"),
    ),
    SeedRule(
        "SEED-012",
        "trajectory DOWN + CRT sum=10",
        "UP",
        117,
        0.5897,
        "r012",
        tags=("moderate_reversal",),
    ),
    SeedRule(
        "SEED-013",
        "trajectory DOWN + candlestick mid context",
        "UP",
        272,
        0.5772,
        "r013",
        tags=("moderate_reversal", "higher_support"),
    ),
    SeedRule(
        "SEED-014",
        "trajectory UP + trajectory dw1 + CRT a=2",
        "DOWN",
        89,
        0.6517,
        "r014",
        tags=("reversal", "medium_support"),
    ),
    SeedRule(
        "SEED-015",
        "trajectory lin + CRT a=2 + repeated CRT",
        "DOWN",
        60,
        0.7167,
        "r015",
        tags=("structural_context",),
    ),
    SeedRule(
        "SEED-016",
        "exact lin trajectory dw0 + CRT c=3 + nonrepeating CRT",
        "DOWN",
        60,
        0.6500,
        "r016",
        tags=("exact_context",),
    ),
    SeedRule(
        "SEED-017",
        "trajectory quad + calm + CRT a=0",
        "UP",
        86,
        0.6279,
        "r017",
        tags=("structural_context",),
    ),
)


def _sig(outputs: Sequence[EngineOutput], engine: str) -> Optional[str]:
    for o in outputs:
        if (
            o.engine == engine
            and isinstance(o.pattern_signature, str)
            and o.pattern_signature.strip()
        ):
            return o.pattern_signature
    return None


def _direction(outputs: Sequence[EngineOutput], engine: str) -> str:
    for o in outputs:
        if o.engine == engine:
            return o.direction
    return "NO_SIGNAL"


def _raw(outputs: Sequence[EngineOutput], engine: str) -> Mapping[str, Any]:
    for o in outputs:
        if o.engine == engine and isinstance(o.raw_state, dict):
            return o.raw_state
    return {}


def _parsed(outputs: Sequence[EngineOutput], engine: str) -> Dict[str, str]:
    s = _sig(outputs, engine)
    return parse_signature_features(engine, s) if s else {}


def _crt_values(outputs: Sequence[EngineOutput]) -> Dict[str, str]:
    return _parsed(outputs, "crt")


def _contains_digit_token(signature: Optional[str], token: str) -> bool:
    if not signature:
        return False
    return token in signature.split("_")


def _candlestick_has_d_context(outputs: Sequence[EngineOutput]) -> bool:
    sig = _sig(outputs, "candlestick") or ""
    raw = _raw(outputs, "candlestick")
    if raw:
        for value in raw.values():
            if isinstance(value, str) and value.upper() == "D":
                return True
    return "_D_" in sig or sig.endswith("_D_st0") or "_D_" in sig


def _candlestick_mid(outputs: Sequence[EngineOutput]) -> bool:
    sig = _sig(outputs, "candlestick") or ""
    return "mid" in sig


def _candlestick_body(outputs: Sequence[EngineOutput]) -> bool:
    sig = _sig(outputs, "candlestick") or ""
    raw = _raw(outputs, "candlestick")
    if "body" in sig:
        return True
    return any(
        isinstance(v, str) and v.lower() == "body"
        for v in raw.values()
    )


def _trajectory_model(outputs: Sequence[EngineOutput]) -> Optional[str]:
    p = _parsed(outputs, "trajectory")
    return p.get("model")


def _trajectory_dw(outputs: Sequence[EngineOutput]) -> Optional[str]:
    p = _parsed(outputs, "trajectory")
    return p.get("dw")


def _eval_rule(rule_name: str, outputs: Sequence[EngineOutput]) -> bool:
    tr_dir = _direction(outputs, "trajectory")
    tr_sig = _sig(outputs, "trajectory")
    crt = _crt_values(outputs)
    trig = _parsed(outputs, "trig_euler")
    ns = _parsed(outputs, "navier_stokes")
    lab = _parsed(outputs, "labouchere")
    sf = _parsed(outputs, "superformula")
    ent = _sig(outputs, "entropy_regime")
    mom = _parsed(outputs, "momentum")
    trail = _sig(outputs, "trail_tracer")
    cg = _sig(outputs, "curve_geometry")
    flow = _sig(outputs, "ns_flow")
    cs = _sig(outputs, "candlestick")

    if rule_name == "r001":
        return (
            tr_dir == "DOWN"
            and _candlestick_has_d_context(outputs)
            and crt.get("c") == "4"
            and trig.get("rel") == "5"
        )

    if rule_name == "r002":
        return _eval_rule("r001", outputs) and ns.get("p") == "1"

    if rule_name == "r003":
        return (
            tr_dir == "UP"
            and crt.get("b") == "6"
            and lab.get("sum") == "3"
            and ns.get("v") == "1"
        )

    if rule_name == "r004":
        return _eval_rule("r003", outputs) and sf.get("d") == "D"

    if rule_name == "r005":
        return (
            tr_dir == "DOWN"
            and _candlestick_has_d_context(outputs)
            and crt.get("c") == "4"
            and lab.get("sum") == "7"
            and ns.get("p") == "1"
        )

    if rule_name == "r006":
        return (
            ent == "ent:p3_r2_dU_dw0_v0"
            and ns.get("exact") == "ns:v1_p0_d0_a0_dw0"
            and flow == "nsflow:up:p1"
        )

    if rule_name == "r007":
        return (
            tr_sig == "traj:ZIGZAG_lin_calm_dw0"
            and cs == "cs:marubozu_bull_hi_D_st0"
        )

    if rule_name == "r008":
        return (
            ent == "ent:p3_r2_dU_dw0_v0"
            and sf.get("exact") == "sf:m0_n1_1_n2_0_dD_dw0"
        )

    if rule_name == "r009":
        return (
            trail == "trail:nt6_pep_UD_t0_dw0"
            and cg == "cg:f3_s4_q3_dw1"
        )

    if rule_name == "r010":
        return crt.get("b") == "8" and _candlestick_body(outputs)

    if rule_name == "r011":
        return (
            ent == "ent:p3_r2_dU_dw0_v0"
            and mom.get("m") == "1"
        )

    if rule_name == "r012":
        return tr_dir == "DOWN" and crt.get("sum") == "10"

    if rule_name == "r013":
        return tr_dir == "DOWN" and _candlestick_mid(outputs)

    if rule_name == "r014":
        return (
            tr_dir == "UP"
            and _trajectory_dw(outputs) == "1"
            and crt.get("a") == "2"
        )

    if rule_name == "r015":
        return (
            _trajectory_model(outputs) == "lin"
            and crt.get("a") == "2"
            and crt.get("repeated") == "true"
        )

    if rule_name == "r016":
        return (
            tr_sig == "traj:ZIGZAG_lin_calm_dw0"
            and crt.get("c") == "3"
            and crt.get("repeated") == "false"
        )

    if rule_name == "r017":
        p = _parsed(outputs, "trajectory")
        return (
            p.get("model") == "quad"
            and p.get("state") == "calm"
            and crt.get("a") == "0"
        )

    return False


class InteractionSeedEngine(Engine):
    name = "interaction_seed"
    is_async = True

    async def evaluate(
        self,
        outputs: Sequence[EngineOutput],
        ctx: EngineContext,
    ) -> EngineOutput:
        active = extract_observed_state(outputs)

        if len(
            {
                a.engine
                for a in active
                if a.kind == "exact_signature"
            }
        ) < 2:
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature="seed:insufficient_context",
                raw_state={
                    "reason": "fewer than two signature-bearing engines"
                },
            )

        fired: List[Dict[str, Any]] = []

        for rule in SEED_RULES:
            try:
                if _eval_rule(rule.evaluator, outputs):
                    fired.append(
                        {
                            "rule_id": rule.rule_id,
                            "target": rule.target,
                            "historical_probability": rule.historical_probability,
                            "historical_support": rule.historical_support,
                            "tags": list(rule.tags),
                            "description": rule.description,
                        }
                    )
            except Exception as exc:
                fired.append(
                    {
                        "rule_id": rule.rule_id,
                        "error": str(exc),
                        "status": "NOT_EVALUATED",
                    }
                )

        valid = [
            r for r in fired
            if "target" in r
        ]

        up = [
            r for r in valid
            if r["target"] == "UP"
        ]

        down = [
            r for r in valid
            if r["target"] == "DOWN"
        ]

        # Aggregate using support-weighted historical probabilities.
        def weighted_probability(
            items: List[Dict[str, Any]],
        ) -> Optional[float]:
            if not items:
                return None

            total = sum(
                max(1, int(x["historical_support"]))
                for x in items
            )

            return sum(
                x["historical_probability"]
                * max(1, int(x["historical_support"]))
                for x in items
            ) / total

        p_up = weighted_probability(up)
        p_down = weighted_probability(down)

        direction = "NO_SIGNAL"

        if p_up is not None or p_down is not None:
            up_score = p_up if p_up is not None else 0.0
            down_score = p_down if p_down is not None else 0.0

            if abs(up_score - down_score) >= 0.05:
                direction = (
                    "UP"
                    if up_score > down_score
                    else "DOWN"
                )

        strongest = None

        if valid:
            strongest = max(
                valid,
                key=lambda x: (
                    abs(x["historical_probability"] - 0.5),
                    x["historical_support"],
                ),
            )

        sig = (
            f"seed:{direction.lower()}:rules{len(valid)}"
        )

        return EngineOutput(
            engine=self.name,
            direction=direction,
            confidence=(
                max(p_up or 0.0, p_down or 0.0)
                if direction != "NO_SIGNAL"
                else None
            ),
            pattern_signature=sig,
            raw_state={
                "seed_rule_version": "2026-09-28-seed-v1",
                "fired_rules": fired,
                "fired_rule_count": len(valid),
                "up_rule_count": len(up),
                "down_rule_count": len(down),
                "seed_probability_up": p_up,
                "seed_probability_down": p_down,
                "strongest_rule": strongest,
                "active_engine_count": len(
                    {
                        a.engine
                        for a in active
                        if a.kind == "exact_signature"
                    }
                ),
            },
        )

    def run(self, ctx: EngineContext) -> EngineOutput:
        raise NotImplementedError(
            "InteractionSeedEngine is async; use evaluate()."
        )

    async def run_async(
        self,
        ctx: EngineContext,
    ) -> EngineOutput:
        # Base Engine cannot see sibling EngineOutputs. The prediction
        # orchestrator must call evaluate(outputs, ctx) as a post-pass.
        return EngineOutput(
            engine=self.name,
            direction="NO_SIGNAL",
            pattern_signature="seed:requires_postpass",
            raw_state={
                "reason": "requires complete engine output set"
            },
        )
