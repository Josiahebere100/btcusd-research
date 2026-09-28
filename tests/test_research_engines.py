import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "staging"))

from backend.app.engines.base import EngineContext, EngineOutput
from backend.app.engines.interaction_features import extract_observed_state, canonical_state_json
from backend.app.engines.interaction_seed import InteractionSeedEngine
from backend.app.engines.research_ensemble import ResearchEnsemble
from backend.app.engines.interaction_sway import InteractionSwayEngine


def make(engine, direction, sig):
    return EngineOutput(engine=engine, direction=direction,
                        pattern_signature=sig, raw_state={})


def test_no_signal_signature_retained():
    outs = [
        make("trajectory", "UP", "traj:ZIGZAG_lin_calm_dw0"),
        make("candlestick", "NO_SIGNAL", "cs:doji_mid_U_st0"),
    ]
    atoms = extract_observed_state(outs)
    canon = [a.canonical for a in atoms]
    assert "sig.candlestick=cs:doji_mid_U_st0" in canon
    assert not any(a.canonical == "dir.candlestick=NO_SIGNAL" for a in atoms)


def test_exact_signature_difference():
    a = extract_observed_state([make("trajectory", "UP", "traj:ZIGZAG_lin_calm_dw0")])
    b = extract_observed_state([make("trajectory", "UP", "traj:ZIGZAG_lin_calm_dw1")])
    assert canonical_state_json([make("trajectory", "UP", "traj:ZIGZAG_lin_calm_dw0")]) != canonical_state_json([make("trajectory", "UP", "traj:ZIGZAG_lin_calm_dw1")])
    assert any(x.value.endswith("dw0") for x in a)
    assert any(x.value.endswith("dw1") for x in b)


def test_seed_context_rule_fires():
    outs = [
        make("trajectory", "UP", "traj:ZIGZAG_four2_normal_dw0"),
        make("projectile", "DOWN", "proj:a1_c3_h0_dw0"),
        make("candlestick", "NO_SIGNAL", "cs:body_bull_hi_D_st0"),
        make("crt", "NO_SIGNAL", "crt:0-8-4_dw0"),
    ]
    # Seed 010 should fire: CRT b=8 + candlestick body.
    out = asyncio.run(InteractionSeedEngine().evaluate(outs, EngineContext(
        session_id="s", round_id="1", round_start_timestamp=None,
        trade_cutoff_timestamp=None, price_start_timestamp=None,
        price_end_timestamp=None, current_time=None,
        current_price=0.0, recent_ticks=[])))
    rules = [x for x in out.raw_state["fired_rules"] if x.get("rule_id") == "SEED-010"]
    assert rules


class FakeMemory:
    async def stats_for_atoms(self, atoms, max_candidates=6000):
        return [{
            "hash": "h1",
            "key": ["sig.trajectory=traj:ZIGZAG_lin_calm_dw0", "dir.trajectory=UP"],
            "n": 100, "up": 35, "down": 65,
        }]

    async def upsert_observation(self, **kwargs):
        return None

    async def record_prediction(self, **kwargs):
        return None


def test_adaptive_engine_uses_context_and_reversal():
    outs = [
        make("trajectory", "UP", "traj:ZIGZAG_lin_calm_dw0"),
        make("projectile", "DOWN", "proj:a1_c3_h0_dw0"),
        make("candlestick", "NO_SIGNAL", "cs:doji_mid_U_st0"),
    ]
    ctx = EngineContext("s", "1", None, None, None, None, None, 0.0, [])
    out = asyncio.run(InteractionSwayEngine(memory=FakeMemory(), min_support=5).evaluate(outs, ctx))
    assert out.direction == "DOWN"
    assert out.raw_state["sway_direction"] == "DOWN"
    assert out.raw_state["sway_delta"] > 0


def test_ensemble_preserves_conflict():
    ctx = EngineContext("s", "1", None, None, None, None, None, 0.0, [])
    seed = EngineOutput("interaction_seed", "DOWN", .8, "seed:x", {"seed_probability_up": .2, "fired_rules": []})
    adaptive = EngineOutput("interaction_sway", "UP", .7, "sway:up", {"p_up": .7, "matched_interactions": []})
    out = asyncio.run(ResearchEnsemble().evaluate([], seed, adaptive, ctx))
    assert out.raw_state["conflict"] is True
    assert out.raw_state["agreement"] is False


if __name__ == "__main__":
    test_no_signal_signature_retained()
    test_exact_signature_difference()
    test_seed_context_rule_fires()
    test_ensemble_preserves_conflict()
    print("ALL TESTS PASSED")
