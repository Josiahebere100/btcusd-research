import asyncio
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# The research engines import ResearchMemory at module load time.
# Provide a test-only DB surface so this suite does not require live
# database credentials.
dbmod = types.ModuleType("backend.app.db")


async def _no_db():
    raise RuntimeError("test-only: no database")


dbmod.get_pool = _no_db
sys.modules.setdefault("backend.app.db", dbmod)


from backend.app.engines.base import EngineContext, EngineOutput
from backend.app.engines.interaction_features import (
    atom_values,
    canonical_state_json,
    extract_observed_state,
    select_research_atoms,
)
from backend.app.engines.interaction_seed import InteractionSeedEngine
from backend.app.engines.interaction_sway import InteractionSwayEngine
from backend.app.engines.research_ensemble import ResearchEnsemble
from backend.app.services.research_memory import (
    bounded_interactions,
    canonical_interaction_key,
    interaction_hash,
)


def make(engine, direction, sig, raw_state=None):
    return EngineOutput(
        engine=engine,
        direction=direction,
        pattern_signature=sig,
        raw_state=raw_state or {},
    )


def make_context(round_id="1", session_id="s"):
    return EngineContext(
        session_id=session_id,
        round_id=round_id,
        round_start_timestamp=None,
        trade_cutoff_timestamp=None,
        price_start_timestamp=None,
        price_end_timestamp=None,
        current_time=None,
        current_price=0.0,
        recent_ticks=[],
    )


def test_no_signal_signature_is_retained_as_context():
    outputs = [
        make(
            "trajectory",
            "UP",
            "traj:ZIGZAG_lin_calm_dw0",
        ),
        make(
            "candlestick",
            "NO_SIGNAL",
            "cs:doji_mid_U_st0",
        ),
    ]

    atoms = extract_observed_state(outputs)
    canonical = atom_values(atoms)

    assert "sig.candlestick=cs:doji_mid_U_st0" in canonical

    # NO_SIGNAL is contextual state, never a directional vote.
    assert "dir.candlestick=NO_SIGNAL" not in canonical


def test_exact_signature_difference_is_preserved():
    output_a = make(
        "trajectory",
        "UP",
        "traj:ZIGZAG_lin_calm_dw0",
    )

    output_b = make(
        "trajectory",
        "UP",
        "traj:ZIGZAG_lin_calm_dw1",
    )

    state_a = canonical_state_json([output_a])
    state_b = canonical_state_json([output_b])

    assert state_a != state_b

    atoms_a = extract_observed_state([output_a])
    atoms_b = extract_observed_state([output_b])

    assert any(atom.value.endswith("dw0") for atom in atoms_a)
    assert any(atom.value.endswith("dw1") for atom in atoms_b)


def test_no_signal_engines_do_not_create_direction_atoms():
    outputs = [
        make(
            "trajectory",
            "NO_SIGNAL",
            "traj:ZIGZAG_lin_calm_dw0",
        ),
        make(
            "candlestick",
            "NO_SIGNAL",
            "cs:doji_mid_U_st0",
        ),
        make(
            "crt",
            "NO_SIGNAL",
            "crt:0-8-4_dw0",
        ),
    ]

    atoms = extract_observed_state(outputs)

    assert all(
        not (
            atom.kind == "direction"
            and atom.direction == "NO_SIGNAL"
        )
        for atom in atoms
    )


def test_research_atom_selector_keeps_exact_signatures():
    outputs = [
        make(
            "trajectory",
            "UP",
            "traj:ZIGZAG_four2_normal_dw0",
        ),
        make(
            "candlestick",
            "NO_SIGNAL",
            "cs:body_bull_hi_D_st0",
        ),
        make(
            "crt",
            "NO_SIGNAL",
            "crt:0-8-4_dw0",
        ),
    ]

    atoms = extract_observed_state(outputs)
    selected = select_research_atoms(atoms, max_atoms=28)

    assert "sig.trajectory=traj:ZIGZAG_four2_normal_dw0" in selected
    assert "sig.candlestick=cs:body_bull_hi_D_st0" in selected
    assert "sig.crt=crt:0-8-4_dw0" in selected


def test_seed_context_rule_fires():
    outputs = [
        make(
            "trajectory",
            "UP",
            "traj:ZIGZAG_four2_normal_dw0",
        ),
        make(
            "projectile",
            "DOWN",
            "proj:a1_c3_h0_dw0",
        ),
        make(
            "candlestick",
            "NO_SIGNAL",
            "cs:body_bull_hi_D_st0",
        ),
        make(
            "crt",
            "NO_SIGNAL",
            "crt:0-8-4_dw0",
        ),
    ]

    out = asyncio.run(
        InteractionSeedEngine().evaluate(
            outputs,
            make_context(),
        )
    )

    rules = [
        rule
        for rule in out.raw_state["fired_rules"]
        if rule.get("rule_id") == "SEED-010"
    ]

    assert rules
    assert rules[0]["target"] == "UP"


class FakeMemory:
    async def ensure_schema(self):
        return None

    async def stats_for_atoms(
        self,
        atoms,
        max_candidates=6000,
    ):
        return [
            {
                "hash": "h1",
                "key": [
                    "sig.trajectory=traj:ZIGZAG_lin_calm_dw0",
                    "dir.trajectory=UP",
                ],
                "n": 100,
                "up": 35,
                "down": 65,
            }
        ]

    async def upsert_observation(self, **kwargs):
        return None

    async def record_prediction(self, **kwargs):
        return None


def test_adaptive_engine_uses_context_and_reversal():
    outputs = [
        make(
            "trajectory",
            "UP",
            "traj:ZIGZAG_lin_calm_dw0",
        ),
        make(
            "projectile",
            "DOWN",
            "proj:a1_c3_h0_dw0",
        ),
        make(
            "candlestick",
            "NO_SIGNAL",
            "cs:doji_mid_U_st0",
        ),
    ]

    out = asyncio.run(
        InteractionSwayEngine(
            memory=FakeMemory(),
            min_support=5,
        ).evaluate(
            outputs,
            make_context(),
        )
    )

    assert out.direction == "DOWN"
    assert out.raw_state["sway_direction"] == "DOWN"
    assert out.raw_state["sway_delta"] > 0


def test_sway_negative_delta_means_continuation():
    outputs = [
        make(
            "trajectory",
            "UP",
            "traj:ZIGZAG_lin_calm_dw0",
        ),
        make(
            "projectile",
            "DOWN",
            "proj:a1_c3_h0_dw0",
        ),
        make(
            "candlestick",
            "NO_SIGNAL",
            "cs:doji_mid_U_st0",
        ),
    ]

    class ContinuationMemory(FakeMemory):
        async def stats_for_atoms(
            self,
            atoms,
            max_candidates=6000,
        ):
            return [
                {
                    "hash": "h2",
                    "key": [
                        "sig.trajectory=traj:ZIGZAG_lin_calm_dw0"
                    ],
                    "n": 100,
                    "up": 80,
                    "down": 20,
                }
            ]

    out = asyncio.run(
        InteractionSwayEngine(
            memory=ContinuationMemory(),
            min_support=5,
        ).evaluate(
            outputs,
            make_context(round_id="2"),
        )
    )

    assert out.raw_state["sway_mode"] == "continuation"
    assert out.raw_state["sway_direction"] == "UP"
    assert out.raw_state["sway_delta"] < 0


def test_bounded_interactions_are_deterministic_and_capped():
    atoms = [
        "sig.a=A",
        "sig.b=B",
        "sig.c=C",
        "sig.d=D",
        "sig.e=E",
    ]

    first = bounded_interactions(
        atoms,
        min_order=2,
        max_order=4,
        max_atoms=5,
        max_candidates=20,
    )

    second = bounded_interactions(
        list(reversed(atoms)),
        min_order=2,
        max_order=4,
        max_atoms=5,
        max_candidates=20,
    )

    assert first == second
    assert len(first) <= 20


def test_interaction_key_and_hash_are_stable():
    key_a = canonical_interaction_key(
        [
            "sig.b=B",
            "sig.a=A",
            "sig.a=A",
        ]
    )

    key_b = canonical_interaction_key(
        [
            "sig.a=A",
            "sig.b=B",
        ]
    )

    assert key_a == key_b
    assert interaction_hash(key_a) == interaction_hash(key_b)


def test_ensemble_preserves_conflict():
    seed = EngineOutput(
        "interaction_seed",
        "DOWN",
        0.8,
        "seed:x",
        {
            "seed_probability_up": 0.2,
            "seed_probability_down": 0.8,
            "fired_rules": [],
        },
    )

    adaptive = EngineOutput(
        "interaction_sway",
        "UP",
        0.7,
        "sway:up",
        {
            "p_up": 0.7,
            "p_down": 0.3,
            "matched_interactions": [],
        },
    )

    out = asyncio.run(
        ResearchEnsemble().evaluate(
            [],
            seed,
            adaptive,
            make_context(),
        )
    )

    assert out.raw_state["conflict"] is True
    assert out.raw_state["agreement"] is False


def test_research_resolver_uses_actual_previous_round():
    source = (
        ROOT
        / "backend"
        / "app"
        / "services"
        / "research_memory.py"
    ).read_text(encoding="utf-8")

    # The unsafe "current - 1" lookup must be gone.
    assert "round_id::numeric - 1" not in source

    # The resolver must first identify the actual previous round.
    assert "FROM rounds r_prev" in source
    assert "r_prev.session_id = %s" in source
    assert "ORDER BY r_prev.round_id::numeric DESC" in source

    # It must then require an observation for that exact round.
    assert "o.round_id = %s" in source
    assert "o.next_outcome IS NULL" in source


def test_research_predictions_are_resolved_with_next_round():
    source = (
        ROOT
        / "backend"
        / "app"
        / "services"
        / "research_memory.py"
    ).read_text(encoding="utf-8")

    assert "actual_next_outcome" in source
    assert "next_round_id" in source
    assert "resolved_at=NOW()" in source or "resolved_at = NOW()" in source


def test_collector_resolution_is_independent_of_prediction_creation():
    source = (
        ROOT
        / "backend"
        / "app"
        / "routes"
        / "collector.py"
    ).read_text(encoding="utf-8")

    # Research resolution is scheduled directly from round ingestion.
    assert "_schedule_research_resolution" in source

    # The deduplication key includes session + round.
    assert "(r.session_id, r.round_id) not in state.seen_round_ids" in source
    assert "state.seen_round_ids.add((r.session_id, r.round_id))" in source


def test_production_research_is_outside_primary_direction_selection():
    source = (
        ROOT
        / "backend"
        / "app"
        / "services"
        / "prediction.py"
    ).read_text(encoding="utf-8")

    # Research must run as an independent post-pass.
    assert (
        "asyncio.create_task("
        "_run_research_postpass(outputs=outputs, ctx=ctx)"
        ")" in source
    )

    # Primary direction selection still uses the production outputs.
    assert "await _pick_direction(outputs)" in source

    # Research predictions must not be appended into the frozen production
    # output list before _pick_direction().
    assert "outputs.append(seed_out)" not in source
    assert "outputs.append(sway_out)" not in source
    assert "outputs.append(ensemble_out)" not in source


if __name__ == "__main__":
    test_no_signal_signature_is_retained_as_context()
    test_exact_signature_difference_is_preserved()
    test_no_signal_engines_do_not_create_direction_atoms()
    test_research_atom_selector_keeps_exact_signatures()
    test_seed_context_rule_fires()
    test_adaptive_engine_uses_context_and_reversal()
    test_sway_negative_delta_means_continuation()
    test_bounded_interactions_are_deterministic_and_capped()
    test_interaction_key_and_hash_are_stable()
    test_ensemble_preserves_conflict()
    test_research_resolver_uses_actual_previous_round()
    test_research_predictions_are_resolved_with_next_round()
    test_collector_resolution_is_independent_of_prediction_creation()
    test_production_research_is_outside_primary_direction_selection()

    print("ALL TESTS PASSED")
