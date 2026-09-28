import asyncio

from backend.app.engines.base import EngineContext, EngineOutput
from backend.app.engines.research_ensemble import ResearchEnsemble


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


def test_down_seed_probability_is_converted_before_fusion():
    seed = EngineOutput(
        "interaction_seed",
        "DOWN",
        0.65,
        "seed:down:rules1",
        {
            "seed_probability_up": None,
            "seed_probability_down": 0.65,
            "fired_rules": [],
        },
    )

    adaptive = EngineOutput(
        "interaction_sway",
        "UP",
        0.616292613452465,
        "sway:up",
        {
            "p_up": 0.616292613452465,
            "p_down": 1.0 - 0.616292613452465,
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

    expected_p_up = (0.35 * 0.45) + (0.616292613452465 * 0.55)

    assert abs(out.raw_state["seed_probability_up"] - 0.35) < 1e-12
    assert out.raw_state["seed_probability_down"] == 0.65
    assert abs(out.raw_state["probability_up"] - expected_p_up) < 1e-12
    assert abs(out.raw_state["probability_down"] - (1.0 - expected_p_up)) < 1e-12

    # 0.49646... is inside the NO_SIGNAL interval.
    assert out.direction == "NO_SIGNAL"


def test_up_seed_probability_is_fused_normally():
    seed = EngineOutput(
        "interaction_seed",
        "UP",
        0.65,
        "seed:up:rules1",
        {
            "seed_probability_up": 0.65,
            "seed_probability_down": None,
            "fired_rules": [],
        },
    )

    adaptive = EngineOutput(
        "interaction_sway",
        "UP",
        0.55,
        "sway:up",
        {
            "p_up": 0.55,
            "p_down": 0.45,
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

    expected_p_up = (0.65 * 0.45) + (0.55 * 0.55)

    assert out.raw_state["seed_probability_up"] == 0.65
    assert out.raw_state["seed_probability_down"] is None
    assert abs(out.raw_state["probability_up"] - expected_p_up) < 1e-12
    assert out.direction == "UP"


def test_down_seed_works_when_adaptive_has_no_probability():
    seed = EngineOutput(
        "interaction_seed",
        "DOWN",
        0.65,
        "seed:down:rules1",
        {
            "seed_probability_up": None,
            "seed_probability_down": 0.65,
            "fired_rules": [],
        },
    )

    adaptive = EngineOutput(
        "interaction_sway",
        "NO_SIGNAL",
        None,
        "sway:no_signal",
        {
            "p_up": None,
            "p_down": None,
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

    assert abs(out.raw_state["probability_up"] - 0.35) < 1e-12
    assert abs(out.raw_state["probability_down"] - 0.65) < 1e-12
    assert out.direction == "DOWN"
