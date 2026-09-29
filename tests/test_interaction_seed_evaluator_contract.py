import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


from backend.app.engines.interaction_features import (
    parse_signature_features,
)
from backend.app.engines.interaction_seed import _eval_rule
from backend.app.engines.base import EngineOutput


def test_longer_prefixed_tokens_are_parsed_before_shorter_tokens():
    sf = parse_signature_features(
        "superformula",
        "sf:m0_n1_1_n2_0_dD_dw0",
    )

    ns = parse_signature_features(
        "navier_stokes",
        "ns:v1_p1_d0_a0_dw0",
    )

    assert sf["d"] == "D"
    assert sf["dw"] == "0"

    assert ns["d"] == "0"
    assert ns["dw"] == "0"


def test_seed_006_requires_exact_navier_stokes_signature():
    outputs = [
        EngineOutput(
            engine="entropy_regime",
            direction="NO_SIGNAL",
            confidence=None,
            pattern_signature="ent:p3_r2_dU_dw0_v0",
            raw_state={},
        ),
        EngineOutput(
            engine="navier_stokes",
            direction="NO_SIGNAL",
            confidence=None,
            pattern_signature="ns:v1_p0_d0_a0_dw0",
            raw_state={},
        ),
        EngineOutput(
            engine="ns_flow",
            direction="NO_SIGNAL",
            confidence=None,
            pattern_signature="nsflow:up:p1",
            raw_state={},
        ),
    ]

    assert _eval_rule("r006", outputs) is True


def test_seed_006_rejects_different_navier_stokes_dwell_signature():
    outputs = [
        EngineOutput(
            engine="entropy_regime",
            direction="NO_SIGNAL",
            confidence=None,
            pattern_signature="ent:p3_r2_dU_dw0_v0",
            raw_state={},
        ),
        EngineOutput(
            engine="navier_stokes",
            direction="NO_SIGNAL",
            confidence=None,
            pattern_signature="ns:v1_p0_d0_a0_dw1",
            raw_state={},
        ),
        EngineOutput(
            engine="ns_flow",
            direction="NO_SIGNAL",
            confidence=None,
            pattern_signature="nsflow:up:p1",
            raw_state={},
        ),
    ]

    assert _eval_rule("r006", outputs) is False


def test_seed_008_uses_exact_superformula_signature():
    outputs = [
        EngineOutput(
            engine="entropy_regime",
            direction="UP",
            confidence=None,
            pattern_signature="ent:p3_r2_dU_dw0_v0",
            raw_state={},
        ),
        EngineOutput(
            engine="superformula",
            direction="DOWN",
            confidence=None,
            pattern_signature="sf:m0_n1_1_n2_0_dD_dw0",
            raw_state={},
        ),
    ]

    assert _eval_rule("r008", outputs) is True


def test_seed_008_does_not_match_wrong_superformula_signature():
    outputs = [
        EngineOutput(
            engine="entropy_regime",
            direction="UP",
            confidence=None,
            pattern_signature="ent:p3_r2_dU_dw0_v0",
            raw_state={},
        ),
        EngineOutput(
            engine="superformula",
            direction="DOWN",
            confidence=None,
            pattern_signature="sf:m1_n1_1_n2_0_dD_dw0",
            raw_state={},
        ),
    ]

    assert _eval_rule("r008", outputs) is False
