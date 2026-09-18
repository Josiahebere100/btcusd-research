"""Stub engine.

Returns NO_SIGNAL always. Used during Phase 11 to build and test the
prediction snapshot skeleton before any real engines are plugged in.

This will be replaced by the CRT, Labouchere, and Trig/Euler engines in
later phases. It exists so the pipeline can run end-to-end with a valid
(if uninformative) engine.
"""
from .base import Engine, EngineContext, EngineOutput


class StubEngine(Engine):
    name = "stub"

    def run(self, ctx: EngineContext) -> EngineOutput:
        return EngineOutput(
            engine=self.name,
            direction="NO_SIGNAL",
            confidence=None,
            pattern_signature=None,
            raw_state={
                "reason": "stub engine returns NO_SIGNAL by design",
                "recent_ticks_count": len(ctx.recent_ticks),
            },
        )
