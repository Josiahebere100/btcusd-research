"""Human-readable integration patch for the existing prediction.py.

This patch intentionally does not replace the existing primary-direction
selection. The new research engines become live independent research/
prediction sources first, with their own persistence and evaluation. A later
configuration flag can promote the research ensemble if desired.
"""

PATCH = r'''
# --- imports --------------------------------------------------------------
from ..engines.interaction_seed import InteractionSeedEngine
from ..engines.interaction_sway import InteractionSwayEngine
from ..engines.research_ensemble import ResearchEnsemble

_research_seed = InteractionSeedEngine()
_research_sway = InteractionSwayEngine()
_research_ensemble = ResearchEnsemble()

# --- after `outputs = await _run_engines(ctx)` and before the existing
#     Ensemble Kalman post-pass --------------------------------------------

research_seed_out = await _research_seed.evaluate(outputs, ctx)
research_sway_out = await _research_sway.evaluate(outputs, ctx)
research_out = await _research_ensemble.evaluate(
    outputs,
    research_seed_out,
    research_sway_out,
    ctx,
)

# Preserve the research engines as independent outputs.
# IMPORTANT: leave the existing `_pick_direction(outputs)` behaviour unchanged
# initially. This means the research engines are LIVE and independently
# evaluable without silently changing the production primary direction.
outputs.extend([
    research_seed_out,
    research_sway_out,
    research_out,
])

# If later you explicitly want the research ensemble to become the primary
# application prediction, add a separate configuration-controlled branch
# rather than silently changing `_pick_direction`.
'''

print(PATCH)
