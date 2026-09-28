# Interaction / Sway Research Engine Bundle

This bundle contains the first implementation of the live dual-purpose research layer:

- `InteractionSeedEngine`: today's discovered scenarios as explicit hypotheses.
- `InteractionSwayEngine`: adaptive interaction/reversal research over the live complete engine state.
- `ResearchEnsemble`: evidence-preserving fusion of seed and adaptive outputs.
- `ResearchMemory`: Postgres persistence and next-round online learning.
- `replay_interactions.py`: chronological bootstrap/replay from the all-engine CSV.
- `research_schema.sql`: dedicated research tables.
- `integration_patch_prediction.py`: post-pass integration point for the existing prediction orchestrator.

## Data semantics

- Every non-empty exact signature is retained in contextual state, including NO_SIGNAL engines.
- NO_SIGNAL is never emitted as a directional vote.
- UP/DOWN directions remain separate from exact signatures.
- No timestamp is used as a predictive feature.
- The intended label is the authoritative next-round `rounds.raw_direction` in the same session.
- Research persistence is separate from the existing evaluator's `outcomes` table.

## Integration

The existing `_run_engines(ctx)` must complete first. Then call the three research components as a post-pass over the complete `outputs` list. The included patch deliberately keeps the existing primary direction unchanged initially; research predictions are independently persisted/evaluated.

## Seed rules

The 17 seed hypotheses are stored in `interaction_seed.py` and never modified by online learning.

## Adaptive learning

`InteractionSwayEngine` generates bounded 2-, 3-, and 4-way conjunctions of exact/context atoms. It queries historical interaction statistics, applies conservative beta shrinkage, and produces reversal/sway diagnostics relative to the current trajectory direction.

The adaptive memory updates only after the authoritative next-round outcome becomes available.

## Verification status

The package was syntax-checked. Unit tests cover exact-token preservation, NO_SIGNAL contextual signatures, seed firing, adaptive reversal, and seed/adaptive conflict handling. Database integration is not executed in this sandbox because the user's live DATABASE_URL is not available here.
