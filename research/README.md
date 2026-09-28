# BTC/USD Research — Live Integration Patch v6

This patch was prepared against the current `main` tree of `Josiahebere100/btcusd-research` after the repository became public.

## Files to replace

- `backend/app/services/prediction.py`
- `backend/app/services/research_memory.py`
- `backend/app/engines/interaction_sway.py`
- `backend/app/routes/collector.py`
- `backend/app/state.py` (only annotates the existing `seen_round_ids` set as session/round keys)

The research feature modules already present in the repository remain compatible.

## What v6 fixes

1. Research observes the complete production engine output set, including Ensemble Kalman, but is launched independently and never enters `_pick_direction()`.
2. Research observation capture is independent of `prediction_id`, so a production snapshot insert failure cannot erase the research state.
3. Every non-empty signature remains contextual state, including NO_SIGNAL signatures. Only UP/DOWN produces directional atoms.
4. Adaptive Sway distinguishes reversal from continuation correctly: positive reversal delta means a reversal signal; negative reversal delta means continuation of the current trajectory direction.
5. Research observation persistence and research prediction persistence are orchestrated once, avoiding duplicate Sway writes in the live path.
6. Interaction learning uses `executemany()` for the bounded candidate set.
7. The resolver requires the exact immediate predecessor round in the same session and will not silently label round t with t+2 after a missing round.
8. The resolver keeps polling until both the authoritative current outcome and predecessor research observation exist.
9. Collector round deduplication is session-aware, preventing identical round IDs in different sessions from suppressing prediction/research work.
10. Research resolution runs from round ingestion and does not depend on the current round having a prediction snapshot.

## Validation

Local validation on the patched source:

- `pytest -q` → **11 passed**
- `python -m py_compile` → passed for patched production modules

No live Postgres/Supabase credentials are present in this execution environment, so database I/O was not connected to the production database.

## Important deployment behavior

The primary production prediction remains unchanged. The research layer is an independent research/prediction stream and learns against the authoritative `rounds.raw_direction` of the immediate next round in the same session.

After replacing the files, restart the backend so the new import graph is loaded. `ResearchMemory.ensure_schema()` creates the dedicated research tables if they do not already exist.
