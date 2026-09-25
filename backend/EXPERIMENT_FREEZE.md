# TRIG_EULER FROZEN EXPERIMENT

Freeze date: 2026-09-25
Evaluation end: 2026-10-25

## Frozen
- engine code (trig_euler.py)
- feature calculations
- pattern signature format
- lookup logic (Wilson CI, thresholds)
- minimum occurrence requirements
- confidence rules
- pattern-memory write path for trig_euler

## Not frozen
- baselines module (`baselines.py`) — a new file; read-only observers; do not affect trig_euler
- prediction.py — one block added at the end; appends a `baseline` key to the JSON after direction is decided; does not affect trig_euler's decision
- evaluator.py — one optional guard line; no behavioral change for trig_euler
- other engines — may be tuned, but their changes must not touch trig_euler's inputs
- dashboards, queries, documentation

## Metric
Primary: trig_euler accuracy over the 30-day window
Secondary: trig_euler accuracy minus max(baseline accuracies)
Tertiary: trig_euler coverage vs. baseline coverage

## Note on the signature JSON
The signature JSON now contains a top-level `"baseline"` key. Example:

    {
      "crt": "crt:5-8-7_dw0",
      "trig_euler": "trig:rel5_...",
      ...,
      "baseline": {
        "baseline_always_up": {"dir": "UP", "made_prediction": true},
        "baseline_always_down": {"dir": "DOWN", "made_prediction": true},
        ...
      },
      "_combination": "..."
    }

The `"baseline"` key is ignored by every reader of the signature JSON
except the comparison query. It cannot affect any engine.

## No performance-driven modifications to trig_euler during this window.
