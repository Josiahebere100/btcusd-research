# FROZEN EXPERIMENTS

## TRIG_EULER
Freeze date: 2026-09-25
Evaluation end: 2026-10-25

Frozen:
- engine code (trig_euler.py)
- feature calculations
- pattern signature format
- lookup logic (Wilson CI, thresholds)
- minimum occurrence requirements
- confidence rules
- pattern-memory write path

Not frozen:
- baselines (read-only observers)
- prediction.py (adds baseline/trajectory_state keys)
- other engines
- dashboards

Primary metric: trig_euler accuracy over the 30-day window
Secondary: trig_euler accuracy minus max(baseline accuracies)

## TRAJECTORY ENGINE v6.1
Freeze date: 2026-09-25
Evaluation end: 2026-10-25

Frozen:
- engine code (trajectory.py)
- 12 model families and their transforms
- leak-free rolling-origin validation
- structural signature format
- Architecture B decision layer

Frozen data contract:
- Each trajectory prediction writes UP TO 12 rows to
  trajectory_model_outcomes
- Each row stores two distinct direction labels:
    * internal_direction_correct — sign(p_{t+3} - p_t)
    * target_direction_correct   — 15-second outcome
- Each row stores trajectory_rms and endpoint_error
  in real price units

Known limitations:
- trajectory_rms assumes the last observed tick spacing continues.
  Actual tick arrival is not guaranteed uniform; the metric is
  next-three-tick point error under assumed spacing, not pure
  temporal path error. Actual tick timestamps are not currently
  stored in trajectory_model_outcomes. Temporal realignment is
  deferred to later data collection.
- Internal forecast horizon (3 ticks ≈ 1.5s) differs from the
  project decision horizon (15s). Intentional. The two direction
  labels measure different things.

Deferred until data accumulates:
- Persistent per-hypothesis weights
- Contextual weights per regime
- Meta-learner on hypothesis_vector
- Trajectory renderer
- Calibration of ensemble_strength

## RULES
- No performance-driven modifications during the freeze window
- Baselines and read-only dashboards may evolve
- Any change to a frozen file resets the sample
