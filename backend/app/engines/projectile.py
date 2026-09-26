"""Projectile Engine v2.

Treats the tick path within the play period as a kinematic system in
graph coordinates.

PLAY PERIOD:
  Defined purely by time interval [round_start, min(trade_cutoff, now)].
  All ticks falling inside this interval are used. Never falls back
  to ticks outside the interval. Never downsamples.

COORDINATE SYSTEM:
  x = normalized time within the play period, mapped to [0, 1] over
      the full play-period duration (typically 15s)
  y = price delta from play-period start, scaled by a FIXED unit
      (10 bps of the starting price). The unit does not change as new
      ticks arrive, so historical points do not get rescaled.
  t = real seconds elapsed since play-period start (parameter)

NOTE ON ax:
  Because x is derived from time, dx²/dt² ≈ 0 by construction.
  The engine reports ax but flags it in raw_state as "trivial"
  when |ax| < 1e-6. The useful kinematic signal is in y(t): vy0, ay,
  and the shape of the price arc.

KINEMATIC FIT:
  y(t) = y0 + vy0*t + 0.5*ay*t²

CONSISTENCY SCORE:
  Combined fit quality + acceleration stability + concavity penalty:
    consistency = exp(-5*(rms_y + ay_std)) * (1 if ay<0 else 0.5)
  Where ay<0 is the condition for a "gravity-like" concave-down arc.

DIRECTION:
  Forecast y at the play-period cutoff using the fitted kinematics.
  Direction = sign(y_cutoff - y_current).

SIGNATURE (structural only):
  proj:a{angle}_c{cons}_h{height}_dw{dwell}

State space: 5 x 4 x 3 x 4 = 240.
"""
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._lookup import lookup_direction
from ._temporal import dwell_bin
from .base import Engine, EngineContext, EngineOutput


# ============================================================
# Configuration
# ============================================================

DWELL_MAX_BACK = 10
Y_UNIT_BPS = 10.0  # 10 bps of the starting price = 1 unit of y
INF = float("inf")

PROJECTILE_MODE = os.environ.get("PROJECTILE_MODE", "active").strip().lower()


# ============================================================
# Play period (time-only)
# ============================================================

def _extract_play_period(ctx: EngineContext):
    """Return (ticks, meta) or (None, None) if play period is undefined.

    Play period is defined by round_start and trade_cutoff timestamps.
    Ticks falling outside [start, min(cutoff, now)] are excluded.
    """
    if ctx.round_start_timestamp is None or ctx.trade_cutoff_timestamp is None:
        return None, None
    if not ctx.recent_ticks:
        return None, None

    start_s = ctx.round_start_timestamp.timestamp()
    cutoff_s = ctx.trade_cutoff_timestamp.timestamp()
    now_s = ctx.recent_ticks[-1][0]
    end_s = min(cutoff_s, now_s)

    ticks = [(t, p) for t, p in ctx.recent_ticks if start_s <= t <= end_s]
    if not ticks:
        return None, None

    meta = {
        "start": start_s,
        "cutoff": cutoff_s,
        "end": end_s,
        "now": now_s,
        "duration_full_s": max(cutoff_s - start_s, 1e-9),
        "duration_elapsed_s": max(end_s - start_s, 1e-9),
        "n_ticks": len(ticks),
    }
    return ticks, meta


# ============================================================
# Coordinates
# ============================================================

def _build_coordinates(ticks, meta):
    """Fixed-scale coordinates. No dynamic normalization.

    x = time fraction of the full play period, in [0, 1]
    y = (price - p0) / (Y_UNIT_BPS * 0.0001 * p0)
    t = seconds since play-period start (kept as parameter)
    """
    t0, p0 = ticks[0]
    if p0 <= 0:
        return None

    t_rel = np.array([t - t0 for t, _ in ticks], dtype=np.float64)
    p_rel = np.array([p - p0 for _, p in ticks], dtype=np.float64)

    span_t = meta["duration_full_s"]
    y_unit = (Y_UNIT_BPS * 0.0001) * p0  # 10 bps of p0

    xs = t_rel / span_t
    ys = p_rel / y_unit

    return {
        "t": t_rel,
        "x": xs,
        "y": ys,
        "p0": float(p0),
        "span_t": float(span_t),
        "y_unit": float(y_unit),
    }


# ============================================================
# Quadratic fits
# ============================================================

def _fit_quadratic(t, arr):
    if len(t) < 3 or len(t) != len(arr):
        return None
    try:
        c = np.polyfit(t, arr, 2)
        return (float(c[0]), float(c[1]), float(c[2]))
    except Exception:
        return None


def _rms(actual, fitted):
    if len(actual) != len(fitted) or len(actual) == 0:
        return INF
    return float(np.sqrt(np.mean((actual - fitted) ** 2)))


# ============================================================
# Kinematic solve
# ============================================================

def _kinematic_solve(coords, meta):
    """Fit y(t) = y0 + vy0*t + 0.5*ay*t².

    Returns a dict with whatever quantities are computable, with None
    for anything that requires more data than available.
    """
    t = coords["t"]
    y = coords["y"]
    n = len(t)

    result: Dict[str, Any] = {
        "n_points": n,
        "fit_available": False,
        "peak_available": False,
        "return_available": False,
        "range_available": False,
    }

    if n < 3:
        return result

    cy = _fit_quadratic(t, y)
    if cy is None:
        return result
    ay_half, vy0, y0 = cy
    ay = 2.0 * ay_half

    y_pred = np.polyval([ay_half, vy0, y0], t)
    rms_y = _rms(y, y_pred)

    # Local acceleration stability
    if n >= 4:
        dt = np.diff(t)
        dt = np.where(dt < 1e-9, 1e-9, dt)
        vy_local = np.diff(y) / dt
        if len(vy_local) >= 2:
            ay_local = np.diff(vy_local) / dt[1:]
            ay_std = float(np.std(ay_local)) if len(ay_local) > 1 else 0.0
        else:
            ay_std = 0.0
    else:
        ay_std = 0.0

    result.update({
        "fit_available": True,
        "y0": float(y0),
        "vy0": float(vy0),
        "ay": float(ay),
        "rms_y": float(rms_y),
        "ay_std": float(ay_std),
    })

    # Peak (only when ay < 0)
    if ay < -1e-9:
        t_peak = -vy0 / ay
        if t_peak > 0:
            y_peak = y0 + vy0 * t_peak + 0.5 * ay * t_peak * t_peak
            result["peak_available"] = True
            result["t_peak_s"] = float(t_peak)
            result["peak_height_units"] = float(y_peak)
            # Peak height in real dollars
            result["peak_height_dollars"] = float(y_peak * coords["y_unit"])

    # Time of flight to return to y = 0 (start level)
    # Solve 0.5*ay*t² + vy0*t + y0 = 0
    if ay < -1e-9:
        disc = vy0 * vy0 - 2.0 * ay * y0
        if disc >= 0:
            root = math.sqrt(disc)
            t1 = (-vy0 + root) / ay
            t2 = (-vy0 - root) / ay
            candidates = [x for x in (t1, t2) if x > 0]
            if candidates:
                t_ret = max(candidates)
                result["return_available"] = True
                result["model_return_time_s"] = float(t_ret)
                # Where on the horizontal axis does the arc return to y=0?
                x_ret = t_ret / coords["span_t"]
                result["model_return_x"] = float(x_ret)

    # Consistency
    penalty_fit = rms_y
    penalty_accel = ay_std
    concave_factor = 1.0 if ay < 0 else 0.5
    consistency = math.exp(-5.0 * (penalty_fit + penalty_accel)) * concave_factor
    result["consistency"] = float(max(0.0, min(1.0, consistency)))

    return result


# ============================================================
# Binning
# ============================================================

def _slope_bin(vy0, span_t):
    """Slope in y-units per full play period."""
    slope = vy0 * span_t  # total y drift at end of play period
    if slope < -1.0: return 0
    if slope < -0.2: return 1
    if slope < 0.2: return 2
    if slope < 1.0: return 3
    return 4


def _cons_bin(c):
    if c < 0.2: return 0
    if c < 0.5: return 1
    if c < 0.75: return 2
    return 3


def _height_bin(kin):
    h = kin.get("peak_height_units")
    if h is None:
        return 0
    if h < 0.5: return 1
    if h < 2.0: return 2
    return 3


# ============================================================
# Signature (structural only)
# ============================================================

def _core_signature(ticks, meta):
    coords = _build_coordinates(ticks, meta)
    if coords is None:
        return None

    kin = _kinematic_solve(coords, meta)

    # If no kinematic fit possible, still return a signature of the
    # observed shape? No — the engine is defined by its kinematic fit.
    # Return None so the engine emits NO_SIGNAL for this play period.
    if not kin.get("fit_available"):
        return None

    a = _slope_bin(kin["vy0"], coords["span_t"])
    c = _cons_bin(kin["consistency"])
    h = _height_bin(kin)

    sig = f"proj:a{a}_c{c}_h{h}"

    # Forecast to end of play period (cutoff)
    t_cutoff_rel = meta["duration_full_s"]
    y_forecast = kin["y0"] + kin["vy0"] * t_cutoff_rel + 0.5 * kin["ay"] * t_cutoff_rel * t_cutoff_rel
    y_current = float(coords["y"][-1])

    delta = y_forecast - y_current
    if delta > 1e-4:
        direction = "UP"
    elif delta < -1e-4:
        direction = "DOWN"
    else:
        direction = None

    # Flag whether ax is meaningful
    ax_trivial = True  # by construction (x is derived from time)

    features = {
        "signature": sig,
        "kinematics": {
            "y0": kin["y0"],
            "vy0": kin["vy0"],
            "ay": kin["ay"],
            "ax_trivial": ax_trivial,
            "ax_note": "x is derived from time; ax ≈ 0 by construction",
        },
        "derived": {
            "t_peak_s": kin.get("t_peak_s"),
            "peak_height_units": kin.get("peak_height_units"),
            "peak_height_dollars": kin.get("peak_height_dollars"),
            "model_return_time_s": kin.get("model_return_time_s"),
            "model_return_x": kin.get("model_return_x"),
        },
        "fit": {
            "rms_y": kin.get("rms_y"),
            "ay_std": kin.get("ay_std"),
            "consistency": kin.get("consistency"),
        },
        "forecast": {
            "t_cutoff_s": float(meta["duration_full_s"]),
            "y_forecast": float(y_forecast),
            "y_current": y_current,
            "delta": float(delta),
            "direction": direction,
        },
        "play_period": {
            "duration_full_s": float(meta["duration_full_s"]),
            "duration_elapsed_s": float(meta["duration_elapsed_s"]),
            "n_ticks": int(meta["n_ticks"]),
            "start": float(meta["start"]),
            "cutoff": float(meta["cutoff"]),
        },
        "coordinates": {
            "p0": float(coords["p0"]),
            "y_unit": float(coords["y_unit"]),
            "span_t_s": float(coords["span_t"]),
        },
        "path_x": [round(float(v), 6) for v in coords["x"]],
        "path_y": [round(float(v), 6) for v in coords["y"]],
        "path_t": [round(float(v), 4) for v in coords["t"]],
    }

    return sig, features, direction, kin["consistency"]


# ============================================================
# Engine
# ============================================================

class ProjectileEngine(Engine):
    name = "projectile"

    def run(self, ctx):
        raise NotImplementedError("ProjectileEngine is async; use run_async()")

    async def run_async(self, ctx: EngineContext) -> EngineOutput:
        if PROJECTILE_MODE != "active":
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature=None,
                raw_state={"reason": f"PROJECTILE_MODE={PROJECTILE_MODE}"},
            )

        ticks, meta = _extract_play_period(ctx)
        if ticks is None:
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature=None,
                raw_state={"reason": "play period undefined (no round timestamps)"},
            )

        result = _core_signature(ticks, meta)
        if result is None:
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature=None,
                raw_state={
                    "reason": "kinematic fit unavailable",
                    "n_ticks": len(ticks),
                },
            )

        core_sig, features, kin_dir, kin_conf = result

        # Dwell: how long has this structural signature persisted?
        dwell_s = 0.0
        now_ts = ticks[-1][0]
        for back in range(1, DWELL_MAX_BACK + 1):
            sub = ticks[:-back]
            if len(sub) < 3:
                break
            sub_meta = dict(meta)
            sub_meta["end"] = sub[-1][0]
            sub_meta["duration_elapsed_s"] = max(sub[-1][0] - meta["start"], 1e-9)
            sub_meta["n_ticks"] = len(sub)
            sub_result = _core_signature(sub, sub_meta)
            if sub_result is None:
                break
            if sub_result[0] != core_sig:
                dwell_s = now_ts - sub[-1][0]
                break

        full_sig = core_sig + "_dw" + str(dwell_bin(dwell_s))

        raw = dict(features)
        raw["dwell_s"] = dwell_s

        pattern_lookup = None
        try:
            pattern_lookup = await lookup_direction(full_sig)
        except Exception as e:
            raw["lookup_error"] = str(e)

        # Decision layer (Architecture B, same as trajectory)
        if kin_dir is None:
            if pattern_lookup is not None:
                p_dir, p_occ, p_conf = pattern_lookup
                return EngineOutput(
                    engine=self.name,
                    direction=p_dir,
                    confidence=p_conf,
                    pattern_signature=full_sig,
                    raw_state={
                        **raw,
                        "decision_source": "pattern_fallback",
                        "lookup": {"occurrences": p_occ,
                                   "confidence": p_conf,
                                   "chosen_direction": p_dir},
                    },
                )
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature=full_sig,
                raw_state={**raw, "decision_source": "both_silent"},
            )

        if pattern_lookup is None:
            return EngineOutput(
                engine=self.name,
                direction=kin_dir,
                confidence=kin_conf,
                pattern_signature=full_sig,
                raw_state={
                    **raw,
                    "decision_source": "kinematic_only",
                    "lookup": "insufficient_history",
                },
            )

        p_dir, p_occ, p_conf = pattern_lookup

        if p_dir == kin_dir:
            combined_conf = min(1.0, kin_conf * (0.5 + 0.5 * p_conf))
            return EngineOutput(
                engine=self.name,
                direction=kin_dir,
                confidence=combined_conf,
                pattern_signature=full_sig,
                raw_state={
                    **raw,
                    "decision_source": "kinematic_and_pattern_agree",
                    "kinematic_confidence": kin_conf,
                    "pattern_confidence": p_conf,
                    "lookup": {"occurrences": p_occ,
                               "confidence": p_conf,
                               "chosen_direction": p_dir},
                },
            )

        return EngineOutput(
            engine=self.name,
            direction="NO_SIGNAL",
            pattern_signature=full_sig,
            raw_state={
                **raw,
                "decision_source": "kinematic_pattern_disagree",
                "kinematic_direction": kin_dir,
                "pattern_direction": p_dir,
                "kinematic_confidence": kin_conf,
                "pattern_confidence": p_conf,
                "lookup": {"occurrences": p_occ,
                           "confidence": p_conf,
                           "chosen_direction": p_dir},
            },
        )
