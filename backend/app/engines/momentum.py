"""Momentum engine."""
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._lookup import lookup_direction
from ._temporal import dwell_bin
from .base import Engine, EngineContext, EngineOutput


CANDLE_SECONDS = 5.0
MIN_CANDLES = 90
MAX_CANDLES = 240

RSI_PERIOD = 60
MACD_FAST = 30
MACD_SLOW = 60
MACD_SIGNAL = 20
STOCH_PERIOD = 30
CCI_PERIOD = 30

RSI_OVERSOLD = 30.0
RSI_OVERBOUGHT = 70.0
STOCH_OVERSOLD = 20.0
STOCH_OVERBOUGHT = 80.0
CCI_STRONG = 100.0

DWELL_MAX_BACK = 12


def _build_candles(ticks, seconds):
    if not ticks:
        return []
    candles = []
    bucket = None
    o = h = l = c = None
    for ts, p in ticks:
        b = int(ts // seconds)
        if bucket is None:
            bucket, o, h, l, c = b, p, p, p, p
        elif b == bucket:
            if p > h:
                h = p
            if p < l:
                l = p
            c = p
        else:
            candles.append({"o": o, "h": h, "l": l, "c": c, "start": bucket * seconds})
            bucket, o, h, l, c = b, p, p, p, p
    return candles


def _ema(series, period):
    alpha = 2.0 / (period + 1.0)
    out = np.empty_like(series, dtype=np.float64)
    out[0] = series[0]
    for i in range(1, len(series)):
        out[i] = alpha * series[i] + (1.0 - alpha) * out[i - 1]
    return out


def _rsi(closes, period):
    if len(closes) < period + 1:
        return None
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = float(gains[:period].mean())
    avg_loss = float(losses[:period].mean())
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss <= 1e-12:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


def _macd(closes, fast, slow, signal):
    if len(closes) < slow + signal + 1:
        return None
    ema_fast = _ema(closes, fast)
    ema_slow = _ema(closes, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    hist = macd_line - signal_line
    return (float(macd_line[-1]), float(signal_line[-1]),
            float(hist[-1]), float(hist[-2]))


def _stoch_rsi(closes, period):
    if len(closes) < period * 2 + 1:
        return None
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    n = len(deltas)
    if n < period:
        return None
    avg_gain = float(gains[:period].mean())
    avg_loss = float(losses[:period].mean())
    rsi_values = []
    if avg_loss <= 1e-12:
        rsi_values.append(100.0)
    else:
        rs = avg_gain / avg_loss
        rsi_values.append(100.0 - (100.0 / (1.0 + rs)))
    for i in range(period, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss <= 1e-12:
            rsi_values.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsi_values.append(100.0 - (100.0 / (1.0 + rs)))
    if len(rsi_values) < period:
        return None
    recent = np.array(rsi_values[-period:], dtype=np.float64)
    lo = float(recent.min())
    hi = float(recent.max())
    if hi - lo <= 1e-9:
        return 50.0
    return float((recent[-1] - lo) / (hi - lo) * 100.0)


def _cci(highs, lows, closes, period):
    if len(closes) < period:
        return None
    tp = (highs + lows + closes) / 3.0
    recent = tp[-period:]
    sma = float(recent.mean())
    mad = float(np.abs(recent - sma).mean())
    if mad <= 1e-12:
        return 0.0
    return float((recent[-1] - sma) / (0.015 * mad))


def _rsi_bin(rsi):
    if rsi < RSI_OVERSOLD:
        return 0
    if rsi < 50.0:
        return 1
    if rsi < RSI_OVERBOUGHT:
        return 2
    return 3


def _macd_bin(hist, prev_hist):
    if hist > 0 and hist >= prev_hist:
        return 0
    if hist > 0 and hist < prev_hist:
        return 1
    if hist < 0 and hist <= prev_hist:
        return 2
    return 3


def _stoch_bin(stoch):
    if stoch < STOCH_OVERSOLD:
        return 0
    if stoch < STOCH_OVERBOUGHT:
        return 1
    return 2


def _cci_bin(cci):
    if cci < -CCI_STRONG:
        return 0
    if cci < CCI_STRONG:
        return 1
    return 2


def _core_signature(rsi, hist, prev_hist, stoch, cci):
    return (
        f"mom:r{_rsi_bin(rsi)}"
        f"_m{_macd_bin(hist, prev_hist)}"
        f"_s{_stoch_bin(stoch)}"
        f"_c{_cci_bin(cci)}"
    )


def _compute_signature(candles):
    if len(candles) < MIN_CANDLES:
        return None
    closes = np.array([c["c"] for c in candles], dtype=np.float64)
    highs = np.array([c["h"] for c in candles], dtype=np.float64)
    lows = np.array([c["l"] for c in candles], dtype=np.float64)
    rsi = _rsi(closes, RSI_PERIOD)
    macd = _macd(closes, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    stoch = _stoch_rsi(closes, STOCH_PERIOD)
    cci = _cci(highs, lows, closes, CCI_PERIOD)
    if rsi is None or macd is None or stoch is None or cci is None:
        return None
    macd_line, signal_line, hist, prev_hist = macd
    sig = _core_signature(rsi, hist, prev_hist, stoch, cci)
    features = {
        "rsi": rsi,
        "macd_hist": hist,
        "macd_line": macd_line,
        "signal_line": signal_line,
        "stoch_rsi": stoch,
        "cci": cci,
    }
    return sig, features


def _compute_dwell_s(candles, current_core_sig, max_back=DWELL_MAX_BACK):
    if len(candles) < MIN_CANDLES + 1:
        return 0.0
    for back in range(1, max_back + 1):
        sub = candles[:-back]
        if len(sub) < MIN_CANDLES:
            return (back - 1) * CANDLE_SECONDS
        result = _compute_signature(sub)
        if result is None:
            return (back - 1) * CANDLE_SECONDS
        sub_sig, _ = result
        if sub_sig != current_core_sig:
            return (back - 1) * CANDLE_SECONDS
    return max_back * CANDLE_SECONDS


class MomentumEngine(Engine):
    name = "momentum"

    def run(self, ctx):
        raise NotImplementedError("Momentum engine is async; use run_async()")

    async def run_async(self, ctx: EngineContext) -> EngineOutput:
        if not ctx.recent_ticks or len(ctx.recent_ticks) < 200:
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature=None,
                raw_state={"reason": "insufficient ticks"},
            )

        candles = _build_candles(ctx.recent_ticks, CANDLE_SECONDS)
        if len(candles) > MAX_CANDLES:
            candles = candles[-MAX_CANDLES:]

        result = _compute_signature(candles)
        if result is None:
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature=None,
                raw_state={
                    "reason": "insufficient candles",
                    "n_candles": len(candles),
                },
            )

        core_sig, features = result
        dwell_s = _compute_dwell_s(candles, core_sig, max_back=DWELL_MAX_BACK)
        full_sig = f"{core_sig}_dw{dwell_bin(dwell_s)}"

        raw = {**features, "dwell_s": dwell_s, "n_candles": len(candles)}

        try:
            lookup = await lookup_direction(full_sig)
        except Exception as e:
            raw["lookup_error"] = str(e)
            lookup = None

        if lookup is None:
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature=full_sig,
                raw_state={**raw, "lookup": "insufficient_history"},
            )

        direction, occurrences, confidence = lookup
        return EngineOutput(
            engine=self.name,
            direction=direction,
            confidence=confidence,
            pattern_signature=full_sig,
            raw_state={
                **raw,
                "lookup": {
                    "occurrences": occurrences,
                    "confidence": confidence,
                    "chosen_direction": direction,
                },
            },
        )
