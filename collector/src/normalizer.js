'use strict';

// Normalizer: converts raw BC.GAME messages into our internal format.
// Also performs source/symbol validation. Rejects anything that isn't
// the exact market we're supposed to be observing.

const SOURCE = 'BC.GAME';
const TICK_SYMBOL = 'BTC-USD';
const ROUND_SYMBOL = 'BTC/USD';

// ---- Validators -----------------------------------------------------------

function isFiniteNumber(v) {
  return typeof v === 'number' && Number.isFinite(v);
}

function isPlausiblePrice(v) {
  if (!isFiniteNumber(v)) return false;
  // BTC/USD has been between ~$1k and ~$1M historically. Be generous.
  return v > 100 && v < 10_000_000;
}

function isPlausibleEpochMs(v) {
  if (!isFiniteNumber(v)) return false;
  // Between 2020-01-01 and 2100-01-01
  return v > 1_577_836_800_000 && v < 4_102_444_800_000;
}

function toIso(epochMs) {
  if (!isPlausibleEpochMs(epochMs)) return null;
  return new Date(epochMs).toISOString();
}

// ---- Tick normalization ---------------------------------------------------

// Raw tick message shape:
// { cmd: "/kline/BTC-USD/ticker",
//   resp: { c, p, s: "BTC-USD", t: <epoch_ms> },
//   status: 0 }
function normalizeTick(raw, sessionId, receivedAt) {
  if (!raw || raw.cmd !== '/kline/BTC-USD/ticker') {
    return { ok: false, reason: 'unknown_tick_cmd' };
  }
  const r = raw.resp;
  if (!r) return { ok: false, reason: 'missing_resp' };
  if (r.s !== TICK_SYMBOL) {
    return { ok: false, reason: `wrong_tick_symbol:${r.s}` };
  }
  const price = parseFloat(r.p);
  if (!isPlausiblePrice(price)) {
    return { ok: false, reason: `implausible_tick_price:${r.p}` };
  }
  const tickTs = Number(r.t);
  if (!isPlausibleEpochMs(tickTs)) {
    return { ok: false, reason: `implausible_tick_timestamp:${r.t}` };
  }

  const receivedMs = receivedAt.getTime();
  const latencyMs = receivedMs - tickTs;

  return {
    ok: true,
    tick: {
      source: SOURCE,
      symbol: TICK_SYMBOL,
      stream: '/kline/BTC-USD/ticker',
      price,
      tick_timestamp: toIso(tickTs),
      received_at: receivedAt.toISOString(),
      latency_ms: latencyMs,
      session_id: sessionId,
      mode: 'LIVE',
      raw_change: r.c ?? null
    }
  };
}

// ---- Round normalization --------------------------------------------------

// Raw round message shape:
// { cmd: "/contest/BTC/USD/5/ticker",
//   resp: { currentTime, endPrice, feeRate, id, period,
//           previousRoundResult: [...], priceEndTime, priceStartTime,
//           roundStartTime, startPrice, status, statusChangeTime,
//           symbol: "BTC/USD", tradeCutoffTime, winSide },
//   status: 0 }
function normalizeRound(raw, sessionId) {
  if (!raw || raw.cmd !== '/contest/BTC/USD/5/ticker') {
    return { ok: false, reason: 'unknown_round_cmd' };
  }
  const r = raw.resp;
  if (!r) return { ok: false, reason: 'missing_resp' };
  if (r.symbol !== ROUND_SYMBOL) {
    return { ok: false, reason: `wrong_round_symbol:${r.symbol}` };
  }

  const roundId = String(r.id ?? '');
  if (!roundId) return { ok: false, reason: 'missing_round_id' };

  const roundStartMs = Number(r.roundStartTime);
  const tradeCutoffMs = Number(r.tradeCutoffTime);
  const priceStartMs = Number(r.priceStartTime);
  const priceEndMs = Number(r.priceEndTime);

  if (!isPlausibleEpochMs(roundStartMs)) {
    return { ok: false, reason: `bad_round_start:${r.roundStartTime}` };
  }
  if (!isPlausibleEpochMs(tradeCutoffMs)) {
    return { ok: false, reason: `bad_cutoff:${r.tradeCutoffTime}` };
  }

  const startPrice = Number(r.startPrice);
  const endPrice = Number(r.endPrice);
  const winSide = Number(r.winSide);

  // winSide: 0=unresolved, 1=UP, 2=DOWN
  let rawDirection = 'UNRESOLVED';
  if (winSide === 1) rawDirection = 'UP';
  else if (winSide === 2) rawDirection = 'DOWN';

  return {
    ok: true,
    round: {
      source: SOURCE,
      symbol: ROUND_SYMBOL,
      stream: '/contest/BTC/USD/5/ticker',
      round_id: roundId,
      round_start_timestamp: toIso(roundStartMs),
      trade_cutoff_timestamp: toIso(tradeCutoffMs),
      price_start_timestamp: toIso(priceStartMs),
      price_end_timestamp: toIso(priceEndMs),
      start_price: isPlausiblePrice(startPrice) ? startPrice : null,
      end_price: isPlausiblePrice(endPrice) ? endPrice : null,
      win_side: winSide,
      raw_direction: rawDirection,
      status_code: Number(r.status),
      status_change_at: toIso(Number(r.statusChangeTime)),
      current_time: toIso(Number(r.currentTime)),
      session_id: sessionId,
      mode: 'LIVE',
      previous_rounds: Array.isArray(r.previousRoundResult)
        ? r.previousRoundResult.slice(0, 6).map(x => ({
            round_id: String(x.id),
            start_price: Number(x.startPrice),
            end_price: Number(x.endPrice),
            win_side: Number(x.winSide)
          }))
        : []
    }
  };
}

module.exports = {
  normalizeTick,
  normalizeRound,
  SOURCE,
  TICK_SYMBOL,
  ROUND_SYMBOL
};