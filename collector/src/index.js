'use strict';

// Collector entry point.
// Wires adapter -> session manager -> normalizer -> cloud forwarder.

const { BCGameFeedAdapter } = require('./bcgameAdapter');
const { CloudForwarder } = require('./cloudForwarder');
const { SessionManager } = require('./sessionManager');
const { Health } = require('./health');
const { normalizeTick, normalizeRound } = require('./normalizer');

// ---- Config ---------------------------------------------------------------

function required(name) {
  const v = process.env[name];
  if (!v || v.trim() === '') {
    console.error(`FATAL: missing required env var ${name}`);
    process.exit(1);
  }
  return v.trim();
}

function optionalNumber(name, fallback) {
  const v = process.env[name];
  if (v === undefined || v === '') return fallback;
  const n = Number(v);
  if (!Number.isFinite(n)) {
    console.error(`FATAL: env var ${name} must be a number`);
    process.exit(1);
  }
  return n;
}

const config = {
  bcgameWsUrl: required('BCGAME_WS_URL'),
  subscriptions: (process.env.BCGAME_SUBSCRIPTIONS || '')
    .split(',')
    .map(s => s.trim())
    .filter(Boolean),
  heartbeatMs: optionalNumber('BCGAME_HEARTBEAT_MS', 20_000),
  reconnectMinMs: optionalNumber('BCGAME_RECONNECT_MIN_MS', 1_000),
  reconnectMaxMs: optionalNumber('BCGAME_RECONNECT_MAX_MS', 60_000),
  cloudApiUrl: required('CLOUD_API_URL'),
  collectorApiKey: required('COLLECTOR_API_KEY'),
  sessionGapMs: optionalNumber('SESSION_GAP_MS', 60_000),
  healthBeaconMs: optionalNumber('HEALTH_BEACON_MS', 30_000),
  cloudBatchMs: optionalNumber('CLOUD_BATCH_MS', 1_000),
  cloudBatchMax: optionalNumber('CLOUD_BATCH_MAX', 200),
  verbose: process.env.VERBOSE === 'true'
};

if (config.subscriptions.length === 0) {
  config.subscriptions = undefined; // use adapter defaults
}

// ---- Logging --------------------------------------------------------------

function log(level, msg) {
  const line = `[${new Date().toISOString()}] [${level}] ${msg}`;
  if (level === 'ERROR') console.error(line);
  else if (level === 'WARN') console.warn(line);
  else console.log(line);
}
const logger = {
  info: (m) => log('INFO', m),
  warn: (m) => log('WARN', m),
  error: (m) => log('ERROR', m)
};

// ---- Wire everything ------------------------------------------------------

const health = new Health();
const sessionManager = new SessionManager({ gapMs: config.sessionGapMs, logger });
const forwarder = new CloudForwarder({
  apiUrl: config.cloudApiUrl,
  apiKey: config.collectorApiKey,
  logger,
  batchMs: config.cloudBatchMs,
  batchMax: config.cloudBatchMax
});
const adapter = new BCGameFeedAdapter({
  wsUrl: config.bcgameWsUrl,
  subscriptions: config.subscriptions,
  heartbeatMs: config.heartbeatMs,
  reconnectMinMs: config.reconnectMinMs,
  reconnectMaxMs: config.reconnectMaxMs,
  verbose: config.verbose,
  logger
});

forwarder.onTickResult = (n, ok) => {
  if (ok) for (let i = 0; i < n; i++) health.noteTickForwarded();
  else for (let i = 0; i < n; i++) health.noteTickForwardFailed();
};
forwarder.onRoundResult = (n, ok) => {
  if (ok) for (let i = 0; i < n; i++) health.noteRoundForwarded();
  else for (let i = 0; i < n; i++) health.noteRoundForwardFailed();
};

adapter.on('state', (s) => health.setWsState(s));
adapter.on('disconnect', ({ reason }) => health.noteDisconnect(reason));
adapter.on('connected', () => {
  logger.info('[adapter] connected');
});

adapter.on('stream', (msg) => {
  const now = new Date();

  // --- tick stream ---
  if (msg.cmd === '/kline/BTC-USD/ticker') {
    // Sessions are driven by ticks (they're the highest-frequency signal).
    const session = sessionManager.noteTick(now);
    health.setSessionId(session.session_id);

    const result = normalizeTick(msg, session.session_id, now);
    if (!result.ok) {
      health.noteTickRejected();
      if (config.verbose) logger.warn(`[tick] rejected: ${result.reason}`);
      return;
    }
    health.noteTickReceived(result.tick);
    forwarder.enqueueTick(result.tick);
    return;
  }

  // --- round stream ---
  if (msg.cmd === '/contest/BTC/USD/5/ticker') {
    const sessionId = sessionManager.currentSessionId;
    if (!sessionId) {
      // No active session yet — drop until first tick arrives.
      return;
    }
    const result = normalizeRound(msg, sessionId);
    if (!result.ok) {
      health.noteRoundRejected();
      if (config.verbose) logger.warn(`[round] rejected: ${result.reason}`);
      return;
    }
    health.noteRoundReceived(result.round);
    forwarder.enqueueRound(result.round);
    return;
  }
});

// ---- Session emission -----------------------------------------------------
// When a session ends, push the snapshot before the new one starts.
// We hook this via polling rather than events to keep sessionManager simple.
let lastEmittedSessionId = null;
setInterval(() => {
  const snap = sessionManager.snapshot();
  if (snap && snap.session_id !== lastEmittedSessionId) {
    lastEmittedSessionId = snap.session_id;
    forwarder.enqueueSession(snap);
  }
}, 5_000).unref();

// ---- Health beacon --------------------------------------------------------
// POST to /api/collector/health every N ms with the collector's internal state.
async function sendHealthBeacon() {
  try {
    const res = await fetch(`${config.cloudApiUrl.replace(/\/+$/, '')}/api/collector/health`, {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        authorization: `Bearer ${config.collectorApiKey}`
      },
      body: JSON.stringify(health.snapshot())
    });
    if (!res.ok) {
      logger.warn(`[health] beacon failed HTTP ${res.status}`);
    }
  } catch (e) {
    logger.warn(`[health] beacon error: ${e.message}`);
  }
}
setInterval(sendHealthBeacon, config.healthBeaconMs).unref();

// ---- Start ----------------------------------------------------------------

logger.info('[collector] starting');
logger.info(`[collector] cloud=${config.cloudApiUrl}`);
logger.info(`[collector] heartbeat=${config.heartbeatMs}ms gap=${config.sessionGapMs}ms`);
adapter.start();

// ---- Graceful shutdown ----------------------------------------------------

async function shutdown(signal) {
  logger.info(`[collector] received ${signal}, shutting down`);
  adapter.stop();
  sessionManager.close('shutdown');
  const snap = sessionManager.snapshot();
  // sessionManager.close() nulls current; use last one via forwarder flush
  await forwarder.stop();
  logger.info('[collector] stopped');
  process.exit(0);
}

process.on('SIGTERM', () => shutdown('SIGTERM'));
process.on('SIGINT', () => shutdown('SIGINT'));
process.on('uncaughtException', (err) => {
  logger.error(`[collector] uncaught: ${err.stack || err.message}`);
});
process.on('unhandledRejection', (err) => {
  logger.error(`[collector] unhandled rejection: ${err && err.stack ? err.stack : err}`);
});