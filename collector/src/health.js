'use strict';

// Tracks counters and timestamps the cloud can use to determine
// whether the collector is truly alive. We never claim verification
// ourselves — that's the backend's job.

class Health {
  constructor() {
    this.startedAt = new Date();
    this.lastTickAt = null;
    this.lastRoundAt = null;
    this.lastTickPrice = null;
    this.lastTickLatencyMs = null;
    this.lastRoundId = null;
    this.ticksReceived = 0;
    this.roundsReceived = 0;
    this.ticksRejected = 0;
    this.roundsRejected = 0;
    this.tickForwarded = 0;
    this.roundForwarded = 0;
    this.tickForwardFailed = 0;
    this.roundForwardFailed = 0;
    this.reconnectCount = 0;
    this.disconnectCount = 0;
    this.lastDisconnectReason = null;
    this.wsState = 'DISCONNECTED';
    this.currentSessionId = null;
  }

  noteTickReceived(normalized) {
    this.ticksReceived += 1;
    this.lastTickAt = new Date();
    this.lastTickPrice = normalized.price;
    this.lastTickLatencyMs = normalized.latency_ms;
  }
  noteTickRejected() { this.ticksRejected += 1; }
  noteTickForwarded() { this.tickForwarded += 1; }
  noteTickForwardFailed() { this.tickForwardFailed += 1; }

  noteRoundReceived(normalized) {
    this.roundsReceived += 1;
    this.lastRoundAt = new Date();
    this.lastRoundId = normalized.round_id;
  }
  noteRoundRejected() { this.roundsRejected += 1; }
  noteRoundForwarded() { this.roundForwarded += 1; }
  noteRoundForwardFailed() { this.roundForwardFailed += 1; }

  noteReconnect() { this.reconnectCount += 1; }
  noteDisconnect(reason) {
    this.disconnectCount += 1;
    this.lastDisconnectReason = reason;
  }
  setWsState(state) { this.wsState = state; }
  setSessionId(sessionId) { this.currentSessionId = sessionId; }

  snapshot() {
    return {
      collector_started_at: this.startedAt.toISOString(),
      collector_uptime_ms: Date.now() - this.startedAt.getTime(),
      ws_state: this.wsState,
      last_tick_at: this.lastTickAt ? this.lastTickAt.toISOString() : null,
      last_round_at: this.lastRoundAt ? this.lastRoundAt.toISOString() : null,
      last_tick_price: this.lastTickPrice,
      last_tick_latency_ms: this.lastTickLatencyMs,
      last_round_id: this.lastRoundId,
      ticks_received: this.ticksReceived,
      rounds_received: this.roundsReceived,
      ticks_rejected: this.ticksRejected,
      rounds_rejected: this.roundsRejected,
      ticks_forwarded: this.tickForwarded,
      rounds_forwarded: this.roundForwarded,
      ticks_forward_failed: this.tickForwardFailed,
      rounds_forward_failed: this.roundForwardFailed,
      reconnect_count: this.reconnectCount,
      disconnect_count: this.disconnectCount,
      last_disconnect_reason: this.lastDisconnectReason,
      current_session_id: this.currentSessionId,
      mode: 'LIVE'
    };
  }
}

module.exports = { Health };