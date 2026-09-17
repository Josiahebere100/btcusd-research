'use strict';

const crypto = require('crypto');

// One session = one continuous recording period.
// Any gap >= SESSION_GAP_MS between received ticks starts a new session.
// Every record carries a session_id.

class SessionManager {
  constructor({ gapMs = 60_000, logger }) {
    this.gapMs = gapMs;
    this.logger = logger || console;
    this.current = null;
    this.lastTickAt = null;
    this.previousSessionId = null;
  }

  // Called on every tick with the wall-clock time we received it.
  noteTick(receivedAt) {
    const now = receivedAt.getTime();
    if (this.current === null) {
      return this._startSession(now, null, 'startup');
    }
    const gap = now - this.lastTickAt;
    if (gap >= this.gapMs) {
      const old = this.current.session_id;
      this._endSession(this.lastTickAt, 'gap');
      const next = this._startSession(now, gap, 'gap');
      this.logger.warn(
        `[session] gap=${gap}ms old=${old} new=${next.session_id}`
      );
      return next;
    }
    this.lastTickAt = now;
    return this.current;
  }

  _startSession(nowMs, gapBeforeMs, reason) {
    const sessionId = crypto.randomUUID();
    this.current = {
      session_id: sessionId,
      start_timestamp: new Date(nowMs).toISOString(),
      end_timestamp: null,
      gap_before_ms: gapBeforeMs,
      status: 'ACTIVE',
      reason
    };
    this.lastTickAt = nowMs;
    this.logger.info(`[session] started ${sessionId} (reason=${reason})`);
    return this.current;
  }

  _endSession(endMs, reason) {
    if (!this.current) return;
    this.current.end_timestamp = new Date(endMs).toISOString();
    this.current.status = reason === 'gap' ? 'INTERRUPTED' : 'CLOSED';
    this.current.reason = reason;
    this.previousSessionId = this.current.session_id;
    this.current = null;
  }

  // Called on graceful shutdown.
  close(reason = 'shutdown') {
    this._endSession(Date.now(), reason);
  }

  get currentSessionId() {
    return this.current ? this.current.session_id : null;
  }

  // Emit a snapshot for the collector to forward to the cloud
  // whenever a session starts or ends.
  snapshot() {
    return this.current ? { ...this.current } : null;
  }
}

module.exports = { SessionManager };