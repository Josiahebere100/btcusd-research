'use strict';

// Forwards normalized records to our cloud ingestion API.
// Authenticates with COLLECTOR_API_KEY. Batches for efficiency.

class CloudForwarder {
  constructor({ apiUrl, apiKey, logger, batchMs = 1000, batchMax = 200 }) {
    this.apiUrl = apiUrl.replace(/\/+$/, ''); // strip trailing slashes
    this.apiKey = apiKey;
    this.logger = logger || console;
    this.batchMs = batchMs;
    this.batchMax = batchMax;

    this.tickQueue = [];
    this.roundQueue = [];
    this.sessionQueue = [];

    this.timer = setInterval(() => this._flush(), this.batchMs);
    if (this.timer.unref) this.timer.unref();

    this.onTickResult = null;
    this.onRoundResult = null;
    this.onSessionResult = null;
  }

  enqueueTick(tick) {
    this.tickQueue.push(tick);
    if (this.tickQueue.length >= this.batchMax) this._flushTicks();
  }
  enqueueRound(round) {
    this.roundQueue.push(round);
    if (this.roundQueue.length >= this.batchMax) this._flushRounds();
  }
  enqueueSession(session) {
    this.sessionQueue.push(session);
    if (this.sessionQueue.length >= this.batchMax) this._flushSessions();
  }

  _flush() {
    this._flushTicks();
    this._flushRounds();
    this._flushSessions();
  }

  async _post(path, body) {
    const url = `${this.apiUrl}${path}`;
    const res = await fetch(url, {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        authorization: `Bearer ${this.apiKey}`
      },
      body: JSON.stringify(body)
    });
    if (!res.ok) {
      const text = await res.text().catch(() => '');
      throw new Error(`HTTP ${res.status} ${res.statusText}: ${text.slice(0, 200)}`);
    }
    return res.json().catch(() => ({}));
  }

  async _flushTicks() {
    if (this.tickQueue.length === 0) return;
    const batch = this.tickQueue.splice(0, this.tickQueue.length);
    try {
      await this._post('/api/collector/tick', { ticks: batch });
      if (this.onTickResult) this.onTickResult(batch.length, true);
    } catch (err) {
      this.logger.error(`[forwarder] tick post failed: ${err.message}`);
      if (this.onTickResult) this.onTickResult(batch.length, false);
      // Do not requeue: losing some ticks is acceptable; blocking is not.
    }
  }

  async _flushRounds() {
    if (this.roundQueue.length === 0) return;
    const batch = this.roundQueue.splice(0, this.roundQueue.length);
    try {
      await this._post('/api/collector/round', { rounds: batch });
      if (this.onRoundResult) this.onRoundResult(batch.length, true);
    } catch (err) {
      this.logger.error(`[forwarder] round post failed: ${err.message}`);
      if (this.onRoundResult) this.onRoundResult(batch.length, false);
    }
  }

  async _flushSessions() {
    if (this.sessionQueue.length === 0) return;
    const batch = this.sessionQueue.splice(0, this.sessionQueue.length);
    try {
      await this._post('/api/collector/session', { sessions: batch });
      if (this.onSessionResult) this.onSessionResult(batch.length, true);
    } catch (err) {
      this.logger.error(`[forwarder] session post failed: ${err.message}`);
      if (this.onSessionResult) this.onSessionResult(batch.length, false);
    }
  }

  async stop() {
    clearInterval(this.timer);
    await Promise.all([this._flushTicks(), this._flushRounds(), this._flushSessions()]);
  }
}

module.exports = { CloudForwarder };