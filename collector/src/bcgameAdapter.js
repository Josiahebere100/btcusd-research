'use strict';

// BCGameFeedAdapter
// - Opens a WSS connection to BC.GAME with the token in the query string
// - Sends subscriptions for the streams we care about
// - Sends application-level heartbeats ({"cmd":"ping",...})
// - Decompresses incoming zlib-compressed JSON
// - Reconnects with exponential backoff
// - Emits normalized events to the caller
//
// All BC.GAME-specific knowledge lives here. Nothing else in the collector
// needs to know the wire format.

const WebSocket = require('ws');
const zlib = require('zlib');
const crypto = require('crypto');
const { EventEmitter } = require('events');

const DEFAULT_UA =
  'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 ' +
  '(KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36';

const DEFAULT_SUBSCRIPTIONS = [
  '/kline/BTC-USD/ticker/subscribe',
  '/contest/BTC/USD/5/ticker/subscribe'
];

class BCGameFeedAdapter extends EventEmitter {
  constructor(opts) {
    super();
    this.wsUrl = opts.wsUrl;
    this.subscriptions = opts.subscriptions || DEFAULT_SUBSCRIPTIONS;
    this.heartbeatMs = opts.heartbeatMs ?? 20_000;
    this.reconnectMinMs = opts.reconnectMinMs ?? 1_000;
    this.reconnectMaxMs = opts.reconnectMaxMs ?? 60_000;
    this.verbose = !!opts.verbose;
    this.logger = opts.logger || console;

    this.ws = null;
    this.reconnectDelay = this.reconnectMinMs;
    this.heartbeatTimer = null;
    this.closed = false;
    this.token = null;
    this.cid = Buffer.from(DEFAULT_UA).toString('base64');

    this._extractToken();
  }

  _extractToken() {
    try {
      const u = new URL(this.wsUrl);
      this.token = u.searchParams.get('token');
      if (!this.token) {
        throw new Error('No "token" query parameter in BCGAME_WS_URL');
      }
    } catch (e) {
      throw new Error(`Invalid BCGAME_WS_URL: ${e.message}`);
    }
  }

  start() {
    this.closed = false;
    this._connect();
  }

  _connect() {
    if (this.closed) return;
    this.emit('state', 'CONNECTING');
    this.logger.info('[ws] connecting...');

    try {
      this.ws = new WebSocket(this.wsUrl, {
        handshakeTimeout: 15_000,
        perMessageDeflate: false
      });
    } catch (e) {
      this.logger.error(`[ws] construction failed: ${e.message}`);
      return this._scheduleReconnect('construction_error');
    }

    this.ws.on('open', () => {
      this.reconnectDelay = this.reconnectMinMs;
      this.emit('state', 'OPEN');
      this.logger.info('[ws] open');
    });

    this.ws.on('message', (data, isBinary) => this._onMessage(data, isBinary));
    this.ws.on('error', (err) => {
      this.logger.warn(`[ws] error: ${err.message}`);
    });
    this.ws.on('close', (code, reasonBuf) => {
      const reason = reasonBuf ? reasonBuf.toString() : '';
      this.logger.warn(`[ws] closed code=${code} reason=${reason}`);
      this._stopHeartbeat();
      this.emit('state', 'CLOSED');
      this.emit('disconnect', { code, reason });
      this._scheduleReconnect('close');
    });
  }

  _scheduleReconnect(reason) {
    if (this.closed) return;
    const delay = this.reconnectDelay;
    this.logger.info(`[ws] reconnect in ${delay}ms (reason=${reason})`);
    setTimeout(() => this._connect(), delay);
    this.reconnectDelay = Math.min(this.reconnectDelay * 2, this.reconnectMaxMs);
  }

  _onMessage(data, isBinary) {
    let text;
    try {
      const buf = Buffer.isBuffer(data) ? data : Buffer.from(data);
      if (isBinary) {
        text = this._decompress(buf);
      } else {
        text = buf.toString('utf8');
      }
    } catch (e) {
      this.logger.warn(`[ws] decompress failed: ${e.message}`);
      return;
    }
    if (!text) return;

    let msg;
    try {
      msg = JSON.parse(text);
    } catch {
      this.logger.warn(`[ws] non-JSON message: ${text.slice(0, 120)}`);
      return;
    }

    if (this.verbose) {
      this.logger.info(`[ws] << ${JSON.stringify(msg).slice(0, 300)}`);
    }

    // --- connect ack -> send subscribes ---
    if (msg.cmd === 'connect') {
      if (msg.status === 200) {
        this.emit('connected', msg);
        this._sendSubscriptions();
        this._startHeartbeat();
      } else {
        // 408 = "Not active for a long time" — we'll reconnect
        this.logger.warn(`[ws] connect status=${msg.status} error=${msg.error}`);
      }
      return;
    }

    // --- application-level data messages ---
    if (typeof msg.cmd === 'string' && msg.cmd.startsWith('/')) {
      this.emit('stream', msg);
      return;
    }

    this.emit('other', msg);
  }

  _decompress(buf) {
    try {
      return zlib.inflateSync(buf).toString('utf8');
    } catch {
      try {
        return zlib.inflateRawSync(buf).toString('utf8');
      } catch {
        try {
          return zlib.gunzipSync(buf).toString('utf8');
        } catch (e) {
          throw new Error('zlib inflate/inflateRaw/gunzip all failed');
        }
      }
    }
  }

  _sendRaw(obj) {
    const payload = zlib.deflateSync(Buffer.from(JSON.stringify(obj), 'utf8'));
    this.ws.send(payload);
  }

  _sendSubscriptions() {
    for (const cmd of this.subscriptions) {
      const msg = {
        cmd,
        token: this.token,
        cid: this.cid,
        reqId: crypto.randomUUID()
      };
      try {
        this._sendRaw(msg);
        this.logger.info(`[ws] >> subscribe ${cmd}`);
      } catch (e) {
        this.logger.error(`[ws] subscribe failed: ${e.message}`);
      }
    }
  }

  _startHeartbeat() {
    this._stopHeartbeat();
    this.heartbeatTimer = setInterval(() => {
      if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
      const msg = {
        cmd: 'ping',
        token: this.token,
        cid: this.cid,
        reqId: crypto.randomUUID()
      };
      try {
        this._sendRaw(msg);
        if (this.verbose) this.logger.info('[ws] >> ping');
      } catch (e) {
        this.logger.warn(`[ws] ping failed: ${e.message}`);
      }
    }, this.heartbeatMs);
    if (this.heartbeatTimer.unref) this.heartbeatTimer.unref();
  }

  _stopHeartbeat() {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  stop() {
    this.closed = true;
    this._stopHeartbeat();
    if (this.ws) {
      try { this.ws.close(); } catch {}
    }
  }
}

module.exports = { BCGameFeedAdapter };