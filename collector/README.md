# BTC/USD Collector

External long-lived Node.js process that connects to BC.GAME's real
BTC/USD WebSocket feed, normalizes the messages, and forwards them to
our cloud ingestion API.

This is the **only** component allowed to talk to BC.GAME. The frontend
and the backend never do.

## What it does

1. Opens a WebSocket to BC.GAME with a JWT in the query string.
2. Subscribes to:
   - `/kline/BTC-USD/ticker`
   - `/contest/BTC/USD/5/ticker`
3. Sends application-level heartbeats (`{"cmd":"ping",...}`) every 20 seconds.
4. Decompresses (zlib) and parses incoming JSON messages.
5. Validates source, symbol, and price sanity.
6. Detects session boundaries (gap ≥ 60s = new session).
7. Batches and forwards normalized records to the backend:
   - `POST /api/collector/tick`
   - `POST /api/collector/round`
   - `POST /api/collector/session`
8. Posts a health beacon to `POST /api/collector/health`.
9. Reconnects with exponential backoff on disconnects.
10. Shuts down gracefully on SIGTERM/SIGINT.

## Environment variables

See `.env.example`. Required:

| Variable | Purpose |
|---|---|
| `BCGAME_WS_URL` | Full WSS URL **including** `?token=...` |
| `CLOUD_API_URL` | Base URL of our backend |
| `COLLECTOR_API_KEY` | Shared secret sent to the backend |

Optional (defaults shown):

| Variable | Default |
|---|---|
| `BCGAME_SUBSCRIPTIONS` | `/kline/BTC-USD/ticker/subscribe,/contest/BTC/USD/5/ticker/subscribe` |
| `BCGAME_HEARTBEAT_MS` | `20000` |
| `BCGAME_RECONNECT_MIN_MS` | `1000` |
| `BCGAME_RECONNECT_MAX_MS` | `60000` |
| `SESSION_GAP_MS` | `60000` |
| `HEALTH_BEACON_MS` | `30000` |
| `CLOUD_BATCH_MS` | `1000` |
| `CLOUD_BATCH_MAX` | `200` |
| `VERBOSE` | `false` |

## Local run

```bash
npm install
cp .env.example .env
# edit .env with real values
npm start