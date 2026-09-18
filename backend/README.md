# BTC/USD Research Backend

FastAPI service that receives normalized data from the external collector
and writes it to the Supabase Postgres database.

## Routes

All routes require `Authorization: Bearer <COLLECTOR_API_KEY>`.

| Method | Path | Purpose |
|--------|------|---------|
| GET    | `/api/collector/health` | Return current feed state |
| POST   | `/api/collector/health` | Receive a health beacon from the collector |
| POST   | `/api/collector/tick`   | Receive a batch of ticks |
| POST   | `/api/collector/round`  | Receive a batch of round updates |
| POST   | `/api/collector/session`| Receive a batch of session records |

## Environment

See `.env.example`. Required:
- `COLLECTOR_API_KEY`
- `DATABASE_URL`

## Local run

```bash
pip install -r requirements.txt
cp .env.example .env
# edit .env
uvicorn main:app --host 0.0.0.0 --port 8000