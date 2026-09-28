"""Environment configuration."""
import os

from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: int) -> int:
    v = os.getenv(name, "")
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        return default


COLLECTOR_API_KEY: str = os.getenv("COLLECTOR_API_KEY", "").strip()
DATABASE_URL: str = os.getenv("DATABASE_URL", "").strip()

FEED_TICK_MAX_AGE_MS: int = _int("FEED_TICK_MAX_AGE_MS", 5000)
FEED_ROUND_MAX_AGE_MS: int = _int("FEED_ROUND_MAX_AGE_MS", 15000)
FEED_MAX_LATENCY_MS: int = _int("FEED_MAX_LATENCY_MS", 3000)

# Prediction defaults (spec sections 14 and 16)
PREDICTION_HORIZON_SECONDS: int = _int("PREDICTION_HORIZON_SECONDS", 15)
PREDICTION_SAFETY_BUFFER_MS: int = _int("PREDICTION_SAFETY_BUFFER_MS", 2000)
