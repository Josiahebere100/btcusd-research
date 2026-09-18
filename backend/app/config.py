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
