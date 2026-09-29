"""Async Postgres pool using psycopg3."""
from typing import Optional

from psycopg_pool import AsyncConnectionPool

from .config import DATABASE_URL

_pool: Optional[AsyncConnectionPool] = None


async def get_pool() -> AsyncConnectionPool:
    global _pool

    if _pool is None:
        if not DATABASE_URL:
            raise RuntimeError("DATABASE_URL is not configured")

        _pool = AsyncConnectionPool(
            conninfo=DATABASE_URL,
            min_size=1,
            max_size=10,
            open=False,
            timeout=30.0,
            kwargs={
                "autocommit": False,
                # Supabase Transaction Pooler (port 6543) does not
                # support prepared statements.
                "prepare_threshold": None,
            },
        )

        await _pool.open()

    return _pool


async def close_pool() -> None:
    global _pool

    if _pool is not None:
        await _pool.close()
        _pool = None
