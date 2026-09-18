"""Async Postgres pool (Supabase)."""
import urllib.parse
from typing import Optional

import asyncpg

from .config import DATABASE_URL

_pool: Optional[asyncpg.Pool] = None


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        if not DATABASE_URL:
            raise RuntimeError("DATABASE_URL is not configured")

        # Manually parse the connection string to avoid asyncpg's DSN
        # parser, which fails on some pooler hostnames in Railway.
        parsed = urllib.parse.urlparse(DATABASE_URL)

        host = parsed.hostname
        port = parsed.port or 5432
        user = urllib.parse.unquote(parsed.username) if parsed.username else None
        password = urllib.parse.unquote(parsed.password) if parsed.password else None
        database = parsed.path.lstrip("/") if parsed.path else "postgres"

        if not host:
            raise RuntimeError("DATABASE_URL is missing a host")

        _pool = await asyncpg.create_pool(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            min_size=1,
            max_size=10,
            command_timeout=30,
            ssl="require",
        )
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
