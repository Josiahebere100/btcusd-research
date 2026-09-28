"""Persistent online memory for interaction/sway research.

The service stores observations separately from predictions so the adaptive
engine can learn from every eligible previous-round state, not merely from
states on which the research engine happened to emit UP/DOWN.
"""
from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..db import get_pool


def canonical_interaction_key(atoms: Sequence[str]) -> str:
    return json.dumps(sorted(set(atoms)), separators=(",", ":"), ensure_ascii=True)


def interaction_hash(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def bounded_interactions(
    atoms: Sequence[str],
    min_order: int = 2,
    max_order: int = 4,
    max_atoms: int = 28,
    max_candidates: int = 6000,
) -> List[Tuple[str, ...]]:
    """Generate deterministic bounded conjunction candidates.

    To prevent combinatorial explosion, atoms are sorted and capped. This is
    a research-engine safeguard, not a claim that later combinations are
    unimportant.
    """
    base = sorted(set(atoms))[:max_atoms]
    out: List[Tuple[str, ...]] = []
    for order in range(min_order, max_order + 1):
        for combo in itertools.combinations(base, order):
            out.append(combo)
            if len(out) >= max_candidates:
                return out
    return out


CREATE_SQL = """
CREATE TABLE IF NOT EXISTS research_state_observations (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    round_id TEXT NOT NULL,
    active_engine_count INTEGER NOT NULL,
    directional_engine_count INTEGER NOT NULL,
    atoms_json JSONB NOT NULL,
    state_json JSONB NOT NULL,
    next_round_id TEXT,
    next_outcome TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved_at TIMESTAMPTZ,
    UNIQUE(session_id, round_id)
);

CREATE TABLE IF NOT EXISTS research_interaction_stats (
    interaction_hash TEXT PRIMARY KEY,
    interaction_key JSONB NOT NULL,
    observation_count BIGINT NOT NULL DEFAULT 0,
    up_count BIGINT NOT NULL DEFAULT 0,
    down_count BIGINT NOT NULL DEFAULT 0,
    first_seen TIMESTAMPTZ,
    last_seen TIMESTAMPTZ,
    UNIQUE(interaction_hash)
);

CREATE TABLE IF NOT EXISTS research_predictions (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    round_id TEXT NOT NULL,
    engine TEXT NOT NULL,
    direction TEXT NOT NULL,
    p_up DOUBLE PRECISION,
    p_down DOUBLE PRECISION,
    confidence DOUBLE PRECISION,
    sway_direction TEXT,
    sway_delta DOUBLE PRECISION,
    matched_interactions JSONB NOT NULL DEFAULT '[]'::jsonb,
    state_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    next_round_id TEXT,
    actual_next_outcome TEXT,
    resolved_at TIMESTAMPTZ,
    UNIQUE(session_id, round_id, engine)
);

CREATE INDEX IF NOT EXISTS idx_research_interaction_stats_hash
    ON research_interaction_stats(interaction_hash);
CREATE INDEX IF NOT EXISTS idx_research_state_observations_session_round
    ON research_state_observations(session_id, round_id);
CREATE INDEX IF NOT EXISTS idx_research_predictions_session_round
    ON research_predictions(session_id, round_id);
"""


class ResearchMemory:
    def __init__(self, min_support: int = 5, max_interaction_order: int = 4, max_candidates: int = 6000) -> None:
        self.min_support = min_support
        self.max_interaction_order = max_interaction_order
        self.max_candidates = max(1, int(max_candidates))

    _schema_ready = False
    _schema_lock = asyncio.Lock()

    async def ensure_schema(self) -> None:
        if self._schema_ready:
            return
        async with self._schema_lock:
            if self._schema_ready:
                return
            pool = await get_pool()
            async with pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(CREATE_SQL)
                await conn.commit()
            self._schema_ready = True

    async def upsert_observation(
        self,
        *,
        session_id: str,
        round_id: str,
        atoms: Sequence[str],
        state_json: str,
        active_engine_count: int,
        directional_engine_count: int,
    ) -> None:
        pool = await get_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO research_state_observations
                      (session_id, round_id, active_engine_count,
                       directional_engine_count, atoms_json, state_json)
                    VALUES (%s,%s,%s,%s,%s::jsonb,%s::jsonb)
                    ON CONFLICT (session_id, round_id) DO NOTHING
                    """,
                    (
                        session_id, round_id, active_engine_count,
                        directional_engine_count, json.dumps(list(atoms)), state_json,
                    ),
                )
            await conn.commit()

    async def record_prediction(
        self,
        *,
        session_id: str,
        round_id: str,
        engine: str,
        direction: str,
        p_up: Optional[float],
        p_down: Optional[float],
        confidence: Optional[float],
        sway_direction: Optional[str],
        sway_delta: Optional[float],
        matched_interactions: Sequence[Mapping[str, Any]],
        state_json: str,
    ) -> None:
        pool = await get_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO research_predictions
                      (session_id, round_id, engine, direction, p_up, p_down,
                       confidence, sway_direction, sway_delta,
                       matched_interactions, state_json)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb)
                    ON CONFLICT (session_id, round_id, engine)
                    DO UPDATE SET
                      direction=EXCLUDED.direction,
                      p_up=EXCLUDED.p_up,
                      p_down=EXCLUDED.p_down,
                      confidence=EXCLUDED.confidence,
                      sway_direction=EXCLUDED.sway_direction,
                      sway_delta=EXCLUDED.sway_delta,
                      matched_interactions=EXCLUDED.matched_interactions,
                      state_json=EXCLUDED.state_json
                    """,
                    (
                        session_id, round_id, engine, direction, p_up, p_down,
                        confidence, sway_direction, sway_delta,
                        json.dumps(list(matched_interactions)), state_json,
                    ),
                )
            await conn.commit()

    async def _load_stats(self, keys: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        if not keys:
            return {}
        hashes = [(interaction_hash(k),) for k in keys]
        pool = await get_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT interaction_hash, interaction_key,
                           observation_count, up_count, down_count
                    FROM research_interaction_stats
                    WHERE interaction_hash = ANY(%s)
                    """,
                    ([h[0] for h in hashes],),
                )
                rows = await cur.fetchall()
        out: Dict[str, Dict[str, Any]] = {}
        for h, key_json, n, up, down in rows:
            key = json.dumps(key_json, separators=(",", ":"), ensure_ascii=True)
            out[key] = {"hash": h, "key": key_json, "n": int(n), "up": int(up), "down": int(down)}
        return out

    async def stats_for_atoms(self, atoms: Sequence[str], *, max_candidates: int = 6000) -> List[Dict[str, Any]]:
        await self.ensure_schema()
        combos = bounded_interactions(atoms, min_order=2, max_order=self.max_interaction_order, max_candidates=max_candidates)
        keys = [canonical_interaction_key(c) for c in combos]
        raw = await self._load_stats(keys)
        return [v for v in raw.values() if v["n"] >= self.min_support]

    async def record_resolved_observation(
        self, observation_id: int, outcome: str, next_round_id: Optional[str] = None
    ) -> None:
        if outcome not in {"UP", "DOWN"}:
            return
        pool = await get_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT session_id, round_id, atoms_json
                    FROM research_state_observations
                    WHERE id = %s AND next_outcome IS NULL
                    FOR UPDATE
                    """,
                    (observation_id,),
                )
                row = await cur.fetchone()
                if row is None:
                    return
                session_id, round_id, atoms = row
                if isinstance(atoms, str):
                    atoms = json.loads(atoms)

                combos = bounded_interactions(
                    atoms,
                    min_order=2,
                    max_order=self.max_interaction_order,
                    max_candidates=self.max_candidates,
                )
                up_inc = 1 if outcome == "UP" else 0
                dn_inc = 1 if outcome == "DOWN" else 0
                rows = []
                for combo in combos:
                    key = canonical_interaction_key(combo)
                    h = interaction_hash(key)
                    rows.append((h, key, up_inc, dn_inc))
                if rows:
                    await cur.executemany(
                        """
                        INSERT INTO research_interaction_stats
                          (interaction_hash, interaction_key, observation_count,
                           up_count, down_count, first_seen, last_seen)
                        VALUES (%s,%s::jsonb,1,%s,%s,NOW(),NOW())
                        ON CONFLICT (interaction_hash)
                        DO UPDATE SET
                          observation_count = research_interaction_stats.observation_count + 1,
                          up_count = research_interaction_stats.up_count + EXCLUDED.up_count,
                          down_count = research_interaction_stats.down_count + EXCLUDED.down_count,
                          last_seen = NOW()
                        """,
                        rows,
                    )

                await cur.execute(
                    """
                    UPDATE research_state_observations
                    SET next_round_id=%s, next_outcome=%s, resolved_at=NOW()
                    WHERE id=%s AND next_outcome IS NULL
                    """,
                    (next_round_id, outcome, observation_id),
                )

                # Resolve every research prediction attached to this observed
                # round.  This is separate from the legacy `outcomes` table.
                if next_round_id is not None:
                    await cur.execute(
                        """
                        UPDATE research_predictions
                        SET next_round_id=%s, actual_next_outcome=%s, resolved_at=NOW()
                        WHERE session_id=%s
                          AND round_id=%s
                          AND actual_next_outcome IS NULL
                        """,
                        (next_round_id, outcome, session_id, round_id),
                    )
            await conn.commit()

    async def resolve_previous_round(
        self,
        *,
        session_id: str,
        current_round_id: str,
        timeout_seconds: float = 90.0,
        poll_seconds: float = 1.0,
    ) -> None:
        """Resolve the observation for the immediate previous round.

        The current round's authoritative `raw_direction` is the label for
        the previous-round state. This function waits for that current round
        to settle, but never crosses session boundaries.
        """
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        while asyncio.get_running_loop().time() < deadline:
            try:
                pool = await get_pool()
                async with pool.connection() as conn:
                    async with conn.cursor() as cur:
                        await cur.execute(
                            """
                            SELECT r.round_id, r.raw_direction
                            FROM rounds r
                            WHERE r.session_id=%s
                              AND r.round_id::numeric=%s::numeric
                              AND r.raw_direction IN ('UP','DOWN')
                            LIMIT 1
                            """,
                            (session_id, current_round_id),
                        )
                        current = await cur.fetchone()
                        obs_id = None
                        outcome = None
                        settled_round = current_round_id
                        if current:
                            outcome = current[1]
                            # Resolve ONLY the immediate previous round in this
                            # session.  Never use the latest unresolved row as a
                            # fallback: a collector gap must not turn t+2 into
                            # the label for t.
                            await cur.execute(
                                """
                                SELECT o.id, o.round_id
                            FROM research_state_observations o
                            WHERE o.session_id=%s
                              AND o.next_outcome IS NULL
                              AND o.round_id::numeric = (%s::numeric - 1)
                            LIMIT 1
                                """,
                                (session_id, current_round_id),
                            )
                            obs = await cur.fetchone()
                            if obs:
                                obs_id = int(obs[0])
                    await conn.commit()
                if obs_id is not None:
                    await self.record_resolved_observation(
                        obs_id, outcome, next_round_id=current_round_id
                    )
                    return
                # Keep polling until both the authoritative current outcome and
                # the immediate predecessor observation are present.
                await asyncio.sleep(poll_seconds)
                continue
            except Exception:
                # A research DB problem must never stop the live engine loop.
                return
