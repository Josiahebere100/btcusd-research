"""Persistent online memory for interaction/sway research.

The service stores observations separately from predictions so the adaptive
engine can learn from every eligible previous-round state, not merely from
states on which the research engine happened to emit UP/DOWN.

Important labeling rule
-----------------------
A research observation made on round N is labeled only by the authoritative
raw_direction of the immediate next round N+1 within the SAME session.

The resolver must never:
* skip over a missing round observation,
* use an older unresolved observation as a fallback,
* cross a session boundary,
* use the legacy `outcomes` table as the research label source.
"""
from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..db import get_pool


def canonical_interaction_key(atoms: Sequence[str]) -> str:
    """Return the canonical serialized identity of an interaction."""
    return json.dumps(
        sorted(set(atoms)),
        separators=(",", ":"),
        ensure_ascii=True,
    )


def interaction_hash(key: str) -> str:
    """Return the stable SHA-256 identity for a canonical interaction key."""
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
    def __init__(
        self,
        min_support: int = 5,
        max_interaction_order: int = 4,
        max_candidates: int = 6000,
    ) -> None:
        self.min_support = min_support
        self.max_interaction_order = max_interaction_order
        self.max_candidates = max(1, int(max_candidates))

    _schema_ready = False
    _schema_lock = asyncio.Lock()

    async def ensure_schema(self) -> None:
        """Create the research tables if they do not already exist."""
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
        """Store one immutable round-state observation.

        The unique (session_id, round_id) constraint prevents duplicate
        observations for the same round.
        """
        pool = await get_pool()

        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO research_state_observations
                      (
                        session_id,
                        round_id,
                        active_engine_count,
                        directional_engine_count,
                        atoms_json,
                        state_json
                      )
                    VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb)
                    ON CONFLICT (session_id, round_id) DO NOTHING
                    """,
                    (
                        session_id,
                        round_id,
                        active_engine_count,
                        directional_engine_count,
                        json.dumps(list(atoms)),
                        state_json,
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
        """Store or update one research-engine prediction for a round."""
        pool = await get_pool()

        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO research_predictions
                      (
                        session_id,
                        round_id,
                        engine,
                        direction,
                        p_up,
                        p_down,
                        confidence,
                        sway_direction,
                        sway_delta,
                        matched_interactions,
                        state_json
                      )
                    VALUES (
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s::jsonb,
                        %s::jsonb
                    )
                    ON CONFLICT (session_id, round_id, engine)
                    DO UPDATE SET
                      direction = EXCLUDED.direction,
                      p_up = EXCLUDED.p_up,
                      p_down = EXCLUDED.p_down,
                      confidence = EXCLUDED.confidence,
                      sway_direction = EXCLUDED.sway_direction,
                      sway_delta = EXCLUDED.sway_delta,
                      matched_interactions = EXCLUDED.matched_interactions,
                      state_json = EXCLUDED.state_json
                    """,
                    (
                        session_id,
                        round_id,
                        engine,
                        direction,
                        p_up,
                        p_down,
                        confidence,
                        sway_direction,
                        sway_delta,
                        json.dumps(list(matched_interactions)),
                        state_json,
                    ),
                )

            await conn.commit()

    async def _load_stats(
        self,
        keys: Sequence[str],
    ) -> Dict[str, Dict[str, Any]]:
        """Load persisted interaction statistics for canonical keys."""
        if not keys:
            return {}

        hashes = [interaction_hash(key) for key in keys]

        pool = await get_pool()

        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT
                        interaction_hash,
                        interaction_key,
                        observation_count,
                        up_count,
                        down_count
                    FROM research_interaction_stats
                    WHERE interaction_hash = ANY(%s)
                    """,
                    (hashes,),
                )

                rows = await cur.fetchall()

        out: Dict[str, Dict[str, Any]] = {}

        for h, key_json, n, up, down in rows:
            key = json.dumps(
                key_json,
                separators=(",", ":"),
                ensure_ascii=True,
            )

            out[key] = {
                "hash": h,
                "key": key_json,
                "n": int(n),
                "up": int(up),
                "down": int(down),
            }

        return out

    async def stats_for_atoms(
        self,
        atoms: Sequence[str],
        *,
        max_candidates: int = 6000,
    ) -> List[Dict[str, Any]]:
        """Return learned interaction statistics for the current atom set."""
        await self.ensure_schema()

        combos = bounded_interactions(
            atoms,
            min_order=2,
            max_order=self.max_interaction_order,
            max_candidates=max_candidates,
        )

        keys = [
            canonical_interaction_key(combo)
            for combo in combos
        ]

        raw = await self._load_stats(keys)

        return [
            value
            for value in raw.values()
            if value["n"] >= self.min_support
        ]

    async def record_resolved_observation(
        self,
        observation_id: int,
        outcome: str,
        next_round_id: Optional[str] = None,
    ) -> None:
        """Resolve one observation and update its interaction statistics.

        The observation is updated exactly once because the row is selected
        with `next_outcome IS NULL`.
        """
        if outcome not in {"UP", "DOWN"}:
            return

        pool = await get_pool()

        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT
                        session_id,
                        round_id,
                        atoms_json
                    FROM research_state_observations
                    WHERE id = %s
                      AND next_outcome IS NULL
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

                    rows.append(
                        (
                            h,
                            key,
                            up_inc,
                            dn_inc,
                        )
                    )

                if rows:
                    await cur.executemany(
                        """
                        INSERT INTO research_interaction_stats
                          (
                            interaction_hash,
                            interaction_key,
                            observation_count,
                            up_count,
                            down_count,
                            first_seen,
                            last_seen
                          )
                        VALUES (
                            %s,
                            %s::jsonb,
                            1,
                            %s,
                            %s,
                            NOW(),
                            NOW()
                        )
                        ON CONFLICT (interaction_hash)
                        DO UPDATE SET
                          observation_count =
                            research_interaction_stats.observation_count + 1,
                          up_count =
                            research_interaction_stats.up_count
                            + EXCLUDED.up_count,
                          down_count =
                            research_interaction_stats.down_count
                            + EXCLUDED.down_count,
                          last_seen = NOW()
                        """,
                        rows,
                    )

                await cur.execute(
                    """
                    UPDATE research_state_observations
                    SET
                        next_round_id = %s,
                        next_outcome = %s,
                        resolved_at = NOW()
                    WHERE id = %s
                      AND next_outcome IS NULL
                    """,
                    (
                        next_round_id,
                        outcome,
                        observation_id,
                    ),
                )

                # Resolve every research prediction attached to the observed
                # round. This is separate from the legacy `outcomes` table.
                if next_round_id is not None:
                    await cur.execute(
                        """
                        UPDATE research_predictions
                        SET
                            next_round_id = %s,
                            actual_next_outcome = %s,
                            resolved_at = NOW()
                        WHERE session_id = %s
                          AND round_id = %s
                          AND actual_next_outcome IS NULL
                        """,
                        (
                            next_round_id,
                            outcome,
                            session_id,
                            round_id,
                        ),
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
        """Resolve the immediate previous round inside the same session.

        The current round's authoritative `rounds.raw_direction` is the label
        for the immediate previous round's research state.

        Resolution sequence:

            observation(N) -> current round N+1 raw_direction

        The function explicitly identifies the immediately preceding round
        from the `rounds` table before looking for its research observation.

        Therefore:

            round 100 observation
            round 101 observation missing
            round 102 current

        does NOT produce:

            round 100 -> outcome(102)  [incorrect]

        Instead, no observation is resolved for round 102 until an observation
        actually exists for the exact previous round 101.
        """
        deadline = (
            asyncio.get_running_loop().time()
            + timeout_seconds
        )

        while asyncio.get_running_loop().time() < deadline:
            try:
                pool = await get_pool()

                async with pool.connection() as conn:
                    async with conn.cursor() as cur:

                        # --------------------------------------------------
                        # 1. Obtain the authoritative result for the current
                        #    round.
                        # --------------------------------------------------
                        await cur.execute(
                            """
                            SELECT
                                r.round_id,
                                r.raw_direction
                            FROM rounds r
                            WHERE r.session_id = %s
                              AND r.round_id::numeric = %s::numeric
                              AND r.raw_direction IN ('UP', 'DOWN')
                            LIMIT 1
                            """,
                            (
                                session_id,
                                current_round_id,
                            ),
                        )

                        current = await cur.fetchone()

                        obs_id: Optional[int] = None
                        outcome: Optional[str] = None

                        # --------------------------------------------------
                        # 2. Only after the current round has a settled
                        #    authoritative direction do we locate its
                        #    immediate predecessor.
                        # --------------------------------------------------
                        if current:
                            outcome = current[1]

                            await cur.execute(
                                """
                                SELECT
                                    r_prev.round_id
                                FROM rounds r_prev
                                WHERE r_prev.session_id = %s
                                  AND r_prev.round_id::numeric
                                      < %s::numeric
                                ORDER BY r_prev.round_id::numeric DESC
                                LIMIT 1
                                """,
                                (
                                    session_id,
                                    current_round_id,
                                ),
                            )

                            previous_round = await cur.fetchone()

                            # ------------------------------------------------
                            # 3. If a previous round exists, resolve ONLY
                            #    an observation belonging to that exact
                            #    round.
                            # ------------------------------------------------
                            if previous_round:
                                previous_round_id = str(
                                    previous_round[0]
                                )

                                await cur.execute(
                                    """
                                    SELECT
                                        o.id
                                    FROM research_state_observations o
                                    WHERE o.session_id = %s
                                      AND o.round_id = %s
                                      AND o.next_outcome IS NULL
                                    LIMIT 1
                                    """,
                                    (
                                        session_id,
                                        previous_round_id,
                                    ),
                                )

                                obs = await cur.fetchone()

                                if obs:
                                    obs_id = int(obs[0])

                    await conn.commit()

                # ----------------------------------------------------------
                # 4. Perform the actual learning update outside the lookup
                #    transaction.
                # ----------------------------------------------------------
                if obs_id is not None and outcome is not None:
                    await self.record_resolved_observation(
                        obs_id,
                        outcome,
                        next_round_id=current_round_id,
                    )
                    return

                # ----------------------------------------------------------
                # 5. Keep polling while either the current authoritative
                #    outcome or the exact predecessor observation is not
                #    available yet.
                # ----------------------------------------------------------
                await asyncio.sleep(poll_seconds)

            except Exception:
                # Research-memory failures must never stop the live collector
                # or production prediction loop.
                return
