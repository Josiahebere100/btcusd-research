import asyncio
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# Provide the production DB module expected by ResearchMemory.
dbmod = types.ModuleType("backend.app.db")


async def _placeholder_get_pool():
    raise RuntimeError("test-only placeholder")


dbmod.get_pool = _placeholder_get_pool
sys.modules.setdefault("backend.app.db", dbmod)


from backend.app.services import research_memory as mod


class Cursor:
    def __init__(self):
        self.calls = []
        self.observation_calls = 0

    async def execute(self, sql, params=None):
        self.calls.append((sql, params))

    async def fetchone(self):
        if not self.calls:
            raise AssertionError("fetchone() called before execute()")

        sql, _ = self.calls[-1]

        # Current-round lookup.
        if "FROM rounds r" in sql and "r.round_id::numeric" in sql:
            return ("102", "DOWN")

        # Immediate-predecessor lookup.
        if "FROM rounds r_prev" in sql:
            return ("101",)

        # Exact research observation lookup.
        if "FROM research_state_observations o" in sql:
            self.observation_calls += 1

            if self.observation_calls == 1:
                return None

            return (55,)

        raise AssertionError(f"Unexpected SQL:\n{sql}")

    async def fetchall(self):
        return []

    async def executemany(self, sql, rows):
        self.calls.append((sql, list(rows)))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class Conn:
    def __init__(self, cursor):
        self.cursor_obj = cursor

    def cursor(self):
        return self.cursor_obj

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def commit(self):
        return None


class Pool:
    def __init__(self, cursor):
        self.cursor_obj = cursor

    def connection(self):
        return Conn(self.cursor_obj)


def test_resolution_waits_for_exact_immediate_predecessor(monkeypatch):
    cursor = Cursor()
    pool = Pool(cursor)

    async def fake_get_pool():
        return pool

    monkeypatch.setattr(mod, "get_pool", fake_get_pool)

    class TestMemory(mod.ResearchMemory):
        def __init__(self):
            super().__init__(max_candidates=1)
            self.resolved = None

        async def record_resolved_observation(
            self,
            observation_id,
            outcome,
            next_round_id=None,
        ):
            self.resolved = (
                observation_id,
                outcome,
                next_round_id,
            )

    memory = TestMemory()

    asyncio.run(
        memory.resolve_previous_round(
            session_id="s1",
            current_round_id="102",
            timeout_seconds=1,
            poll_seconds=0.001,
        )
    )

    assert memory.resolved == (
        55,
        "DOWN",
        "102",
    )

    sql_text = "\n".join(sql for sql, _ in cursor.calls)

    # The new resolver must find the previous round from the rounds table.
    assert "FROM rounds r_prev" in sql_text
    assert "r_prev.session_id = %s" in sql_text
    assert "ORDER BY r_prev.round_id::numeric DESC" in sql_text

    # It must then require an observation for that exact round.
    assert "FROM research_state_observations o" in sql_text
    assert "o.round_id = %s" in sql_text
    assert "o.next_outcome IS NULL" in sql_text

    # The old unsafe shortcut must not exist.
    assert "round_id::numeric - 1" not in sql_text


def test_resolution_does_not_use_older_unresolved_observation(monkeypatch):
    cursor = Cursor()
    pool = Pool(cursor)

    async def fake_get_pool():
        return pool

    monkeypatch.setattr(mod, "get_pool", fake_get_pool)

    class NoResolveMemory(mod.ResearchMemory):
        def __init__(self):
            super().__init__(max_candidates=1)
            self.resolved = False

        async def record_resolved_observation(
            self,
            observation_id,
            outcome,
            next_round_id=None,
        ):
            self.resolved = True

    memory = NoResolveMemory()

    # Modify the cursor so the exact predecessor observation never appears.
    original_fetchone = cursor.fetchone

    async def no_observation_fetchone():
        if cursor.calls:
            sql, _ = cursor.calls[-1]

            if "FROM research_state_observations o" in sql:
                return None

        return await original_fetchone()

    cursor.fetchone = no_observation_fetchone

    # Use a very short timeout because we deliberately never provide
    # the required exact predecessor observation.
    asyncio.run(
        memory.resolve_previous_round(
            session_id="s1",
            current_round_id="102",
            timeout_seconds=0.01,
            poll_seconds=0.001,
        )
    )

    assert memory.resolved is False


def test_new_resolver_is_session_scoped():
    source = (
        ROOT
        / "backend"
        / "app"
        / "services"
        / "research_memory.py"
    ).read_text(encoding="utf-8")

    # Both current-round and predecessor queries must scope by session.
    assert "r.session_id = %s" in source
    assert "r_prev.session_id = %s" in source

    # Observation lookup is also session scoped.
    assert "o.session_id = %s" in source


if __name__ == "__main__":
    test_resolution_waits_for_exact_immediate_predecessor(
        type(
            "MonkeyPatch",
            (),
            {
                "setattr": staticmethod(
                    lambda obj, name, value: setattr(obj, name, value)
                )
            },
        )()
    )

    print("RESOLUTION TESTS PASSED")
