import asyncio
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class Cursor:
    def __init__(self):
        self.calls = []
        self.obs_polls = 0

    async def execute(self, sql, params=None):
        self.calls.append((sql, params))

    async def fetchone(self):
        sql, _ = self.calls[-1]
        if "FROM rounds r" in sql:
            return ("101", "DOWN")
        if "FROM research_state_observations o" in sql:
            self.obs_polls += 1
            if self.obs_polls == 1:
                return None
            return (55, "100")
        raise AssertionError(sql)

    async def fetchall(self):
        return []

    async def executemany(self, sql, rows):
        self.calls.append((sql, list(rows)))

    async def __aenter__(self): return self
    async def __aexit__(self, *args): return False


class Conn:
    def __init__(self, cursor): self.cursor_obj = cursor
    def cursor(self): return self.cursor_obj
    async def __aenter__(self): return self
    async def __aexit__(self, *args): return False
    async def commit(self): return None


class Pool:
    def __init__(self, cursor): self.cursor_obj = cursor
    def connection(self): return Conn(self.cursor_obj)


def test_resolution_waits_for_observation_and_uses_exact_predecessor(monkeypatch):
    cursor = Cursor()
    pool = Pool(cursor)
    from backend.app.services import research_memory as mod
    async def get_pool(): return pool
    monkeypatch.setattr(mod, "get_pool", get_pool)
    ResearchMemory = mod.ResearchMemory

    class TestMemory(ResearchMemory):
        def __init__(self):
            super().__init__(max_candidates=1)
            self.resolved = None

        async def record_resolved_observation(self, observation_id, outcome, next_round_id=None):
            self.resolved = (observation_id, outcome, next_round_id)

    mem = TestMemory()
    asyncio.run(mem.resolve_previous_round(
        session_id='s1',
        current_round_id='101',
        timeout_seconds=1,
        poll_seconds=0.001,
    ))

    assert mem.resolved == (55, 'DOWN', '101')
    obs_queries = [sql for sql, _ in cursor.calls if 'FROM research_state_observations o' in sql]
    assert obs_queries
    assert 'o.round_id::numeric = (%s::numeric - 1)' in obs_queries[-1]
    assert 'SELECT MAX(r.round_id::numeric)' not in obs_queries[-1]
