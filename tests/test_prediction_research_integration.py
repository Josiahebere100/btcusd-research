import asyncio
import sys
import types
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def install_module(monkeypatch, name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items(): setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


class FakeCursor:
    def __init__(self): self.last_params = None
    async def execute(self, sql, params=None): self.last_params = params
    async def fetchone(self): return (123,)
    async def __aenter__(self): return self
    async def __aexit__(self,*args): return False

class FakeConn:
    def __init__(self, cursor): self.cursor_obj = cursor
    def cursor(self): return self.cursor_obj
    async def __aenter__(self): return self
    async def __aexit__(self,*args): return False
    async def commit(self): pass

class FakePool:
    def __init__(self): self.cursor = FakeCursor()
    def connection(self): return FakeConn(self.cursor)

class FailingCursor(FakeCursor):
    async def execute(self, sql, params=None):
        if "INSERT INTO prediction_snapshots" in sql:
            raise RuntimeError("production insert failed")
        self.last_params = params

class FailingPool:
    def __init__(self): self.cursor = FailingCursor()
    def connection(self): return FakeConn(self.cursor)

class FakeMemory:
    def __init__(self): self.events=[]
    async def ensure_schema(self): self.events.append('schema')
    async def upsert_observation(self, **kwargs): self.events.append(('obs',kwargs))
    async def record_prediction(self, **kwargs): self.events.append(('pred',kwargs))
    async def stats_for_atoms(self,*args,**kwargs): return []


def test_production_schedules_research_without_changing_primary(monkeypatch):
    from backend.app.engines.base import EngineOutput
    install_module(monkeypatch, 'backend.app.config', PREDICTION_HORIZON_SECONDS=15, PREDICTION_SAFETY_BUFFER_MS=2000)

    # Stub only the unrelated production engine modules imported by prediction.py.
    class Stub:
        def __init__(self,name): self.name=name
        def run(self,ctx): return EngineOutput(self.name, 'UP', 0.60, f'{self.name}:sig', {})
    engine_specs=[
        ('candlestick','CandlestickEngine','candlestick'),('crt','CRTEngine','crt'),
        ('curve_geometry','CurveGeometryEngine','curve_geometry'),('entropy_regime','EntropyRegimeEngine','entropy_regime'),
        ('labouchere','LabouchereEngine','labouchere'),('momentum','MomentumEngine','momentum'),
        ('navier_stokes','NavierStokesAutomatonEngine','navier_stokes'),('ns_flow','NSFlowEngine','ns_flow'),
        ('projectile','ProjectileEngine','projectile'),('superformula','SuperformulaEngine','superformula'),
        ('trail_tracer','TrailTracerEngine','trail_tracer'),('trajectory','TrajectoryEngine','trajectory'),
        ('trig_euler','TrigEulerEngine','trig_euler')]
    for mod, cls, name in engine_specs:
        install_module(monkeypatch, f'backend.app.engines.{mod}', **{cls:type(cls,(Stub,),{'__init__':lambda self,n=name: Stub.__init__(self,n)})})
    install_module(monkeypatch,'backend.app.engines.ensemble_kalman',run_ensemble_kalman=lambda outputs,ctx: asyncio.sleep(0,result=None))
    # Lazy baseline import occurs during prediction, so provide it too.
    install_module(monkeypatch,'backend.app.engines.baselines',run_baselines=lambda ctx: asyncio.sleep(0,result={}))
    install_module(monkeypatch,'backend.app.services.combination',combination_key=lambda outputs:'COMB',lookup_combination=lambda key: asyncio.sleep(0,result=None))
    install_module(monkeypatch,'backend.app.services.evaluator',evaluate_prediction=lambda *a,**k: asyncio.sleep(0))

    dbmod=types.ModuleType('backend.app.db')
    pool=FakePool()
    async def get_pool(): return pool
    dbmod.get_pool=get_pool
    monkeypatch.setitem(sys.modules,'backend.app.db',dbmod)

    # Reload the prediction module cleanly using the injected stubs.
    for name in list(sys.modules):
        if name == 'backend.app.services.prediction':
            del sys.modules[name]
    import backend.app.services.prediction as pred

    memory=FakeMemory()
    class FakeSeed:
        async def evaluate(self, outputs, ctx):
            self.seen=outputs
            return EngineOutput('interaction_seed','UP',0.60,'seed:test',{'seed_probability_up':0.60,'seed_probability_down':0.40,'fired_rules':[]})
    class FakeSway:
        async def evaluate(self, outputs, ctx):
            self.seen=outputs
            return EngineOutput('interaction_sway','DOWN',0.65,'sway:test',{'p_up':0.35,'p_down':0.65,'matched_interactions':[],'sway_direction':'DOWN','sway_delta':0.1})
    class FakeEnsemble:
        async def evaluate(self, outputs, seed, sway, ctx):
            return EngineOutput('research_ensemble','NO_SIGNAL',None,'research:none:seedC',{'probability_up':0.47,'probability_down':0.53,'adaptive_interactions':[],'agreement':False,'conflict':True})

    monkeypatch.setattr(pred,'_research_memory',memory)
    monkeypatch.setattr(pred,'_research_seed',FakeSeed())
    monkeypatch.setattr(pred,'_research_sway',FakeSway())
    monkeypatch.setattr(pred,'_research_ensemble',FakeEnsemble())

    async def run():
        now=datetime.now(timezone.utc)
        pid=await pred.create_prediction_for_round(
            session_id='s',round_id='1',round_start=now,
            trade_cutoff=now-timedelta(seconds=1),
            price_start=now,price_end=now,current_price=100.0,recent_ticks=[]
        )
        await asyncio.sleep(0.01)
        return pid

    pid=asyncio.run(run())
    assert pid==123
    assert memory.events and memory.events[0]=='schema'
    pred_events=[e for e in memory.events if isinstance(e,tuple) and e[0]=='pred']
    assert {e[1]['engine'] for e in pred_events}=={'interaction_seed','interaction_sway','research_ensemble'}
    assert pool.cursor.last_params[7]=='UP'


def test_research_runs_even_when_primary_snapshot_insert_fails(monkeypatch):
    from backend.app.engines.base import EngineOutput
    from datetime import datetime, timezone, timedelta

    memory=FakeMemory()
    class Seed:
        async def evaluate(self, outputs, ctx):
            return EngineOutput('interaction_seed','UP',0.6,'seed:test',{'seed_probability_up':0.6,'seed_probability_down':0.4,'fired_rules':[]})
    class Sway:
        async def evaluate(self, outputs, ctx):
            return EngineOutput('interaction_sway','NO_SIGNAL',None,'sway:none',{'p_up':None,'p_down':None,'matched_interactions':[]})
    class Ensemble:
        async def evaluate(self, outputs, seed, sway, ctx):
            return EngineOutput('research_ensemble','UP',0.6,'research:up:seedN',{'probability_up':0.6,'probability_down':0.4,'adaptive_interactions':[]})

    # Import an isolated prediction module under the same stubs used above.
    # The existing module is already loaded by the preceding test; reuse it.
    import backend.app.services.prediction as pred
    monkeypatch.setattr(pred,'_research_memory',memory)
    monkeypatch.setattr(pred,'_research_seed',Seed())
    monkeypatch.setattr(pred,'_research_sway',Sway())
    monkeypatch.setattr(pred,'_research_ensemble',Ensemble())
    pool=FailingPool()
    async def get_pool(): return pool
    monkeypatch.setattr(pred,'get_pool',get_pool)

    async def run():
        now=datetime.now(timezone.utc)
        try:
            await pred.create_prediction_for_round(
                session_id='s',round_id='2',round_start=now,
                trade_cutoff=now-timedelta(seconds=1),
                price_start=now,price_end=now,current_price=100.0,recent_ticks=[]
            )
        except RuntimeError:
            pass
        await asyncio.sleep(0.01)

    asyncio.run(run())
    pred_events=[e for e in memory.events if isinstance(e,tuple) and e[0]=='pred']
    assert {e[1]['engine'] for e in pred_events}=={'interaction_seed','interaction_sway','research_ensemble'}
