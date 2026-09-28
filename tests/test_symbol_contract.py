from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def read_repo_file(relative_path: str) -> str:
    return (
        ROOT / relative_path
    ).read_text(encoding="utf-8")


def test_bcgame_normalizer_preserves_feed_native_symbols():
    source = read_repo_file("collector/src/normalizer.js")

    assert "const SOURCE = 'BC.GAME';" in source

    # BC.GAME exposes the tick and round streams under different symbols.
    assert "const TICK_SYMBOL = 'BTC-USD';" in source
    assert "const ROUND_SYMBOL = 'BTC/USD';" in source

    assert "cmd !== '/kline/BTC-USD/ticker'" in source
    assert "cmd !== '/contest/BTC/USD/5/ticker'" in source


def test_backend_collector_preserves_feed_native_symbols():
    source = read_repo_file("backend/app/routes/collector.py")

    # Tick ingestion accepts the BC.GAME tick representation.
    assert 't.symbol != "BTC-USD"' in source

    # Round ingestion accepts the BC.GAME contest representation.
    assert 'r.symbol != "BTC/USD"' in source


def test_feed_state_validates_each_stream_against_its_native_symbol():
    source = read_repo_file("backend/app/state.py")

    assert '_state.last_tick_symbol != "BTC-USD"' in source
    assert '_state.last_round_symbol != "BTC/USD"' in source


def test_prediction_snapshot_uses_round_market_symbol():
    source = read_repo_file("backend/app/services/prediction.py")

    # Prediction snapshots correspond to the contest/round representation.
    assert "'BTC/USD'" in source


def test_market_tick_queries_use_tick_market_symbol():
    source = read_repo_file("backend/app/services/evaluator.py")

    # Market tick storage/querying uses the tick-stream representation.
    assert "symbol = 'BTC-USD'" in source


def test_symbol_contract_is_not_silently_collapsed():
    normalizer = read_repo_file("collector/src/normalizer.js")

    assert "const TICK_SYMBOL = 'BTC-USD';" in normalizer
    assert "const ROUND_SYMBOL = 'BTC/USD';" in normalizer

    # These are intentionally distinct feed identifiers.
    assert "const TICK_SYMBOL = ROUND_SYMBOL" not in normalizer
    assert "const ROUND_SYMBOL = TICK_SYMBOL" not in normalizer
