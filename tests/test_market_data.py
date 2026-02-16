"""Tests for market data feed."""

from hft_system.market_data import MarketDataFeed


def test_initialize_symbols():
    feed = MarketDataFeed()
    feed.initialize_symbols(["AAPL", "GOOGL"])
    assert len(feed._base_prices) == 2


def test_generate_tick():
    feed = MarketDataFeed()
    feed.initialize_symbols(["AAPL"])
    tick = feed.generate_tick("AAPL")

    assert tick.symbol == "AAPL"
    assert tick.bid > 0
    assert tick.ask > tick.bid
    assert tick.last_price > 0


def test_price_history():
    feed = MarketDataFeed()
    feed.initialize_symbols(["AAPL"])

    for _ in range(50):
        feed.generate_tick("AAPL")

    prices = feed.get_price_history("AAPL", 20)
    assert len(prices) == 20
    assert all(p > 0 for p in prices)


def test_generate_ticks_all_symbols():
    feed = MarketDataFeed()
    feed.initialize_symbols(["AAPL", "GOOGL", "MSFT"])
    ticks = feed.generate_ticks_all_symbols()

    assert len(ticks) == 3
    symbols = {t.symbol for t in ticks}
    assert symbols == {"AAPL", "GOOGL", "MSFT"}


def test_historical_ohlcv():
    feed = MarketDataFeed()
    feed.initialize_symbols(["AAPL"])
    bars = feed.generate_historical_ohlcv("AAPL", periods=100)

    assert len(bars) == 100
    for bar in bars:
        assert bar.high >= bar.low
        assert bar.volume > 0
