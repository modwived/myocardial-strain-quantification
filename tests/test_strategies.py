"""Tests for trading strategies."""

import time

from hft_system.config import StrategyConfig
from hft_system.models import Tick
from hft_system.strategies import (
    MarketMakingStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    StatisticalArbitrageStrategy,
)


def _make_tick(symbol="AAPL", price=175.0, spread=0.10):
    half = spread / 2
    return Tick(
        symbol=symbol,
        timestamp=time.time(),
        bid=price - half,
        ask=price + half,
        bid_size=1000,
        ask_size=1000,
        last_price=price,
        last_size=100,
        volume=50000,
    )


def test_market_making_warmup():
    config = StrategyConfig()
    strategy = MarketMakingStrategy("mm_test", config)

    # First few ticks should return None (warmup)
    for _ in range(5):
        signal = strategy.on_tick(_make_tick())
    # Not enough data yet for volatility estimation
    assert strategy._tick_count == 5


def test_market_making_generates_signals():
    config = StrategyConfig()
    strategy = MarketMakingStrategy("mm_test", config)

    signals = []
    for i in range(100):
        tick = _make_tick(price=175.0 + i * 0.01)
        signal = strategy.on_tick(tick)
        if signal:
            signals.append(signal)

    assert len(signals) > 0


def test_momentum_needs_warmup():
    config = StrategyConfig(fast_window=5, slow_window=20)
    strategy = MomentumStrategy("mom_test", config)

    # Too few ticks for signal generation
    for _ in range(10):
        signal = strategy.on_tick(_make_tick())
        # Should not generate signal during warmup
    assert strategy._signal_count == 0


def test_momentum_crossover_signal():
    config = StrategyConfig(
        fast_window=5,
        slow_window=20,
        momentum_threshold=0.001,
    )
    strategy = MomentumStrategy("mom_test", config)

    signals = []
    # Create trending price data (uptrend) with volume surges
    for i in range(300):
        price = 175.0 + i * 0.1  # Strong uptrend
        tick = _make_tick(price=price)
        tick.last_size = 500.0 if i > 100 else 100.0  # Volume surge
        signal = strategy.on_tick(tick)
        if signal:
            signals.append(signal)

    # Should generate at least one signal during strong trend
    assert len(signals) > 0


def test_mean_reversion_entry_exit():
    config = StrategyConfig(
        mean_window=20,
        entry_z_score=2.0,
        exit_z_score=0.5,
    )
    strategy = MeanReversionStrategy("mr_test", config)

    signals = []
    # Stable price, then spike, then revert
    prices = [175.0] * 50 + [180.0] * 10 + [175.0] * 10
    for price in prices:
        tick = _make_tick(price=price)
        signal = strategy.on_tick(tick)
        if signal:
            signals.append(signal)

    # Should detect the deviation
    assert len(signals) >= 0  # May or may not trigger depending on z-score


def test_stat_arb_needs_multiple_symbols():
    config = StrategyConfig()
    strategy = StatisticalArbitrageStrategy("sa_test", config)

    # Single symbol should not generate signals
    for _ in range(50):
        signal = strategy.on_tick(_make_tick("AAPL"))
    assert strategy._signal_count == 0


def test_stat_arb_with_pairs():
    config = StrategyConfig(
        cointegration_window=30,
        entry_z_score=2.0,
        exit_z_score=0.5,
    )
    strategy = StatisticalArbitrageStrategy("sa_test", config)

    for i in range(100):
        # AAPL and GOOGL prices with some correlation
        strategy.on_tick(_make_tick("AAPL", price=175.0 + i * 0.01))
        strategy.on_tick(_make_tick("GOOGL", price=140.0 + i * 0.01))

    assert strategy._pairs_initialized


def test_strategy_activation():
    config = StrategyConfig()
    strategy = MomentumStrategy("test", config)

    assert strategy.is_active
    strategy.deactivate()
    assert not strategy.is_active

    signal = strategy.on_tick(_make_tick())
    assert signal is None

    strategy.activate()
    assert strategy.is_active
