"""Tests for the backtesting framework."""

from hft_system.backtester import Backtester, BacktestResult
from hft_system.config import SystemConfig


def test_backtest_basic():
    config = SystemConfig(tick_interval_ms=0)
    backtester = Backtester(config)
    backtester.create_default_strategies()

    result = backtester.run(num_ticks=500)

    assert isinstance(result, BacktestResult)
    assert result.total_ticks > 0
    assert result.duration_seconds >= 0


def test_backtest_metrics():
    config = SystemConfig(tick_interval_ms=0)
    backtester = Backtester(config)
    result = backtester.run(num_ticks=1000)

    assert result.metrics is not None
    assert result.equity_curve is not None
    assert len(result.equity_curve) > 0


def test_backtest_custom_symbols():
    config = SystemConfig(tick_interval_ms=0)
    config.strategy.symbols = ["AAPL", "GOOGL"]
    backtester = Backtester(config)
    result = backtester.run(num_ticks=200, symbols=["AAPL", "GOOGL"])

    assert result.total_ticks > 0


def test_backtest_result_str():
    config = SystemConfig(tick_interval_ms=0)
    backtester = Backtester(config)
    result = backtester.run(num_ticks=200)

    output = str(result)
    assert "BACKTEST RESULTS" in output
    assert "Total P&L" in output
    assert "Sharpe Ratio" in output
