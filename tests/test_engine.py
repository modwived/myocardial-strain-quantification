"""Tests for the main trading engine."""

from hft_system.config import SystemConfig
from hft_system.engine import TradingEngine


def test_engine_initialization():
    config = SystemConfig()
    engine = TradingEngine(config)
    engine.initialize()

    assert len(engine._strategies) >= 3
    assert not engine._running


def test_engine_run_cycles():
    config = SystemConfig(tick_interval_ms=0)  # No delay for testing
    engine = TradingEngine(config)
    engine.initialize()

    engine.run(max_cycles=100)

    assert engine._cycle_count == 100
    assert engine._tick_count > 0


def test_engine_status():
    config = SystemConfig(tick_interval_ms=0)
    engine = TradingEngine(config)
    engine.initialize()
    engine.run(max_cycles=50)

    status = engine.get_status()
    assert "portfolio" in status
    assert "risk" in status
    assert "execution" in status
    assert "strategies" in status
    assert status["cycle_count"] == 50


def test_engine_report():
    config = SystemConfig(tick_interval_ms=0)
    engine = TradingEngine(config)
    engine.initialize()
    engine.run(max_cycles=50)

    report = engine.print_report()
    assert "PORTFOLIO SUMMARY" in report
    assert "EXECUTION STATS" in report
    assert "RISK SUMMARY" in report
    assert "STRATEGIES" in report


def test_engine_custom_symbols():
    config = SystemConfig()
    config.strategy.symbols = ["AAPL", "MSFT"]
    engine = TradingEngine(config)
    engine.initialize(symbols=["AAPL", "MSFT"])

    engine.run(max_cycles=10)
    assert engine._tick_count > 0
