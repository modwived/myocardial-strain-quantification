"""Tests for execution engine."""

import time

from hft_system.config import (
    ExecutionConfig,
    OrderSide,
    OrderType,
    RiskConfig,
    StrategyConfig,
)
from hft_system.execution import ExecutionEngine
from hft_system.market_data import MarketDataFeed
from hft_system.models import Order, Signal
from hft_system.portfolio import PortfolioManager
from hft_system.risk_manager import RiskManager


def _setup_engine():
    exec_config = ExecutionConfig()
    risk_config = RiskConfig()
    risk_manager = RiskManager(risk_config, 100000)
    portfolio = PortfolioManager(100000)
    market_data = MarketDataFeed()
    market_data.initialize_symbols(["AAPL"])

    # Generate a tick so we have market data
    market_data.generate_tick("AAPL")

    engine = ExecutionEngine(exec_config, risk_manager, portfolio, market_data)
    return engine, market_data


def test_submit_market_order():
    engine, market_data = _setup_engine()

    tick = market_data.get_last_tick("AAPL")
    order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=100,
    )
    fills = engine.submit_order(order)

    assert len(fills) == 1
    assert fills[0].quantity == 100


def test_process_signal():
    engine, market_data = _setup_engine()

    signal = Signal(
        symbol="AAPL",
        side=OrderSide.BUY,
        strength=0.5,
        strategy_id="test",
        metadata={"type": "test"},
    )
    fills = engine.process_signal(signal)

    # Should generate and fill an order
    assert isinstance(fills, list)


def test_cancel_all_orders():
    engine, market_data = _setup_engine()
    cancelled = engine.cancel_all_orders()
    assert cancelled == 0  # No orders to cancel


def test_execution_stats():
    engine, market_data = _setup_engine()
    stats = engine.get_stats()

    assert stats["total_orders"] == 0
    assert stats["total_fills"] == 0
    assert stats["rejected_orders"] == 0


def test_risk_rejection():
    exec_config = ExecutionConfig()
    risk_config = RiskConfig(max_order_size=10)  # Very small limit
    risk_manager = RiskManager(risk_config, 100000)
    portfolio = PortfolioManager(100000)
    market_data = MarketDataFeed()
    market_data.initialize_symbols(["AAPL"])
    market_data.generate_tick("AAPL")

    engine = ExecutionEngine(exec_config, risk_manager, portfolio, market_data)

    order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=100,  # Exceeds limit of 10
    )
    fills = engine.submit_order(order)
    assert len(fills) == 0
    assert engine.get_stats()["rejected_orders"] == 1
