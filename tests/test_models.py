"""Tests for data models."""

import time

from hft_system.config import OrderSide, OrderStatus, OrderType
from hft_system.models import Fill, Order, Position, Signal, Tick


def test_tick_properties():
    tick = Tick(
        symbol="AAPL",
        timestamp=time.time(),
        bid=174.50,
        ask=174.60,
        bid_size=100,
        ask_size=200,
        last_price=174.55,
        last_size=50,
        volume=1000,
    )
    assert tick.mid_price == 174.55
    assert abs(tick.spread - 0.10) < 1e-6
    assert tick.spread_bps > 0


def test_order_remaining_quantity():
    order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=100,
        price=175.0,
    )
    assert order.remaining_quantity == 100
    assert order.is_active

    order.filled_quantity = 50
    order.status = OrderStatus.PARTIAL_FILL
    assert order.remaining_quantity == 50
    assert order.is_active

    order.filled_quantity = 100
    order.status = OrderStatus.FILLED
    assert order.remaining_quantity == 0
    assert not order.is_active


def test_position_pnl():
    pos = Position(symbol="AAPL", quantity=100, avg_entry_price=175.0)
    pos.update_market_price(180.0)
    assert pos.unrealized_pnl == 500.0
    assert pos.market_value == 18000.0

    pos.update_market_price(170.0)
    assert pos.unrealized_pnl == -500.0


def test_signal_creation():
    signal = Signal(
        symbol="AAPL",
        side=OrderSide.BUY,
        strength=0.75,
        strategy_id="test_strategy",
    )
    assert signal.strength == 0.75
    assert signal.side == OrderSide.BUY
