"""Tests for risk management system."""

from hft_system.config import OrderSide, OrderType, RiskConfig
from hft_system.models import Fill, Order, Signal
from hft_system.risk_manager import RiskManager


def _make_risk_manager():
    config = RiskConfig(
        max_position_size=10000,
        max_order_size=500,
        max_daily_loss=2000,
        max_drawdown_pct=0.10,
        position_limit_per_symbol=5000,
    )
    return RiskManager(config, initial_capital=100000)


def test_validate_valid_order():
    rm = _make_risk_manager()
    order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=100,
        price=175.0,
    )
    valid, reason = rm.validate_order(order)
    assert valid
    assert reason == ""


def test_reject_oversized_order():
    rm = _make_risk_manager()
    order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=1000,  # Exceeds max_order_size of 500
        price=175.0,
    )
    valid, reason = rm.validate_order(order)
    assert not valid
    assert "exceeds max" in reason


def test_reject_zero_quantity():
    rm = _make_risk_manager()
    order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0,
    )
    valid, reason = rm.validate_order(order)
    assert not valid


def test_on_fill_updates_position():
    rm = _make_risk_manager()
    fill = Fill(
        order_id="test",
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=100,
        price=175.0,
        commission=0.50,
    )
    rm.on_fill(fill)

    pos = rm.get_position("AAPL")
    assert pos is not None
    assert pos.quantity == 100
    assert pos.avg_entry_price == 175.0


def test_daily_loss_limit():
    config = RiskConfig(
        max_position_size=100000,
        max_order_size=10000,
        max_daily_loss=2000,
        max_drawdown_pct=0.10,
        position_limit_per_symbol=100000,
    )
    rm = RiskManager(config, initial_capital=100000)

    # Simulate a large loss by buying and seeing price drop
    buy_fill = Fill(
        order_id="buy1",
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=100,
        price=175.0,
        commission=0.50,
    )
    rm.on_fill(buy_fill)

    # Update market price to simulate loss > $2000
    rm.update_market_prices("AAPL", 150.0)

    # Now daily P&L should be very negative
    order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=10,
        price=150.0,
    )
    valid, reason = rm.validate_order(order)
    assert not valid
    assert "Daily loss" in reason or "drawdown" in reason.lower()


def test_risk_summary():
    rm = _make_risk_manager()
    summary = rm.get_risk_summary()
    assert summary["capital"] == 100000
    assert summary["open_positions"] == 0


def test_validate_signal():
    rm = _make_risk_manager()
    signal = Signal(
        symbol="AAPL",
        side=OrderSide.BUY,
        strength=0.5,
        strategy_id="test",
    )
    valid, reason = rm.validate_signal(signal)
    assert valid
