"""Tests for portfolio management."""

from hft_system.config import OrderSide
from hft_system.models import Fill
from hft_system.portfolio import PortfolioManager


def test_initial_state():
    pm = PortfolioManager(100000)
    assert pm.total_equity == 100000
    assert pm.cash == 100000
    assert pm.total_pnl == 0


def test_buy_fill():
    pm = PortfolioManager(100000)
    fill = Fill(
        order_id="test",
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=100,
        price=175.0,
        commission=0.50,
    )
    pm.on_fill(fill)

    pos = pm.get_position("AAPL")
    assert pos is not None
    assert pos.quantity == 100
    assert pos.avg_entry_price == 175.0
    # Cash decreases by cost + commission
    assert pm.cash == 100000 - 175.0 * 100 - 0.50


def test_round_trip_trade():
    pm = PortfolioManager(100000)

    # Buy
    pm.on_fill(Fill(
        order_id="buy1",
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=100,
        price=175.0,
        commission=0.50,
    ))

    # Sell at higher price
    pm.on_fill(Fill(
        order_id="sell1",
        symbol="AAPL",
        side=OrderSide.SELL,
        quantity=100,
        price=180.0,
        commission=0.50,
    ))

    pos = pm.get_position("AAPL")
    assert pos.quantity == 0
    # Profit = (180 - 175) * 100 - sell_commission = 500 - 0.50 = 499.50
    assert abs(pos.realized_pnl - 499.5) < 0.01


def test_performance_metrics():
    pm = PortfolioManager(100000)

    # Do a profitable trade
    pm.on_fill(Fill("b1", "AAPL", OrderSide.BUY, 100, 175.0, 0.50))
    pm.on_fill(Fill("s1", "AAPL", OrderSide.SELL, 100, 180.0, 0.50))

    # Do a losing trade
    pm.on_fill(Fill("b2", "GOOGL", OrderSide.BUY, 50, 140.0, 0.25))
    pm.on_fill(Fill("s2", "GOOGL", OrderSide.SELL, 50, 138.0, 0.25))

    metrics = pm.get_performance_metrics()
    assert metrics.total_trades == 2
    assert metrics.winning_trades == 1
    assert metrics.losing_trades == 1
    assert metrics.win_rate == 0.5


def test_portfolio_summary():
    pm = PortfolioManager(100000)
    summary = pm.summary()
    assert "PORTFOLIO SUMMARY" in summary
    assert "100,000.00" in summary


def test_active_positions():
    pm = PortfolioManager(100000)
    pm.on_fill(Fill("b1", "AAPL", OrderSide.BUY, 100, 175.0, 0.50))
    pm.on_fill(Fill("b2", "GOOGL", OrderSide.BUY, 50, 140.0, 0.25))

    active = pm.get_active_positions()
    assert len(active) == 2
    assert "AAPL" in active
    assert "GOOGL" in active
