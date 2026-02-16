"""Tests for order book and matching engine."""

import time

from hft_system.config import OrderSide, OrderStatus, OrderType
from hft_system.models import Order, Tick
from hft_system.order_book import OrderBook


def _make_tick(symbol="AAPL", bid=174.50, ask=174.60):
    return Tick(
        symbol=symbol,
        timestamp=time.time(),
        bid=bid,
        ask=ask,
        bid_size=1000,
        ask_size=1000,
        last_price=(bid + ask) / 2,
        last_size=100,
        volume=50000,
    )


def test_market_order_buy():
    book = OrderBook("AAPL")
    book.update_market_data(_make_tick())

    order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=100,
    )
    fills = book.submit_order(order)

    assert len(fills) == 1
    assert fills[0].price == 174.60  # Filled at ask
    assert fills[0].quantity == 100
    assert order.status == OrderStatus.FILLED


def test_market_order_sell():
    book = OrderBook("AAPL")
    book.update_market_data(_make_tick())

    order = Order(
        symbol="AAPL",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=50,
    )
    fills = book.submit_order(order)

    assert len(fills) == 1
    assert fills[0].price == 174.50  # Filled at bid
    assert order.status == OrderStatus.FILLED


def test_limit_order_crosses_spread():
    book = OrderBook("AAPL")
    book.update_market_data(_make_tick())

    # Buy limit above ask -> immediate fill
    order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=100,
        price=175.00,
    )
    fills = book.submit_order(order)
    assert len(fills) == 1
    assert order.status == OrderStatus.FILLED


def test_limit_order_rests_in_book():
    book = OrderBook("AAPL")
    book.update_market_data(_make_tick())

    # Buy limit below bid -> rests in book
    order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=100,
        price=174.00,
    )
    fills = book.submit_order(order)
    assert len(fills) == 0
    assert order.status == OrderStatus.SUBMITTED

    active = book.get_active_orders()
    assert len(active) == 1


def test_cancel_order():
    book = OrderBook("AAPL")
    book.update_market_data(_make_tick())

    order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=100,
        price=174.00,
    )
    book.submit_order(order)

    result = book.cancel_order(order.order_id)
    assert result is True
    assert order.status == OrderStatus.CANCELLED


def test_order_book_properties():
    book = OrderBook("AAPL")
    tick = _make_tick()
    book.update_market_data(tick)

    assert book.best_bid == 174.50
    assert book.best_ask == 174.60
    assert abs(book.spread - 0.10) < 1e-6
    assert abs(book.mid_price - 174.55) < 1e-6
