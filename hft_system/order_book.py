"""
Order book implementation with price-time priority matching.
"""

import heapq
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from .config import OrderSide, OrderStatus, OrderType
from .models import Fill, Order, Tick

logger = logging.getLogger(__name__)


@dataclass(order=True)
class OrderBookEntry:
    """Entry in the order book, ordered by price-time priority."""

    priority: float
    timestamp: float
    order: Order = field(compare=False)


class OrderBook:
    """
    Simulated order book with price-time priority matching engine.

    Maintains bid and ask sides, processes incoming orders, and
    generates fills when orders cross the spread.
    """

    def __init__(self, symbol: str, commission_per_share: float = 0.005):
        self.symbol = symbol
        self.commission_per_share = commission_per_share
        self._bids: list[OrderBookEntry] = []  # max-heap (negated prices)
        self._asks: list[OrderBookEntry] = []  # min-heap
        self._orders: dict[str, Order] = {}
        self._fills: list[Fill] = []
        self._last_tick: Optional[Tick] = None

    def update_market_data(self, tick: Tick) -> None:
        self._last_tick = tick

    @property
    def best_bid(self) -> Optional[float]:
        if self._last_tick:
            return self._last_tick.bid
        while self._bids:
            entry = self._bids[0]
            if entry.order.is_active:
                return -entry.priority
            heapq.heappop(self._bids)
        return None

    @property
    def best_ask(self) -> Optional[float]:
        if self._last_tick:
            return self._last_tick.ask
        while self._asks:
            entry = self._asks[0]
            if entry.order.is_active:
                return entry.priority
            heapq.heappop(self._asks)
        return None

    @property
    def spread(self) -> Optional[float]:
        bid = self.best_bid
        ask = self.best_ask
        if bid is not None and ask is not None:
            return ask - bid
        return None

    @property
    def mid_price(self) -> Optional[float]:
        bid = self.best_bid
        ask = self.best_ask
        if bid is not None and ask is not None:
            return (bid + ask) / 2.0
        return None

    def submit_order(self, order: Order) -> list[Fill]:
        """Submit an order and attempt to match it."""
        order.status = OrderStatus.SUBMITTED
        order.updated_at = time.time()
        self._orders[order.order_id] = order

        fills = self._match_order(order)

        if order.remaining_quantity > 0 and order.order_type == OrderType.LIMIT:
            self._add_to_book(order)
        elif order.remaining_quantity > 0 and order.order_type == OrderType.MARKET:
            # Market orders that can't be fully filled get cancelled
            if order.filled_quantity > 0:
                order.status = OrderStatus.PARTIAL_FILL
            else:
                order.status = OrderStatus.CANCELLED
            order.updated_at = time.time()

        return fills

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an active order."""
        order = self._orders.get(order_id)
        if order and order.is_active:
            order.status = OrderStatus.CANCELLED
            order.updated_at = time.time()
            logger.debug("Cancelled order %s", order_id[:8])
            return True
        return False

    def get_order(self, order_id: str) -> Optional[Order]:
        return self._orders.get(order_id)

    def get_active_orders(self) -> list[Order]:
        return [o for o in self._orders.values() if o.is_active]

    def _match_order(self, order: Order) -> list[Fill]:
        """Attempt to match an order against the book or market data."""
        fills = []

        if self._last_tick is None:
            return fills

        tick = self._last_tick

        if order.order_type == OrderType.MARKET:
            fill_price = tick.ask if order.side == OrderSide.BUY else tick.bid
            fill = self._create_fill(order, order.quantity, fill_price)
            fills.append(fill)

        elif order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY and order.price >= tick.ask:
                fill_price = tick.ask
                fill = self._create_fill(order, order.quantity, fill_price)
                fills.append(fill)
            elif order.side == OrderSide.SELL and order.price <= tick.bid:
                fill_price = tick.bid
                fill = self._create_fill(order, order.quantity, fill_price)
                fills.append(fill)

        elif order.order_type == OrderType.STOP:
            if order.side == OrderSide.BUY and tick.last_price >= order.stop_price:
                fill = self._create_fill(order, order.quantity, tick.ask)
                fills.append(fill)
            elif order.side == OrderSide.SELL and tick.last_price <= order.stop_price:
                fill = self._create_fill(order, order.quantity, tick.bid)
                fills.append(fill)

        return fills

    def _create_fill(self, order: Order, quantity: float, price: float) -> Fill:
        """Create a fill for an order."""
        commission = quantity * self.commission_per_share

        fill = Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=quantity,
            price=price,
            commission=commission,
        )

        # Update order state
        total_cost = order.avg_fill_price * order.filled_quantity + price * quantity
        order.filled_quantity += quantity
        order.avg_fill_price = total_cost / order.filled_quantity
        order.updated_at = time.time()

        if order.remaining_quantity <= 0:
            order.status = OrderStatus.FILLED
        else:
            order.status = OrderStatus.PARTIAL_FILL

        self._fills.append(fill)
        logger.debug(
            "Fill: %s %s %.2f @ %.4f (commission: %.4f)",
            order.side.value,
            order.symbol,
            quantity,
            price,
            commission,
        )
        return fill

    def _add_to_book(self, order: Order) -> None:
        """Add a resting order to the book."""
        if order.side == OrderSide.BUY:
            entry = OrderBookEntry(
                priority=-order.price,
                timestamp=order.created_at,
                order=order,
            )
            heapq.heappush(self._bids, entry)
        else:
            entry = OrderBookEntry(
                priority=order.price,
                timestamp=order.created_at,
                order=order,
            )
            heapq.heappush(self._asks, entry)
