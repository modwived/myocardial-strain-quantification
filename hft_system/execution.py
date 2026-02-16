"""
Execution engine with smart order routing for the HFT system.

Converts trading signals into orders, applies execution logic,
and manages order lifecycle.
"""

import logging
import time
from typing import Optional

from .config import ExecutionConfig, OrderSide, OrderType, TimeInForce
from .market_data import MarketDataFeed
from .models import Fill, Order, Signal, Tick
from .order_book import OrderBook
from .portfolio import PortfolioManager
from .risk_manager import RiskManager

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """
    Execution engine that processes signals, creates orders,
    routes them through the order book, and manages fills.

    Features:
    - Signal-to-order conversion with size calculation
    - Smart order routing with slippage modeling
    - Order lifecycle management
    - Fill processing and portfolio updates
    """

    def __init__(
        self,
        config: ExecutionConfig,
        risk_manager: RiskManager,
        portfolio: PortfolioManager,
        market_data: MarketDataFeed,
    ):
        self.config = config
        self.risk_manager = risk_manager
        self.portfolio = portfolio
        self.market_data = market_data
        self._order_books: dict[str, OrderBook] = {}
        self._active_orders: dict[str, Order] = {}
        self._total_fills = 0
        self._total_orders = 0
        self._rejected_orders = 0

    def get_or_create_order_book(self, symbol: str) -> OrderBook:
        if symbol not in self._order_books:
            self._order_books[symbol] = OrderBook(
                symbol, self.config.commission_per_share
            )
        return self._order_books[symbol]

    def process_signal(self, signal: Signal) -> list[Fill]:
        """
        Convert a signal to an order, validate it, and execute.
        Returns list of fills generated.
        """
        # Validate signal through risk manager
        valid, reason = self.risk_manager.validate_signal(signal)
        if not valid:
            logger.debug("Signal rejected: %s", reason)
            return []

        # Get current market data
        tick = self.market_data.get_last_tick(signal.symbol)
        if not tick:
            logger.debug("No market data for %s", signal.symbol)
            return []

        # Calculate order size based on signal strength
        order_size = self._calculate_order_size(signal, tick)
        if order_size < self.config.min_order_size:
            return []

        # Determine order type and price
        order_type, price = self._determine_order_params(signal, tick)

        # Create order
        order = Order(
            symbol=signal.symbol,
            side=signal.side,
            order_type=order_type,
            quantity=order_size,
            price=price,
            time_in_force=TimeInForce.IOC,
            strategy_id=signal.strategy_id,
            tag=signal.metadata.get("type", ""),
        )

        return self.submit_order(order)

    def submit_order(self, order: Order) -> list[Fill]:
        """Submit an order for execution."""
        self._total_orders += 1

        # Risk validation
        valid, reason = self.risk_manager.validate_order(order)
        if not valid:
            logger.debug("Order rejected: %s", reason)
            self._rejected_orders += 1
            return []

        # Apply slippage model
        if self.config.slippage_bps > 0 and order.price:
            slippage = order.price * self.config.slippage_bps / 10000
            if order.side == OrderSide.BUY:
                order.price += slippage
            else:
                order.price -= slippage

        # Route to order book
        book = self.get_or_create_order_book(order.symbol)

        # Update order book with latest market data
        tick = self.market_data.get_last_tick(order.symbol)
        if tick:
            book.update_market_data(tick)

        fills = book.submit_order(order)

        # Process fills
        for fill in fills:
            self._process_fill(fill)

        if order.is_active:
            self._active_orders[order.order_id] = order

        return fills

    def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """Cancel all active orders, optionally filtered by symbol."""
        cancelled = 0
        for order_id, order in list(self._active_orders.items()):
            if symbol and order.symbol != symbol:
                continue
            book = self._order_books.get(order.symbol)
            if book and book.cancel_order(order_id):
                cancelled += 1
                del self._active_orders[order_id]
        return cancelled

    def _calculate_order_size(self, signal: Signal, tick: Tick) -> float:
        """Calculate order size based on signal strength and risk limits."""
        base_size = self.config.min_order_size
        max_size = self.risk_manager.config.max_order_size

        # Scale by signal strength
        size = base_size + (max_size - base_size) * abs(signal.strength) * 0.3

        # Check position limits
        position = self.risk_manager.get_position(signal.symbol)
        if position:
            pos_limit = self.risk_manager.config.position_limit_per_symbol
            current_value = abs(position.quantity * tick.last_price)
            remaining = max(0, pos_limit - current_value)
            max_qty = remaining / tick.last_price if tick.last_price > 0 else 0

            # If reducing position, allow full size
            if (signal.side == OrderSide.SELL and position.quantity > 0) or \
               (signal.side == OrderSide.BUY and position.quantity < 0):
                size = min(size, abs(position.quantity), max_size)
            else:
                size = min(size, max_qty)

        size = min(size, max_size)
        return round(size, 2)

    def _determine_order_params(
        self, signal: Signal, tick: Tick
    ) -> tuple[OrderType, Optional[float]]:
        """Determine order type and price based on signal and market."""
        if signal.strength > 0.7:
            # Strong signal - use market order for guaranteed fill
            return OrderType.MARKET, None

        # Use aggressive limit orders that cross the spread for fills
        if signal.side == OrderSide.BUY:
            # Place limit at or above the ask to get filled
            price = tick.ask + tick.spread * 0.1
        else:
            # Place limit at or below the bid to get filled
            price = tick.bid - tick.spread * 0.1

        return OrderType.LIMIT, round(price, 4)

    def _process_fill(self, fill: Fill) -> None:
        """Process a fill through risk manager and portfolio."""
        self._total_fills += 1
        self.risk_manager.on_fill(fill)
        self.portfolio.on_fill(fill)

        # Remove from active if fully filled
        if fill.order_id in self._active_orders:
            order = self._active_orders[fill.order_id]
            if not order.is_active:
                del self._active_orders[fill.order_id]

    def get_stats(self) -> dict:
        return {
            "total_orders": self._total_orders,
            "rejected_orders": self._rejected_orders,
            "total_fills": self._total_fills,
            "active_orders": len(self._active_orders),
            "fill_rate": (
                self._total_fills / self._total_orders
                if self._total_orders > 0
                else 0
            ),
        }
