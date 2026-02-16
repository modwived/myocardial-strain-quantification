"""
Risk management system for the HFT trading system.

Enforces position limits, drawdown limits, order size limits,
and portfolio-level risk constraints.
"""

import logging
import math
import time
from collections import defaultdict
from typing import Optional

import numpy as np

from .config import OrderSide, RiskConfig
from .models import Fill, Order, Position, Signal

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Centralized risk management that validates all orders and signals
    against configurable risk limits before execution.
    """

    def __init__(self, config: RiskConfig, initial_capital: float):
        self.config = config
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        self.daily_pnl = 0.0
        self.daily_start_capital = initial_capital
        self._positions: dict[str, Position] = {}
        self._daily_fills: list[Fill] = []
        self._order_count = 0
        self._rejected_count = 0
        self._pnl_history: list[float] = []

    def validate_order(self, order: Order) -> tuple[bool, str]:
        """
        Validate an order against all risk limits.
        Returns (is_valid, rejection_reason).
        """
        # Check order size
        if order.quantity > self.config.max_order_size:
            self._rejected_count += 1
            return False, f"Order size {order.quantity} exceeds max {self.config.max_order_size}"

        if order.quantity <= 0:
            self._rejected_count += 1
            return False, "Order quantity must be positive"

        # Check position limits
        position = self._positions.get(order.symbol)
        if position:
            projected_qty = position.quantity
            if order.side == OrderSide.BUY:
                projected_qty += order.quantity
            else:
                projected_qty -= order.quantity

            price = order.price or position.last_price or 0
            projected_value = abs(projected_qty * price)
            if projected_value > self.config.position_limit_per_symbol:
                self._rejected_count += 1
                return False, (
                    f"Projected position value {projected_value:.2f} exceeds "
                    f"limit {self.config.position_limit_per_symbol:.2f}"
                )

        # Check portfolio concentration
        total_value = self._total_portfolio_value()
        if total_value > 0 and order.price:
            order_value = order.quantity * order.price
            if order_value / total_value > self.config.max_concentration_pct:
                self._rejected_count += 1
                return False, "Order would exceed concentration limit"

        # Check daily loss limit
        if self.daily_pnl < -self.config.max_daily_loss:
            self._rejected_count += 1
            return False, f"Daily loss limit reached: {self.daily_pnl:.2f}"

        # Check max drawdown
        drawdown = self._current_drawdown()
        if drawdown > self.config.max_drawdown_pct:
            self._rejected_count += 1
            return False, f"Max drawdown exceeded: {drawdown:.4f}"

        self._order_count += 1
        return True, ""

    def validate_signal(self, signal: Signal) -> tuple[bool, str]:
        """Validate a trading signal before generating orders."""
        if abs(signal.strength) < 0.01:
            return False, "Signal too weak"

        if self.daily_pnl < -self.config.max_daily_loss:
            return False, "Daily loss limit reached"

        drawdown = self._current_drawdown()
        if drawdown > self.config.max_drawdown_pct:
            return False, "Max drawdown exceeded"

        return True, ""

    def on_fill(self, fill: Fill) -> None:
        """Process a fill and update risk state."""
        self._daily_fills.append(fill)

        position = self._positions.get(fill.symbol)
        if not position:
            position = Position(symbol=fill.symbol)
            self._positions[fill.symbol] = position

        if fill.side == OrderSide.BUY:
            if position.quantity >= 0:
                # Adding to long
                total_cost = position.avg_entry_price * position.quantity + fill.price * fill.quantity
                position.quantity += fill.quantity
                if position.quantity > 0:
                    position.avg_entry_price = total_cost / position.quantity
            else:
                # Closing short
                pnl = (position.avg_entry_price - fill.price) * fill.quantity
                position.realized_pnl += pnl - fill.commission
                position.quantity += fill.quantity
                if position.quantity > 0:
                    position.avg_entry_price = fill.price
        else:
            if position.quantity <= 0:
                # Adding to short
                total_cost = abs(position.avg_entry_price * position.quantity) + fill.price * fill.quantity
                position.quantity -= fill.quantity
                if position.quantity < 0:
                    position.avg_entry_price = total_cost / abs(position.quantity)
            else:
                # Closing long
                pnl = (fill.price - position.avg_entry_price) * fill.quantity
                position.realized_pnl += pnl - fill.commission
                position.quantity -= fill.quantity
                if position.quantity < 0:
                    position.avg_entry_price = fill.price

        self._update_capital()

    def update_market_prices(self, symbol: str, price: float) -> None:
        """Update mark-to-market prices for risk calculations."""
        position = self._positions.get(symbol)
        if position:
            position.update_market_price(price)
        self._update_capital()

    def reset_daily(self) -> None:
        """Reset daily counters."""
        self.daily_pnl = 0.0
        self.daily_start_capital = self.current_capital
        self._daily_fills.clear()
        logger.info("Daily risk counters reset")

    def get_position(self, symbol: str) -> Optional[Position]:
        return self._positions.get(symbol)

    def get_all_positions(self) -> dict[str, Position]:
        return dict(self._positions)

    def get_portfolio_value(self) -> float:
        return self._total_portfolio_value()

    def get_risk_summary(self) -> dict:
        return {
            "capital": self.current_capital,
            "peak_capital": self.peak_capital,
            "daily_pnl": self.daily_pnl,
            "drawdown": self._current_drawdown(),
            "total_orders": self._order_count,
            "rejected_orders": self._rejected_count,
            "open_positions": sum(
                1 for p in self._positions.values() if p.quantity != 0
            ),
            "total_exposure": sum(
                abs(p.market_value) for p in self._positions.values()
            ),
        }

    def _total_portfolio_value(self) -> float:
        unrealized = sum(p.unrealized_pnl for p in self._positions.values())
        realized = sum(p.realized_pnl for p in self._positions.values())
        return self.initial_capital + realized + unrealized

    def _current_drawdown(self) -> float:
        current = self._total_portfolio_value()
        if self.peak_capital <= 0:
            return 0.0
        return max(0.0, (self.peak_capital - current) / self.peak_capital)

    def _update_capital(self) -> None:
        self.current_capital = self._total_portfolio_value()
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        self.daily_pnl = self.current_capital - self.daily_start_capital
        self._pnl_history.append(self.current_capital)

    def calculate_var(self, confidence: float = 0.99) -> float:
        """Calculate Value at Risk using historical simulation."""
        if len(self._pnl_history) < 10:
            return 0.0
        returns = np.diff(self._pnl_history) / np.array(self._pnl_history[:-1])
        if len(returns) == 0:
            return 0.0
        percentile = (1 - confidence) * 100
        var = np.percentile(returns, percentile) * self.current_capital
        return abs(var)
