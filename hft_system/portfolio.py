"""
Portfolio and position management for the HFT trading system.

Tracks all positions, calculates P&L, and provides portfolio analytics.
"""

import logging
import time
from collections import defaultdict
from typing import Optional

import numpy as np

from .config import OrderSide
from .models import Fill, PerformanceMetrics, Position

logger = logging.getLogger(__name__)


class PortfolioManager:
    """
    Manages portfolio state, tracks positions, and calculates
    performance metrics across all strategies.
    """

    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self._positions: dict[str, Position] = {}
        self._fills: list[Fill] = []
        self._trade_pnls: list[float] = []
        self._equity_curve: list[tuple[float, float]] = [
            (time.time(), initial_capital)
        ]
        self._peak_equity = initial_capital
        self._max_drawdown = 0.0

    def on_fill(self, fill: Fill) -> None:
        """Process a fill and update portfolio state."""
        self._fills.append(fill)

        position = self._positions.get(fill.symbol)
        if not position:
            position = Position(symbol=fill.symbol)
            self._positions[fill.symbol] = position

        # Track trade P&L for closing trades
        trade_pnl = 0.0

        if fill.side == OrderSide.BUY:
            if position.quantity < 0:
                # Closing short position
                close_qty = min(fill.quantity, abs(position.quantity))
                trade_pnl = (position.avg_entry_price - fill.price) * close_qty - fill.commission
                position.realized_pnl += trade_pnl
                self._trade_pnls.append(trade_pnl)

            # Update position
            if position.quantity >= 0:
                total = position.avg_entry_price * position.quantity + fill.price * fill.quantity
                position.quantity += fill.quantity
                position.avg_entry_price = total / position.quantity if position.quantity > 0 else 0
            else:
                position.quantity += fill.quantity
                if position.quantity > 0:
                    position.avg_entry_price = fill.price

            self.cash -= fill.price * fill.quantity + fill.commission

        else:  # SELL
            if position.quantity > 0:
                # Closing long position
                close_qty = min(fill.quantity, position.quantity)
                trade_pnl = (fill.price - position.avg_entry_price) * close_qty - fill.commission
                position.realized_pnl += trade_pnl
                self._trade_pnls.append(trade_pnl)

            # Update position
            if position.quantity <= 0:
                total = abs(position.avg_entry_price * position.quantity) + fill.price * fill.quantity
                position.quantity -= fill.quantity
                position.avg_entry_price = total / abs(position.quantity) if position.quantity != 0 else 0
            else:
                position.quantity -= fill.quantity
                if position.quantity < 0:
                    position.avg_entry_price = fill.price

            self.cash += fill.price * fill.quantity - fill.commission

        # Update equity curve
        equity = self.total_equity
        self._equity_curve.append((time.time(), equity))
        if equity > self._peak_equity:
            self._peak_equity = equity
        drawdown = (self._peak_equity - equity) / self._peak_equity if self._peak_equity > 0 else 0
        if drawdown > self._max_drawdown:
            self._max_drawdown = drawdown

    def update_market_price(self, symbol: str, price: float) -> None:
        """Update mark-to-market price for a position."""
        position = self._positions.get(symbol)
        if position:
            position.update_market_price(price)

    @property
    def total_equity(self) -> float:
        """Total portfolio value including cash and unrealized P&L."""
        unrealized = sum(p.unrealized_pnl for p in self._positions.values())
        return self.cash + sum(
            abs(p.quantity) * p.avg_entry_price for p in self._positions.values()
        ) + unrealized

    @property
    def total_realized_pnl(self) -> float:
        return sum(p.realized_pnl for p in self._positions.values())

    @property
    def total_unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self._positions.values())

    @property
    def total_pnl(self) -> float:
        return self.total_realized_pnl + self.total_unrealized_pnl

    def get_position(self, symbol: str) -> Optional[Position]:
        return self._positions.get(symbol)

    def get_active_positions(self) -> dict[str, Position]:
        return {s: p for s, p in self._positions.items() if p.quantity != 0}

    def get_performance_metrics(self) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        winning = [p for p in self._trade_pnls if p > 0]
        losing = [p for p in self._trade_pnls if p <= 0]

        total_trades = len(self._trade_pnls)
        win_rate = len(winning) / total_trades if total_trades > 0 else 0
        avg_trade = np.mean(self._trade_pnls) if self._trade_pnls else 0

        # Calculate Sharpe ratio from equity curve
        sharpe = self._calculate_sharpe()

        return PerformanceMetrics(
            total_pnl=self.total_pnl,
            realized_pnl=self.total_realized_pnl,
            unrealized_pnl=self.total_unrealized_pnl,
            total_trades=total_trades,
            winning_trades=len(winning),
            losing_trades=len(losing),
            max_drawdown=self._max_drawdown,
            sharpe_ratio=sharpe,
            win_rate=win_rate,
            avg_trade_pnl=float(avg_trade),
            peak_equity=self._peak_equity,
        )

    def get_equity_curve(self) -> list[tuple[float, float]]:
        return list(self._equity_curve)

    def _calculate_sharpe(self, risk_free_rate: float = 0.05) -> float:
        """Calculate annualized Sharpe ratio from equity curve."""
        if len(self._equity_curve) < 10:
            return 0.0

        equities = [e[1] for e in self._equity_curve]
        returns = np.diff(equities) / np.array(equities[:-1])
        returns = returns[np.isfinite(returns)]

        if len(returns) < 2 or np.std(returns) < 1e-10:
            return 0.0

        # Annualize (assuming ~252 trading days, ~6.5 hrs, ticks per second)
        excess_return = np.mean(returns) - risk_free_rate / 252 / 6.5 / 3600
        annualized_factor = np.sqrt(252 * 6.5 * 3600)
        sharpe = (excess_return / np.std(returns)) * annualized_factor
        return float(np.clip(sharpe, -100, 100))

    def summary(self) -> str:
        """Return a text summary of portfolio state."""
        metrics = self.get_performance_metrics()
        active = self.get_active_positions()

        lines = [
            "=" * 60,
            "PORTFOLIO SUMMARY",
            "=" * 60,
            f"  Initial Capital:  ${self.initial_capital:>12,.2f}",
            f"  Current Equity:   ${self.total_equity:>12,.2f}",
            f"  Cash:             ${self.cash:>12,.2f}",
            f"  Total P&L:        ${metrics.total_pnl:>12,.2f}",
            f"  Realized P&L:     ${metrics.realized_pnl:>12,.2f}",
            f"  Unrealized P&L:   ${metrics.unrealized_pnl:>12,.2f}",
            f"  Max Drawdown:     {metrics.max_drawdown:>12.2%}",
            f"  Sharpe Ratio:     {metrics.sharpe_ratio:>12.2f}",
            f"  Total Trades:     {metrics.total_trades:>12d}",
            f"  Win Rate:         {metrics.win_rate:>12.2%}",
            f"  Avg Trade P&L:    ${metrics.avg_trade_pnl:>12,.2f}",
            "-" * 60,
            "ACTIVE POSITIONS:",
        ]

        if active:
            for sym, pos in active.items():
                lines.append(
                    f"  {sym:>6s}: {pos.quantity:>8.1f} @ ${pos.avg_entry_price:.2f}"
                    f"  P&L: ${pos.total_pnl:>10,.2f}"
                )
        else:
            lines.append("  (no open positions)")
        lines.append("=" * 60)

        return "\n".join(lines)
