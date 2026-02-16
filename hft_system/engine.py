"""
Main trading engine that orchestrates all components of the HFT system.

This is the central coordinator that connects market data, strategies,
risk management, execution, and portfolio management into a cohesive
self-trading system.
"""

import logging
import signal
import time
from typing import Optional

from .config import OrderSide, SystemConfig
from .execution import ExecutionEngine
from .market_data import MarketDataFeed
from .models import Tick
from .portfolio import PortfolioManager
from .risk_manager import RiskManager
from .strategies import (
    BaseStrategy,
    MarketMakingStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    StatisticalArbitrageStrategy,
)

logger = logging.getLogger(__name__)


class TradingEngine:
    """
    Core trading engine that runs the full autonomous trading loop.

    The engine:
    1. Receives market data ticks (simulated or live)
    2. Feeds data to multiple strategies simultaneously
    3. Validates generated signals through risk management
    4. Executes orders through the execution engine
    5. Updates portfolio and position state
    6. Monitors system health and performance

    The system trades autonomously once started, making all
    buy/sell decisions based on the configured algorithms.
    """

    def __init__(self, config: SystemConfig):
        self.config = config
        self._running = False
        self._tick_count = 0
        self._cycle_count = 0

        # Initialize core components
        self.market_data = MarketDataFeed(config.data_feed_buffer_size)
        self.risk_manager = RiskManager(config.risk, config.initial_capital)
        self.portfolio = PortfolioManager(config.initial_capital)
        self.execution = ExecutionEngine(
            config.execution, self.risk_manager, self.portfolio, self.market_data
        )

        # Strategies
        self._strategies: list[BaseStrategy] = []

        # Performance tracking
        self._last_heartbeat = time.time()
        self._start_time: Optional[float] = None

    def initialize(self, symbols: Optional[list[str]] = None) -> None:
        """Initialize the trading engine with symbols and strategies."""
        symbols = symbols or self.config.strategy.symbols

        # Initialize market data
        self.market_data.initialize_symbols(symbols)

        # Create default strategies
        cfg = self.config.strategy
        self._strategies = [
            MarketMakingStrategy("mm_primary", cfg),
            MomentumStrategy("mom_primary", cfg),
            MeanReversionStrategy("mr_primary", cfg),
        ]
        if len(symbols) >= 2:
            self._strategies.append(
                StatisticalArbitrageStrategy("sa_primary", cfg)
            )

        logger.info(
            "Trading engine initialized: %d symbols, %d strategies",
            len(symbols),
            len(self._strategies),
        )

    def add_strategy(self, strategy: BaseStrategy) -> None:
        """Add a custom strategy to the engine."""
        self._strategies.append(strategy)

    def run(self, max_cycles: Optional[int] = None) -> None:
        """
        Start the autonomous trading loop.

        Args:
            max_cycles: Maximum trading cycles to run (None = unlimited)
        """
        self._running = True
        self._start_time = time.time()
        logger.info("Trading engine started (paper=%s)", self.config.paper_trading)

        try:
            while self._running:
                self._trading_cycle()
                self._cycle_count += 1

                if max_cycles and self._cycle_count >= max_cycles:
                    logger.info("Reached max cycles (%d), stopping", max_cycles)
                    break

                # Heartbeat
                now = time.time()
                if now - self._last_heartbeat >= self.config.heartbeat_interval_seconds:
                    self._heartbeat()
                    self._last_heartbeat = now

                # Tick interval
                time.sleep(self.config.tick_interval_ms / 1000.0)

        except KeyboardInterrupt:
            logger.info("Shutdown signal received")
        finally:
            self.stop()

    def _trading_cycle(self) -> None:
        """Execute one full trading cycle."""
        # 1. Generate market data
        ticks = self.market_data.generate_ticks_all_symbols()

        for tick in ticks:
            self._tick_count += 1

            # 2. Update risk and portfolio with latest prices
            self.risk_manager.update_market_prices(tick.symbol, tick.last_price)
            self.portfolio.update_market_price(tick.symbol, tick.last_price)

            # 3. Feed tick to all active strategies
            for strategy in self._strategies:
                if not strategy.is_active:
                    continue

                signal = strategy.on_tick(tick)
                if signal is None:
                    continue

                # 4. Execute signal through the execution engine
                fills = self.execution.process_signal(signal)

                # 5. Update strategy state (e.g., inventory for market making)
                if isinstance(strategy, MarketMakingStrategy):
                    for fill in fills:
                        qty = (
                            fill.quantity
                            if fill.side == OrderSide.BUY
                            else -fill.quantity
                        )
                        strategy.update_inventory(fill.symbol, qty)

    def _heartbeat(self) -> None:
        """Log periodic system status."""
        metrics = self.portfolio.get_performance_metrics()
        risk = self.risk_manager.get_risk_summary()
        exec_stats = self.execution.get_stats()

        logger.info(
            "HEARTBEAT | Equity: $%.2f | P&L: $%.2f | Trades: %d | "
            "Orders: %d | Fills: %d | Drawdown: %.4f",
            self.portfolio.total_equity,
            metrics.total_pnl,
            metrics.total_trades,
            exec_stats["total_orders"],
            exec_stats["total_fills"],
            risk["drawdown"],
        )

    def stop(self) -> None:
        """Stop the trading engine gracefully."""
        self._running = False

        # Cancel all open orders
        cancelled = self.execution.cancel_all_orders()
        logger.info("Cancelled %d open orders", cancelled)

        # Final report
        elapsed = time.time() - self._start_time if self._start_time else 0
        logger.info(
            "Trading engine stopped. Ran for %.2fs, %d cycles, %d ticks",
            elapsed,
            self._cycle_count,
            self._tick_count,
        )

    def get_status(self) -> dict:
        """Get comprehensive system status."""
        return {
            "running": self._running,
            "tick_count": self._tick_count,
            "cycle_count": self._cycle_count,
            "uptime_seconds": (
                time.time() - self._start_time if self._start_time else 0
            ),
            "portfolio": {
                "equity": self.portfolio.total_equity,
                "cash": self.portfolio.cash,
                "total_pnl": self.portfolio.total_pnl,
                "active_positions": len(self.portfolio.get_active_positions()),
            },
            "risk": self.risk_manager.get_risk_summary(),
            "execution": self.execution.get_stats(),
            "strategies": [
                {
                    "id": s.strategy_id,
                    "name": s.name,
                    "active": s.is_active,
                    "signals": s._signal_count,
                }
                for s in self._strategies
            ],
        }

    def print_report(self) -> str:
        """Generate and return a full trading report."""
        report = self.portfolio.summary()
        exec_stats = self.execution.get_stats()
        risk = self.risk_manager.get_risk_summary()

        report += "\n\nEXECUTION STATS:\n"
        report += f"  Total Orders:   {exec_stats['total_orders']:>10,}\n"
        report += f"  Total Fills:    {exec_stats['total_fills']:>10,}\n"
        report += f"  Rejected:       {exec_stats['rejected_orders']:>10,}\n"
        report += f"  Fill Rate:      {exec_stats['fill_rate']:>10.2%}\n"

        report += "\nRISK SUMMARY:\n"
        report += f"  Drawdown:       {risk['drawdown']:>10.4f}\n"
        report += f"  Open Positions: {risk['open_positions']:>10,}\n"
        report += f"  Total Exposure: ${risk['total_exposure']:>10,.2f}\n"

        report += "\nSTRATEGIES:\n"
        for s in self._strategies:
            report += f"  {s.strategy_id:>15s} ({s.name}): "
            report += f"{'ACTIVE' if s.is_active else 'INACTIVE'} | "
            report += f"Signals: {s._signal_count}\n"

        return report
