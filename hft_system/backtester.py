"""
Backtesting framework for the HFT trading system.

Replays historical or simulated data through strategies and
measures performance with realistic execution modeling.
"""

import logging
import time
from typing import Optional

import numpy as np

from .config import ExecutionConfig, RiskConfig, StrategyConfig, StrategyType, SystemConfig
from .execution import ExecutionEngine
from .market_data import MarketDataFeed
from .models import PerformanceMetrics, Tick
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


class BacktestResult:
    """Container for backtest results."""

    def __init__(
        self,
        metrics: PerformanceMetrics,
        equity_curve: list[tuple[float, float]],
        portfolio_summary: str,
        execution_stats: dict,
        risk_summary: dict,
        total_ticks: int,
        duration_seconds: float,
    ):
        self.metrics = metrics
        self.equity_curve = equity_curve
        self.portfolio_summary = portfolio_summary
        self.execution_stats = execution_stats
        self.risk_summary = risk_summary
        self.total_ticks = total_ticks
        self.duration_seconds = duration_seconds

    def __str__(self) -> str:
        lines = [
            "",
            "=" * 60,
            "BACKTEST RESULTS",
            "=" * 60,
            f"  Duration:         {self.duration_seconds:.2f}s",
            f"  Total Ticks:      {self.total_ticks:,}",
            f"  Total P&L:        ${self.metrics.total_pnl:>12,.2f}",
            f"  Realized P&L:     ${self.metrics.realized_pnl:>12,.2f}",
            f"  Max Drawdown:     {self.metrics.max_drawdown:>12.2%}",
            f"  Sharpe Ratio:     {self.metrics.sharpe_ratio:>12.2f}",
            f"  Total Trades:     {self.metrics.total_trades:>12,}",
            f"  Win Rate:         {self.metrics.win_rate:>12.2%}",
            f"  Avg Trade P&L:    ${self.metrics.avg_trade_pnl:>12,.2f}",
            "",
            "  Execution Stats:",
            f"    Orders:         {self.execution_stats.get('total_orders', 0):>12,}",
            f"    Fills:          {self.execution_stats.get('total_fills', 0):>12,}",
            f"    Rejected:       {self.execution_stats.get('rejected_orders', 0):>12,}",
            f"    Fill Rate:      {self.execution_stats.get('fill_rate', 0):>12.2%}",
            "",
            "  Risk Summary:",
            f"    Final Capital:  ${self.risk_summary.get('capital', 0):>12,.2f}",
            f"    Peak Capital:   ${self.risk_summary.get('peak_capital', 0):>12,.2f}",
            f"    Drawdown:       {self.risk_summary.get('drawdown', 0):>12.4f}",
            "=" * 60,
        ]
        return "\n".join(lines)


class Backtester:
    """
    Backtesting engine that simulates market conditions and
    runs strategies with realistic execution modeling.
    """

    def __init__(self, config: SystemConfig):
        self.config = config
        self._strategies: list[BaseStrategy] = []

    def add_strategy(self, strategy: BaseStrategy) -> None:
        self._strategies.append(strategy)

    def create_default_strategies(self) -> None:
        """Create strategies based on system config."""
        cfg = self.config.strategy
        strategies = [
            MarketMakingStrategy("mm_1", cfg),
            MomentumStrategy("mom_1", cfg),
            MeanReversionStrategy("mr_1", cfg),
        ]
        if len(cfg.symbols) >= 2:
            strategies.append(StatisticalArbitrageStrategy("sa_1", cfg))
        self._strategies = strategies

    def run(
        self,
        num_ticks: int = 10000,
        symbols: Optional[list[str]] = None,
    ) -> BacktestResult:
        """
        Run a backtest simulation.

        Args:
            num_ticks: Number of ticks to simulate
            symbols: List of symbols to trade (uses config if None)
        """
        start_time = time.time()
        symbols = symbols or self.config.strategy.symbols

        # Initialize components
        market_data = MarketDataFeed(self.config.data_feed_buffer_size)
        market_data.initialize_symbols(symbols)

        risk_manager = RiskManager(self.config.risk, self.config.initial_capital)
        portfolio = PortfolioManager(self.config.initial_capital)
        execution = ExecutionEngine(
            self.config.execution, risk_manager, portfolio, market_data
        )

        if not self._strategies:
            self.create_default_strategies()

        logger.info(
            "Starting backtest: %d ticks, %d symbols, %d strategies",
            num_ticks,
            len(symbols),
            len(self._strategies),
        )

        # Main simulation loop
        tick_count = 0
        for i in range(num_ticks):
            # Generate ticks for all symbols
            ticks = market_data.generate_ticks_all_symbols()

            for tick in ticks:
                tick_count += 1

                # Update risk manager with latest prices
                risk_manager.update_market_prices(tick.symbol, tick.last_price)
                portfolio.update_market_price(tick.symbol, tick.last_price)

                # Feed tick to all strategies
                for strategy in self._strategies:
                    if not strategy.is_active:
                        continue

                    signal = strategy.on_tick(tick)
                    if signal:
                        fills = execution.process_signal(signal)

                        # Update market making inventory
                        if isinstance(strategy, MarketMakingStrategy):
                            for fill in fills:
                                qty = fill.quantity if fill.side == OrderSide.BUY else -fill.quantity
                                strategy.update_inventory(fill.symbol, qty)

            # Progress logging
            if (i + 1) % (num_ticks // 10 or 1) == 0:
                pct = (i + 1) / num_ticks * 100
                logger.info(
                    "Backtest progress: %.0f%% | Equity: $%.2f | Trades: %d",
                    pct,
                    portfolio.total_equity,
                    portfolio.get_performance_metrics().total_trades,
                )

        duration = time.time() - start_time

        # Collect results
        metrics = portfolio.get_performance_metrics()
        result = BacktestResult(
            metrics=metrics,
            equity_curve=portfolio.get_equity_curve(),
            portfolio_summary=portfolio.summary(),
            execution_stats=execution.get_stats(),
            risk_summary=risk_manager.get_risk_summary(),
            total_ticks=tick_count,
            duration_seconds=duration,
        )

        logger.info("Backtest complete: %s", result)
        return result


# Need this import for the isinstance check in run()
from .config import OrderSide  # noqa: E402
