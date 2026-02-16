#!/usr/bin/env python3
"""
High-Frequency Trading System - Main Entry Point

This system autonomously trades using multiple algorithmic strategies
including market making, momentum, mean reversion, and statistical arbitrage.

Usage:
    python main.py                    # Run live trading simulation
    python main.py --backtest         # Run backtest
    python main.py --cycles 5000      # Run for 5000 cycles
    python main.py --symbols AAPL MSFT GOOGL  # Trade specific symbols
"""

import argparse
import logging
import sys

from hft_system.backtester import Backtester
from hft_system.config import (
    ExecutionConfig,
    RiskConfig,
    StrategyConfig,
    StrategyType,
    SystemConfig,
)
from hft_system.engine import TradingEngine


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="High-Frequency Trading System"
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run in backtest mode",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=1000,
        help="Number of trading cycles to run (default: 1000)",
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=10000,
        help="Number of ticks for backtest (default: 10000)",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["AAPL", "GOOGL", "MSFT"],
        help="Symbols to trade",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100000.0,
        help="Initial capital (default: $100,000)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser.parse_args()


def create_config(args: argparse.Namespace) -> SystemConfig:
    """Create system configuration from command-line arguments."""
    return SystemConfig(
        risk=RiskConfig(
            max_position_size=args.capital * 0.1,
            max_order_size=args.capital * 0.01,
            max_daily_loss=args.capital * 0.05,
            max_drawdown_pct=0.10,
            position_limit_per_symbol=args.capital * 0.05,
        ),
        strategy=StrategyConfig(
            symbols=args.symbols,
            lookback_period=100,
            spread_bps=10.0,
            fast_window=10,
            slow_window=50,
            mean_window=20,
            entry_z_score=2.0,
            exit_z_score=0.5,
        ),
        execution=ExecutionConfig(
            slippage_bps=1.0,
            commission_per_share=0.005,
        ),
        initial_capital=args.capital,
        paper_trading=True,
        tick_interval_ms=10,
    )


def run_live(config: SystemConfig, max_cycles: int) -> None:
    """Run the live trading simulation."""
    engine = TradingEngine(config)
    engine.initialize()

    print("\n" + "=" * 60)
    print("  HIGH-FREQUENCY TRADING SYSTEM")
    print("  Mode: Paper Trading (Simulated)")
    print(f"  Symbols: {', '.join(config.strategy.symbols)}")
    print(f"  Capital: ${config.initial_capital:,.2f}")
    print(f"  Strategies: Market Making, Momentum, Mean Reversion, Stat Arb")
    print(f"  Max Cycles: {max_cycles}")
    print("=" * 60 + "\n")

    engine.run(max_cycles=max_cycles)

    report = engine.print_report()
    print(report)


def run_backtest(config: SystemConfig, num_ticks: int) -> None:
    """Run a backtest."""
    backtester = Backtester(config)
    backtester.create_default_strategies()

    print("\n" + "=" * 60)
    print("  BACKTEST MODE")
    print(f"  Symbols: {', '.join(config.strategy.symbols)}")
    print(f"  Capital: ${config.initial_capital:,.2f}")
    print(f"  Ticks: {num_ticks:,}")
    print("=" * 60 + "\n")

    result = backtester.run(num_ticks=num_ticks)

    print(str(result))
    print(result.portfolio_summary)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    config = create_config(args)

    if args.backtest:
        run_backtest(config, args.ticks)
    else:
        run_live(config, args.cycles)


if __name__ == "__main__":
    main()
