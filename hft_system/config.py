"""
Configuration for the HFT trading system.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class StrategyType(Enum):
    MARKET_MAKING = "market_making"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL_FILL = "partial_fill"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class TimeInForce(Enum):
    GTC = "good_till_cancel"
    IOC = "immediate_or_cancel"
    FOK = "fill_or_kill"
    DAY = "day"


@dataclass
class RiskConfig:
    max_position_size: float = 10000.0
    max_order_size: float = 1000.0
    max_daily_loss: float = 5000.0
    max_drawdown_pct: float = 0.05
    max_concentration_pct: float = 0.25
    max_open_orders: int = 50
    position_limit_per_symbol: float = 5000.0
    var_confidence: float = 0.99
    var_lookback_days: int = 252


@dataclass
class StrategyConfig:
    strategy_type: StrategyType = StrategyType.MARKET_MAKING
    symbols: list[str] = field(default_factory=lambda: ["AAPL", "GOOGL", "MSFT"])
    lookback_period: int = 100
    signal_threshold: float = 0.02
    # Market making
    spread_bps: float = 10.0
    inventory_limit: float = 1000.0
    quote_size: float = 100.0
    # Momentum
    fast_window: int = 10
    slow_window: int = 50
    momentum_threshold: float = 0.01
    # Mean reversion
    mean_window: int = 20
    entry_z_score: float = 2.0
    exit_z_score: float = 0.5
    # Statistical arbitrage
    cointegration_window: int = 60
    hedge_ratio_window: int = 30


@dataclass
class ExecutionConfig:
    slippage_bps: float = 1.0
    commission_per_share: float = 0.005
    min_order_size: float = 1.0
    max_retry_attempts: int = 3
    order_timeout_seconds: float = 5.0
    use_smart_routing: bool = True


@dataclass
class SystemConfig:
    risk: RiskConfig = field(default_factory=RiskConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    tick_interval_ms: int = 100
    data_feed_buffer_size: int = 10000
    log_level: str = "INFO"
    paper_trading: bool = True
    initial_capital: float = 100000.0
    heartbeat_interval_seconds: float = 1.0
