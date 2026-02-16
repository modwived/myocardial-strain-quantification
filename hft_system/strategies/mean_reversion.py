"""
Mean Reversion Strategy

Identifies overbought/oversold conditions using z-score analysis
and trades the expected return to the mean.
"""

import logging
from collections import deque
from typing import Optional

import numpy as np

from ..config import OrderSide, StrategyConfig
from ..models import Signal, Tick
from .base import BaseStrategy

logger = logging.getLogger(__name__)


class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion strategy based on Bollinger Band / z-score analysis.

    Features:
    - Z-score based entry and exit signals
    - Adaptive mean and standard deviation calculation
    - Half-life estimation for mean reversion speed
    - Position scaling based on deviation magnitude
    """

    def __init__(self, strategy_id: str, config: StrategyConfig):
        super().__init__(strategy_id, config)
        self._price_history: dict[str, deque] = {}
        self._in_position: dict[str, Optional[OrderSide]] = {}

    def on_tick(self, tick: Tick) -> Optional[Signal]:
        if not self.is_active:
            return None

        self._tick_count += 1
        symbol = tick.symbol

        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=2000)
        self._price_history[symbol].append(tick.last_price)

        min_periods = self.config.mean_window + 20
        if len(self._price_history[symbol]) < min_periods:
            return None

        # Only check every few ticks
        if self._tick_count % 3 != 0:
            return None

        prices = np.array(self._price_history[symbol])

        # Calculate z-score
        window = self.config.mean_window
        rolling_mean = np.mean(prices[-window:])
        rolling_std = np.std(prices[-window:])

        if rolling_std < 1e-8:
            return None

        current_price = prices[-1]
        z_score = (current_price - rolling_mean) / rolling_std

        # Estimate half-life of mean reversion (Ornstein-Uhlenbeck)
        half_life = self._estimate_half_life(prices[-window:])

        current_position = self._in_position.get(symbol)

        signal = None

        if current_position is None:
            # Entry signals
            if z_score < -self.config.entry_z_score:
                # Price significantly below mean - buy
                strength = min(abs(z_score) / (self.config.entry_z_score * 2), 1.0)
                signal = Signal(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    strength=strength,
                    strategy_id=self.strategy_id,
                    metadata={
                        "type": "mean_reversion_entry",
                        "z_score": z_score,
                        "half_life": half_life,
                        "rolling_mean": rolling_mean,
                        "rolling_std": rolling_std,
                    },
                )
                self._in_position[symbol] = OrderSide.BUY
                self._signal_count += 1

            elif z_score > self.config.entry_z_score:
                # Price significantly above mean - sell
                strength = min(abs(z_score) / (self.config.entry_z_score * 2), 1.0)
                signal = Signal(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    strength=strength,
                    strategy_id=self.strategy_id,
                    metadata={
                        "type": "mean_reversion_entry",
                        "z_score": z_score,
                        "half_life": half_life,
                        "rolling_mean": rolling_mean,
                        "rolling_std": rolling_std,
                    },
                )
                self._in_position[symbol] = OrderSide.SELL
                self._signal_count += 1

        else:
            # Exit signals - price reverted to mean
            if current_position == OrderSide.BUY and z_score > -self.config.exit_z_score:
                signal = Signal(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    strength=0.8,
                    strategy_id=self.strategy_id,
                    metadata={
                        "type": "mean_reversion_exit",
                        "z_score": z_score,
                    },
                )
                self._in_position[symbol] = None
                self._signal_count += 1

            elif current_position == OrderSide.SELL and z_score < self.config.exit_z_score:
                signal = Signal(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    strength=0.8,
                    strategy_id=self.strategy_id,
                    metadata={
                        "type": "mean_reversion_exit",
                        "z_score": z_score,
                    },
                )
                self._in_position[symbol] = None
                self._signal_count += 1

        return signal

    def on_bar(self, prices: list[float], symbol: str) -> Optional[Signal]:
        """Process bar data for mean reversion signals."""
        if len(prices) < self.config.mean_window + 10:
            return None

        arr = np.array(prices)
        window = self.config.mean_window
        rolling_mean = np.mean(arr[-window:])
        rolling_std = np.std(arr[-window:])

        if rolling_std < 1e-8:
            return None

        z_score = (arr[-1] - rolling_mean) / rolling_std

        if abs(z_score) < self.config.entry_z_score:
            return None

        side = OrderSide.BUY if z_score < 0 else OrderSide.SELL
        strength = min(abs(z_score) / (self.config.entry_z_score * 2), 1.0)

        return Signal(
            symbol=symbol,
            side=side,
            strength=strength,
            strategy_id=self.strategy_id,
            metadata={"type": "mean_reversion_bar", "z_score": z_score},
        )

    @staticmethod
    def _estimate_half_life(prices: np.ndarray) -> float:
        """Estimate mean reversion half-life using OLS on lagged prices."""
        if len(prices) < 10:
            return float("inf")

        log_prices = np.log(prices)
        lag = log_prices[:-1]
        diff = np.diff(log_prices)

        if np.std(lag) < 1e-10:
            return float("inf")

        # OLS: diff = alpha + beta * lag + epsilon
        beta = np.cov(diff, lag)[0, 1] / np.var(lag) if np.var(lag) > 0 else 0

        if beta >= 0:
            return float("inf")  # Not mean reverting

        half_life = -np.log(2) / beta
        return max(half_life, 1.0)
