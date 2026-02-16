"""
Momentum Strategy

Uses dual moving average crossover with volume confirmation
to identify and trade trending markets.
"""

import logging
from collections import deque
from typing import Optional

import numpy as np

from ..config import OrderSide, StrategyConfig
from ..models import Signal, Tick
from .base import BaseStrategy

logger = logging.getLogger(__name__)


class MomentumStrategy(BaseStrategy):
    """
    Momentum strategy using exponential moving average crossover.

    Features:
    - Fast/slow EMA crossover signals
    - Volume-weighted momentum confirmation
    - Trend strength filtering
    - Adaptive threshold based on recent volatility
    """

    def __init__(self, strategy_id: str, config: StrategyConfig):
        super().__init__(strategy_id, config)
        self._price_history: dict[str, deque] = {}
        self._volume_history: dict[str, deque] = {}
        self._prev_signal: dict[str, Optional[OrderSide]] = {}
        self._fast_ema: dict[str, float] = {}
        self._slow_ema: dict[str, float] = {}

    def on_tick(self, tick: Tick) -> Optional[Signal]:
        if not self.is_active:
            return None

        self._tick_count += 1
        symbol = tick.symbol

        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=1000)
            self._volume_history[symbol] = deque(maxlen=1000)
        self._price_history[symbol].append(tick.last_price)
        self._volume_history[symbol].append(tick.last_size)

        min_periods = max(self.config.slow_window + 10, 60)
        if len(self._price_history[symbol]) < min_periods:
            return None

        # Only generate signals every N ticks to avoid overtrading
        if self._tick_count % 5 != 0:
            return None

        prices = np.array(self._price_history[symbol])

        # Calculate EMAs
        fast_ema = self._calc_ema(prices, self.config.fast_window)
        slow_ema = self._calc_ema(prices, self.config.slow_window)

        self._fast_ema[symbol] = fast_ema
        self._slow_ema[symbol] = slow_ema

        # Calculate momentum metrics
        ema_diff = (fast_ema - slow_ema) / slow_ema
        returns = np.diff(np.log(prices[-20:]))
        volatility = np.std(returns) if len(returns) > 0 else 0.001
        adaptive_threshold = max(self.config.momentum_threshold, volatility * 1.5)

        # Volume confirmation
        volumes = np.array(self._volume_history[symbol])
        avg_volume = np.mean(volumes[-20:])
        recent_volume = np.mean(volumes[-5:])
        volume_surge = recent_volume / avg_volume if avg_volume > 0 else 1.0

        # Generate signal on crossover with volume confirmation
        signal = None
        if ema_diff > adaptive_threshold and volume_surge > 1.1:
            # Bullish crossover with volume
            strength = min(abs(ema_diff) / adaptive_threshold * 0.5, 1.0)
            strength *= min(volume_surge / 2.0, 1.0)

            if self._prev_signal.get(symbol) != OrderSide.BUY:
                signal = Signal(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    strength=strength,
                    strategy_id=self.strategy_id,
                    metadata={
                        "type": "momentum_crossover",
                        "ema_diff": ema_diff,
                        "volume_surge": volume_surge,
                        "volatility": volatility,
                        "fast_ema": fast_ema,
                        "slow_ema": slow_ema,
                    },
                )
                self._prev_signal[symbol] = OrderSide.BUY
                self._signal_count += 1

        elif ema_diff < -adaptive_threshold and volume_surge > 1.1:
            # Bearish crossover with volume
            strength = min(abs(ema_diff) / adaptive_threshold * 0.5, 1.0)
            strength *= min(volume_surge / 2.0, 1.0)

            if self._prev_signal.get(symbol) != OrderSide.SELL:
                signal = Signal(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    strength=strength,
                    strategy_id=self.strategy_id,
                    metadata={
                        "type": "momentum_crossover",
                        "ema_diff": ema_diff,
                        "volume_surge": volume_surge,
                        "volatility": volatility,
                        "fast_ema": fast_ema,
                        "slow_ema": slow_ema,
                    },
                )
                self._prev_signal[symbol] = OrderSide.SELL
                self._signal_count += 1

        return signal

    def on_bar(self, prices: list[float], symbol: str) -> Optional[Signal]:
        """Process bar data for momentum signals."""
        if len(prices) < self.config.slow_window + 10:
            return None

        arr = np.array(prices)
        fast_ema = self._calc_ema(arr, self.config.fast_window)
        slow_ema = self._calc_ema(arr, self.config.slow_window)
        ema_diff = (fast_ema - slow_ema) / slow_ema

        if abs(ema_diff) < self.config.momentum_threshold:
            return None

        side = OrderSide.BUY if ema_diff > 0 else OrderSide.SELL
        strength = min(abs(ema_diff) / self.config.momentum_threshold * 0.5, 1.0)

        return Signal(
            symbol=symbol,
            side=side,
            strength=strength,
            strategy_id=self.strategy_id,
            metadata={"type": "momentum_bar", "ema_diff": ema_diff},
        )

    @staticmethod
    def _calc_ema(prices: np.ndarray, window: int) -> float:
        """Calculate exponential moving average."""
        if len(prices) < window:
            return float(np.mean(prices))
        multiplier = 2.0 / (window + 1)
        ema = float(np.mean(prices[:window]))
        for price in prices[window:]:
            ema = (price - ema) * multiplier + ema
        return ema
