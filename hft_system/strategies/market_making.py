"""
Market Making Strategy

Places simultaneous buy and sell limit orders around the mid price,
profiting from the bid-ask spread while managing inventory risk.
"""

import logging
from collections import deque
from typing import Optional

import numpy as np

from ..config import OrderSide, StrategyConfig
from ..models import Signal, Tick
from .base import BaseStrategy

logger = logging.getLogger(__name__)


class MarketMakingStrategy(BaseStrategy):
    """
    Market making strategy that quotes both sides of the order book.

    Features:
    - Dynamic spread adjustment based on volatility
    - Inventory-aware quoting (skews prices to reduce inventory)
    - Volatility-based position sizing
    """

    def __init__(self, strategy_id: str, config: StrategyConfig):
        super().__init__(strategy_id, config)
        self._inventory: dict[str, float] = {}
        self._price_history: dict[str, deque] = {}
        self._vol_estimate: dict[str, float] = {}

    def on_tick(self, tick: Tick) -> Optional[Signal]:
        if not self.is_active:
            return None

        self._tick_count += 1
        symbol = tick.symbol

        # Maintain price history
        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=500)
        self._price_history[symbol].append(tick.mid_price)

        # Need enough data for vol estimation
        if len(self._price_history[symbol]) < 20:
            return None

        # Estimate short-term volatility
        prices = np.array(self._price_history[symbol])
        returns = np.diff(np.log(prices[-50:])) if len(prices) >= 50 else np.diff(np.log(prices))
        vol = np.std(returns) if len(returns) > 0 else 0.001
        self._vol_estimate[symbol] = vol

        # Calculate inventory skew
        inventory = self._inventory.get(symbol, 0.0)
        inventory_ratio = inventory / self.config.inventory_limit if self.config.inventory_limit else 0
        inventory_skew = -inventory_ratio * self.config.spread_bps * 0.5

        # Dynamic spread based on volatility
        base_spread_bps = self.config.spread_bps
        vol_adjustment = vol * 10000 * 2  # Widen spread in volatile markets
        adjusted_spread = base_spread_bps + vol_adjustment

        # Generate signal based on inventory needs
        if abs(inventory_ratio) > 0.8:
            # Need to reduce inventory
            side = OrderSide.SELL if inventory > 0 else OrderSide.BUY
            strength = min(abs(inventory_ratio), 1.0)
            signal = Signal(
                symbol=symbol,
                side=side,
                strength=strength,
                strategy_id=self.strategy_id,
                metadata={
                    "type": "inventory_reduction",
                    "spread_bps": adjusted_spread,
                    "inventory": inventory,
                    "volatility": vol,
                },
            )
            self._signal_count += 1
            return signal

        # Normal market making - quote both sides
        # Alternate between buy and sell signals
        if self._tick_count % 2 == 0:
            side = OrderSide.BUY
            strength = 0.3 + inventory_skew * 0.1
        else:
            side = OrderSide.SELL
            strength = 0.3 - inventory_skew * 0.1

        strength = max(0.1, min(abs(strength), 0.5))

        signal = Signal(
            symbol=symbol,
            side=side,
            strength=strength,
            strategy_id=self.strategy_id,
            metadata={
                "type": "market_making",
                "spread_bps": adjusted_spread,
                "inventory": inventory,
                "volatility": vol,
                "skew": inventory_skew,
            },
        )
        self._signal_count += 1
        return signal

    def on_bar(self, prices: list[float], symbol: str) -> Optional[Signal]:
        """Not used for tick-level market making."""
        return None

    def update_inventory(self, symbol: str, quantity_change: float) -> None:
        current = self._inventory.get(symbol, 0.0)
        self._inventory[symbol] = current + quantity_change
