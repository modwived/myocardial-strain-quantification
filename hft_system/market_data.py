"""
Market data feed handler with simulated and historical data support.
"""

import logging
import math
import random
import time
from collections import defaultdict, deque
from typing import Callable, Optional

import numpy as np

from .models import OHLCV, Tick

logger = logging.getLogger(__name__)


class MarketDataFeed:
    """
    Market data feed that generates simulated tick data.

    Supports multiple symbols, subscribers, and maintains
    rolling price history for strategy calculations.
    """

    def __init__(self, buffer_size: int = 10000):
        self.buffer_size = buffer_size
        self._subscribers: list[Callable[[Tick], None]] = []
        self._tick_history: dict[str, deque[Tick]] = defaultdict(
            lambda: deque(maxlen=buffer_size)
        )
        self._ohlcv_history: dict[str, deque[OHLCV]] = defaultdict(
            lambda: deque(maxlen=buffer_size)
        )
        self._last_tick: dict[str, Tick] = {}
        self._base_prices: dict[str, float] = {}
        self._volatilities: dict[str, float] = {}
        self._tick_count = 0

    def initialize_symbols(
        self,
        symbols: list[str],
        base_prices: Optional[dict[str, float]] = None,
        volatilities: Optional[dict[str, float]] = None,
    ) -> None:
        """Initialize symbols with base prices and volatilities."""
        default_prices = {
            "AAPL": 175.0,
            "GOOGL": 140.0,
            "MSFT": 380.0,
            "AMZN": 180.0,
            "META": 500.0,
            "TSLA": 245.0,
            "NVDA": 800.0,
            "JPM": 190.0,
            "V": 280.0,
            "SPY": 500.0,
        }

        for symbol in symbols:
            if base_prices and symbol in base_prices:
                self._base_prices[symbol] = base_prices[symbol]
            elif symbol in default_prices:
                self._base_prices[symbol] = default_prices[symbol]
            else:
                self._base_prices[symbol] = 100.0

            if volatilities and symbol in volatilities:
                self._volatilities[symbol] = volatilities[symbol]
            else:
                self._volatilities[symbol] = 0.02  # 2% daily vol

        logger.info("Initialized %d symbols for market data", len(symbols))

    def subscribe(self, callback: Callable[[Tick], None]) -> None:
        self._subscribers.append(callback)

    def generate_tick(self, symbol: str) -> Tick:
        """Generate a simulated tick using geometric Brownian motion."""
        last = self._last_tick.get(symbol)
        base_price = self._base_prices.get(symbol, 100.0)
        vol = self._volatilities.get(symbol, 0.02)

        if last:
            current_price = last.last_price
        else:
            current_price = base_price

        # GBM step: dS = mu*S*dt + sigma*S*dW
        dt = 0.0001  # Fraction of trading day per tick
        drift = 0.0
        shock = vol * math.sqrt(dt) * random.gauss(0, 1)
        new_price = current_price * (1 + drift * dt + shock)
        new_price = max(new_price, 0.01)

        # Generate bid/ask around the new price
        half_spread = new_price * random.uniform(0.0001, 0.0005)
        bid = round(new_price - half_spread, 4)
        ask = round(new_price + half_spread, 4)

        bid_size = round(random.uniform(100, 5000), 0)
        ask_size = round(random.uniform(100, 5000), 0)
        last_size = round(random.uniform(1, 500), 0)
        volume = (last.volume + last_size) if last else last_size

        tick = Tick(
            symbol=symbol,
            timestamp=time.time(),
            bid=bid,
            ask=ask,
            bid_size=bid_size,
            ask_size=ask_size,
            last_price=round(new_price, 4),
            last_size=last_size,
            volume=volume,
        )

        self._last_tick[symbol] = tick
        self._tick_history[symbol].append(tick)
        self._tick_count += 1

        return tick

    def generate_ticks_all_symbols(self) -> list[Tick]:
        """Generate ticks for all initialized symbols."""
        ticks = []
        for symbol in self._base_prices:
            tick = self.generate_tick(symbol)
            ticks.append(tick)
            for callback in self._subscribers:
                callback(tick)
        return ticks

    def get_price_history(
        self, symbol: str, periods: int = 100
    ) -> list[float]:
        """Get recent price history for a symbol."""
        history = self._tick_history.get(symbol, deque())
        prices = [t.last_price for t in history]
        return prices[-periods:] if len(prices) >= periods else prices

    def get_mid_prices(self, symbol: str, periods: int = 100) -> list[float]:
        """Get recent mid price history."""
        history = self._tick_history.get(symbol, deque())
        prices = [t.mid_price for t in history]
        return prices[-periods:] if len(prices) >= periods else prices

    def get_last_tick(self, symbol: str) -> Optional[Tick]:
        return self._last_tick.get(symbol)

    def get_tick_count(self) -> int:
        return self._tick_count

    def generate_historical_ohlcv(
        self, symbol: str, periods: int = 252, interval_seconds: float = 86400
    ) -> list[OHLCV]:
        """Generate synthetic historical OHLCV data for backtesting."""
        base_price = self._base_prices.get(symbol, 100.0)
        vol = self._volatilities.get(symbol, 0.02)
        bars = []
        price = base_price
        now = time.time()
        start_time = now - periods * interval_seconds

        for i in range(periods):
            timestamp = start_time + i * interval_seconds
            daily_return = random.gauss(0.0002, vol)
            open_price = price
            intraday_moves = [random.gauss(0, vol * 0.3) for _ in range(10)]
            prices_intraday = [open_price]
            for move in intraday_moves:
                prices_intraday.append(prices_intraday[-1] * (1 + move))

            high = max(prices_intraday)
            low = min(prices_intraday)
            close = open_price * (1 + daily_return)
            close = max(close, 0.01)
            volume = random.uniform(1_000_000, 50_000_000)

            bar = OHLCV(
                symbol=symbol,
                timestamp=timestamp,
                open=round(open_price, 4),
                high=round(high, 4),
                low=round(low, 4),
                close=round(close, 4),
                volume=round(volume, 0),
            )
            bars.append(bar)
            self._ohlcv_history[symbol].append(bar)
            price = close

        # Update base price to last close
        self._base_prices[symbol] = price
        return bars
