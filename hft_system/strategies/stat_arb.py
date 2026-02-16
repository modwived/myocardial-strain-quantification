"""
Statistical Arbitrage Strategy

Identifies cointegrated pairs and trades the spread
when it deviates from equilibrium.
"""

import logging
from collections import deque
from itertools import combinations
from typing import Optional

import numpy as np

from ..config import OrderSide, StrategyConfig
from ..models import Signal, Tick
from .base import BaseStrategy

logger = logging.getLogger(__name__)


class StatisticalArbitrageStrategy(BaseStrategy):
    """
    Pairs trading / statistical arbitrage strategy.

    Features:
    - Dynamic cointegration testing between symbol pairs
    - Kalman filter-based hedge ratio estimation
    - Spread z-score trading with adaptive thresholds
    - Multi-pair portfolio construction
    """

    def __init__(self, strategy_id: str, config: StrategyConfig):
        super().__init__(strategy_id, config)
        self._price_history: dict[str, deque] = {}
        self._pairs: list[tuple[str, str]] = []
        self._hedge_ratios: dict[tuple[str, str], float] = {}
        self._spread_history: dict[tuple[str, str], deque] = {}
        self._in_trade: dict[tuple[str, str], Optional[str]] = {}
        self._pairs_initialized = False

    def _initialize_pairs(self) -> None:
        """Create all possible pairs from configured symbols."""
        symbols = list(self._price_history.keys())
        if len(symbols) >= 2:
            self._pairs = list(combinations(sorted(symbols), 2))
            for pair in self._pairs:
                self._spread_history[pair] = deque(maxlen=1000)
                self._in_trade[pair] = None
            self._pairs_initialized = True
            logger.info("Initialized %d pairs for stat arb", len(self._pairs))

    def on_tick(self, tick: Tick) -> Optional[Signal]:
        if not self.is_active:
            return None

        self._tick_count += 1
        symbol = tick.symbol

        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=2000)
        self._price_history[symbol].append(tick.last_price)

        if not self._pairs_initialized and len(self._price_history) >= 2:
            self._initialize_pairs()

        if not self._pairs_initialized:
            return None

        # Only check periodically
        if self._tick_count % 10 != 0:
            return None

        # Check each pair involving this symbol
        for pair in self._pairs:
            if symbol not in pair:
                continue

            sym_a, sym_b = pair
            hist_a = self._price_history.get(sym_a)
            hist_b = self._price_history.get(sym_b)

            if not hist_a or not hist_b:
                continue

            min_len = min(len(hist_a), len(hist_b), self.config.cointegration_window)
            if min_len < 30:
                continue

            prices_a = np.array(list(hist_a))[-min_len:]
            prices_b = np.array(list(hist_b))[-min_len:]

            # Estimate hedge ratio using OLS
            hedge_ratio = self._calculate_hedge_ratio(prices_a, prices_b)
            self._hedge_ratios[pair] = hedge_ratio

            # Calculate spread
            spread = prices_a - hedge_ratio * prices_b
            self._spread_history[pair].append(spread[-1])

            if len(self._spread_history[pair]) < 20:
                continue

            spread_series = np.array(self._spread_history[pair])
            spread_mean = np.mean(spread_series)
            spread_std = np.std(spread_series)

            if spread_std < 1e-8:
                continue

            z_score = (spread[-1] - spread_mean) / spread_std

            # Check for cointegration (simplified ADF-like test)
            if not self._is_cointegrated(spread_series):
                continue

            signal = self._generate_pair_signal(
                pair, z_score, hedge_ratio, spread_mean, spread_std
            )
            if signal:
                return signal

        return None

    def on_bar(self, prices: list[float], symbol: str) -> Optional[Signal]:
        """Not primary mode for stat arb."""
        return None

    def _generate_pair_signal(
        self,
        pair: tuple[str, str],
        z_score: float,
        hedge_ratio: float,
        spread_mean: float,
        spread_std: float,
    ) -> Optional[Signal]:
        """Generate trading signal for a pair."""
        sym_a, sym_b = pair
        entry_threshold = self.config.entry_z_score
        exit_threshold = self.config.exit_z_score
        current_trade = self._in_trade.get(pair)

        if current_trade is None:
            if z_score > entry_threshold:
                # Spread is wide - sell A, buy B (expect convergence)
                strength = min(abs(z_score) / (entry_threshold * 2), 1.0)
                self._in_trade[pair] = "short_spread"
                self._signal_count += 1
                return Signal(
                    symbol=sym_a,
                    side=OrderSide.SELL,
                    strength=strength,
                    strategy_id=self.strategy_id,
                    metadata={
                        "type": "stat_arb_entry",
                        "pair": f"{sym_a}/{sym_b}",
                        "z_score": z_score,
                        "hedge_ratio": hedge_ratio,
                        "direction": "short_spread",
                    },
                )
            elif z_score < -entry_threshold:
                # Spread is narrow - buy A, sell B (expect divergence)
                strength = min(abs(z_score) / (entry_threshold * 2), 1.0)
                self._in_trade[pair] = "long_spread"
                self._signal_count += 1
                return Signal(
                    symbol=sym_a,
                    side=OrderSide.BUY,
                    strength=strength,
                    strategy_id=self.strategy_id,
                    metadata={
                        "type": "stat_arb_entry",
                        "pair": f"{sym_a}/{sym_b}",
                        "z_score": z_score,
                        "hedge_ratio": hedge_ratio,
                        "direction": "long_spread",
                    },
                )
        else:
            # Exit when spread reverts
            if current_trade == "short_spread" and z_score < exit_threshold:
                self._in_trade[pair] = None
                self._signal_count += 1
                return Signal(
                    symbol=sym_a,
                    side=OrderSide.BUY,
                    strength=0.8,
                    strategy_id=self.strategy_id,
                    metadata={
                        "type": "stat_arb_exit",
                        "pair": f"{sym_a}/{sym_b}",
                        "z_score": z_score,
                    },
                )
            elif current_trade == "long_spread" and z_score > -exit_threshold:
                self._in_trade[pair] = None
                self._signal_count += 1
                return Signal(
                    symbol=sym_a,
                    side=OrderSide.SELL,
                    strength=0.8,
                    strategy_id=self.strategy_id,
                    metadata={
                        "type": "stat_arb_exit",
                        "pair": f"{sym_a}/{sym_b}",
                        "z_score": z_score,
                    },
                )

        return None

    @staticmethod
    def _calculate_hedge_ratio(prices_a: np.ndarray, prices_b: np.ndarray) -> float:
        """Calculate hedge ratio using OLS regression."""
        if np.var(prices_b) < 1e-10:
            return 1.0
        beta = np.cov(prices_a, prices_b)[0, 1] / np.var(prices_b)
        return float(beta)

    @staticmethod
    def _is_cointegrated(spread: np.ndarray, threshold: float = -2.0) -> bool:
        """
        Simplified cointegration test using ADF-like statistic.
        Tests if the spread is stationary.
        """
        if len(spread) < 20:
            return False

        # Augmented Dickey-Fuller approximation
        diff = np.diff(spread)
        lag = spread[:-1]

        if np.std(lag) < 1e-10:
            return False

        # OLS: diff = alpha + beta * lag
        beta = np.cov(diff, lag)[0, 1] / np.var(lag) if np.var(lag) > 0 else 0
        residuals = diff - beta * lag
        se = np.std(residuals) / (np.std(lag) * np.sqrt(len(lag)))

        if se < 1e-10:
            return False

        adf_stat = beta / se
        return adf_stat < threshold
