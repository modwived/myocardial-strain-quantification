"""
Base strategy interface for all trading strategies.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

from ..config import StrategyConfig
from ..models import Signal, Tick

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.

    Each strategy receives market data ticks and generates
    trading signals with direction and strength.
    """

    def __init__(self, strategy_id: str, config: StrategyConfig):
        self.strategy_id = strategy_id
        self.config = config
        self.is_active = True
        self._tick_count = 0
        self._signal_count = 0

    @abstractmethod
    def on_tick(self, tick: Tick) -> Optional[Signal]:
        """Process a tick and optionally return a signal."""

    @abstractmethod
    def on_bar(self, prices: list[float], symbol: str) -> Optional[Signal]:
        """Process a price bar series and optionally return a signal."""

    def activate(self) -> None:
        self.is_active = True
        logger.info("Strategy %s activated", self.strategy_id)

    def deactivate(self) -> None:
        self.is_active = False
        logger.info("Strategy %s deactivated", self.strategy_id)

    def reset(self) -> None:
        """Reset strategy state."""
        self._tick_count = 0
        self._signal_count = 0

    @property
    def name(self) -> str:
        return self.__class__.__name__
