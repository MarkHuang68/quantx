# quantx/market/provider/base.py
"""Base class for market data providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd

class MarketProvider(ABC):
    """Abstract base class for providers supplying OHLCV data."""

    @abstractmethod
    def fetch_klines(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data.

        Returns
        -------
        pandas.DataFrame
            DatetimeIndex, columns = ['open','high','low','close','volume'].
            Empty DataFrame if no data.
        """
        raise NotImplementedError
    
    @abstractmethod
    def submit_order(self, symbol: str, order_type: str, side: str, qty: float,
                    price: Optional[float] = None, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Submit an order to the exchange.

        Returns
        -------
        dict
            A dictionary containing the order information.
        """
        raise NotImplementedError
