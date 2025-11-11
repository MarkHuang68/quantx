"""Simulated market data provider.

This provider generates synthetic OHLCV data for demonstration and
testing purposes.  The generated data resembles a random walk with
added noise.  Because no real market data is available in this
environment, the simulated provider serves as a drop‑in replacement
for strategy development and backtesting.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import numpy as np

from .base import MarketProvider
from ...core.timeframe import parse_tf_minutes


class SimulatedProvider(MarketProvider):
    """Provider that synthesises OHLCV bars using a simple random walk."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self.random_state = np.random.RandomState(seed)

    def fetch_klines(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        tf: str = "1m",
        interval_m: int = 0,
    ) -> pd.DataFrame:
        # Generate a DateTimeIndex for the requested period
        minutes = parse_tf_minutes(tf)
        freq = f"{interval_m if interval_m > 0 else minutes}min"
        idx = pd.date_range(start=start, end=end, freq=freq)
        if len(idx) == 0:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        # Generate a random walk price series
        # Start price around 100 + noise
        price = 100 + self.random_state.randn()
        prices = [price]
        for _ in range(1, len(idx)):
            # small drift and noise
            change = self.random_state.randn() * 0.1
            price = max(0.1, price + change)
            prices.append(price)
        prices = np.array(prices)
        # Construct OHLC from the walk: open=previous close, high=+noise, low=-noise
        open_prices = prices.copy()
        open_prices[1:] = prices[:-1]
        high = np.maximum(prices, open_prices) + self.random_state.rand(len(prices)) * 0.2
        low = np.minimum(prices, open_prices) - self.random_state.rand(len(prices)) * 0.2
        close = prices
        volume = self.random_state.rand(len(prices)) * 10
        df = pd.DataFrame(
            {
                "open": open_prices,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            },
            index=idx,
        )
        return df