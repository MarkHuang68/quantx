"""Streaming aggregator for live data.

The ``StreamAggregator`` collects incoming 1‑minute bars and
dynamically resamples them to the configured higher timeframe.  This is
useful for live trading where strategy timeframes are larger than the
base feed.  The aggregator maintains a rolling DataFrame for each
symbol and periodically emits the resampled bar when a complete
interval is available.
"""

from __future__ import annotations

import pandas as pd
from collections import defaultdict
from typing import Callable, Dict, Optional

from ...core.timeframe import resample_ohlcv


class StreamAggregator:
    """Aggregates 1m bars into longer timeframes on the fly."""

    def __init__(self, tf: str) -> None:
        self.tf = tf
        self.buffer: Dict[str, pd.DataFrame] = defaultdict(lambda: pd.DataFrame())

    def update(self, symbol: str, bar: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Add a new 1‑minute bar and return a resampled bar if completed.

        Parameters
        ----------
        symbol : str
            Symbol identifier.
        bar : DataFrame
            A single row DataFrame representing the latest 1‑minute bar.

        Returns
        -------
        DataFrame or None
            A one‑row DataFrame for the aggregated bar when a new bar is
            complete; otherwise ``None``.
        """
        if bar.empty:
            return None
        # Append to buffer
        buf = self.buffer[symbol]
        if buf.empty:
            self.buffer[symbol] = bar.copy()
        else:
            self.buffer[symbol] = pd.concat([buf, bar]).sort_index()
        # Determine if a new higher timeframe bar can be emitted
        resampled = resample_ohlcv(self.buffer[symbol], self.tf)
        if resampled.empty:
            return None
        # Keep only the last row for next interval; drop emitted bars
        last_timestamp = resampled.index[-1]
        # bars up to last_timestamp inclusive have been emitted
        self.buffer[symbol] = self.buffer[symbol][self.buffer[symbol].index > last_timestamp]
        return resampled.tail(1)