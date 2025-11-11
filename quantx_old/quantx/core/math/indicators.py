"""Simple technical indicators.

These indicator functions operate on pandas Series and return pandas
Series aligned with the input.  Each indicator intentionally avoids
future data leakage by using only past and present values.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def sma(series: pd.Series, length: int) -> pd.Series:
    """Simple moving average.

    Parameters
    ----------
    series : Series
        Input series.
    length : int
        Number of periods to average.

    Returns
    -------
    Series
        Moving average of the series.
    """
    return series.rolling(window=length, min_periods=1).mean()


def zscore(series: pd.Series, length: int) -> pd.Series:
    """Rolling z‑score of a series.

    Calculates the z‑score of the latest value relative to the trailing
    window of the specified length.  When fewer than ``length``
    observations are available the standard deviation is computed over
    the available observations.

    Parameters
    ----------
    series : Series
        Input series.
    length : int
        Lookback window length.

    Returns
    -------
    Series
        Z‑score series.
    """
    rolling_mean = series.rolling(window=length, min_periods=1).mean()
    rolling_std = series.rolling(window=length, min_periods=1).std(ddof=0)
    z = (series - rolling_mean) / rolling_std.replace(0, np.nan)
    return z.fillna(0)


def rsi(series: pd.Series, length: int) -> pd.Series:
    """Relative Strength Index (RSI).

    A simple RSI implementation that computes gains and losses and
    applies the Wilder smoothing method.  Returns a series between 0
    and 100.  Where insufficient data exists the RSI is computed on
    the available observations.

    Parameters
    ----------
    series : Series
        Input price series (typically closes).
    length : int
        Lookback period.

    Returns
    -------
    Series
        RSI values.
    """
    delta = series.diff().fillna(0)
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=length, min_periods=1).mean()
    avg_loss = loss.rolling(window=length, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)