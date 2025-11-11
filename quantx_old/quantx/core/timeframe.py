# -*- coding: utf-8 -*-
# quantx/core/timeframe.py

import pandas as pd
import numpy as np
from typing import Literal
import calendar
from datetime import datetime, timezone

def parse_tf_minutes(tf: str, ref_dt: datetime = None) -> int:
    """
    將 '15m'/'30m'/'1h'/'4h'/'1d'/'1w'/'1M' 解析為分鐘數。
    - m/h/d/w → 固定換算
    - M → 根據 ref_dt 當月的實際天數換算 (精確版)
    Args:
        tf (str): 時間框字串，例如 "15m", "1h", "1d", "1w", "1M"
        ref_dt (datetime, optional): 基準日期時間，若為 None 則取當前 UTC
    Returns:
        int: 對應的分鐘數
    """
    tf = str(tf).strip()

    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    if tf.endswith("d"):
        return int(tf[:-1]) * 60 * 24
    if tf.endswith("w"):
        return int(tf[:-1]) * 60 * 24 * 7
    if tf.endswith("M"):
        months = int(tf[:-1])
        if ref_dt is None:
            ref_dt = datetime.now(timezone.utc)
        year, month = ref_dt.year, ref_dt.month
        days_in_month = calendar.monthrange(year, month)[1]
        return months * days_in_month * 24 * 60
    raise ValueError(f"Unsupported timeframe: {tf}")

def parse_tf_seconds(tf: str, ref_dt: datetime = None) -> int:
    if tf.endswith("m"):
        return int(tf[:-1]) * 60
    elif tf.endswith("h"):
        return int(tf[:-1]) * 3600
    elif tf.endswith("d"):
        return int(tf[:-1]) * 86400
    elif tf.endswith("w"):
        return int(tf[:-1]) * 7 * 86400
    elif tf.endswith("M"):
        months = int(tf[:-1])
        if ref_dt is None:
            ref_dt = datetime.now(timezone.utc)
        year, month = ref_dt.year, ref_dt.month
        days_in_month = calendar.monthrange(year, month)[1]
        return months * days_in_month * 86400
    else:
        raise ValueError(f"Unsupported tf: {tf}")

def bars_per_year(tf: str) -> float:
    """Approximate number of bars in a year for a given timeframe.

    Assumes continuous trading (24×7) for cryptocurrency markets.

    Parameters
    ----------
    tf : str
        Timeframe string (see :func:`parse_tf_minutes`).

    Returns
    -------
    float
        Approximate number of bars per year.
    """
    minutes = parse_tf_minutes(tf)
    return 365 * 24 * 60 / minutes

def resample_ohlcv(
    df: pd.DataFrame,
    tf: str,
    *,
    closed: Literal["right", "left"] = "right",
    label: Literal["right", "left"] = "right",
) -> pd.DataFrame:
    """Resample OHLCV data to a new timeframe.

    The input DataFrame must have a DatetimeIndex and the columns
    ``open``, ``high``, ``low``, ``close`` and ``volume``.  Missing
    values are forward filled for OHLC fields and set to zero for
    volume.  This helper uses pandas' built-in resampling; the
    parameters ``closed`` and ``label`` follow pandas semantics.

    Parameters
    ----------
    df : DataFrame
        Input 1m OHLCV data indexed by datetime.
    tf : str
        Target timeframe string. e.g. "15m", "1h", "1d", "1w", "1M"
    closed, label : {"right", "left"}, optional
        Behaviour of the resampler, see :meth:`pandas.DataFrame.resample`.

    Returns
    -------
    DataFrame
        Resampled OHLCV data.
    """
    if df.empty:
        return df.copy()
    # Ensure sorted index
    df = df.sort_index()
    # Forward fill OHLC fields and fill volume with zero
    df_ffill = df.copy()
    for col in ["open", "high", "low", "close"]:
        df_ffill[col] = df_ffill[col].ffill()
    df_ffill["volume"] = df_ffill["volume"].fillna(0)

    def _to_pandas_rule(tf: str) -> str:
        if tf.endswith("m"):  # 分鐘
            return f"{int(tf[:-1])}min"
        if tf.endswith("h"):  # 小時
            return f"{int(tf[:-1])}h"
        if tf.endswith("d"):  # 日
            return f"{int(tf[:-1])}d"
        if tf.endswith("w"):  # 週
            return f"{int(tf[:-1])}W" if tf[:-1].isdigit() else "W"
        if tf.endswith("M"):  # 月
            return f"{int(tf[:-1])}M" if tf[:-1].isdigit() else "M"
        return tf  # 其他原樣

    rule = _to_pandas_rule(tf)
    agg_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    resampled = df_ffill.resample(rule, closed=closed, label=label).agg(agg_dict)
    # Drop rows with all NaNs (in case of incomplete last bar)
    resampled = resampled.dropna(how="any")
    return resampled
