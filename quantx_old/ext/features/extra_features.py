# ext/features/extra_features.py
# 說明：
# - 提供額外特徵給 ML pipeline
# - 需要實作 build_extra_features(df)，輸入/輸出都是 DataFrame
# - 新增的欄位會自動被 features.py 的 maybe_add_extra_features() 合併

import pandas as pd
import numpy as np
import ta  # pip install ta

def build_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    輸入 df (至少包含 open/high/low/close/volume)，
    回傳包含額外技術指標的新 DataFrame。
    """
    out = df.copy()

    # 移動平均線
    out["sma_20"] = out["close"].rolling(20, min_periods=1).mean()
    out["ema_20"] = out["close"].ewm(span=20, adjust=False).mean()

    # RSI (14)
    out["rsi_14"] = ta.momentum.RSIIndicator(close=out["close"], window=14).rsi()

    # ATR (14)
    out["atr_14"] = ta.volatility.AverageTrueRange(
        high=out["high"], low=out["low"], close=out["close"], window=14
    ).average_true_range()

    # 布林通道寬度 (20)
    bb = ta.volatility.BollingerBands(close=out["close"], window=20, window_dev=2)
    out["bb_width"] = bb.bollinger_hband() - bb.bollinger_lband()

    # 成交量移動平均
    out["vol_ma20"] = out["volume"].rolling(20, min_periods=1).mean()

    return out
