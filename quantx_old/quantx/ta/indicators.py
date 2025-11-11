# quantx/ta/indicators.py
# -*- coding: utf-8 -*-
# 檔案: quantx/ta/indicators.py
# 版本: v5 (新增 ATR 函數)
# 說明: 包含策略檔案 ema_trend_follower.py 中引用的所有技術指標的實現。

from __future__ import annotations

import pandas as pd
import numpy as np


# --- 輔助函數 ---
def _series_prep(series: pd.Series) -> pd.Series:
    """確保輸入 Series 為 float 類型，並處理潛在錯誤。"""
    return pd.to_numeric(series, errors='coerce').astype(float)


# --- 單純移動平均線 ---
def SMA(series: pd.Series, length: int) -> pd.Series:
    """Simple moving average (SMA)."""
    series = _series_prep(series)
    return series.rolling(window=length, min_periods=1).mean()


# --- 指數移動平均線 (策略主要使用) ---
def EMA(series: pd.Series, length: int) -> pd.Series:
    """Exponential moving average (EMA)."""
    series = _series_prep(series)
    # 使用 adjust=False 以符合傳統 TA 的 EMA 遞迴公式
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


# --- Z-Score (原始檔案中存在) ---
def Zscore(series: pd.Series, length: int) -> pd.Series:
    """Rolling z‑score."""
    series = _series_prep(series)
    
    # 使用 min_periods=1
    rolling_mean = series.rolling(window=length, min_periods=1).mean()
    # ddof=0 為標準差 (Population Standard Deviation)
    rolling_std = series.rolling(window=length, min_periods=1).std(ddof=0)
    
    rolling_std = rolling_std.replace(0, np.nan)
    z = (series - rolling_mean) / rolling_std
    
    # 如果無法計算 (例如 std=0)，則填補為 0
    return z.fillna(0)


# --- 相對強弱指數 (RSI) ---
def RSI(series: pd.Series, length: int = 14) -> pd.Series:
    """Relative Strength Index (RSI)."""
    series = _series_prep(series)
    
    delta = series.diff().fillna(0)
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    # 使用 EMA 進行平滑 (更符合標準 RSI 實現)
    avg_gain = gain.ewm(span=length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(span=length, adjust=False, min_periods=length).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


# --- 交叉信號 (策略主要使用) ---
def CrossUp(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    """A 向上穿越 B。"""
    a = _series_prep(series_a)
    b = _series_prep(series_b)
    
    # 向上穿越: 前一期 A <= B 且 當前期 A > B
    condition = (a.shift(1) <= b.shift(1)) & (a > b)
    return condition.fillna(False)


def CrossDown(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    """A 向下穿越 B。"""
    a = _series_prep(series_a)
    b = _series_prep(series_b)
    
    # 向下穿越: 前一期 A >= B 且 當前期 A < B
    condition = (a.shift(1) >= b.shift(1)) & (a < b)
    return condition.fillna(False)


# --- 複雜指標 (策略中引用，現補齊) ---

def HeikinAshi(open_s: pd.Series, high_s: pd.Series, low_s: pd.Series, close_s: pd.Series) -> pd.DataFrame:
    """Heikin Ashi (HA) K線圖。
    key HA_Open
        HA_High
        HA_Low
        HA_Close
    """
    
    ha_close = (open_s + high_s + low_s + close_s) / 4
    
    # 初始化 HA 開盤價為 SMA
    ha_open = SMA(close_s, length=1).shift(1)
    ha_open.iloc[0] = open_s.iloc[0] # 第一根 K 棒使用原始開盤價
    
    # 計算 HA 開盤價
    for i in range(1, len(open_s)):
        ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2

    # 計算 HA 高點和低點
    ha_high = pd.DataFrame({'high': high_s, 'ha_open': ha_open, 'ha_close': ha_close}).max(axis=1)
    ha_low = pd.DataFrame({'low': low_s, 'ha_open': ha_open, 'ha_close': ha_close}).min(axis=1)
    
    df = pd.DataFrame({
        'HA_Open': ha_open,
        'HA_High': ha_high,
        'HA_Low': ha_low,
        'HA_Close': ha_close
    }, index=close_s.index)
    return df


def BBands(series: pd.Series, length: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands (BBands).
    key BB_Upper
        BB_Middle
        BB_Lower
    """
    series = _series_prep(series)
    
    df = pd.DataFrame(index=series.index)
    df['Middle'] = SMA(series, length=length)
    df['StdDev'] = series.rolling(window=length, min_periods=length).std()
    df['Upper'] = df['Middle'] + (df['StdDev'] * num_std)
    df['Lower'] = df['Middle'] - (df['StdDev'] * num_std)
    
    # 命名符合標準庫習慣
    df_result = pd.DataFrame(index=series.index)
    df_result['BB_Upper'] = df['Upper']
    df_result['BB_Middle'] = df['Middle']
    df_result['BB_Lower'] = df['Lower']
    return df_result


def MACD(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Moving Average Convergence Divergence (MACD).
    key MACD_macd
        MACD_signal
        MACD_histogram
    """
    series = _series_prep(series)
    
    fast_ema = EMA(series, length=fast)
    slow_ema = EMA(series, length=slow)
    
    macd_line = fast_ema - slow_ema
    signal_line = EMA(macd_line, length=signal)
    
    # Histogram 雖然策略中沒用，但通常會一起計算
    histogram = macd_line - signal_line
    
    df = pd.DataFrame(index=series.index)
    df['MACD_macd'] = macd_line
    df['MACD_signal'] = signal_line
    df['MACD_histogram'] = histogram
    return df


# 🟢 === 新增 ATR 函數 === 🟢
def ATR(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """Average True Range (ATR).
    使用 Welles Wilder 平滑 (EWM, alpha=1/length)。
    """
    high_s = _series_prep(high)
    low_s = _series_prep(low)
    close_s = _series_prep(close)
    
    # 1. 計算 True Range (TR)
    # (此邏輯借鑒自下方 ADX 函數的實現)
    tr_df = pd.DataFrame({
        'h_l': high_s - low_s,
        'h_c': (high_s - close_s.shift(1)).abs(),
        'l_c': (low_s - close_s.shift(1)).abs()
    })
    tr = tr_df.max(axis=1)
    
    # 2. 平滑 TR (使用 EWM, alpha=1/length)
    # (此邏輯借鑒自下方 ADX 函數中的 _smooth 實現)
    atr_line = tr.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    
    return atr_line
# 🟢 ======================= 🟢


def ADX(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.DataFrame:
    """Average Directional Index (ADX). 簡化實現。
    key ADX
        PDI
        MDI
    """
    # 確保所有 K線數據都是 float
    high_s = _series_prep(high)
    low_s = _series_prep(low)
    close_s = _series_prep(close)
    
    # 1. 計算 True Range (TR)
    tr = pd.DataFrame({
        'h_l': high_s - low_s,
        'h_c': (high_s - close_s.shift(1)).abs(),
        'l_c': (low_s - close_s.shift(1)).abs()
    }).max(axis=1)
    
    # 2. 計算 Directional Movement (DM)
    up_move = high_s - high_s.shift(1)
    down_move = low_s.shift(1) - low_s
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    # 3. 平滑 DM 和 TR (使用 EMA 或 Welles Wilder 平滑)
    # 這裡使用 Ema 作為 Welles Wilder 平滑的近似
    def _smooth(s, length):
        return s.ewm(alpha=1/length, adjust=False).mean()
    
    plus_di = _smooth(pd.Series(plus_dm, index=close_s.index), length) / _smooth(tr, length) * 100
    minus_di = _smooth(pd.Series(minus_dm, index=close_s.index), length) / _smooth(tr, length) * 100
    
    # 4. 計算 Directional Index (DX)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).abs() * 100
    
    # 5. 計算 Average Directional Index (ADX)
    adx_line = _smooth(dx, length)
    
    df = pd.DataFrame(index=close_s.index)
    df['ADX'] = adx_line
    df['PDI'] = plus_di
    df['MDI'] = minus_di
    return df


def StochRSI(series: pd.Series, length: int = 14, rsi_length: int = 14, k: int = 3, d: int = 3) -> pd.DataFrame:
    """Stochastic RSI (StochRsi).
    key StochRSI_k
        StochRSI_d
    """
    series = _series_prep(series)
    
    # 1. 計算 RSI
    rsi_line = RSI(series, length=rsi_length)
    
    # 2. 計算 StochRSI 的 %K
    # StochRSI = (RSI - Min(RSI)) / (Max(RSI) - Min(RSI))
    lowest_rsi = rsi_line.rolling(window=length, min_periods=length).min()
    highest_rsi = rsi_line.rolling(window=length, min_periods=length).max()
    
    stoch_rsi = (rsi_line - lowest_rsi) / (highest_rsi - lowest_rsi)
    stoch_rsi = stoch_rsi.fillna(0.5) # 避免除以零或 NaN
    
    # 3. 平滑得到 %K 和 %D
    stoch_k = SMA(stoch_rsi * 100, length=k)
    stoch_d = SMA(stoch_k, length=d)
    
    df = pd.DataFrame(index=series.index)
    df['StochRSI_k'] = stoch_k
    df['StochRSI_d'] = stoch_d
    return df