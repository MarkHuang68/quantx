# -*- coding: utf-8 -*-
"""
特徵引擎 (Feature Engine)
... (docstring remains the same) ...
"""
import pandas as pd
import ta
import xgboost as xgb
import numpy as np
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# **導入路徑修正**: 使用相對導入
from smc_features import (
    find_swing_highs_lows,
    find_break_of_structure,
    find_choch,
    find_order_blocks,
    find_high_quality_ob,
    find_order_flow,
    find_fair_value_gaps,
    add_smc_state_features
)

# ... (rest of the file remains the same) ...

# feature_engine.py → 完整 add_smc_features
def add_smc_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    df_final = df.copy()

    # 1. 擺盪高低點
    df_final = find_swing_highs_lows(df_final, window=window)

    # 2. BOS / CHoCH ← 這行你漏了！
    df_final = find_break_of_structure(df_final)

    # 3. 訂單塊
    df_final = find_order_blocks(df_final)

    # 4. 訂單流
    df_final = find_order_flow(df_final)

    # 5. 公平價值缺口
    df_final = find_fair_value_gaps(df_final)

    # 6. 高品質 OB
    df_final = find_high_quality_ob(df_final)

    # 7. 狀態機特徵
    df_final = add_smc_state_features(df_final)

    # === 特徵欄位清單 ===
    feature_cols = [
        'swing_high', 'swing_low',
        'bos_bullish', 'bos_bearish', 'bos_bullish_idx', 'bos_bearish_idx',
        'choch_bullish', 'choch_bearish', 'choch_bullish_idx', 'choch_bearish_idx',
        'bullish_ob_top', 'bullish_ob_bottom', 'bearish_ob_top', 'bearish_ob_bottom',
        'bullish_fvg_top', 'bullish_fvg_bottom', 'bearish_fvg_top', 'bearish_fvg_bottom',
        'hq_bullish_ob_top', 'hq_bullish_ob_bottom', 'hq_bearish_ob_top', 'hq_bearish_ob_bottom',
        'bullish_of_top', 'bullish_of_bottom', 'bearish_of_top', 'bearish_of_bottom',
        'poi_bullish', 'poi_bearish', 'feat_has_hq_ob'
    ]

    # 確保所有欄位存在
    for col in feature_cols:
        if col not in df_final.columns:
            df_final[col] = np.nan

    window_df = df_final[feature_cols].copy()
    df_final = pd.concat([df_final, window_df], axis=1)
    df_final.fillna(0, inplace=True)

    return df_final

def add_technical_indicators(df: pd.DataFrame, timeframes: list = [14, 30, 50]):
    """
    為數據幀添加多種技術指標。
    **已擴充以包含大師模型所需的所有指標。**
    """
    # --- 修正 & 優化 ---
    # 為所有指定的時間週期產生 TA 指標 (RSI, SMA, BB)，並使用正確的後綴。
    timeframe_suffixes = ['s', 'm', 'l']
    for i, t in enumerate(timeframes):
        if i < len(timeframe_suffixes):
            tf_str = timeframe_suffixes[i]
            df[f'rsi_{tf_str}'] = ta.momentum.RSIIndicator(close=df['close'], window=t).rsi()
            df[f'sma_{tf_str}'] = ta.trend.SMAIndicator(close=df['close'], window=t).sma_indicator()
            bollinger = ta.volatility.BollingerBands(close=df['close'], window=t, window_dev=2)
            df[f'bb_high_{tf_str}'] = bollinger.bollinger_hband()
            df[f'bb_low_{tf_str}'] = bollinger.bollinger_lband()

    macd = ta.trend.MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()

    adx = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['adx'] = adx.adx()

    stoch_rsi = ta.momentum.StochRSIIndicator(close=df['close'], window=14, smooth1=3, smooth2=3)
    df['stoch_rsi_k'] = stoch_rsi.stochrsi_k()
    df['stoch_rsi_d'] = stoch_rsi.stochrsi_d()

    df['ema_spread_s_m'] = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator() - \
                                 ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
    df['ema_spread_m_l'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator() - \
                                  ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()

    df['close_price'] = df['close']

    return df

def add_all_features(df: pd.DataFrame):
    """
    主特徵工程流程，按正確順序調用所有特徵生成函數。
    [v5.0 簡化版]:
    1.  添加 TA 指標 (RSI, SMA etc.)。
    2.  添加「單一窗口 (window=5)」的 SMC 基礎特徵。
    3.  運行 SMC 狀態機 (SMC State Features)。
    """
    df_copy = df.copy()
    
    # 1. 添加 TA 指標
    df_copy = add_technical_indicators(df_copy, timeframes=[14, 30, 50])
    
    # 2. 添加「單一窗口」SMC 基礎特徵 (window=5)
    #    這會產生 add_smc_state_features 需要的「沒有後綴」的欄位
    df_copy = add_smc_features(df_copy, window=5)

    # 3. 運行 SMC 狀態機 (SMC 大腦)
    #    現在它可以 100% 找到它需要的基礎欄位了
    df_copy = add_smc_state_features(df_copy)

    # --- POI 距離特徵 ---
    df_copy['feat_dist_to_poi'] = 0.0
    bull_mask = df_copy['poi_bullish'].notna()
    bear_mask = df_copy['poi_bearish'].notna()
    df_copy.loc[bull_mask, 'feat_dist_to_poi'] = df_copy['close'] - df_copy['poi_bullish']
    df_copy.loc[bear_mask, 'feat_dist_to_poi'] = df_copy['poi_bearish'] - df_copy['close']

    # 數據清理：用0填充所有剩餘的 NaN 值
    df_copy.fillna(0, inplace=True)
    return df_copy