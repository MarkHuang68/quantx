# 檔案: common_utils.py
# 這裡是「唯一」的特徵工程函數 (您「寫死」的黃金配方)
# 您的「訓練」和「實盤」腳本都會 100% 引用這裡

import ccxt
import pandas as pd
import ta
import numpy as np
import time
import config  # <--- *** 1. 引用您的「設定檔」 ***

def fetch_data(symbol, timeframe, total_limit):
    """
    從 Coinbase 獲取大量 OHLCV 資料 (使用迴圈)。
    """
    print(f"--- 步驟 1: 正在獲取 {symbol} {timeframe} 資料 (目標 {total_limit} 筆) ---")
    exchange = ccxt.coinbase({'rateLimit': 1200, 'enableRateLimit': True})
    
    limit_per_request = 300 # Coinbase max limit is 300
    all_ohlcv = []

    needed_limit = total_limit

    while needed_limit > 0:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=min(needed_limit, limit_per_request))
            if not ohlcv: break
            all_ohlcv = ohlcv + all_ohlcv
            needed_limit -= len(ohlcv)

        except Exception as e:
            print(f"獲取資料時發生未知錯誤: {e}")
            time.sleep(5)
    
    if not all_ohlcv: return None
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df = df.drop_duplicates(subset=['timestamp'])
    if len(df) > total_limit: df = df.tail(total_limit)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    print("DataFrame 處理完成。")
    return df

def create_features_trend(df, ema=20, sma=60, rsi=14, bbands=10):
    """
    (模型 B) 1h 趨勢特徵。
    *** 您的「黃金配方」就「寫死」在這裡 ***
    """
    print("\n--- 正在計算「1h 趨勢特徵」(黃金配方) ---")
    try:
        df['EMA'] = ta.trend.ema_indicator(df['Close'], window=20)
        df['SMA'] = ta.trend.sma_indicator(df['Close'], window=60)
        df['Maybe'] = (df['EMA'] > df['SMA']).astype(int)
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)

        bb = ta.volatility.BollingerBands(close=df['Close'], window=10, window_dev=2)
        df['BB_Width'] = bb.bollinger_wband()
        
        df_features = df.dropna() 

        features_list = [
            'EMA',
            'SMA',
            'Maybe',
            'RSI',
            'BB_Width'
        ]

        return df_features, features_list
    except Exception as e:
        print(f"計算 1h 特徵時發生錯誤: {e}")
        return None, None

def create_features_entry(df):
    """
    (模型 A) 5m 進場特徵 (報酬率預測模型)。
    """
    if df is None:
        return None, None
        
    print("\n--- 正在計算「5m 進場特徵」---")

    try:
        # --- 時間特徵 ---
        df['HOUR'] = df.index.hour
        df['MONTH'] = df.index.month

        # --- TA 指標 ---
        df = ta.add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)

        df['Close_SMA'] = ta.trend.sma_indicator(df['Close'], window=3)
        df['EMA'] = ta.trend.ema_indicator(df['Close'], window=20)
        df['CLOSE_EMA'] = (df['Close'] - df['EMA']) / df['EMA']

        # --- (特徵列表) ---
        features_list = [
            'HOUR', 'MONTH', 'EMA', 'CLOSE_EMA', 'volatility_atr', 'momentum_rsi',
            'trend_adx', 'trend_adx_pos', 'trend_adx_neg', 'momentum_stoch', 'momentum_stoch_signal',
            'trend_macd', 'trend_macd_signal', 'trend_macd_diff', 'volatility_bbw', 'volatility_bbp',
            'volume_obv', 'Volume', 'volume_cmf'
        ]

        df_features = df.dropna()
        return df_features, features_list

    except Exception as e:
        print(f"計算進場特徵時發生錯誤: {e}")
        return None, None

def create_sequences(data, target, lookback_window=24):
    """ (這是 LSTM 專用的，保持不變) """
    print(f"\n--- 正在建立 3D 序列 (回看 {lookback_window} 根 K 棒)... ---")
    X = []
    y = []
    for i in range(lookback_window, len(data)):
        X.append(data[i-lookback_window:i, :])
        y.append(target[i])
    X, y = np.array(X), np.array(y)
    return X, y
