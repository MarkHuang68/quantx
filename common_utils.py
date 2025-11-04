# 檔案: common_utils.py
# 這裡是「唯一」的特徵工程函數 (您「寫死」的黃金配方)
# 您的「訓練」和「實盤」腳本都會 100% 引用這裡

import ccxt
import pandas as pd
import talib
import numpy as np
import time
import config  # <--- *** 1. 引用您的「設定檔」 ***

def fetch_data(symbol, timeframe, total_limit):
    """
    從幣安 (Binance) 獲取大量 OHLCV 資料 (使用迴圈)。
    """
    print(f"--- 步驟 1: 正在獲取 {symbol} {timeframe} 資料 (目標 {total_limit} 筆) ---")
    exchange = ccxt.binance({'rateLimit': 1200, 'enableRateLimit': True})
    
    try:
        timeframe_duration_ms = exchange.parse_timeframe(timeframe) * 1000
    except Exception as e:
        print(f"Timeframe 格式錯誤: {e}。")
        return None
        
    limit_per_request = 1000
    all_ohlcv = []
    total_duration_ms = total_limit * timeframe_duration_ms
    since_timestamp = exchange.milliseconds() - total_duration_ms

    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since_timestamp, limit=limit_per_request)
            if not ohlcv: break
            all_ohlcv.extend(ohlcv)
            last_timestamp = ohlcv[-1][0]
            since_timestamp = last_timestamp + timeframe_duration_ms
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
        close_prices = df['Close'].values.astype(float)
        
        df['EMA'] = talib.EMA(close_prices, timeperiod=20)
        df['SMA'] = talib.SMA(close_prices, timeperiod=60)
        df['Maybe'] = (df['EMA'] > df['SMA']).astype(int)
        df['RSI'] = talib.RSI(close_prices, timeperiod=14)
        upperband, middleband, lowerband = talib.BBANDS(close_prices, timeperiod=10, nbdevup=2, nbdevdn=2, matype=0)
        df['BB_Width'] = (upperband - lowerband) / (middleband + 1e-10)
        
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
        return None

def create_features_price(df, rsi=14, wma=2, macd_fast=6, macd_slow=13, adx=14, bbands=2):
    """
    (模型 A) 5m 價格特徵。
    *** 您的「RMSE=8.00」特徵就「寫死」在這裡 ***
    """
    print("\n--- 正在計算「5m 價格特徵」---")
    try:
        close_prices = df['Close'].values.astype(float)
        high_prices = df['High'].values.astype(float)
        low_prices = df['Low'].values.astype(float)
        volume = df['Volume'].values.astype(float)
        
        df['RSI'] = talib.RSI(close_prices, timeperiod=rsi)
        df['WMA_close_2'] = talib.WMA(close_prices, timeperiod=wma)
        df['WMA_high_2'] = talib.WMA(high_prices, timeperiod=wma)
        df['WMA_low_2'] = talib.WMA(low_prices, timeperiod=wma)

        macd, macdsignal, _ = talib.MACD(close_prices, 
                                         fastperiod=macd_fast, 
                                         slowperiod=macd_slow, 
                                         signalperiod=9)
        df['MACD'] = macd
        df['MACD_signal'] = macdsignal
        df['OBV'] = talib.OBV(close_prices, volume)

        df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=adx)
        df['ADX_hist'] = talib.PLUS_DI(df['High'], df['Low'], df['Close'], timeperiod=adx) - talib.MINUS_DI(df['High'], df['Low'], df['Close'], timeperiod=adx)

        upperband, middleband, lowerband = talib.BBANDS(close_prices, 
                                                        timeperiod=bbands, 
                                                        nbdevup=2, 
                                                        nbdevdn=2, 
                                                        matype=0)
        
        df['BB_Width'] = (upperband - lowerband) / (middleband + 1e-10)
        df['BB_Percent'] = (close_prices - lowerband) / (upperband - lowerband + 1e-10)

        df['Volume'] = volume

        df_features = df.dropna()

        features_list = [
            'RSI',
            'WMA_close_2', 'WMA_high_2', 'WMA_low_2',
            'MACD', 'MACD_signal', 'OBV',
            'ADX', 'ADX_hist',
            'BB_Width',
            'BB_Percent',
            'Volume'
        ]

        return df_features, features_list
    except Exception as e:
        print(f"計算 5m 特徵時出錯: {e}")
        return None

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