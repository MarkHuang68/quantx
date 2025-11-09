# 檔案: common_utils.py
# 這裡是「唯一」的特徵工程函數 (您「寫死」的黃金配方)
# 您的「訓練」和「實盤」腳本都會 100% 引用這裡

import os
import ccxt
import pandas as pd
import ta
import numpy as np
import time
import talib

def fetch_data(symbol, timeframe, start_date=None, end_date=None, total_limit=None):
    """ 獲取 OHLCV 資料（含快取邏輯）。
    - 若 start_date 及 end_date 皆設，先查本地 data/ CSV。
    - 無則抓取並存至 data/{symbol_safe}_{timeframe}_{start_date}_{end_date}.csv。
    - 確保 data/ 存在。
    """
    print(f"--- 正在獲取 {symbol} {timeframe} 資料 ---")
    symbol_safe = symbol.replace('/', '')  # 安全檔名，如 ETHUSDT
    
    cache_file = None
    if start_date and end_date:
        cache_dir = 'data'
        os.makedirs(cache_dir, exist_ok=True)  # 確保目錄存在
        cache_file = os.path.join(cache_dir, f"{symbol_safe}_{timeframe}_{start_date}_{end_date}.csv")
        
        if os.path.exists(cache_file):
            print(f"✅ 找到快取 {cache_file}，載入中...")
            df = pd.read_csv(cache_file, index_col='timestamp', parse_dates=True)
            print("DataFrame 從快取載入完成。")
            return df

    exchange = ccxt.bybit({'rateLimit': 1200, 'enableRateLimit': True})
    timeframe_ms = exchange.parse_timeframe(timeframe) * 1000
    limit_per_request = 1000
    all_ohlcv = []
    
    if start_date:
        since_timestamp = int(pd.to_datetime(start_date).timestamp() * 1000)
    else:
        since_timestamp = exchange.milliseconds() - (total_limit * timeframe_ms) if total_limit else None
    
    end_timestamp = int(pd.to_datetime(end_date).timestamp() * 1000) if end_date else None
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since_timestamp, limit=limit_per_request)
            if not ohlcv: break
            all_ohlcv.extend(ohlcv)
            last_timestamp = ohlcv[-1][0]
            since_timestamp = last_timestamp + timeframe_ms
            if end_timestamp and last_timestamp >= end_timestamp: break
            time.sleep(1)  # 避免 rate limit
        except Exception as e:
            print(f"獲取錯誤: {e}")
            time.sleep(5)
    
    if not all_ohlcv: return None
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df = df.drop_duplicates(subset=['timestamp'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    if end_timestamp:
        df = df[df.index <= pd.to_datetime(end_timestamp, unit='ms')]
    
    if total_limit and len(df) > total_limit:
        df = df.tail(total_limit)
    
    if cache_file:
        print(f"✅ 保存快取至 {cache_file}")
        df.to_csv(cache_file)
    
    print("DataFrame 處理完成。")
    return df

def create_features_trend(df):
    """
    (模型 A) 進場特徵 (報酬率預測模型)。
    """
    if df is None:
        return None, None
        
    # print("\n--- 正在計算「進場特徵」---")

    close_prices = df['Close']
    high_prices = df['High']
    low_prices = df['Low']
    volume = df['Volume']

    try:
        ema_s = talib.EMA(close_prices, timeperiod=8)
        ema_m = talib.EMA(close_prices, timeperiod=20)
        ema_l = talib.EMA(close_prices, timeperiod=60)

        df['HOUR'] = df.index.hour
        df['D_OF_W'] = df.index.dayofweek
        df['EMA_S'] = ema_s
        df['EMA_M'] = ema_m
        df['EMA_L'] = ema_l
        df['CLOSE_EMA_S'] = (close_prices - ema_s) / ema_s
        df['CLOSE_EMA_M'] = (close_prices - ema_m) / ema_m
        df['CLOSE_EMA_L'] = (close_prices - ema_l) / ema_l
        df['EMA_S_EMA_M'] = (ema_s - ema_m) / ema_m
        df['EMA_M_EMA_L'] = (ema_m - ema_l) / ema_l
        df['TREND'] = (ema_s > ema_m) & (ema_m > ema_l)
        df['TREND1'] = (ema_s < ema_m) & (ema_m > ema_l)
        df['TREND2'] = (ema_s < ema_m) & (ema_m < ema_l)
        df['TREND3'] = (ema_s > ema_m) & (ema_m < ema_l)
        df['ATR'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
        df['RSI'] = talib.RSI(close_prices, timeperiod=14)
        df['MOM'] = talib.MOM(close_prices, timeperiod=10)
        df['ADX'] = talib.ADX(high_prices, low_prices, close_prices)
        df['ADX_hist'] = talib.PLUS_DI(high_prices, low_prices, close_prices) - talib.MINUS_DI(high_prices, low_prices, close_prices)
        df['ADX_hist_ema'] = talib.EMA(df['ADX_hist'], 10)
        df['WILLR'] = talib.WILLR(high_prices, low_prices, close_prices, timeperiod=20)

        df['KDJ_K'], df['KDJ_D'] = talib.STOCH(
                                    high_prices, 
                                    low_prices, 
                                    close_prices, 
                                    fastk_period=9,
                                    slowk_period=3,
                                    slowk_matype=0, # 設為 0 使用 SMA
                                    slowd_period=3,
                                    slowd_matype=0  # 設為 0 使用 SMA
                                    )
        df['KDJ_J'] = (3 * df['KDJ_K']) - (2 * df['KDJ_D'])
        
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(close_prices, 
                                                                    fastperiod=12, 
                                                                    slowperiod=26, 
                                                                    signalperiod=9)

        upperband, middleband, lowerband = talib.BBANDS(close_prices, 
                                                        timeperiod=20, 
                                                        nbdevup=2, 
                                                        nbdevdn=2, 
                                                        matype=0)
        
        df['BB_Width'] = (upperband - lowerband) / (middleband + 1e-10)
        df['BB_Percent'] = (close_prices - lowerband) / (upperband - lowerband + 1e-10)

        df['OBV'] = talib.OBV(close_prices, volume)
        df['VOLUME_CHANGE'] = df['Volume'].pct_change()
        df['VOLUME_CHANGE'] = df['VOLUME_CHANGE'].replace([np.inf, -np.inf], np.nan)

        # 以下是幫助ML學習忽略抄襲

        # 平穩化: 用對數 Close (轉換價格為對數，使序列更平穩，減低極端波動影響，幫助模型捕捉百分比變化)
        df['log_close'] = np.log(df['Close'])

        # 加滯後特徵 (加入1-5期滯後報酬，捕捉時間依賴，讓模型學習序列模式，避免只抄當前值)
        for lag in range(1, 6):
            df[f'lag_return_{lag}'] = df['Close'].pct_change().shift(lag)

        # 加波動率 (計算14期標準差，測量價格波動，提供風險信號，幫助預測轉折或趨勢強度)
        df['volatility'] = df['Close'].rolling(14).std()

        # --- (特徵列表) ---
        features_list = [
            # 'HOUR',
            # 'D_OF_W',

            # 'EMA_S',
            # 'EMA_M',
            # 'EMA_L',
            # 'CLOSE_EMA_S',
            # 'CLOSE_EMA_M',
            'CLOSE_EMA_L',
            # 'EMA_S_EMA_M',
            # 'EMA_M_EMA_L',
            # 'TREND',
            # 'TREND1',
            # 'TREND2',
            # 'TREND3',
            # 'ATR',
            'RSI',
            'MOM',
            # 'ADX',
            # 'ADX_hist',
            # 'ADX_hist_ema',
            # 'WILLR',
            # 'KDJ_K',
            # 'KDJ_D',
            # 'KDJ_J',

            'MACD', 'MACD_signal',
            'MACD_hist',
            
            'BB_Width',
            'BB_Percent',

            # 'OBV',
            # 'Volume',
            # 'VOLUME_CHANGE',

            # 'log_close',
            # 'lag_return_1',
            # 'lag_return_2',
            # 'lag_return_3',
            # 'lag_return_4',
            # 'lag_return_5',
            # 'volatility'
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
