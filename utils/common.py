# 檔案: common.py
# 這裡是「唯一」的特徵工程函數 (V3 MTF DMI 黃金配方)
# 您的「訓練」和「實盤」腳本都會 100% 引用這裡

import os
import ccxt
import pandas as pd
import ta
import numpy as np
import time
import talib

# ⭐ 新增函數：抓取單一衍生品指標的歷史數據 (保持不變)
def fetch_derivative_data(exchange, symbol, timeframe, since_timestamp, end_timestamp, derivative_type, timeframe_ms):
    all_data = []
    current_timestamp = since_timestamp
    limit_per_request = 1000
    
    if derivative_type == 'funding_rate':
        fetch_method = exchange.fetch_funding_rate_history
    elif derivative_type == 'open_interest':
        return None 

    print(f"--- 正在獲取 {symbol} {derivative_type} 資料 ---")
    
    while current_timestamp < end_timestamp:
        try:
            if derivative_type == 'funding_rate':
                data = fetch_method(symbol, since=current_timestamp, limit=limit_per_request)
            else:
                return None 
            
            if not data: break
            
            all_data.extend(data)
            last_timestamp = data[-1]['timestamp']
            current_timestamp = last_timestamp + timeframe_ms # 下一根 K 棒
            
            if current_timestamp >= end_timestamp: break
            time.sleep(1)
            
        except Exception as e:
            print(f"獲取 {derivative_type} 錯誤: {e}")
            break

    if not all_data: return None
    
    if derivative_type == 'funding_rate':
        df = pd.DataFrame(all_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.rename(columns={'fundingRate': 'FundingRate'}, inplace=True)
        return df[['timestamp', 'FundingRate']].set_index('timestamp')
        
    return None

def fetch_data(symbol, timeframe, start_date=None, end_date=None, total_limit=None):
    """ 獲取 OHLCV 和衍生品資料（含快取邏輯）。"""
    print(f"--- 正在獲取 {symbol} {timeframe} 資料 ---")
    symbol_safe = symbol.replace('/', '')
    
    # --- 計算時間範圍 ---
    exchange = ccxt.bybit({'rateLimit': 1200, 'enableRateLimit': True})
    timeframe_ms = exchange.parse_timeframe(timeframe) * 1000
    limit_per_request = 1000
    
    end_timestamp = int(pd.to_datetime(end_date).timestamp() * 1000) if end_date else exchange.milliseconds()
    
    if start_date:
        since_timestamp = int(pd.to_datetime(start_date).timestamp() * 1000)
    else:
        since_timestamp = end_timestamp - (total_limit * timeframe_ms) if total_limit else None

    start_date_str = pd.to_datetime(since_timestamp, unit='ms').strftime('%Y-%m-%d') if since_timestamp else 'start'
    end_date_str = pd.to_datetime(end_timestamp, unit='ms').strftime('%Y-%m-%d')
    
    # --- 1. OHLCV 抓取 ---
    cache_file_ohlcv = os.path.join('data', f"{symbol_safe}_{timeframe}_{start_date_str}_{end_date_str}.csv")
    
    if os.path.exists(cache_file_ohlcv):
        print(f"✅ 找到快取 {cache_file_ohlcv}，載入中...")
        df = pd.read_csv(cache_file_ohlcv, index_col='timestamp', parse_dates=True)
    else:
        all_ohlcv = []
        current_since = since_timestamp
        
        while True:
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=limit_per_request)
                if not ohlcv: break
                all_ohlcv.extend(ohlcv)
                last_timestamp = ohlcv[-1][0]
                current_since = last_timestamp + timeframe_ms
                if last_timestamp >= end_timestamp: break
                time.sleep(1)
            except Exception as e:
                print(f"獲取 OHLCV 錯誤: {e}")
                time.sleep(5)
                break
        
        if not all_ohlcv: return None
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df = df.drop_duplicates(subset=['timestamp'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df[df.index <= pd.to_datetime(end_timestamp, unit='ms')]
        
        print(f"✅ 保存 OHLCV 快取至 {cache_file_ohlcv}")
        df.to_csv(cache_file_ohlcv)
    
    # --- 2. 衍生品抓取 (資金費率) ---
    df_funding = fetch_derivative_data(exchange, symbol, timeframe, since_timestamp, end_timestamp, 'funding_rate', timeframe_ms)
    
    if df_funding is not None:
        df_funding_resampled = df_funding.resample(timeframe).ffill()
        df = df.join(df_funding_resampled, how='left')
        df['FundingRate'] = df['FundingRate'].ffill() 
        
    print("DataFrame 處理完成。")
    return df

# -------------------------- 【V3 MTF DMI 特徵開始】 --------------------------

def calculate_dmi_hist(df, len_di=14, len_ma=10, len_adx_smooth=14):
    """計算 DMI 柱體、MA 線和 ADX 數值，並加入長度檢查。"""
    df_temp = df.copy()
    if len(df_temp) < len_di + 30: 
        for col in ['histValue', 'histMa', 'histMa_Up', 'histMa_BelowZero', 'ADX']: df_temp[col] = np.nan
        return df_temp
    
    plus_di  = talib.PLUS_DI(df_temp['High'], df_temp['Low'], df_temp['Close'], timeperiod=len_di)
    minus_di = talib.MINUS_DI(df_temp['High'], df_temp['Low'], df_temp['Close'], timeperiod=len_di)
    adx_val  = talib.ADX(df_temp['High'], df_temp['Low'], df_temp['Close'], timeperiod=len_di)
    
    df_temp['histValue'] = plus_di - minus_di
    alpha = 1 / len_ma
    df_temp['histMa'] = df_temp['histValue'].ewm(alpha=alpha, adjust=False).mean()
    df_temp['histMa_Up'] = (df_temp['histMa'] >= df_temp['histMa'].shift(1)).astype(float) 
    df_temp['histMa_BelowZero'] = (df_temp['histMa'] < 0).astype(float)
    df_temp['ADX'] = adx_val
    return df_temp

def create_mtf_dmi_features(df_15m_raw, DMI_DI_LEN=14, **feature_params):
    """產生跨時間框架的 DMI 趨勢特徵並合併到 15M K 線上 (含 ADX)。"""
    df_15m = df_15m_raw.copy()
    
    # 重新採樣數據
    df_4h = df_15m_raw.resample('4H').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
    df_1h = df_15m_raw.resample('1H').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
    
    # 計算 MTF 特徵
    df_4h = calculate_dmi_hist(df_4h, DMI_DI_LEN, feature_params.get('dmi_len_ma_4h', 10))
    df_1h = calculate_dmi_hist(df_1h, DMI_DI_LEN, feature_params.get('dmi_len_ma_1h', 10))
    df_15m = calculate_dmi_hist(df_15m, DMI_DI_LEN, feature_params.get('dmi_len_ma_15m', 10))

    # 合併 4H 特徵
    df_15m = df_15m.merge(
        df_4h[['histMa', 'histMa_Up', 'histMa_BelowZero', 'ADX']].shift(1).rename(
            columns={'histMa': '4H_Ma', 'histMa_Up': '4H_Ma_Up', 'histMa_BelowZero': '4H_Ma_DownTrend', 'ADX': '4H_ADX'}
        ), left_index=True, right_index=True, how='left'
    ).ffill()
    df_15m['4H_Ma_Up'] = df_15m['4H_Ma_Up'].astype(float)
    df_15m['4H_Ma_DownTrend'] = df_15m['4H_Ma_DownTrend'].astype(float)
    
    # 合併 1H 特徵
    df_15m = df_15m.merge(
        df_1h[['histMa', 'histMa_Up', 'histMa_BelowZero', 'ADX']].shift(1).rename(
            columns={'histMa': '1H_Ma', 'histMa_Up': '1H_Ma_Up', 'histMa_BelowZero': '1H_Ma_DownTrend', 'ADX': '1H_ADX'}
        ), left_index=True, right_index=True, how='left'
    ).ffill()
    df_15m['1H_Ma_Up'] = df_15m['1H_Ma_Up'].astype(float)
    df_15m['1H_Ma_DownTrend'] = df_15m['1H_Ma_DownTrend'].astype(float)
    
    df_15m['15M_Ma_Delta'] = df_15m['histMa'] - df_15m['histMa'].shift(1)

    mtf_features = [
        '4H_Ma', '4H_Ma_Up', '4H_Ma_DownTrend', '4H_ADX',
        '1H_Ma', '1H_Ma_Up', '1H_Ma_DownTrend', '1H_ADX',
        'histValue', 'histMa', 'histMa_Up', '15M_Ma_Delta', 'ADX',
    ]
    return df_15m, mtf_features

def create_features_trend(df_raw, **feature_params):
    """ (V3 模型) 產生所有特徵並合併 MTF DMI 邏輯。 """
    if df_raw is None: return None, None
    df = df_raw.copy()
    
    # 1. 執行 MTF DMI 特徵計算
    try:
        df, mtf_features = create_mtf_dmi_features(df, **feature_params)
    except Exception as e:
        print(f"❌ MTF DMI 特徵計算失敗: {e}")
        mtf_features = []

    # 2. 傳統/衍生品特徵計算
    close_prices = df['Close']
    
    # EMA 結構特徵 (用於捕捉反轉風險)
    ema_s, ema_m, ema_l = talib.EMA(close_prices, 10), talib.EMA(close_prices, 30), talib.EMA(close_prices, 60)
    df['CLOSE_EMA_L'] = (close_prices - ema_l) / ema_l
    df['EMA_M_EMA_L'] = (ema_m - ema_l) / ema_l
    
    # RSI
    df['RSI'] = talib.RSI(close_prices, timeperiod=feature_params.get('rsi_period', 14))
    
    # 衍生品特徵 (FundingRate)
    if 'FundingRate' in df.columns:
        df['FR_ROC'] = df['FundingRate'].pct_change().replace([np.inf, -np.inf], np.nan)
        df['FR_ABS'] = df['FundingRate'] * 1000 
        # 處理 NaN，防止影響特徵計算
        df['FR_ROC'] = df['FR_ROC'].fillna(0.0) 
        df['FR_ABS'] = df['FR_ABS'].fillna(0.0)
    else:
        df['FR_ROC'], df['FR_ABS'] = 0.0, 0.0

    # 3. 組合最終特徵列表
    features_list = [
        *mtf_features, 
        'CLOSE_EMA_L', 'EMA_M_EMA_L', 'RSI', 
        'FR_ROC', 'FR_ABS'
    ]
    
    features_list = list(set(features_list))
    df_features = df.dropna(subset=features_list)
    return df_features, features_list

# -------------------------- 【V3 MTF DMI 特徵結束】 --------------------------

def convert_symbol_to_ccxt(symbol: str) -> str:
    if '/' in symbol: return symbol
    base = symbol.replace('USDT', '')
    return f"{base}/USDT:USDT"

def convert_symbol_from_ccxt(ccxt_symbol: str) -> str:
    return ccxt_symbol.split('/')[0]