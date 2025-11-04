import ccxt
import pandas as pd
import talib
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import numpy as np
import warnings

# 忽略 pandas 的未來警告 (非必需，但可保持輸出整潔)
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- 步驟 1: 使用 CCXT 獲取資料 ---

def fetch_data(symbol='BTC/USDT', timeframe='1d', limit=500):
    """
    從幣安 (Binance) 獲取 OHLCV 資料。
    """
    print(f"--- 步驟 1: 正在從 Binance 獲取 {symbol} {timeframe} 資料 (最近 {limit} 筆) ---")
    try:
        exchange = ccxt.binance({
            'rateLimit': 1200,
            'enableRateLimit': True,
        })
        
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        if not ohlcv:
            print("未獲取到資料。")
            return None

        df = pd.DataFrame(ohlcv, columns=[
            'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # 轉換為數值類型
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        print("資料獲取成功。")
        return df

    except ccxt.NetworkError as e:
        print(f"網路錯誤: {e}")
    except ccxt.ExchangeError as e:
        print(f"交易所錯誤: {e}")
    except Exception as e:
        print(f"獲取資料時發生錯誤: {e}")
        return None

# --- 步驟 2: 使用 TA-Lib 進行特徵工程 ---

def create_features(df):
    """
    計算技術指標作為特徵。
    """
    if df is None:
        return None
        
    print("\n--- 步驟 2: 正在計算技術指標 (特徵工程) ---")
    
    # 確保資料是 float 類型
    close_prices = df['Close'].values.astype(float)
    high_prices = df['High'].values.astype(float)
    low_prices = df['Low'].values.astype(float)
    volume = df['Volume'].values.astype(float)

    try:
        df['RSI'] = talib.RSI(close_prices, timeperiod=14)
        df['SMA_10'] = talib.SMA(close_prices, timeperiod=10)
        df['SMA_50'] = talib.SMA(close_prices, timeperiod=50)
        df['EMA_12'] = talib.EMA(close_prices, timeperiod=12)
        df['EMA_26'] = talib.EMA(close_prices, timeperiod=26)
        
        macd, macdsignal, _ = talib.MACD(close_prices, 
                                         fastperiod=12, 
                                         slowperiod=26, 
                                         signalperiod=9)
        df['MACD'] = macd
        df['MACD_signal'] = macdsignal
        
        df['OBV'] = talib.OBV(close_prices, volume)

        # 去除因計算指標 (如 SMA_50) 而產生的 NaN 值
        original_len = len(df)
        df_features = df.dropna()
        print(f"已去除 {original_len - len(df_features)} 筆舊資料 (因計算指標產生 NaN)。")
        
        return df_features

    except Exception as e:
        print(f"計算特徵時發生錯誤: {e}")
        return None

# --- 步驟 3 & 4: 訓練 XGBoost 並預測 ---

def train_and_predict(df_features):
    """
    準備資料、訓練 XGBoost 模型並預測下一天的價格。
    """
    if df_features is None or df_features.empty:
        print("沒有足夠的特徵資料進行訓練。")
        return

    print("\n--- 步驟 3: 準備資料並訓練 XGBoost 模型 ---")

    # (a) 定義特徵欄位
    features = [
        'RSI', 'SMA_10', 'SMA_50', 'EMA_12', 'EMA_26', 
        'MACD', 'MACD_signal', 'OBV'
    ]
    
    # (b) 創建 y (目標)：預測 '下一天' 的收盤價
    df_model = df_features.copy()
    df_model['target'] = df_model['Close'].shift(-1)

    # (c) 刪除最後一行 (因為它沒有 'target')
    df_model = df_model.dropna()

    X = df_model[features]
    y = df_model['target']

    # (d) 分割資料 (時間序列必須保持順序)
    test_size = 0.2
    if len(X) * test_size < 1:
        print("警告：資料太少，無法分割測試集。")
        split_index = len(X)
    else:
        split_index = int(len(X) * (1 - test_size))

    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    print(f"訓練集筆數: {len(X_train)}, 測試集筆數: {len(X_test)}")

    # (e) 訓練模型
    xgb_reg = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        objective='reg:squarederror',
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=50
    )

    if not X_test.empty:
        # 如果有測試集，使用 early stopping
        xgb_reg.fit(
            X_train, 
            y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False
        )
        
        # --- 評估模型 ---
        y_pred = xgb_reg.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"模型評估 (RMSE): {rmse:.2f} (預測平均誤差)")

    else:
        # 如果沒有測試集 (資料太少)，直接訓練
        print("資料量不足以進行 early stopping，直接訓練...")
        xgb_reg.fit(X_train, y_train)

    print("模型訓練完成。")

    # --- 步驟 4: 預測 '真正' 的明天 ---
    print("\n--- 步驟 4: 預測 '明天' 的價格 ---")
    
    # (a) 獲取 '今天' (資料中最後一筆) 的特徵
    # (注意：我們使用 'df_features'，即 *原始* 的、未 shift target 的 DataFrame)
    latest_features = df_features[features].iloc[-1:]

    print("用於預測的 '今天' (最新) 特徵:")
    print(latest_features)

    # (b) 進行預測
    prediction_for_tomorrow = xgb_reg.predict(latest_features)

    print("\n=========================================================")
    print(f"📈 預測 '明天' (下一根 K 線) 的 BTCUSDT 收盤價: ${prediction_for_tomorrow[0]:.2f}")
    print(f"(基於 '今天' 的收盤價: ${df_features['Close'].iloc[-1]:.2f})")
    print("=========================================================")


# --- 主執行流程 ---
if __name__ == "__main__":
    # 1. 獲取資料
    raw_df = fetch_data(symbol='BTC/USDT', timeframe='1d', limit=500)
    
    # 2. 特徵工程
    df_with_features = create_features(raw_df)
    
    # 3. 訓練與預測
    train_and_predict(df_with_features)