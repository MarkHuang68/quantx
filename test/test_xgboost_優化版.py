import ccxt
import pandas as pd
import talib
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import numpy as np
import warnings
import matplotlib.pyplot as plt
import time # --- 新增: 處理網路延遲 ---

# 忽略 pandas 的未來警告
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- 步驟 1: 使用 CCXT 獲取資料 (***重大更新***) ---

def fetch_data(symbol='BTC/USDT', timeframe='5m', total_limit=10000):
    """
    從幣安 (Binance) 獲取大量 OHLCV 資料 (使用迴圈)。
    """
    print(f"--- 步驟 1: 正在從 Binance 獲取 {symbol} {timeframe} 資料 (目標 {total_limit} 筆) ---")
    
    # 1. 初始化交易所
    exchange = ccxt.binance({'rateLimit': 1200, 'enableRateLimit': True})
    
    # 2. 計算時間
    try:
        # 將 '5m', '1h' 等轉換為毫秒
        timeframe_duration_ms = exchange.parse_timeframe(timeframe) * 1000
    except Exception as e:
        print(f"Timeframe 格式錯誤: {e}。請使用 1m, 3m, 5m, 15m, 1h, 4h, 1d...")
        return None
        
    limit_per_request = 1000  # 幣安 API 每次請求的上限
    all_ohlcv = []

    # 3. 計算起始時間 (從多久以前開始抓)
    # 總筆數 * 每筆 K 棒的時間 = 總時長
    total_duration_ms = total_limit * timeframe_duration_ms
    since_timestamp = exchange.milliseconds() - total_duration_ms
    
    print(f"將從 {pd.to_datetime(since_timestamp, unit='ms')} (大約) 開始獲取資料...")

    # 4. 迴圈獲取資料 (向前獲取，直到「現在」)
    while True:
        try:
            # 獲取 K 線 (從 'since_timestamp' 開始)
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since_timestamp, limit=limit_per_request)
            
            if not ohlcv:
                # 沒有更多資料了 (已抓到最新)
                print("獲取完成 (已達最新資料)。")
                break
                
            all_ohlcv.extend(ohlcv)
            
            # 更新下一次迴圈的 'since' (從最後一根 K 棒的時間戳 + 1 開始)
            last_timestamp = ohlcv[-1][0]
            since_timestamp = last_timestamp + timeframe_duration_ms
            
            print(f"已獲取 {len(all_ohlcv)} 筆資料...")

        except ccxt.NetworkError as e:
            print(f"網路錯誤: {e}，5 秒後重試...")
            time.sleep(5) # 等待 5 秒
        except ccxt.ExchangeError as e:
            print(f"交易所錯誤: {e}")
            return None
        except Exception as e:
            print(f"獲取資料時發生未知錯誤: {e}")
            return None

    print(f"--- 資料獲取完畢，總共 {len(all_ohlcv)} 筆 ---")

    # 5. 轉換為 DataFrame
    if not all_ohlcv:
        print("最終未獲取到任何資料。")
        return None
        
    df = pd.DataFrame(all_ohlcv, columns=[
        'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'
    ])
    
    # 移除重複 (以防 API 錯誤)
    df = df.drop_duplicates(subset=['timestamp'])
    
    # 6. (重要!) 裁剪為最新的 N 筆
    # 因為我們是從 "過去" 抓到 "現在"，資料量可能多於 total_limit
    if len(df) > total_limit:
        print(f"資料量過多 ({len(df)})，將裁剪為最新的 {total_limit} 筆。")
        df = df.tail(total_limit)
    elif len(df) < total_limit:
        print(f"警告：交易所提供的資料不足 {total_limit} 筆，僅有 {len(df)} 筆。")
        
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    print("DataFrame 處理完成。")
    return df

# --- 步驟 2: 使用 TA-Lib 進行特徵工程 ---
# (與您上一版相同，保持不變)
def create_features(df):
    """
    計算技術指標作為特徵。
    """
    if df is None:
        return None
        
    print("\n--- 步驟 2: 正在計算技術指標 (特徵工程) ---")
    
    close_prices = df['Close'].values.astype(float)
    high_prices = df['High'].values.astype(float)
    low_prices = df['Low'].values.astype(float)
    volume = df['Volume'].values.astype(float)

    try:
        # 您的指標 (WMA + 快速 MACD + 快速 BBANDS)
        df['RSI'] = talib.RSI(close_prices, timeperiod=14)
        df['WMA_close_2'] = talib.WMA(close_prices, timeperiod=2)
        df['WMA_high_2'] = talib.WMA(high_prices, timeperiod=2)
        df['WMA_low_2'] = talib.WMA(low_prices, timeperiod=2)
        
        macd, macdsignal, _ = talib.MACD(close_prices, 
                                         fastperiod=6, 
                                         slowperiod=13, 
                                         signalperiod=9)
        df['MACD'] = macd
        df['MACD_signal'] = macdsignal

        df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'])
        df['ADX_hist'] = talib.PLUS_DI(df['High'], df['Low'], df['Close']) - talib.MINUS_DI(df['High'], df['Low'], df['Close'])
        
        df['OBV'] = talib.OBV(close_prices, volume)

        upperband, middleband, lowerband = talib.BBANDS(close_prices, 
                                                        timeperiod=2, 
                                                        nbdevup=2, 
                                                        nbdevdn=2, 
                                                        matype=0)
        
        df['BB_Width'] = (upperband - lowerband) / (middleband + 1e-10)
        df['BB_Percent'] = (close_prices - lowerband) / (upperband - lowerband + 1e-10)
        # --------------------------------

        original_len = len(df)
        df_features = df.dropna() 
        print(f"已去除 {original_len - len(df_features)} 筆舊資料 (因計算指標產生 NaN)。")
        
        return df_features

    except Exception as e:
        print(f"計算特徵時發生錯誤: {e}")
        return None

# --- 步驟 3 & 4: 訓練 XGBoost 並預測 ---
# (與您上一版相同，保持不變)
def train_and_predict(df_features):
    """
    準備資料、訓練 XGBoost 模型並預測下一天的價格。
    """
    if df_features is None or df_features.empty:
        print("沒有足夠的特徵資料進行訓練。")
        return None, None

    print("\n--- 步驟 3: 準備資料並訓練 XGBoost 模型 ---")

    features = [
        'RSI',
        'WMA_close_2', 'WMA_high_2', 'WMA_low_2',
        'MACD', 'MACD_signal', 'OBV',
        'ADX', 'ADX_hist',
        'Volume',
        'BB_Width',
        'BB_Percent'
    ]
    
    df_model = df_features.copy()
    df_model['target'] = df_model['Close'].shift(-1)
    df_model = df_model.dropna()

    X = df_model[features]
    y = df_model['target']

    # (重要) 測試集比例仍然是 20%。
    # 10,000 筆資料 -> 8,000 筆訓練，2,000 筆測試
    # 您的回測將會「長得多」也「可靠得多」
    test_size = 0.2 
    split_index = int(len(X) * (1 - test_size))

    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    print(f"訓練集筆數: {len(X_train)}, 測試集筆數: {len(X_test)}") # <--- 這裡會顯示新數字

    xgb_reg = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        objective='reg:squarederror',
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=50
    )

    y_test_data = None
    y_pred_data = None

    if not X_test.empty:
        xgb_reg.fit(
            X_train, 
            y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False
        )
        
        y_pred = xgb_reg.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"模型評估 (RMSE): {rmse:.2f} (預測平均誤差)")

        y_test_data = y_test
        y_pred_data = pd.Series(y_pred, index=y_test.index, name="Predicted")

    else:
        print("資料量不足以進行 early stopping，直接訓練...")
        xgb_reg.fit(X_train, y_train)

    print("模型訓練完成。")

    # --- 步驟 4: 預測 '真正' 的明天 ---
    print("\n--- 步驟 4: 預測 '明天' 的價格 ---")
    
    latest_features = df_features[features].iloc[-1:]
    print("用於預測的 '今天' (最新) 特徵:")
    print(latest_features)

    prediction_for_tomorrow = xgb_reg.predict(latest_features)

    print("\n=========================================================")
    print(f"📈 預測 '明天' (下一根 K 線) 的 BTCUSDT 收盤價: ${prediction_for_tomorrow[0]:.2f}")
    print(f"(基於 '今天' 的收盤價: ${df_features['Close'].iloc[-1]:.2f})")
    print("=========================================================")
    
    return y_test_data, y_pred_data

# --- 步驟 5: 繪製回測圖表 ---
# (與您上一版相同，保持不變)
def plot_backtest(actual, predicted):
    """
    使用 matplotlib 繪製真實價格與預測價格。
    """
    if actual is None or predicted is None:
        print("\n沒有回測資料可供繪圖 (測試集可能為空)。")
        return

    print("\n--- 步驟 5: 正在繪製回測結果 ---")
    
    plt.figure(figsize=(15, 7))
    
    plt.plot(actual.index, actual, label='Actual Price (真實價格)', color='blue', alpha=0.8)
    plt.plot(predicted.index, predicted, label='Predicted Price (預測價格)', color='red', linestyle='--', alpha=0.9)
    
    plt.title('XGBoost Backtest on BTCUSDT (測試集回測)')
    plt.xlabel('Date (日期)')
    plt.ylabel('Price (USDT)')
    plt.legend()
    plt.grid(True)
    
    print("正在顯示圖表... (請查看彈出視窗，可能在 Python 圖示)")
    plt.show()

# --- 主執行流程 (***重大更新***) ---
if __name__ == "__main__":
    
    # 1. 獲取資料
    # (修改 timeframe 和 total_limit 參數)
    # 抓取 10,000 筆 5 分鐘 K 線 (約 34.7 天的資料)
    raw_df = fetch_data(symbol='ETH/USDT', timeframe='5m', total_limit=10000)
    
    # (您可以嘗試抓更多，例如 1 小時 K 線)
    # raw_df = fetch_data(symbol='BTC/USDT', timeframe='1h', total_limit=10000)
    
    # 2. 特徵工程
    df_with_features = create_features(raw_df)
    
    # 3. 訓練與預測 (並接收回測資料)
    actual_prices, predicted_prices = train_and_predict(df_with_features)
    
    # 4. 繪製回測圖表
    plot_backtest(actual_prices, predicted_prices)