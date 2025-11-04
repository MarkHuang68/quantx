import ccxt
import pandas as pd
import talib
import numpy as np
import warnings
import matplotlib.pyplot as plt
import time

# --- 神經網路和資料處理 ---
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

# --- 匯入 class_weight 計算工具 ---
from sklearn.utils.class_weight import compute_class_weight

# 設置 Keras/Tensorflow 的隨機種子
tf.random.set_seed(42)
np.random.seed(42)

# 忽略 pandas 的未來警告
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- 步驟 1: 使用 CCXT 獲取資料 ---
# (與上一版相同，保持不變)
def fetch_data(symbol='BTC/USDT', timeframe='1h', total_limit=10000):
    """
    從幣安 (Binance) 獲取大量 OHLCV 資料 (使用迴圈)。
    """
    print(f"--- 步驟 1: 正在從 Binance 獲取 {symbol} {timeframe} 資料 (目標 {total_limit} 筆) ---")
    
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
    
    print(f"將從 {pd.to_datetime(since_timestamp, unit='ms')} (大約) 開始獲取資料...")

    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since_timestamp, limit=limit_per_request)
            if not ohlcv:
                print("獲取完成 (已達最新資料)。")
                break
            all_ohlcv.extend(ohlcv)
            last_timestamp = ohlcv[-1][0]
            since_timestamp = last_timestamp + timeframe_duration_ms
            print(f"已獲取 {len(all_ohlcv)} 筆資料...")
        except ccxt.NetworkError as e:
            print(f"網路錯誤: {e}，5 秒後重試...")
            time.sleep(5)
        except Exception as e:
            print(f"獲取資料時發生未知錯誤: {e}")
            return None

    print(f"--- 資料獲取完畢，總共 {len(all_ohlcv)} 筆 ---")

    if not all_ohlcv:
        print("最終未獲取到任何資料。")
        return None
        
    df = pd.DataFrame(all_ohlcv, columns=[
        'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'
    ])
    df = df.drop_duplicates(subset=['timestamp'])
    
    if len(df) > total_limit:
        df = df.tail(total_limit)
    elif len(df) < total_limit:
        print(f"警告：交易所提供的資料不足 {total_limit} 筆，僅有 {len(df)} 筆。")
        
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    print("DataFrame 處理完成。")
    return df

# --- 步驟 2: 1 小時「慢速」特徵 ---
# (與上一版相同，保持不變)
def create_features_1h(df):
    """
    計算適用於 1 小時圖的「標準/慢速」動能特徵。
    """
    if df is None:
        return None
        
    print("\n--- 步驟 2: 正在計算 1h「慢速」指標 ---")
    
    close_prices = df['Close'].values.astype(float)
    high_prices = df['High'].values.astype(float)
    low_prices = df['Low'].values.astype(float)
    volume = df['Volume'].values.astype(float)

    try:
        df['EMA'] = talib.EMA(close_prices, timeperiod=20)
        df['SMA'] = talib.SMA(close_prices, timeperiod=60)
        df['RSI_14'] = talib.RSI(close_prices, timeperiod=14)
        macd, macdsignal, _ = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
        df['MACD'] = macd
        df['MACD_signal'] = macdsignal
        df['ADX_14'] = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)
        adx_hist = talib.PLUS_DI(high_prices, low_prices, close_prices, timeperiod=14) - talib.MINUS_DI(high_prices, low_prices, close_prices, timeperiod=14)
        df['ADX_hist_14'] = talib.EMA(adx_hist, timeperiod=100) > 0
        df['OBV'] = talib.OBV(close_prices, volume)
        upperband, middleband, lowerband = talib.BBANDS(close_prices, timeperiod=10, nbdevup=2, nbdevdn=2, matype=0)
        df['BB_Width'] = (upperband - lowerband) / (middleband + 1e-10)
        df['BB_Percent'] = (close_prices - lowerband) / (upperband - lowerband + 1e-10)
        df['MOM_10'] = talib.MOM(close_prices, timeperiod=10)
        df['Volume'] = volume
        df['Maybe'] = (df['EMA'] > df['SMA']).astype(int)

        original_len = len(df)
        df_features = df.dropna() 
        print(f"已去除 {original_len - len(df_features)} 筆舊資料 (因計算指標產生 NaN)。")
        
        return df_features

    except Exception as e:
        print(f"計算特征時發生錯誤: {e}")
        return None

# --- 步驟 3: 建立 LSTM 序列 ---
# (與上一版相同，保持不變)
def create_sequences(data, target, lookback_window=24):
    """
    將 2D 特徵 轉換為 3D 序列資料 (samples, lookback, features)。
    """
    print(f"\n--- 步驟 3: 正在建立 3D 序列 (回看 {lookback_window} 根 K 棒)... ---")
    X = []
    y = []
    for i in range(lookback_window, len(data)):
        X.append(data[i-lookback_window:i, :])
        y.append(target[i])
        
    X, y = np.array(X), np.array(y)
    print(f"序列資料建立完成。X 形狀: {X.shape}, y 形狀: {y.shape}")
    return X, y

# --- 步驟 4: (***修改: 預測 24 小時後***) ---
def build_and_train_lstm(df_features, lookback_window=24, forecast_horizon=24):
    """
    標準化資料、建立序列、並訓練 LSTM 分類模型。
    """
    if df_features is None or df_features.empty:
        print("沒有足夠的特徵資料進行訓練。")
        return None, None, None

    # 1. 定義特徵和目標
    features = [
        'EMA',
        'SMA',
        'Maybe',
        'RSI_14',
        # 'MACD', 'MACD_signal',
        # 'ADX_14',
        # 'ADX_hist_14', 
        # 'OBV',
        'BB_Width', #'BB_Percent',
        # 'MOM_10',
        # 'Volume'
        # 'Close', 'High', 'Low', 'Open', 'Volume'
    ]
    
    df_model = df_features.copy()
    
    # --- 重大修改: 預測 'forecast_horizon' (24) 根 K 棒之後的漲跌 ---
    print(f"\n--- 正在建立目標: 預測 {forecast_horizon} 小時之後的走向 ---")
    df_model['target'] = (df_model['SMA'].shift(-forecast_horizon) > df_model['SMA']).astype(int)
    # ------------------------------------------------------------------
    
    df_model = df_model.dropna() # (這會移除最後 24 筆資料)
    
    # 2. 標準化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(df_model[features])
    target = df_model['target'].values
    
    # 3. 建立 3D 序列
    X_seq, y_seq = create_sequences(scaled_features, target, lookback_window)
    
    # 4. 分割資料
    test_size = 0.2 
    split_index = int(len(X_seq) * (1 - test_size))
    
    X_train, X_test = X_seq[:split_index], X_seq[split_index:]
    y_train, y_test = y_seq[:split_index], y_seq[split_index:]

    print(f"訓練集筆數: {len(X_train)}, 測試集筆數: {len(X_test)}")
    print(f"訓練集中 '漲 (1)' 的比例: {np.mean(y_train):.2%}")
    print(f"測試集中 '漲 (1)' 的比例: {np.mean(y_test):.2%}")

    # --- 新增: 計算類別權重 ---
    print("\n--- 正在計算類別權重 (Class Weights) ---")
    unique_classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=y_train
    )
    class_weight_dict = dict(zip(unique_classes, weights))
    print(f"計算出的權重: {class_weight_dict}")
    # --------------------------------

    # 5. 建立 Keras LSTM 模型
    # 5. (***重大修改: 建立「深度」Keras LSTM 模型***)
    print("\n--- 步驟 4: 正在建立「深度堆疊」LSTM 模型... ---")
    model = Sequential()
    
    # 第一層 (更寬): 100 個神經元, 必須 return_sequences=True
    model.add(Bidirectional(LSTM(
        units=100, 
        return_sequences=True, # <-- 告訴它後面還有 LSTM 層
        input_shape=(X_train.shape[1], X_train.shape[2]) # (lookback, features)
    )))
    model.add(Dropout(0.3)) # 增加 Dropout 防止過度擬合
    
    # 第二層 (更深): 50 個神經元
    model.add(Bidirectional(LSTM(
        units=50
        # (最後一層 LSTM, 不需要 return_sequences)
    )))
    model.add(Dropout(0.3))

    # 決策層 (更寬)
    model.add(Dense(units=50, activation='relu')) # 增加決策神經元
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(
        optimizer='adam', 
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()

    # 6. 訓練模型
    print("\n--- 正在訓練 LSTM 模型 (已加入 class_weight)... ---")
    history = model.fit(
        X_train, 
        y_train,
        epochs=50,
        batch_size=64,
        validation_data=(X_test, y_test),
        shuffle=False,
        # class_weight=class_weight_dict,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        ]
    )
    
    print("模型訓練完成。")

    # 7. 評估模型
    print("\n--- 步驟 5: 評估 1h LSTM 模型 (預測 24h 後) ---")
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n模型在「測試集」上的準確率 (Accuracy): {accuracy:.2%}")
    print("\n--- 詳細分類報告 (Classification Report) ---")
    print(classification_report(y_test, y_pred, target_names=['跌 (0)', '漲 (1)']))
    
    return model, X_test, y_test

# --- 步驟 6: 繪製混淆矩陣 ---
def plot_confusion_matrix(classifier, X_test, y_test):
    """
    繪製混淆矩陣 (Confusion Matrix)
    """
    print("\n--- 步驟 6: 正在繪製混淆矩陣 (Confusion Matrix) ---")
    
    try:
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.set_title('混淆矩陣 (Confusion Matrix) - 1h LSTM (預測 24h 後)')
        
        y_pred = (classifier.predict(X_test) > 0.5).astype(int)
        
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            ax=ax,
            cmap=plt.cm.Blues,
            display_labels=['實際 跌 (0)', '實際 漲 (1)']
        )
        
        ax.xaxis.set_ticklabels(['預測 跌 (0)', '預測 漲 (1)'])
        ax.yaxis.set_ticklabels(['實際 跌 (0)', '實際 漲 (1)'])
        
        print("正在顯示圖表... (請查看彈出視窗，可能在 Python 圖示)")
        plt.show()

    except Exception as e:
        print(f"繪製混淆矩陣時出錯: {e}")

# --- 主執行流程 (修改) ---
if __name__ == "__main__":
    
    # 1. 獲取資料
    raw_df = fetch_data(symbol='BTC/USDT', timeframe='1h', total_limit=10000)
    
    # 2. 特徵工程 (1h 慢速特徵)
    df_with_features = create_features_1h(raw_df)
    
    # 3. 訓練與預測 (LSTM, 回看 24 根 K 棒, 預測 24 根 K 棒)
    best_classifier, X_test_data, y_test_data = build_and_train_lstm(
        df_with_features, 
        lookback_window=24, 
        forecast_horizon=24
    )
    
    # 4. 繪製混淆矩陣
    if best_classifier:
        plot_confusion_matrix(best_classifier, X_test_data, y_test_data)