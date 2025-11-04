import ccxt
import pandas as pd
import talib
import xgboost as xgb
# --- 新增: 匯入分類評估工具 ---
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
# -----------------------------
import numpy as np
import warnings
import matplotlib.pyplot as plt
import time

# --- 新增: 匯入調校工具 ---
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from scipy.stats import uniform, randint
# --------------------------

# 忽略 pandas 的未來警告
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- 步驟 1: 使用 CCXT 獲取資料 ---
# (與上一版相同，保持不變)
def fetch_data(symbol='BTC/USDT', timeframe='5m', total_limit=10000):
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

# --- 步驟 2: 使用 TA-Lib 進行特徵工程 ---
# (與上一版相同，保持不變)
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
        # 您的 5m 極速指標
        df['RSI'] = talib.RSI(close_prices, timeperiod=5)
        # df['WMA_close_2'] = talib.WMA(close_prices, timeperiod=2)
        # df['WMA_high_2'] = talib.WMA(high_prices, timeperiod=2)
        # df['WMA_low_2'] = talib.WMA(low_prices, timeperiod=2)
        macd, macdsignal, _ = talib.MACD(close_prices, fastperiod=6, slowperiod=13, signalperiod=9)
        df['MACD'] = macd
        df['MACD_signal'] = macdsignal
        df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=5)
        df['ADX_hist'] = talib.PLUS_DI(df['High'], df['Low'], df['Close'], timeperiod=5) - talib.MINUS_DI(df['High'], df['Low'], df['Close'],timeperiod=5)
        df['OBV'] = talib.OBV(close_prices, volume)
        upperband, middleband, lowerband = talib.BBANDS(close_prices, timeperiod=2, nbdevup=2, nbdevdn=2, matype=0)
        df['BB_Width'] = (upperband - lowerband) / (middleband + 1e-10)
        # df['BB_Percent'] = (close_prices - lowerband) / (upperband - lowerband + 1e-10)
        df['MOM'] = talib.MOM(close_prices, timeperiod=5)
        # --------------------------------
        
        original_len = len(df)
        df_features = df.dropna() 
        print(f"已去除 {original_len - len(df_features)} 筆舊資料 (因計算指標產生 NaN)。")
        
        return df_features

    except Exception as e:
        print(f"計算特征時發生錯誤: {e}")
        return None

# --- 步驟 3 & 4: (***重大更新: 改為分類模型***) ---

def train_and_predict(df_features):
    """
    準備資料、自動調校並訓練 XGBoost 分類模型。
    """
    if df_features is None or df_features.empty:
        print("沒有足夠的特徵資料進行訓練。")
        return

    print("\n--- 步驟 3: 準備資料 (分類) ---")

    features = [
        'RSI',
        # 'WMA_close_2', 'WMA_high_2', 'WMA_low_2',
        'MACD', 'MACD_signal', 'OBV',
        'ADX', 'ADX_hist',
        'Volume',
        'BB_Width',
        # 'BB_Percent',
        'MOM'
    ]
    
    df_model = df_features.copy()
    
    # --- 修改: 建立分類目標 (target) ---
    # 預測下一根 K 棒是漲 (1) 還是跌 (0)
    df_model['target'] = (df_model['Close'].shift(-1) > df_model['Close']).astype(int)
    # ----------------------------------
    
    df_model = df_model.dropna()

    X = df_model[features]
    y = df_model['target']

    test_size = 0.2 
    split_index = int(len(X) * (1 - test_size))

    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    print(f"訓練集筆數: {len(X_train)}, 測試集筆數: {len(X_test)}")
    print(f"訓練集中 '漲 (1)' 的比例: {y_train.mean():.2%}")
    print(f"測試集中 '漲 (1)' 的比例: {y_test.mean():.2%}")

    # --- 步驟 3a: 超參數調校設定 (分類) ---
    
    param_dist = {
        'n_estimators': randint(500, 1500),
        'learning_rate': uniform(0.01, 0.05),
        'max_depth': randint(3, 8),
        'subsample': uniform(0.7, 0.3),
        'colsample_bytree': uniform(0.7, 0.3)
    }

    # --- 修改: 更換為 XGBClassifier ---
    xgb_clf_base = xgb.XGBClassifier(
        objective='binary:logistic', # <-- 修改
        eval_metric='logloss',       # <-- 修改
        n_jobs=-1,
        random_state=42
    )
    # ---------------------------------

    tscv = TimeSeriesSplit(n_splits=3)

    # --- 修改: 評分標準改為 'accuracy' ---
    random_search = RandomizedSearchCV(
        estimator=xgb_clf_base,
        param_distributions=param_dist,
        n_iter=25,
        cv=tscv,
        scoring='accuracy', # <-- 修改
        n_jobs=-1,
        verbose=2,
        random_state=42
    )
    # -----------------------------------

    print("\n--- 步驟 3b: 開始超參數調校 (分類模型，這會花費 5-15+ 分鐘...) ---")
    
    random_search.fit(X_train, y_train)

    print("\n--- 調校完成! ---")
    print(f"最佳交叉驗證 (CV) 準確率: {random_search.best_score_:.2%}")
    print("找到的最佳參數組合:")
    print(random_search.best_params_)

    # 獲取「最佳分類器」
    xgb_clf = random_search.best_estimator_

    print("\n--- 步驟 3c: 使用「最佳模型」評估測試集 ---")
    
    # --- 修改: 評估分類結果 ---
    y_pred = xgb_clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n模型在「測試集」上的準確率 (Accuracy): {accuracy:.2%}")
    print("\n--- 詳細分類報告 (Classification Report) ---")
    # 這會顯示 '0' (跌) 和 '1' (漲) 的精確度 (Precision) 和召回率 (Recall)
    print(classification_report(y_test, y_pred, target_names=['跌 (0)', '漲 (1)']))
    # ---------------------------

    # --- 步驟 4: 預測 '真正' 的明天 (下一根 K 棒) ---
    print("\n--- 步驟 4: 預測 '明天' 的方向 ---")
    
    latest_features = df_features[features].iloc[-1:]
    print("用於預測的 '今天' (最新) 特徵:")
    print(latest_features)

    prediction_for_tomorrow = xgb_clf.predict(latest_features)
    prediction_proba = xgb_clf.predict_proba(latest_features)

    direction = "漲 (1)" if prediction_for_tomorrow[0] == 1 else "跌 (0)"
    confidence = prediction_proba[0][prediction_for_tomorrow[0]]

    print("\n=========================================================")
    print(f"📈 預測 '明天' (下一根 K 線) 的方向: {direction}")
    print(f"   (模型對此預測的信心指數: {confidence:.2%})")
    print("=========================================================")
    
    # --- 修改: 返回分類器和測試資料，用於繪製混淆矩陣 ---
    return xgb_clf, X_test, y_test

# --- (舊的 plot_backtest 已刪除) ---

# --- 新增: 步驟 5: 繪製混淆矩陣 ---
def plot_confusion_matrix(classifier, X_test, y_test):
    """
    繪製混淆矩陣 (Confusion Matrix)
    """
    print("\n--- 步驟 5: 正在繪製混淆矩陣 (Confusion Matrix) ---")
    
    try:
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.set_title('混淆矩陣 (Confusion Matrix)')
        
        # 繪製
        ConfusionMatrixDisplay.from_estimator(
            classifier, 
            X_test, 
            y_test,
            ax=ax,
            cmap=plt.cm.Blues,
            display_labels=['實際 跌 (0)', '實際 漲 (1)']
        )
        
        # 調整標籤
        ax.xaxis.set_ticklabels(['預測 跌 (0)', '預測 漲 (1)'])
        ax.yaxis.set_ticklabels(['實際 跌 (0)', '實際 漲 (1)'])
        
        print("正在顯示圖表... (請查看彈出視窗，可能在 Python 圖示)")
        print("圖表解釋：")
        print(" [左上] 預測 跌，實際 跌 (猜對)")
        print(" [右下] 預測 漲，實際 漲 (猜對)")
        print(" [左下] 預測 漲，實際 跌 (猜錯)")
        print(" [右上] 預測 跌，實際 漲 (猜錯)")
        
        plt.show()

    except Exception as e:
        print(f"繪製混淆矩陣時出錯: {e}")

# --- 主執行流程 (修改) ---
if __name__ == "__main__":
    
    # 1. 獲取資料
    # (我們繼續使用 5m ETH 來比較)
    raw_df = fetch_data(symbol='ETH/USDT', timeframe='5m', total_limit=10000)
    
    # 2. 特徵工程
    df_with_features = create_features(raw_df)
    
    # 3. 訓練與預測 (分類)
    best_classifier, X_test_data, y_test_data = train_and_predict(df_with_features)
    
    # 4. 繪製混淆矩陣
    if best_classifier:
        plot_confusion_matrix(best_classifier, X_test_data, y_test_data)