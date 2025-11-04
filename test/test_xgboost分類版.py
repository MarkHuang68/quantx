import ccxt
import pandas as pd
import talib
import xgboost as xgb
# --- 修改: 匯入 from_predictions ---
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, mean_squared_error
# --------------------------------
import numpy as np
import warnings
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from scipy.stats import uniform, randint

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

# --- 步驟 2: (***使用「純動能」特徵***) ---
def create_features(df):
    """
    計算「純」分類模型需要的所有特徵 (動能)。
    """
    if df is None:
        return None
        
    print("\n--- 步驟 2: 正在計算「純動能」指標 ---")
    
    close_prices = df['Close'].values.astype(float)
    high_prices = df['High'].values.astype(float)
    low_prices = df['Low'].values.astype(float)
    volume = df['Volume'].values.astype(float)

    try:
        # --- (移除 WMA_2 和 BB_Percent_2) ---

        # --- 使用快速週期 (5) ---
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
        # --------------------------------

        original_len = len(df)
        df_features = df.dropna() 
        print(f"已去除 {original_len - len(df_features)} 筆舊資料 (因計算指標產生 NaN)。")
        
        return df_features

    except Exception as e:
        print(f"計算特征時發生錯誤: {e}")
        return None

# --- 步驟 3 & 4: (***使用「純動能」分類模型***) ---

def train_and_predict(df_features):
    """
    準備資料、自動調校並訓練 XGBoost 分類模型。
    """
    if df_features is None or df_features.empty:
        print("沒有足夠的特徵資料進行訓練。")
        return None, None, None # <-- 修改

    print("\n--- 步驟 3: 準備資料 (分類) ---")

    # --- (使用「純動能」特徵列表) ---
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
    
    df_model['target'] = (df_model['SMA'].shift(-1) > df_model['SMA']).astype(int)
    df_model = df_model.dropna()

    X = df_model[features]
    y = df_model['target']

    test_size = 0.2 
    split_index = int(len(X) * (1 - test_size))

    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    print(f"訓練集筆數: {len(X_train)}, 測試集筆數: {len(X_test)}")
    print(f"訓練集中 '漲 (1)' 的比例: {y_train.mean():.2%}")
    print(f"測試集中 '漲 (1)' 的比例: {y_test.mean():.2%}")

    # --- 步驟 3a: 超參數調校設定 ---
    
    param_dist = {
        'n_estimators': randint(500, 1500),
        'learning_rate': uniform(0.01, 0.05),
        'max_depth': randint(3, 8),
        'subsample': uniform(0.7, 0.3),
        'colsample_bytree': uniform(0.7, 0.3)
    }

    xgb_clf_base = xgb.XGBClassifier(
        objective='binary:logistic', 
        eval_metric='logloss',       
        n_jobs=-1,
        random_state=42
    )

    tscv = TimeSeriesSplit(n_splits=3)

    random_search = RandomizedSearchCV(
        estimator=xgb_clf_base,
        param_distributions=param_dist,
        n_iter=25,
        cv=tscv,
        scoring='accuracy', 
        n_jobs=-1,
        verbose=2,
        random_state=42
    )

    print("\n--- 步驟 3b: 開始超參數調校 (分類模型，這會花費 5-15+ 分鐘...) ---")
    
    random_search.fit(X_train, y_train)

    print("\n--- 調校完成! ---")
    print(f"最佳交叉驗證 (CV) 準確率: {random_search.best_score_:.2%}")
    print("找到的最佳參數組合:")
    print(random_search.best_params_)

    xgb_clf = random_search.best_estimator_

    print("\n--- 步驟 3c: 使用「最佳模型」評估測試集 (整體) ---")
    
    y_pred = xgb_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n模型在「測試集」上的**整體準確率**: {accuracy:.2%}")
    print("(這就是我們之前 51.30% 的基準線)")

    # --- (省略步驟 4: 即時預測) ---

    # --- 修改: 返回分類器和測試資料 ---
    return xgb_clf, X_test, y_test

# --- (舊的 plot_confusion_matrix 已刪除) ---

# --- 新增: 步驟 5: 高信心回測 (High Confidence Evaluation) ---
def evaluate_with_confidence(classifier, X_test, y_test, threshold=0.60):
    """
    只評估「信心」高於 'threshold' 的交易。
    """
    print(f"\n--- 步驟 5: 正在評估 (信心門檻 > {threshold:.0%}) ---")
    
    try:
        # 1. 獲取信心指數 (e.g., [[0.4, 0.6], [0.9, 0.1]])
        probabilities = classifier.predict_proba(X_test)
        
        # 2. 獲取模型「最有把握」的信心 (e.g., [0.6, 0.9])
        confidence_scores = np.max(probabilities, axis=1)
        
        # 3. 獲取模型的預測 (e.g., [1, 0])
        predictions = classifier.predict(X_test)
        
        # 4. 篩選出高信心的交易
        high_confidence_mask = confidence_scores > threshold
        
        if np.sum(high_confidence_mask) == 0:
            print(f"警告：在 {len(y_test)} 筆資料中，沒有任何預測的信心 > {threshold:.0%}")
            print("請嘗試降低門檻 (例如 0.55)。")
            return

        # 5. 獲取高信心交易的「預測」和「實際結果」
        high_confidence_predictions = predictions[high_confidence_mask]
        high_confidence_actuals = y_test[high_confidence_mask] # <-- 修正: 從 y_test 篩選
        
        # 6. 計算高信心交易的準確率
        new_accuracy = accuracy_score(high_confidence_actuals, high_confidence_predictions)
        
        total_trades = len(y_test)
        trades_taken = len(high_confidence_actuals)
        
        print("\n--- 高信心策略回測結果 ---")
        print(f"總 K 棒 (測試集): {total_trades}")
        print(f"觸發交易 (信心 > {threshold:.0%}): {trades_taken} 次")
        print(f"交易頻率: {trades_taken / total_trades:.2%}")
        print("---------------------------------")
        print(f"高信心交易的準確率 (Accuracy): {new_accuracy:.2%}")
        print("---------------------------------")
        
        # 繪製高信心交易的混淆矩陣
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.set_title(f'混淆矩陣 (Confusion Matrix) - 僅限信心 > {threshold:.0%}')
        
        # (使用 from_predictions，因為我們已經手動篩選了)
        ConfusionMatrixDisplay.from_predictions(
            high_confidence_actuals,
            high_confidence_predictions,
            ax=ax,
            cmap=plt.cm.Blues,
            display_labels=['實際 跌 (0)', '實際 漲 (1)']
        )
        ax.xaxis.set_ticklabels(['預測 跌 (0)', '預測 漲 (1)'])
        ax.yaxis.set_ticklabels(['實際 跌 (0)', '實際 漲 (1)'])
        
        print(f"正在顯示 信心 > {threshold:.0%} 的混淆矩陣...")
        plt.show()

    except Exception as e:
        print(f"執行高信心評估時出錯: {e}")
        import traceback
        traceback.print_exc()

# --- 主執行流程 (修改) ---
if __name__ == "__main__":
    
    # 1. 獲取資料
    raw_df = fetch_data(symbol='ETH/USDT', timeframe='5m', total_limit=10000)
    
    # 2. 特徵工程 (純動能)
    df_with_features = create_features(raw_df)
    
    # 3. 訓練與預測 (分類)
    best_classifier, X_test_data, y_test_data = train_and_predict(df_with_features)
    
    # 4. 執行高信心回測
    if best_classifier:
        # 測試門檻 55%
        evaluate_with_confidence(best_classifier, X_test_data, y_test_data, threshold=0.55)
        
        # 測試門檻 60%
        evaluate_with_confidence(best_classifier, X_test_data, y_test_data, threshold=0.60)
        
        # 測試門檻 65%
        evaluate_with_confidence(best_classifier, X_test_data, y_test_data, threshold=0.65)