import ccxt
import pandas as pd
import talib
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, mean_squared_error
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

# --- 步驟 2: (***重大更新: 建立所有特徵***) ---
def create_features(df):
    """
    計算「迴歸模型」和「分類模型」需要的所有特徵。
    """
    if df is None:
        return None
        
    print("\n--- 步驟 2: 正在計算所有技術指標 ---")
    
    close_prices = df['Close'].values.astype(float)
    high_prices = df['High'].values.astype(float)
    low_prices = df['Low'].values.astype(float)
    volume = df['Volume'].values.astype(float)

    try:
        # --- 特徵 A: 迴歸模型 (價格位置) ---
        df['RSI_14'] = talib.RSI(close_prices, timeperiod=14)
        df['WMA_close_2'] = talib.WMA(close_prices, timeperiod=2)
        df['WMA_high_2'] = talib.WMA(high_prices, timeperiod=2)
        df['WMA_low_2'] = talib.WMA(low_prices, timeperiod=2)
        df['ADX_14'] = talib.ADX(high_prices, low_prices, close_prices)
        df['ADX_hist_14'] = talib.PLUS_DI(high_prices, low_prices, close_prices) - talib.MINUS_DI(high_prices, low_prices, close_prices)
        
        # (我們用週期 2 的 BBANDS)
        upperband_2, middleband_2, lowerband_2 = talib.BBANDS(close_prices, timeperiod=2, nbdevup=2, nbdevdn=2, matype=0)
        df['BB_Width_2'] = (upperband_2 - lowerband_2) / (middleband_2 + 1e-10)
        df['BB_Percent_2'] = (close_prices - lowerband_2) / (upperband_2 - lowerband_2 + 1e-10)

        # --- 特徵 B: 分類模型 (動能/趨勢) ---
        # (我們用週期 5)
        df['RSI_5'] = talib.RSI(close_prices, timeperiod=5)
        df['ADX_5'] = talib.ADX(high_prices, low_prices, close_prices, timeperiod=5)
        df['ADX_hist_5'] = talib.PLUS_DI(high_prices, low_prices, close_prices, timeperiod=5) - talib.MINUS_DI(high_prices, low_prices, close_prices, timeperiod=5)
        df['MOM_5'] = talib.MOM(close_prices, timeperiod=5)
        
        # (MACD 和 OBV 保持原樣)
        macd, macdsignal, _ = talib.MACD(close_prices, fastperiod=6, slowperiod=13, signalperiod=9)
        df['MACD'] = macd
        df['MACD_signal'] = macdsignal
        df['OBV'] = talib.OBV(close_prices, volume)
        df['Volume'] = volume # 確保 Volume 被加入
        # --------------------------------

        original_len = len(df)
        df_features = df.dropna() 
        print(f"已去除 {original_len - len(df_features)} 筆舊資料 (因計算指標產生 NaN)。")
        
        return df_features

    except Exception as e:
        print(f"計算特征時發生錯誤: {e}")
        return None

# --- 步驟 3 & 4: (***重大更新: 模型堆疊***) ---

def train_and_predict_stacked(df_features):
    """
    執行三階段模型堆疊：
    1. 訓練迴歸模型 (預測價格)
    2. 產生價格預測特徵
    3. 訓練分類模型 (預測方向)
    """
    if df_features is None or df_features.empty:
        print("沒有足夠的特徵資料進行訓練。")
        return
    
    print("\n--- 步驟 3: 準備資料 (模型堆疊) ---")

    # --- 步驟 3a: 定義兩組特徵 ---
    features_reg = [ # 迴歸模型的特徵 (價格位置)
        'RSI_14',
        'WMA_close_2', 'WMA_high_2', 'WMA_low_2',
        'ADX_14', 'ADX_hist_14',
        'MACD', 'MACD_signal',
        'BB_Width_2', 'BB_Percent_2',
        'OBV',
        'Volume' # 價格模型也看一下成交量
    ]
    
    features_clf = [ # 分類模型的特徵 (動能)
        'RSI_5',
        'ADX_5', 'ADX_hist_5',
        'MACD', 'MACD_signal',
        'MOM_5',
        'BB_Width_2',
        'OBV',
        'Volume' # 方向模型也看一下成交量
    ]

    # --- 步驟 3b: 建立兩個目標 (Target) ---
    df_model = df_features.copy()
    # 目標 A (迴歸): 下一根 K 棒的收盤價
    df_model['target_price'] = df_model['Close'].shift(-1) 
    # 目標 B (分類): 下一根 K 棒是漲(1)或跌(0)
    df_model['target_direction'] = (df_model['Close'].shift(-1) > df_model['Close']).astype(int)
    
    df_model = df_model.dropna() # 移除最後一行
    
    # --- 步驟 3c: 分割所有資料 ---
    test_size = 0.2 
    split_index = int(len(df_model) * (1 - test_size))

    # 我們分割「整個」 DataFrame
    df_train = df_model.iloc[:split_index]
    df_test = df_model.iloc[split_index:]

    print(f"訓練集筆數: {len(df_train)}, 測試集筆數: {len(df_test)}")

    # --- 階段 1: 訓練「迴歸專家 A」---
    print("\n--- 階段 1: 正在訓練「迴歸專家 A」(預測價格)... ---")
    
    # 準備迴歸資料
    X_reg_train = df_train[features_reg]
    y_reg_train = df_train['target_price']
    X_reg_test = df_test[features_reg]
    y_reg_test = df_test['target_price']

    # (我們使用之前手動調的優秀參數，不再用自動調校，以節省時間)
    xgb_reg = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        objective='reg:squarederror',
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=50 # 使用 Early Stopping
    )

    xgb_reg.fit(
        X_reg_train, 
        y_reg_train,
        eval_set=[(X_reg_test, y_reg_test)], # 用測試集來監控
        verbose=False
    )
    
    # 評估迴歸模型
    y_reg_pred = xgb_reg.predict(X_reg_test)
    rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
    print(f"「迴歸專家 A」在測試集上的 RMSE: {rmse:.2f} (預測平均誤差)")

    # --- 階段 2: 產生「專家意見」特徵 ---
    print("\n--- 階段 2: 正在產生「專家意見」特徵 (價格預測)... ---")
    
    # (重要!) 我們在「訓練集」和「測試集」上都進行預測
    reg_pred_on_train = xgb_reg.predict(X_reg_train)
    reg_pred_on_test = y_reg_pred # (我們在階段 1 已經預測過了)

    # 準備分類資料
    X_clf_train = df_train[features_clf]
    y_clf_train = df_train['target_direction']
    X_clf_test = df_test[features_clf]
    y_clf_test = df_test['target_direction']
    
    # *** 堆疊！ ***
    # 把「專家意見」當作新的一欄，加到分類模型的特徵中
    X_clf_train['reg_expert_opinion'] = reg_pred_on_train
    X_clf_test['reg_expert_opinion'] = reg_pred_on_test
    
    print("已成功將「價格預測」堆疊為新特徵。")

    # --- 階段 3: 訓練「分類主管 B」 (自動調校) ---
    print("\n--- 階段 3: 正在訓練「分類主管 B」(預測方向)... ---")
    print("(這將執行超參數調校，會花費 5-15+ 分鐘...)")
    
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

    # 在「堆疊後」的訓練集上進行調校
    random_search.fit(X_clf_train, y_clf_train)

    print("\n--- 調校完成! ---")
    print(f"最佳交叉驗證 (CV) 準確率: {random_search.best_score_:.2%}")
    print("找到的最佳參數組合:")
    print(random_search.best_params_)

    # 獲取「最佳堆疊分類器」
    xgb_clf_stacked = random_search.best_estimator_

    print("\n--- 步驟 4: 評估「最終堆疊模型」 ---")
    
    # 在「堆疊後」的測試集上進行評估
    y_clf_pred = xgb_clf_stacked.predict(X_clf_test)
    
    accuracy = accuracy_score(y_clf_test, y_clf_pred)
    print(f"\n模型在「測試集」上的最終準確率 (Accuracy): {accuracy:.2%}")
    print("\n--- 詳細分類報告 (Classification Report) ---")
    print(classification_report(y_clf_test, y_clf_pred, target_names=['跌 (0)', '漲 (1)']))

    # --- 預測 '真正' 的明天 ---
    # (這一步會變得更複雜，因為我們需要先跑迴歸模型)
    # (我們暫時專注於回測，省略即時預測)

    return xgb_clf_stacked, X_clf_test, y_clf_test

# --- 步驟 5: 繪製混淆矩陣 ---
# (與上一版相同，保持不變)
def plot_confusion_matrix(classifier, X_test, y_test):
    """
    繪製混淆矩陣 (Confusion Matrix)
    """
    print("\n--- 步驟 5: 正在繪製混淆矩陣 (Confusion Matrix) ---")
    
    try:
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.set_title('混淆矩陣 (Confusion Matrix) - 堆疊模型')
        
        ConfusionMatrixDisplay.from_estimator(
            classifier, 
            X_test, 
            y_test,
            ax=ax,
            cmap=plt.cm.Blues,
            display_labels=['實際 跌 (0)', '實際 漲 (1)']
        )
        
        ax.xaxis.set_ticklabels(['預測 跌 (0)', '預測 漲 (1)'])
        ax.yaxis.set_ticklabels(['實際 跌 (0)', '實際 漲 (1)'])
        
        print("正在顯示圖表... (請查看彈出視窗，可能在 Python 圖示)")
        plt.show()

    except Exception as e:
        print(f"繪製混淆矩G陣時出錯: {e}")

# --- 主執行流程 (修改) ---
if __name__ == "__main__":
    
    # 1. 獲取資料
    raw_df = fetch_data(symbol='ETH/USDT', timeframe='5m', total_limit=10000)
    
    # 2. 特徵工程 (建立所有特徵)
    df_with_features = create_features(raw_df)
    
    # 3. 訓練與預測 (堆疊)
    best_classifier, X_test_data, y_test_data = train_and_predict_stacked(df_with_features)
    
    # 4. 繪製混淆矩陣
    if best_classifier:
        plot_confusion_matrix(best_classifier, X_test_data, y_test_data)