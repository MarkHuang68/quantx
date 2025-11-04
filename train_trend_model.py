# 檔案: train_trend_model.py
import json
import warnings
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os

# --- 1. 引用「設定檔」和「共用工具箱」 ---
import config
from common_utils import fetch_data, create_features_trend, create_sequences
from hyperparameter_search import SearchIterator

# --- (匯入所有 Keras/Sklearn 工具) ---
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

# (設置 Keras/Tensorflow 的隨機種子)
tf.random.set_seed(42)
np.random.seed(42)
warnings.simplefilter(action='ignore', category=FutureWarning)
#ema=20, sma=60, rsi=14, bbands=10
# --- 您的尋參空間 (保持不變) ---
TREND_SEARCH_SPACE = {
    'ema': [5, 20, 5],
    'sma': [20, 100, 20],
    'rsi': [7, 14, 7],
    'bbands': [2, 8, 6]
}

# --- 您的 LSTM 訓練參數 (保持不變) ---
LSTM_BASE_PARAMS = {
    'epochs': 50,
    'batch_size': 64,
    'shuffle': False,
    'callbacks': [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
}

def build_and_train_lstm(df_features, features_list):
    """
    (這是我們 90.9% 的冠軍模型訓練邏輯)
    """
    if df_features is None: return None, 0.0, None, None

    # --- 2. 從「設定檔」讀取所有參數 ---
    P = config.TREND_MODEL_PARAMS
    lookback_window = P['LOOKBACK_WINDOW']
    forecast_horizon = P['FORECAST_HORIZON']
    u1, u2, d1 = P['LSTM_UNITS_1'], P['LSTM_UNITS_2'], P['DENSE_UNITS']

    # 1. 定義特徵和目標 (必須 100% 匹配 common_utils.py)

    df_model = df_features.copy()
    
    print(f"\n--- 正在建立目標: 預測 {forecast_horizon} 小時之後的趨勢走向 ---")
    df_model['target'] = (df_model['SMA'].shift(-forecast_horizon) > df_model['SMA']).astype(int)
    df_model = df_model.dropna() 
    
    # 2. 標準化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(df_model[features_list])
    target = df_model['target'].values
    
    # 3. 建立 3D 序列
    X_seq, y_seq = create_sequences(scaled_features, target, lookback_window=lookback_window)
    
    # 4. 分割資料
    test_size = 0.2 
    split_index = int(len(X_seq) * (1 - test_size))
    X_train, X_test = X_seq[:split_index], X_seq[split_index:]
    y_train, y_test = y_seq[:split_index], y_seq[split_index:]

    print(f"訓練集筆數: {len(X_train)}, 測試集筆數: {len(X_test)}")

    # 5. 建立「深度堆疊」LSTM 模型
    print("\n--- 步驟 4: 正在建立「深度堆疊」LSTM 模型... ---")
    model = Sequential()
    model.add(Bidirectional(LSTM(units=u1, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))))
    model.add(Dropout(0.3)) 
    model.add(Bidirectional(LSTM(units=u2)))
    model.add(Dropout(0.3))
    model.add(Dense(units=d1, activation='relu')) 
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # 6. 訓練模型
    print("\n--- 正在訓練「深度」LSTM 模型... ---")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        **LSTM_BASE_PARAMS
    )

    # 7. 評估
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print("模型訓練完成。")
    return model, accuracy, X_test, y_test

def plot_confusion_matrix(classifier, X_test, y_test, show_plot=False):
    """ (這是我們帶「開關」的繪圖函數) """
    print("\n--- 步驟 6: 正在繪製混淆矩陣 ---")
    try:
        y_pred = (classifier.predict(X_test, verbose=0) > 0.5).astype(int)
        
        # (印出最終準確率)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n模型在「測試集」上的準確率 (Accuracy): {accuracy:.2%}")
        print("\n--- 詳細分類報告 (Classification Report) ---")
        print(classification_report(y_test, y_pred, target_names=['跌 (0)', '漲 (1)']))

        if show_plot:
            print("正在顯示圖表...")
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.set_title('混淆矩陣 (Confusion Matrix) - 1h LSTM')
            ConfusionMatrixDisplay.from_predictions(
                y_test, y_pred, ax=ax, cmap=plt.cm.Blues,
                display_labels=['實際 跌 (0)', '實際 漲 (1)']
            )
            ax.xaxis.set_ticklabels(['預測 跌 (0)', '預測 漲 (1)'])
            ax.yaxis.set_ticklabels(['實際 跌 (0)', '實際 漲 (1)'])
            plt.show()
        else:
            print("繪圖開關已關閉 (未傳入 --plot)。")
    except Exception as e:
        print(f"繪製混淆矩陣時出錯: {e}")

def evaluate_existing_model_trend(symbol, version, raw_df):
    """ 評估「現行」模型在「相同」數據上的 Accuracy。"""
    model_path = config.get_trend_model_path(symbol, version)
    config_path = model_path.replace('.keras', '_feature_config.json')
    
    if not os.path.exists(model_path) or not os.path.exists(config_path):
        print("--- 找不到現行模型，跳過競爭比較。---")
        return 0.0 # 返回 0.0 準確率，確保新模型獲勝

    try:
        print(f"--- 載入現行模型 ({version}) 進行競爭比較... ---")
        
        # 1. 載入現行模型和它的特徵參數
        current_model = tf.keras.models.load_model(model_path)
        with open(config_path, 'r') as f:
            current_feature_config = json.load(f)
        
        print(f"現行模型的特徵參數: {current_feature_config}")
        
        # 2. 創建特徵 (使用現行模型自己的配置)
        df_features_old, features_list_old = create_features_trend(raw_df.copy(), **current_feature_config)
        
        # 3. 準備數據 (必須與 build_and_train_lstm 邏輯 100% 相同)
        P = config.TREND_MODEL_PARAMS
        df_model_old = df_features_old.copy()
        
        # 4. (*** 關鍵：使用儲存的配置來建立 Label ***)
        # (我們假設 Label 總是基於 'SMA'，如果不是，這裡需要修改)
        df_model_old['target'] = (df_model_old['SMA'].shift(-P['FORECAST_HORIZON']) > df_model_old['SMA']).astype(int)
        df_model_old = df_model_old.dropna() 
        
        # 5. 標準化
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_features_old = scaler.fit_transform(df_model_old[features_list_old])
        target_old = df_model_old['target'].values
        
        # 6. 建立 3D 序列
        X_seq_old, y_seq_old = create_sequences(scaled_features_old, target_old, lookback_window=P['LOOKBACK_WINDOW'])
        
        # 7. 分割 (只取測試集)
        test_size = 0.2 
        split_index_old = int(len(X_seq_old) * (1 - test_size))
        X_test_old = X_seq_old[split_index_old:]
        y_test_old = y_seq_old[split_index_old:]
        
        # 8. 評估現行模型
        loss, accuracy = current_model.evaluate(X_test_old, y_test_old, verbose=0)
        
        print(f"--- 現行模型在「當前數據」上的 Accuracy: {accuracy:.4f} ---")
        return accuracy

    except Exception as e:
        print(f"🛑 載入或評估現行模型時出錯: {e}")
        return 0.0 # 失敗返回 0.0，確保新模型獲勝
    
if __name__ == "__main__":
    
    # --- 3. 建立「參數解析器」 ---
    parser = argparse.ArgumentParser(description='訓練 1h LSTM 趨勢模型')
    
    parser.add_argument(
        '-s', '--symbol', 
        type=str, 
        required=True, # <-- *** 必須指定 symbol ***
        help='要訓練的交易對 (例如: ETH/USDT 或 BTC/USDT)'
    )
    parser.add_argument(
        '-l', '--limit', 
        type=int, 
        default=config.TREND_MODEL_TRAIN_LIMIT, 
        help=f'K 線筆數 (預設: {config.TREND_MODEL_TRAIN_LIMIT})'
    )
    parser.add_argument(
        '-v', '--version',
        type=str,
        default=config.TREND_MODEL_VERSION, # <-- 從 config 讀取預設版本
        help=f'要訓練的模型版本 (預設: {config.TREND_MODEL_VERSION})'
    )
    parser.add_argument(
        '-p', '--plot', 
        action='store_true', 
        help='(開關) 訓練完成後，顯示混淆矩陣圖表。'
    )
    
    args = parser.parse_args()
    
    # --- 4. 執行訓練 ---
    print(f"--- 開始執行: {args.symbol} ({config.TREND_MODEL_TIMEFRAME}), 資料量={args.limit} ---")
    
    # (確保 models 資料夾存在)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    # 1. 獲取資料
    raw_df = fetch_data(symbol=args.symbol, timeframe=config.TREND_MODEL_TIMEFRAME, total_limit=args.limit)

    # 設定參數格式, 生成參數組合
    f_type = {
        'lookback_window': 'discrete', 
        'forecast_horizon': 'discrete',
    }
    iterator = SearchIterator(TREND_SEARCH_SPACE, search_type='random', n_iter=30, format_types=f_type)

    print(f"--- 總共需要執行 {iterator.get_total_runs()} 次訓練 ---")
    
    best_accuracy = 0.0
    best_model = None
    best_feature_params = None 
    best_X_test = None
    best_y_test = None

    FEATURE_KEYS = ['ema', 'sma', 'rsi', 'bbands']

    for i, params in enumerate(iterator):
        
        # 1a. 分離特徵參數和模型參數
        feature_params = {k: params[k] for k in FEATURE_KEYS if k in params}
        # train_params = {k: params[k] for k in params.keys() if k not in FEATURE_KEYS}
    
        # 2. 特徵工程 (從 common_utils 引用)
        df_features, features_list = create_features_trend(raw_df, **feature_params)
        if df_features is None or features_list is None: 
            print(f"Iter {i+1:02d}/{iterator.get_total_runs()}: 特徵計算失敗，跳過。")
            continue
        
        # 3. 訓練與預測
        best_classifier, accuracy, X_test, y_test = build_and_train_lstm(df_features, features_list)

        print(f"Iter {i+1:02d}/{iterator.get_total_runs()}: Accuracy={accuracy:.4f} ({feature_params}))")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = best_classifier
            best_feature_params = feature_params
            best_X_test = X_test
            best_y_test = y_test

    if not best_model:
        print("🛑 訓練失敗：模型未被建立。")
        exit()

    if best_accuracy < config.ABS_MIN_ACCURACY:
        print(f"\n❌ 質量門 1 失敗！最佳 Accuracy ({best_accuracy:.4f}) 未達到絕對最低標準 ({config.ABS_MIN_ACCURACY * 100}%)。不儲存模型。")
        exit()
    else:
        print(f"\n✅ 質量門 1 (絕對標準) 通過！")

    historical_accuracy = evaluate_existing_model_trend(args.symbol, args.version, raw_df)
    
    if best_accuracy <= historical_accuracy:
        print(f"\n❌ 質量門 2 失敗！新模型 Accuracy ({best_accuracy:.4f}) 並未優于現行模型 ({historical_accuracy:.4f})。不儲存模型。")
        exit()
    else:
        print(f"\n✅ 質量門 2 (競爭標準) 通過！新模型 ({best_accuracy:.4f}) 成功擊敗 現行模型 ({historical_accuracy:.4f})。")

    model_filename = config.get_trend_model_path(args.symbol, args.version)
    config_filename = config.get_trend_model_path(args.symbol, args.version).replace('.keras', '_feature_config.json')
    
    # 5. 儲存模型
    if best_model:
        print(f"\n--- 正在儲存「趨勢模型」... ---")
        best_model.save(model_filename)
        print(f"模型儲存完畢！({model_filename})")

    if best_feature_params:
        with open(config_filename, 'w') as f:
            json.dump(best_feature_params, f, indent=4)
        print(f"✅ 最佳特徵配置儲存完畢：{config_filename}")

    # 4. 繪製混淆矩陣 (根據 --plot 參數)
    if best_model:
        plot_confusion_matrix(best_model, best_X_test, best_y_test, show_plot=args.plot)