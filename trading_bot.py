# 檔案: trading_bot.py
import ccxt
import pandas as pd
import numpy as np
import warnings
import time
import os

# --- 1. 引用「設定檔」和「共用工具箱」 ---
import config
from common_utils import create_features_trend, create_features_trend, create_sequences

# --- 2. 匯入模型 ---
import tensorflow as tf
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler

warnings.simplefilter(action='ignore', category=FutureWarning)

# --- (*** 全局變數: 已升級為「字典」***) ---
TREND_STATE = {symbol: "NEUTRAL" for symbol in config.SYMBOLS_TO_TRADE}
LAST_1H_CHECK = {symbol: None for symbol in config.SYMBOLS_TO_TRADE}
SCALERS_1H = {symbol: MinMaxScaler(feature_range=(0, 1)) for symbol in config.SYMBOLS_TO_TRADE}

# --- 3. 載入我們訓練好的「所有」模型 (從 config 讀取) ---
print("--- 正在載入所有模型... ---")
MODELS_B_1H_LSTM = {}
MODELS_A_5M_XGB = {}

os.makedirs(config.MODEL_DIR, exist_ok=True)

for symbol in config.SYMBOLS_TO_TRADE:
    print(f"--- 正在載入 {symbol} 的模型 ---")
    
    # 載入趨勢模型 (LSTM)
    try:
        path = config.get_trend_model_path(symbol, config.TREND_MODEL_VERSION)
        MODELS_B_1H_LSTM[symbol] = tf.keras.models.load_model(path)
        print(f"✅ {symbol} 趨勢模型 (Ver: {config.TREND_MODEL_VERSION}) 載入成功！")
    except Exception as e:
        print(f"🛑 錯誤：無法載入 {symbol} 的「趨勢模型」。請先執行: \n python train_trend_model.py --symbol {symbol} --version {config.TREND_MODEL_VERSION}")
        exit()

    # 載入進場模型 (XGBoost)
    try:
        path = config.get_trend_model_path(symbol, config.TREND_MODEL_VERSION)
        xgb_model = xgb.Booster()
        xgb_model.load_model(path)
        MODELS_A_5M_XGB[symbol] = xgb_model
        print(f"✅ {symbol} 進場模型 (Ver: {config.TREND_MODEL_VERSION}) 載入成功！")
    except Exception as e:
        print(f"🛑 錯誤：無法載入 {symbol} 的「進場模型」。請先執行: \n python train_entry_model.py --symbol {symbol} --version {config.TREND_MODEL_VERSION}")
        exit()

print("--- 所有模型載入完畢 ---")

exchange = ccxt.binance()

# --- 4. 決策函數 ---

def get_trend_signal(symbol):
    """
    執行「趨勢模型」，決定「長期走向」。
    """
    global TREND_STATE
    print(f"\n--- (檢查 {symbol} 1h 濾網) ---")
    try:
        P_TREND = config.TREND_MODEL_PARAMS
        lookback = P_TREND['LOOKBACK_WINDOW']
        
        ohlcv_1h = exchange.fetch_ohlcv(symbol, config.TREND_MODEL_TIMEFRAME, limit=100)
        df = pd.DataFrame(ohlcv_1h, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        
        df_with_features, features_list = create_features_trend(df)
        if df_with_features is None or len(df_with_features) < lookback:
            print(f"{symbol} 1h 資料不足，無法建立序列。")
            return

        last_sequence_data = df_with_features[features_list].iloc[-lookback:]
        scaled_sequence = SCALERS_1H[symbol].fit_transform(last_sequence_data)
        X_live = np.array([scaled_sequence])
        
        prediction_proba = MODELS_B_1H_LSTM[symbol].predict(X_live, verbose=0)
        prediction = (prediction_proba > 0.5).astype(int)[0][0]
        
        if prediction == 1:
            TREND_STATE[symbol] = "UP"
            print(f"✅ {symbol} 1h LSTM 濾網: 趨勢向上 (信心 {prediction_proba[0][0]:.2%})")
        else:
            TREND_STATE[symbol] = "DOWN"
            print(f"🛑 {symbol} 1h LSTM 濾網: 趨勢向下 (信心 {1 - prediction_proba[0][0]:.2%})")
            
    except Exception as e:
        print(f"執行 {symbol} 1h LSTM 預測時出錯: {e}")
        TREND_STATE[symbol] = "NEUTRAL"

def get_entry_signal(symbol):
    """
    執行「進場模型」，尋找「短期進場點」。
    """
    print(f"--- (檢查 {symbol} 5m 觸發器) ---")
    try:
        ohlcv_5m = exchange.fetch_ohlcv(symbol, config.TREND_MODEL_TIMEFRAME, limit=100)
        df = pd.DataFrame(ohlcv_5m, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        
        df_with_features, features_list = create_features_trend(df)
        
        last_features = df_with_features[features_list].iloc[-1:]
        X_live = xgb.DMatrix(last_features)
        
        predicted_return = MODELS_A_5M_XGB[symbol].predict(X_live)[0]
        
        print(f"{symbol} 5m XGB: 預測報酬率 {predicted_return:.4%}")

        ENTRY_THRESHOLD = 0.0001
        if predicted_return > ENTRY_THRESHOLD:
            return "BUY"
        elif predicted_return < -ENTRY_THRESHOLD:
            return "SELL"
        else:
            return "HOLD"
            
    except Exception as e:
        print(f"執行 {symbol} 5m XGB 預測時出錯: {e}")
        return "HOLD"

# --- 5. 決策機器人主迴圈 ---
def main_loop():
    global LAST_1H_CHECK, TREND_STATE
    
    while True:
        try:
            current_time = pd.Timestamp.now(tz='UTC')
            print(f"\n==============================================")
            print(f"時間: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"當前所有趨勢: {TREND_STATE}")
            
            for symbol in config.SYMBOLS_TO_TRADE:
                
                print(f"\n--- 正在處理 {symbol} ---")

                if LAST_1H_CHECK[symbol] is None or current_time.hour != LAST_1H_CHECK[symbol]:
                    if current_time.minute < 5: 
                        get_trend_signal(symbol)
                        LAST_1H_CHECK[symbol] = current_time.hour
                
                entry_signal = get_entry_signal(symbol)
                
                print(f"--- ({symbol} 最終決策) ---")
                symbol_trend = TREND_STATE[symbol]
                
                if symbol_trend == "UP" and entry_signal == "BUY":
                    print(f"✅ {symbol} 決策: 執行做多 (Buy)！ (1h 濾網 = UP, 5m 觸發器 = BUY)")
                elif symbol_trend == "DOWN" and entry_signal == "SELL":
                    print(f"🛑 {symbol} 決策: 執行做空 (Sell)！ (1h 濾網 = DOWN, 5m 觸發器 = SELL)")
                else:
                    print(f"⬜ {symbol} 決策: 持有 (Hold)。 (濾網: {symbol_trend}, 觸發器: {entry_signal})")

            print("==============================================")
            time.sleep(config.BOT_LOOP_SLEEP_SECONDS)

        except ccxt.NetworkError as e:
            print(f"網路錯誤: {e}，60 秒後重試...")
            time.sleep(60)
        except Exception as e:
            print(f"主迴圈發生錯誤: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main_loop()
