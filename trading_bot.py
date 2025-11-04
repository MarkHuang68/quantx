# 檔案: trading_bot.py
import ccxt
import pandas as pd
import numpy as np
import warnings
import time
import os

# --- 1. 引用「設定檔」和「共用工具箱」 ---
import config
from common_utils import create_features_trend, create_features_price, create_sequences

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

# (確保 models 資料夾存在)
os.makedirs(config.MODEL_DIR, exist_ok=True)

for symbol in config.SYMBOLS_TO_TRADE:
    symbol_str = symbol.replace('/', '_')
    print(f"--- 正在載入 {symbol} 的模型 ---")
    
    # 載入趨勢模型 (LSTM) - (從 config 讀取「版本號」)
    try:
        path = config.get_trend_model_path(symbol, config.TREND_MODEL_VERSION)
        MODELS_B_1H_LSTM[symbol] = tf.keras.models.load_model(path)
        print(f"✅ {symbol} 趨勢模型 (Ver: {config.TREND_MODEL_VERSION}) 載入成功！")
    except Exception as e:
        print(f"🛑 錯誤：無法載入 {symbol} 的「趨勢模型」。請先執行: \n python train_trend_model.py --symbol {symbol} --version {config.TREND_MODEL_VERSION}")
        # print(e) # (取消註解來看詳細錯誤)
        exit()

    # 載入價格模型 (XGBoost) - (從 config 讀取「版本號」)
    try:
        path = config.get_price_model_path(symbol, config.PRICE_MODEL_VERSION)
        xgb_model = xgb.Booster()
        xgb_model.load_model(path)
        MODELS_A_5M_XGB[symbol] = xgb_model
        print(f"✅ {symbol} 價格模型 (Ver: {config.PRICE_MODEL_VERSION}) 載入成功！")
    except Exception as e:
        print(f"🛑 錯誤：無法載入 {symbol} 的「價格模型」。請先執行: \n python train_price_model.py --symbol {symbol} --version {config.PRICE_MODEL_VERSION}")
        # print(e) # (取消註解來看詳細錯誤)
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
        # 1. 從 config 讀取參數
        P_TREND = config.TREND_MODEL_PARAMS
        lookback = P_TREND['LOOKBACK_WINDOW']
        
        # 2. 獲取足夠的 1h 資料
        ohlcv_1h = exchange.fetch_ohlcv(symbol, P_TREND['TIMEFRAME'], limit=100)
        df = pd.DataFrame(ohlcv_1h, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        
        # 3. 計算 1h 特徵 (從 common_utils 引用)
        df_with_features, features_list = create_features_trend(df)
        if df_with_features is None or len(df_with_features) < lookback:
            print(f"{symbol} 1h 資料不足，無法建立序列。")
            return

        # 4. 準備「最後一筆」序列
        last_sequence_data = df_with_features[features_list].iloc[-lookback:]
        
        # 5. 標準化
        scaled_sequence = SCALERS_1H[symbol].fit_transform(last_sequence_data)
        
        # 6. 轉換為 3D (1, lookback, features)
        X_live = np.array([scaled_sequence])
        
        # 7. 預測！
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

def get_price_signal(symbol, current_price):
    """
    執行「價格模型」，尋找「短期進場點」。
    """
    print(f"--- (檢查 {symbol} 5m 觸發器) ---")
    try:
        # 1. 獲取足夠的 5m 資料
        P_PRICE = config.PRICE_MODEL_PARAMS
        ohlcv_5m = exchange.fetch_ohlcv(symbol, P_PRICE['TIMEFRAME'], limit=10)
        df = pd.DataFrame(ohlcv_5m, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        
        # 2. 計算 5m 特徵 (從 common_utils 引用)
        df_with_features, features_list = create_features_price(df)
        
        # 3. 準備「最後一筆」特徵
        last_features = df_with_features[features_list].iloc[-1:]
        
        # 4. 轉換為 DMatrix (XGBoost 格式)
        X_live = xgb.DMatrix(last_features)
        
        # 5. 預測！
        predicted_price = MODELS_A_5M_XGB[symbol].predict(X_live)[0]
        
        print(f"{symbol} 5m XGB: 當前 ${current_price:.2f}, 預測 ${predicted_price:.2f}")

        # 6. 決策
        # (注意: 您可能需要為 BTC/ETH 設定不同的 RMSE_THRESHOLD)
        if predicted_price > (current_price + config.PRICE_MODEL_RMSE_THRESHOLD):
            return "BUY"
        elif predicted_price < (current_price - config.PRICE_MODEL_RMSE_THRESHOLD):
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
            
            # --- (*** 遍歷所有 Symbols ***) ---
            for symbol in config.SYMBOLS_TO_TRADE:
                
                print(f"\n--- GE正在處理 {symbol} ---")
                current_price = exchange.fetch_ticker(symbol)['last']

                # --- 步驟 A: 每小時的「開頭」更新一次「長期趨勢」 ---
                if LAST_1H_CHECK[symbol] is None or current_time.hour != LAST_1H_CHECK[symbol]:
                    if current_time.minute < 5: 
                        get_trend_signal(symbol)
                        LAST_1H_CHECK[symbol] = current_time.hour
                
                # --- 步驟 B: 每 5 分鐘執行「短期進場」決策 ---
                entry_signal = get_price_signal(symbol, current_price)
                
                # --- 步驟 C: 最終決策 (MTF) ---
                print(f"--- ({symbol} 最終決策) ---")
                symbol_trend = TREND_STATE[symbol]
                
                if symbol_trend == "UP" and entry_signal == "BUY":
                    print(f"✅ {symbol} 決策: 執行做多 (Buy)！ (1h 濾網 = UP, 5m 觸發器 = BUY)")
                    # (*** 在此處貼上您的「交易所下單 (Buy)」程式碼 ***)
                    
                elif symbol_trend == "DOWN" and entry_signal == "SELL":
                    print(f"🛑 {symbol} 決策: 執行做空 (Sell)！ (1h 濾網 = DOWN, 5m 觸發器 = SELL)")
                    # (*** 在此處貼上您的「交易所下單 (Sell)」程式碼 ***)
                    
                else:
                    print(f"⬜ {symbol} 決策: 持有 (Hold)。 (濾網: {symbol_trend}, 觸發器: {entry_signal})")

            print("==============================================")
            time.sleep(config.BOT_LOOP_SLEEP_SECONDS) # (從 config 讀取)

        except ccxt.NetworkError as e:
            print(f"網路錯誤: {e}，60 秒後重試...")
            time.sleep(60)
        except Exception as e:
            print(f"主迴圈發生錯誤: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main_loop()