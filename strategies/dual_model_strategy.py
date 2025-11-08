# 檔案: strategies/dual_model_strategy.py

import pandas as pd
import numpy as np
import tensorflow as tf
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler

from strategies.base_strategy import BaseStrategy
from utils.common import create_features_trend, create_sequences
from config.settings import SYMBOLS_TO_TRADE, TREND_MODEL_VERSION, TREND_MODEL_TIMEFRAME, get_trend_model_path, MODEL_DIR

class DualModelStrategy(BaseStrategy):
    def __init__(self, context, symbols=SYMBOLS_TO_TRADE):
        super().__init__(context)
        self.symbols = symbols
        self.trend_state = {symbol: "NEUTRAL" for symbol in self.symbols}
        self.last_check = {symbol: None for symbol in self.symbols}
        self.scalers = {symbol: MinMaxScaler(feature_range=(0, 1)) for symbol in self.symbols}

        self.trend_models = {}
        self.entry_models = {}
        self._load_models()

    def _load_models(self):
        print("--- 正在載入所有模型... ---")
        for symbol in self.symbols:
            print(f"--- 正在載入 {symbol} 的模型 ---")

            try:
                # 載入趨勢模型 (LSTM)
                trend_model_path = get_trend_model_path(symbol, TREND_MODEL_VERSION)
                self.trend_models[symbol] = tf.keras.models.load_model(trend_model_path)
                print(f"✅ {symbol} 趨勢模型 (Ver: {TREND_MODEL_VERSION}) 載入成功！")
            except Exception as e:
                print(f"🛑 警告：無法載入 {symbol} 的「趨勢模型」。將使用預設行為。")
                pass

            try:
                # 載入進場模型 (XGBoost)
                entry_model_path = get_trend_model_path(symbol, TREND_MODEL_VERSION) # Assuming same path logic
                xgb_model = xgb.Booster()
                xgb_model.load_model(entry_model_path)
                self.entry_models[symbol] = xgb_model
                print(f"✅ {symbol} 進場模型 (Ver: {TREND_MODEL_VERSION}) 載入成功！")
            except Exception as e:
                print(f"🛑 警告：無法載入 {symbol} 的「進場模型」。將使用預設行為。")
                pass

    def on_bar(self, dt):
        for symbol in self.symbols:
            if symbol not in self.trend_models or symbol not in self.entry_models:
                print(f"--- ({symbol}) 缺少模型，跳過 ---")
                continue
            self._process_symbol(symbol, dt)

    def _process_symbol(self, symbol, dt):
        print(f"\n--- 正在處理 {symbol} ---")

        # 每小時的第一個 5 分鐘 K 棒更新趨勢訊號
        if self.last_check[symbol] is None or dt.hour != self.last_check[symbol].hour:
            if dt.minute < 5:
                self._update_trend_signal(symbol)
                self.last_check[symbol] = dt

        entry_signal = self._get_entry_signal(symbol)

        print(f"--- ({symbol} 最終決策) ---")
        symbol_trend = self.trend_state[symbol]

        if symbol_trend == "UP" and entry_signal == "BUY":
            print(f"✅ {symbol} 決策: 執行做多 (Buy)！ (趨勢 = UP, 進場 = BUY)")
            # 在這裡下單
            # self.context.exchange.create_order(symbol, 'market', 'buy', 0.01)
        elif symbol_trend == "DOWN" and entry_signal == "SELL":
            print(f"🛑 {symbol} 決策: 執行做空 (Sell)！ (趨勢 = DOWN, 進場 = SELL)")
            # 在這裡下單
            # self.context.exchange.create_order(symbol, 'market', 'sell', 0.01)
        else:
            print(f"⬜ {symbol} 決策: 持有 (Hold)。 (趨勢: {symbol_trend}, 進場: {entry_signal})")

    def _update_trend_signal(self, symbol):
        print(f"\n--- (檢查 {symbol} {TREND_MODEL_TIMEFRAME} 趨勢) ---")
        try:
            ohlcv = self.context.exchange.get_ohlcv(symbol, TREND_MODEL_TIMEFRAME, limit=200)
            df_with_features, features_list = create_features_trend(ohlcv)

            if df_with_features is None or len(df_with_features) < 60: # 假設 lookback 是 60
                print(f"{symbol} 資料不足，無法更新趨勢。")
                return

            last_sequence_data = df_with_features[features_list].iloc[-60:]
            scaled_sequence = self.scalers[symbol].fit_transform(last_sequence_data)
            X_live = np.array([scaled_sequence])

            prediction_proba = self.trend_models[symbol].predict(X_live, verbose=0)
            prediction = (prediction_proba > 0.5).astype(int)[0][0]

            if prediction == 1:
                self.trend_state[symbol] = "UP"
                print(f"✅ {symbol} {TREND_MODEL_TIMEFRAME} 趨勢: 向上 (信心 {prediction_proba[0][0]:.2%})")
            else:
                self.trend_state[symbol] = "DOWN"
                print(f"🛑 {symbol} {TREND_MODEL_TIMEFRAME} 趨勢: 向下 (信心 {1 - prediction_proba[0][0]:.2%})")

        except Exception as e:
            print(f"執行 {symbol} 趨勢預測時出錯: {e}")
            self.trend_state[symbol] = "NEUTRAL"

    def _get_entry_signal(self, symbol):
        print(f"--- (檢查 {symbol} 5m 進場) ---")
        try:
            ohlcv = self.context.exchange.get_ohlcv(symbol, '5m', limit=100)
            df_with_features, features_list = create_features_trend(ohlcv)

            last_features = df_with_features[features_list].iloc[-1:]
            X_live = xgb.DMatrix(last_features)

            predicted_return = self.entry_models[symbol].predict(X_live)[0]

            print(f"{symbol} 5m XGB: 預測報酬率 {predicted_return:.4%}")

            ENTRY_THRESHOLD = 0.0001
            if predicted_return > ENTRY_THRESHOLD:
                return "BUY"
            elif predicted_return < -ENTRY_THRESHOLD:
                return "SELL"
            else:
                return "HOLD"

        except Exception as e:
            print(f"執行 {symbol} 進場預測時出錯: {e}")
            return "HOLD"
