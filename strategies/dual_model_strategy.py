# 檔案: strategies/dual_model_strategy.py

import pandas as pd
import numpy as np
import tensorflow as tf
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler

from strategies.base_strategy import BaseStrategy
from utils.common import create_features_trend, create_sequences
from config.settings import SYMBOLS_TO_TRADE, TREND_MODEL_VERSION, TREND_MODEL_TIMEFRAME, get_trend_model_path, MODEL_DIR
from core.ppo_manager import PPOManager

class DualModelStrategy(BaseStrategy):
    def __init__(self, context, symbols=SYMBOLS_TO_TRADE, use_ppo=False, ppo_model_path=None):
        super().__init__(context)
        self.symbols = symbols
        self.use_ppo = use_ppo

        if self.use_ppo:
            if not ppo_model_path:
                raise ValueError("使用 PPO 時，必須提供 PPO 模型路徑")
            self.ppo_managers = {symbol: PPOManager(ppo_model_path, symbol) for symbol in self.symbols}
        else:
            self.trend_state = {symbol: "NEUTRAL" for symbol in self.symbols}
            self.last_check = {symbol: None for symbol in self.symbols}
            self.scalers = {symbol: MinMaxScaler(feature_range=(0, 1)) for symbol in self.symbols}
            self.trend_models = {}
            self.entry_models = {}
            self._load_models()

    def _load_models(self):
        print("--- 正在載入 XGBoost 模型... ---")
        for symbol in self.symbols:
            try:
                trend_model_path = get_trend_model_path(symbol, TREND_MODEL_VERSION)
                self.trend_models[symbol] = tf.keras.models.load_model(trend_model_path)

                entry_model_path = get_trend_model_path(symbol, TREND_MODEL_VERSION)
                self.entry_models[symbol] = xgb.Booster()
                self.entry_models[symbol].load_model(entry_model_path)
                print(f"✅ {symbol} 的模型載入成功！")
            except Exception as e:
                print(f"🛑 警告：無法載入 {symbol} 的模型。")
                pass

    def on_bar(self, dt):
        for symbol in self.symbols:
            if self.use_ppo:
                self._process_symbol_with_ppo(symbol, dt)
            else:
                if symbol not in self.trend_models or symbol not in self.entry_models:
                    print(f"--- ({symbol}) 缺少模型，跳過 ---")
                    continue
                self._process_symbol_with_rules(symbol, dt)

    def _process_symbol_with_ppo(self, symbol, dt):
        print(f"\n--- 正在使用 PPO 處理 {symbol} ---")
        ohlcv = self.context.exchange.get_ohlcv(symbol, '1m', limit=200) # 假設 PPO 使用 1m 數據
        if ohlcv.empty:
            return

        portfolio_state = {
            'position': self.context.portfolio.get_positions().get(symbol.split('/')[0], 0),
            'net_worth_ratio': self.context.portfolio.get_total_value() / self.context.initial_capital
        }

        action = self.ppo_managers[symbol].get_action(ohlcv, portfolio_state)
        target_position = self.ppo_managers[symbol].model.env.get_attr('action_map')[0][action]
        current_position = self.context.portfolio.get_positions().get(symbol.split('/')[0], 0)

        # 簡單的倉位管理邏輯
        if target_position > 0 and current_position == 0:
            amount_to_buy = 0.01 * target_position # 根據 PPO 的輸出調整倉位
            print(f"PPO 決策 for {symbol}: 執行做多 (Buy) {amount_to_buy}！")
            self.context.exchange.create_order(symbol, 'market', 'buy', amount_to_buy)
        elif target_position == 0 and current_position > 0:
            print(f"PPO 決策 for {symbol}: 執行平倉 (Sell)！")
            self.context.exchange.create_order(symbol, 'market', 'sell', current_position)
        else:
            print(f"PPO 決策 for {symbol}: 持有 (Hold)。")

    def _process_symbol_with_rules(self, symbol, dt):
        print(f"\n--- 正在處理 {symbol} ---")

        if self.last_check[symbol] is None or dt.hour != self.last_check[symbol].hour:
            if dt.minute < 5:
                self._update_trend_signal(symbol)
                self.last_check[symbol] = dt

        entry_signal = self._get_entry_signal(symbol)

        symbol_trend = self.trend_state[symbol]
        current_position = self.context.portfolio.get_positions().get(symbol.split('/')[0], 0)

        if symbol_trend == "UP" and entry_signal == "BUY" and current_position == 0:
            print(f"✅ {symbol} 決策: 執行做多 (Buy)！")
            self.context.exchange.create_order(symbol, 'market', 'buy', 0.01)
        elif symbol_trend == "DOWN" and entry_signal == "SELL" and current_position > 0:
            print(f"🛑 {symbol} 決策: 執行平倉 (Sell)！")
            self.context.exchange.create_order(symbol, 'market', 'sell', current_position)
        else:
            print(f"⬜ {symbol} 決策: 持有 (Hold)。")

    def _update_trend_signal(self, symbol):
        try:
            ohlcv = self.context.exchange.get_ohlcv(symbol, TREND_MODEL_TIMEFRAME, limit=200)
            df_with_features, features_list = create_features_trend(ohlcv)

            last_sequence_data = df_with_features[features_list].iloc[-60:]
            scaled_sequence = self.scalers[symbol].fit_transform(last_sequence_data)
            X_live = np.array([scaled_sequence])

            prediction_proba = self.trend_models[symbol].predict(X_live, verbose=0)
            prediction = (prediction_proba > 0.5).astype(int)[0][0]

            self.trend_state[symbol] = "UP" if prediction == 1 else "DOWN"
        except Exception as e:
            self.trend_state[symbol] = "NEUTRAL"

    def _get_entry_signal(self, symbol):
        try:
            ohlcv = self.context.exchange.get_ohlcv(symbol, '5m', limit=100)
            df_with_features, features_list = create_features_trend(ohlcv)

            last_features = df_with_features[features_list].iloc[-1:]
            X_live = xgb.DMatrix(last_features)

            predicted_return = self.entry_models[symbol].predict(X_live)[0]

            if predicted_return > 0.0001: return "BUY"
            elif predicted_return < -0.0001: return "SELL"
            else: return "HOLD"
        except Exception:
            return "HOLD"
