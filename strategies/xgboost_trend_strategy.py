# 檔案: strategies/xgboost_trend_strategy.py

import pandas as pd
import numpy as np
import xgboost as xgb

from strategies.base_strategy import BaseStrategy
from utils.common import create_features_trend
from config.settings import SYMBOLS_TO_TRADE, TREND_MODEL_VERSION, get_trend_model_path
from core.ppo_manager import PPOManager

class XGBoostTrendStrategy(BaseStrategy):
    def __init__(self, context, symbols=SYMBOLS_TO_TRADE, use_ppo=False, ppo_model_path=None):
        super().__init__(context)
        self.symbols = symbols
        self.use_ppo = use_ppo
        self.models = {}
        self._load_models()

        if self.use_ppo:
            if not ppo_model_path:
                raise ValueError("使用 PPO 時，必須提供 PPO 模型路徑")
            self.ppo_managers = {symbol: PPOManager(ppo_model_path, symbol) for symbol in self.symbols}

    def _load_models(self):
        print("--- 正在載入 XGBoost 趨勢模型... ---")
        for symbol in self.symbols:
            try:
                model_path = get_trend_model_path(symbol, TREND_MODEL_VERSION)
                model = xgb.Booster()
                model.load_model(model_path)
                self.models[symbol] = model
                print(f"✅ {symbol} 的 XGBoost 模型載入成功！")
            except Exception as e:
                print(f"🛑 警告：無法載入 {symbol} 的模型。")
                pass

    def on_bar(self, dt):
        for symbol in self.symbols:
            if symbol not in self.models:
                print(f"--- ({symbol}) 缺少模型，跳過 ---")
                continue

            if self.use_ppo:
                self._process_symbol_with_ppo(symbol, dt)
            else:
                self._process_symbol_with_rules(symbol, dt)

    def _get_xgb_prediction(self, symbol, ohlcv):
        df_with_features, features_list = create_features_trend(ohlcv)
        dmatrix = xgb.DMatrix(df_with_features[features_list].iloc[-1:])
        # 假設模型輸出為: 0 (做空), 1 (空手), 2 (做多)
        prediction = self.models[symbol].predict(dmatrix)[0]
        return int(prediction)

    def _process_symbol_with_ppo(self, symbol, dt):
        print(f"\n--- 正在使用 PPO 處理 {symbol} ---")
        ohlcv = self.context.exchange.get_ohlcv(symbol, '5m', limit=200) # 假設使用 5m 數據
        if ohlcv.empty:
            return

        portfolio_state = {
            'position': self.context.portfolio.get_positions().get(symbol.split('/')[0], 0),
            'net_worth_ratio': self.context.portfolio.get_total_value() / self.context.initial_capital
        }

        # 將 XGBoost 訊號傳遞給 PPO
        xgb_prediction = self._get_xgb_prediction(symbol, ohlcv)
        action = self.ppo_managers[symbol].get_action(ohlcv, portfolio_state, xgb_prediction)
        target_position = self.ppo_managers[symbol].model.env.get_attr('action_map')[0][action]
        current_position_value = self.context.portfolio.get_positions().get(symbol.split('/')[0], 0)

        # 根據 PPO 的目標倉位調整下單
        # (這是一個簡化的邏輯，實際應用中可能需要更複雜的計算)
        if target_position > 0 and current_position_value == 0:
            amount_to_buy = 0.01 * target_position # 根據 PPO 的輸出調整倉位
            print(f"PPO 決策 for {symbol}: 執行做多 (Buy) {amount_to_buy}！")
            self.context.exchange.create_order(symbol, 'market', 'buy', amount_to_buy)
        elif target_position < 0 and current_position_value == 0:
            amount_to_sell = 0.01 * abs(target_position)
            print(f"PPO 決策 for {symbol}: 執行做空 (Sell) {amount_to_sell}！")
            self.context.exchange.create_order(symbol, 'market', 'sell', amount_to_sell)
        elif target_position == 0 and current_position_value != 0:
            print(f"PPO 決策 for {symbol}: 執行平倉！")
            self.context.exchange.create_order(symbol, 'market', 'sell' if current_position_value > 0 else 'buy', abs(current_position_value))
        else:
            print(f"PPO 決策 for {symbol}: 持有 (Hold)。")

    def _process_symbol_with_rules(self, symbol, dt):
        print(f"\n--- 正在使用規則處理 {symbol} ---")
        ohlcv = self.context.exchange.get_ohlcv(symbol, '5m', limit=200) # 假設使用 5m 數據
        if ohlcv.empty:
            return

        prediction = self._get_xgb_prediction(symbol, ohlcv)
        current_position = self.context.portfolio.get_positions().get(symbol.split('/')[0], 0)

        if prediction == 2 and current_position == 0: # 做多
            print(f"✅ {symbol} 決策: 執行做多 (Buy)！")
            self.context.exchange.create_order(symbol, 'market', 'buy', 0.01)
        elif prediction == 0 and current_position > 0: # 做空 (平多)
            print(f"🛑 {symbol} 決策: 執行平倉 (Sell)！")
            self.context.exchange.create_order(symbol, 'market', 'sell', current_position)
        else: # 空手
            print(f"⬜ {symbol} 決策: 持有 (Hold)。")
