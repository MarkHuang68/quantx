# 檔案: strategies/ppo_strategy.py

import numpy as np
import pandas as pd
import xgboost as xgb
from stable_baselines3 import PPO

from strategies.base_strategy import BaseStrategy
from utils.common import create_features_trend
from train.ppo.ppo_environment import TradingEnvironment # 引用 PPO 環境以了解其狀態空間

class PPOStrategy(BaseStrategy):
    def __init__(self, context, model_path, symbols):
        super().__init__(context)
        self.symbols = symbols
        self.model = self._load_model(model_path)

        # 為了建構 PPO 的觀察狀態 (observation)，我們需要 XGBoost 模型
        self.xgb_models = self._load_xgb_models()

        # 建立一個臨時的 PPO 環境以獲取特徵列表
        # 注意：這是一個簡化的方法，理想情況下，特徵列表應該被明確地管理
        temp_env = TradingEnvironment(symbol=self.symbols[0])
        self.features_list = temp_env.features_list

    def _load_model(self, model_path):
        """
        載入預先訓練好的 PPO 模型。
        """
        print(f"--- 正在載入 PPO 模型: {model_path} ---")
        try:
            model = PPO.load(model_path)
            print("✅ PPO 模型載入成功！")
            return model
        except Exception as e:
            print(f"🛑 錯誤：無法載入 PPO 模型。{e}")
            return None

    def _load_xgb_models(self):
        """
        載入 XGBoost 模型，用於產生 PPO 的輸入特徵。
        (這部分邏輯是基於 ppo_trading_tool.py)
        """
        xgb_models = {}
        # 這裡需要根據您的模型命名規則進行調整
        # 為了簡化，我們假設模型檔案名是固定的
        try:
            xgb_models['short'] = xgb.Booster(model_file='models/entry_model_XGB_ETH_USDT_1m_v1.0.json')
            xgb_models['mid'] = xgb.Booster(model_file='models/entry_model_XGB_ETH_USDT_5m_v1.0.json')
            xgb_models['long'] = xgb.Booster(model_file='models/entry_model_XGB_ETH_USDT_15m_v1.0.json')
            print("✅ XGBoost 特徵模型載入成功！")
        except Exception as e:
            print(f"🛑 錯誤：無法載入 XGBoost 模型。{e}")
        return xgb_models

    def on_bar(self, dt):
        if not self.model or not self.xgb_models:
            print("PPO 策略未初始化，跳過。")
            return

        for symbol in self.symbols:
            self._process_symbol(symbol, dt)

    def _process_symbol(self, symbol, dt):
        try:
            # 1. 獲取市場數據
            ohlcv = self.context.exchange.get_ohlcv(symbol, '1m', limit=200) # 假設 PPO 使用 1m 數據
            if ohlcv.empty:
                return

            # 2. 建立觀察狀態 (Observation)
            observation = self._create_observation(ohlcv)

            # 3. 使用 PPO 模型預測動作
            action, _ = self.model.predict(observation, deterministic=True)

            # 將 PPO 的連續動作轉換為目標倉位 (-1.0 to 1.0)
            target_position = np.clip(action[0], -1, 1)

            # 4. 執行交易
            self._execute_trade(symbol, target_position)

        except Exception as e:
            print(f"在 {symbol} 上執行 PPO 策略時出錯: {e}")

    def _create_observation(self, df):
        """
        根據當前的市場數據建立 PPO 模型的觀察狀態。
        這部分的邏輯需要與 ppo_environment.py 中的 _get_observation 保持一致。
        """
        # a. 計算 XGBoost 預測
        df_features, features_list = create_features_trend(df.copy())
        X_dmatrix = xgb.DMatrix(df_features[features_list])

        df_features['short_pred'] = (self.xgb_models['short'].predict(X_dmatrix) > 0.5).astype(int)
        df_features['mid_pred'] = (self.xgb_models['mid'].predict(X_dmatrix) > 0.5).astype(int)
        df_features['long_pred'] = (self.xgb_models['long'].predict(X_dmatrix) > 0.5).astype(int)

        # b. 獲取最新的特徵
        latest_features = df_features[self.features_list].iloc[-1].values

        # c. 獲取帳戶狀態
        # 注意：這裡的帳戶狀態需要與 PPO 環境訓練時的定義相匹配
        # 為了簡化，我們使用一些預設值
        balance_ratio = self.context.portfolio.get_total_value() / self.context.initial_capital
        current_position = self.context.portfolio.get_positions().get('BTC', 0) # 假設我們交易 BTC

        # d. 組合最終的觀察狀態
        account_state = np.array([balance_ratio, current_position])
        observation = np.concatenate([latest_features, account_state])

        return observation

    def _execute_trade(self, symbol, target_position):
        """
        根據 PPO 模型的目標倉位執行交易。
        """
        # 獲取當前倉位
        # (這裡需要一個更完善的方法來獲取以標的資產計價的倉位比例)
        current_position = self.context.portfolio.get_positions().get(symbol.split('/')[0], 0)

        # 為了簡化，我們假設倉位是 0 或 1
        current_position_ratio = 1 if current_position > 0 else 0

        # 計算需要執行的訂單
        # (這是一個簡化的邏輯，實際的倉位管理會更複雜)
        if target_position > 0.5 and current_position_ratio == 0:
            print(f"PPO 決策：在 {symbol} 上做多")
            # self.context.exchange.create_order(symbol, 'market', 'buy', 0.01)
        elif target_position < -0.5 and current_position_ratio > 0:
            print(f"PPO 決策：在 {symbol} 上平倉")
            # self.context.exchange.create_order(symbol, 'market', 'sell', current_position)
        else:
            print(f"PPO 決策：在 {symbol} 上無操作 (目標倉位: {target_position:.2f})")
