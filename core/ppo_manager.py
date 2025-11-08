# 檔案: core/ppo_manager.py

import numpy as np
import pandas as pd
import xgboost as xgb
from stable_baselines3 import PPO
from sklearn.preprocessing import StandardScaler

from config.settings import TREND_MODEL_VERSION, get_trend_model_path
from utils.common import create_features_trend

class PPOManager:
    def __init__(self, model_path, symbol):
        self.model = self._load_model(model_path)
        self.xgb_models = self._load_xgb_models(symbol)
        self.scaler = StandardScaler()

    def _load_model(self, model_path):
        print(f"--- 正在載入 PPO 模型: {model_path} ---")
        try:
            model = PPO.load(model_path)
            print("✅ PPO 模型載入成功！")
            return model
        except Exception as e:
            print(f"🛑 錯誤：無法載入 PPO 模型。{e}")
            return None

    def _load_xgb_models(self, symbol):
        print(f"--- 正在為 PPO 管理器載入 XGBoost 模型: {symbol} ---")
        xgb_models = {}
        try:
            trend_model_path = get_trend_model_path(symbol, TREND_MODEL_VERSION)
            entry_model_path = get_trend_model_path(symbol, TREND_MODEL_VERSION) # 假設路徑邏輯相同

            xgb_models['trend'] = xgb.Booster()
            xgb_models['trend'].load_model(trend_model_path)

            xgb_models['entry'] = xgb.Booster()
            xgb_models['entry'].load_model(entry_model_path)
            print("✅ XGBoost 模型載入成功！")
        except Exception as e:
            print(f"🛑 錯誤：無法載入 XGBoost 模型。{e}")
        return xgb_models

    def get_action(self, ohlcv_data, portfolio_state):
        if not self.model or not self.xgb_models:
            print("PPO 管理器未初始化，返回預設動作 (空手)。")
            return 0 # 0 對應於空手

        # 1. 建立觀察狀態
        observation = self._create_observation(ohlcv_data, portfolio_state)

        # 2. 使用 PPO 模型預測動作
        action, _ = self.model.predict(observation, deterministic=True)

        return action

    def _create_observation(self, df, portfolio_state):
        # a. 計算 XGBoost 訊號
        df_features, features_list = create_features_trend(df.copy())

        # 標準化
        df_features[features_list] = self.scaler.fit_transform(df_features[features_list])

        dmatrix = xgb.DMatrix(df_features[features_list])
        df_features['trend_signal'] = (self.xgb_models['trend'].predict(dmatrix) > 0.5).astype(int) * 2 - 1
        df_features['entry_signal'] = (self.xgb_models['entry'].predict(dmatrix) > 0.5).astype(int) * 2 - 1

        # b. 獲取最新的特徵
        latest_features = df_features[features_list + ['trend_signal', 'entry_signal']].iloc[-1].values

        # c. 獲取帳戶狀態
        position = portfolio_state.get('position', 0)
        net_worth_ratio = portfolio_state.get('net_worth_ratio', 1.0)
        account_state = np.array([position, net_worth_ratio])

        # d. 組合最終的觀察狀態
        observation = np.concatenate([latest_features, account_state]).astype(np.float32)
        return observation
