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
        self.xgb_model = self._load_xgb_model(symbol)
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

    def _load_xgb_model(self, symbol):
        print(f"--- 正在為 PPO 管理器載入 XGBoost 模型: {symbol} ---")
        try:
            model_path = get_trend_model_path(symbol, TREND_MODEL_VERSION)
            model = xgb.XGBClassifier()
            model.load_model(model_path)
            print("✅ XGBoost 模型載入成功！")
            return model
        except Exception as e:
            print(f"🛑 錯誤：無法載入 XGBoost 模型。{e}")
            return None

    def get_action(self, ohlcv_data, portfolio_state, xgb_prediction):
        if not self.model or not self.xgb_model:
            print("PPO 管理器未初始化，返回預設動作 (空手)。")
            return 2 # 2 對應於空手 (做多、做空、空手)

        # 1. 建立觀察狀態
        observation = self._create_observation(ohlcv_data, portfolio_state, xgb_prediction)

        # 2. 使用 PPO 模型預測動作
        action, _ = self.model.predict(observation, deterministic=True)

        return action

    def _create_observation(self, df, portfolio_state, xgb_prediction):
        # a. 計算特徵
        df_features, features_list = create_features_trend(df.copy())

        # 標準化
        df_features[features_list] = self.scaler.fit_transform(df_features[features_list])

        # b. 獲取最新的特徵
        latest_features = df_features[features_list].iloc[-1].values

        # c. 獲取帳戶狀態
        position = portfolio_state.get('position', 0)
        net_worth_ratio = portfolio_state.get('net_worth_ratio', 1.0)
        account_state = np.array([position, net_worth_ratio])

        # d. 組合最終的觀察狀態 (市場特徵 + XGBoost訊號 + 帳戶狀態)
        observation = np.concatenate([latest_features, [xgb_prediction], account_state]).astype(np.float32)
        return observation
