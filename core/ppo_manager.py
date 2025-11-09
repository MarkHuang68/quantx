# 檔案: core/ppo_manager.py

import numpy as np
import pandas as pd
import xgboost as xgb
from stable_baselines3 import PPO
from sklearn.preprocessing import StandardScaler

import settings
from utils.common import create_features_trend

class PPOManager:
    def __init__(self, model_path, symbol, timeframe, version):
        self.initialized = False
        # 直接硬編碼 action_map
        self.action_map = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float32)
        print(f"✅ Action Map 已設定: {self.action_map}")

        self.model = self._load_model(model_path)
        self.xgb_model = self._load_xgb_model(symbol, timeframe, version)
        self.scaler = StandardScaler()

        if self.model and self.xgb_model:
            self.initialized = True
        else:
            print(f"🛑 PPO 管理器初始化失敗！狀態：model={self.model is not None}, xgb_model={self.xgb_model is not None}")

        print(f"--- PPO Manager 最終初始化狀態 for {symbol}: self.initialized = {self.initialized} ---")

    def _load_model(self, model_path):
        if not model_path:
            print("🛑 錯誤：未提供 PPO 模型路徑。")
            return None
        print(f"--- 正在載入 PPO 模型: {model_path} ---")
        try:
            model = PPO.load(model_path)
            print(f"✅ PPO 模型載入成功！")
            return model
        except Exception as e:
            print(f"🛑 錯誤：無法載入 PPO 模型。{e}")
            return None

    def _load_xgb_model(self, symbol, timeframe, version):
        print(f"--- 正在為 PPO 管理器載入 XGBoost 模型: {symbol} ({timeframe}, v{version}) ---")
        try:
            model_path = settings.get_trend_model_path(symbol, timeframe, version)
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

        # d. 組合最終的觀察狀態 (xgb_prediction 已被標準化為 -1, 0, 1)
        observation = np.concatenate([latest_features, [xgb_prediction], account_state]).astype(np.float32)
        return observation
