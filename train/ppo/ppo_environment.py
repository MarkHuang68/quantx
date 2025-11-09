# 檔案: train/ppo/ppo_environment.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

import settings
from utils.common import create_features_trend, create_sequences

class TradingEnvironment(gym.Env):
    def __init__(self, df_data, initial_balance=10000, leverage=5, commission=0.0004):
        super().__init__()

        self.df_data = df_data
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.commission = commission

        # 環境參數
        self.features = [col for col in df_data.columns if col not in ['Open', 'High', 'Low', 'Close']]
        self.n_features = len(self.features)
        self.max_timesteps = len(self.df_data) - 1

        # 動作空間: 離散的倉位大小 (-1.0 到 1.0, 間隔 0.2)
        self.action_map = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.action_map))

        # 觀察狀態空間: [市場特徵, XGBoost訊號, 倉位, 淨值比例]
        low = np.array([-np.inf] * self.n_features + [-1, 0])
        high = np.array([np.inf] * self.n_features + [1, np.inf])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # 內部狀態
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        return self._get_observation(), {}

    def step(self, action):
        target_position = self.action_map[action]
        price_now = self.df_data['Close'].iloc[self.current_step]
        price_next = self.df_data['Close'].iloc[self.current_step + 1]

        # 計算 PnL 和獎勵
        price_change_ratio = (price_next - price_now) / price_now
        pnl_pct = self.position * price_change_ratio * self.leverage
        reward = pnl_pct - (abs(target_position - self.position) * self.commission) # 簡化的交易成本

        # 更新狀態
        self.net_worth *= (1 + pnl_pct)
        self.position = target_position
        self.entry_price = price_now if target_position != 0 else 0

        self.current_step += 1
        done = self.current_step >= self.max_timesteps or self.net_worth < self.initial_balance * 0.5

        return self._get_observation(), reward, done, False, {'net_worth': self.net_worth}

    def _get_observation(self):
        features = self.df_data[self.features].iloc[self.current_step].values
        account_state = np.array([self.position, self.net_worth / self.initial_balance])
        return np.concatenate([features, account_state]).astype(np.float32)

def prepare_data_for_ppo(symbol, ohlcv_data):
    """
    為 PPO 訓練準備數據，包括計算 XGBoost 訊號。
    """
    print(f"--- 正在為 {symbol} 準備 PPO 訓練數據 ---")

    try:
        # 載入預先訓練好的 XGBoost 模型
        model_path = settings.get_trend_model_path(symbol, settings.TREND_MODEL_VERSION)
        model = xgb.XGBClassifier()
        model.load_model(model_path)
    except Exception as e:
        print(f"🛑 錯誤：無法載入 {symbol} 的 XGBoost 模型。請先訓練模型。 {e}")
        return None

    # 計算特徵
    df_features, features_list = create_features_trend(ohlcv_data.copy())

    # 計算 XGBoost 訊號
    # 假設模型輸出為: 0 (做空), 1 (空手), 2 (做多)
    df_features['xgb_signal'] = model.predict(df_features[features_list]).astype(int)

    # 選取 PPO 的輸入特徵 (包括 XGBoost 訊號)
    ppo_features = features_list + ['xgb_signal', 'Close']
    df_ppo = df_features[ppo_features].dropna()

    # 標準化
    scaler = StandardScaler()
    df_ppo[features_list] = scaler.fit_transform(df_ppo[features_list])

    return df_ppo
