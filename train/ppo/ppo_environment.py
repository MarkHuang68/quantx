# 檔案: train/ppo/ppo_environment.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

import settings
from utils.common import create_features_trend

class TradingEnvironment(gym.Env):
    def __init__(self, df_data, initial_balance=10000, leverage=5, commission=0.0004, reward_window_size=100):
        super().__init__()

        self.df_data = df_data
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.commission = commission
        self.reward_window_size = reward_window_size # 新增：計算波動性的窗口大小

        # 環境參數
        self.features = [col for col in df_data.columns if col not in ['Open', 'High', 'Low', 'Close']]
        self.n_features = len(self.features)
        self.max_timesteps = len(self.df_data) - 1

        # 動作空間
        self.action_map = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.action_map))

        # 觀察狀態空間
        obs_low = [-np.inf] * (self.n_features - 1) + [-1, -1, 0]
        obs_high = [np.inf] * (self.n_features - 1) + [1, 1, np.inf]
        self.observation_space = spaces.Box(low=np.array(obs_low), high=np.array(obs_high), dtype=np.float32)

        # 內部狀態
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance # 新增：用於計算最大回撤
        self.position = 0.0
        self.entry_price = 0.0
        self.pnl_history = [] # 新增：記錄 PnL 歷史
        self.net_worth_history = [] # 新增：記錄淨值歷史
        return self._get_observation(), {}

    def step(self, action):
        target_position = self.action_map[action]
        price_now = self.df_data['Close'].iloc[self.current_step]
        price_next = self.df_data['Close'].iloc[self.current_step + 1]

        # 計算原始 PnL
        price_change_ratio = (price_next - price_now) / price_now
        pnl_pct = self.position * price_change_ratio * self.leverage
        self.pnl_history.append(pnl_pct)

        # 更新狀態
        self.net_worth *= (1 + pnl_pct)
        self.net_worth_history.append(self.net_worth)
        self.max_net_worth = max(self.max_net_worth, self.net_worth) # 更新最高淨值

        # --- 全新的風險調整後獎勵機制 ---
        reward = self._calculate_risk_adjusted_reward(pnl_pct, target_position)

        # 更新倉位
        self.position = target_position
        self.entry_price = price_now if target_position != 0 else 0

        self.current_step += 1
        done = self.current_step >= self.max_timesteps or self.net_worth < self.initial_balance * 0.5

        # 在 episode 結束時，增加對最大回撤的懲罰
        if done:
            max_drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth
            # 懲罰與回撤的平方成正比，以加重對大幅度回撤的懲罰
            drawdown_penalty = 5 * (max_drawdown ** 2)
            reward -= drawdown_penalty

        return self._get_observation(), reward, done, False, {'net_worth': self.net_worth}

    def _calculate_risk_adjusted_reward(self, pnl_pct, target_position):
        """
        計算考慮了風險的獎勵。
        """
        # 交易成本懲罰
        trade_cost = abs(target_position - self.position) * self.commission

        # 使用夏普比率的思想調整獎勵
        if len(self.pnl_history) < self.reward_window_size:
            # 在歷史數據不足時，使用簡化獎勵
            return pnl_pct - trade_cost

        # 計算近期 PnL 的波動性 (標準差)
        recent_pnl = self.pnl_history[-self.reward_window_size:]
        pnl_std = np.std(recent_pnl) + 1e-6 # 加上一個極小值避免除以零

        # 夏普獎勵：回報與風險的比值
        sharpe_reward = pnl_pct / pnl_std

        return sharpe_reward - trade_cost

    def _get_observation(self):
        features = self.df_data[self.features].iloc[self.current_step].values
        account_state = np.array([self.position, self.net_worth / self.initial_balance])
        return np.concatenate([features, account_state]).astype(np.float32)

def prepare_data_for_ppo(symbol, ohlcv_data):
    """
    為 PPO 訓練準備數據，包括計算並標準化 XGBoost 訊號。
    """
    print(f"--- 正在為 {symbol} 準備 PPO 訓練數據 ---")

    try:
        model_path = settings.get_trend_model_path(symbol, '1m', settings.TREND_MODEL_VERSION)
        model = xgb.XGBClassifier()
        model.load_model(model_path)
    except Exception as e:
        print(f"🛑 錯誤：無法載入 {symbol} 的 XGBoost 模型。請先訓練模型。 {e}")
        return None

    df_features, features_list = create_features_trend(ohlcv_data.copy())
    raw_signal = model.predict(df_features[features_list]).astype(int)

    signal_map = {1: 1, 2: -1, 0: 0}
    df_features['xgb_signal'] = pd.Series(raw_signal, index=df_features.index).map(signal_map)

    ppo_features = features_list + ['xgb_signal', 'Close']
    df_ppo = df_features[ppo_features].dropna()

    scaler = StandardScaler()
    df_ppo[features_list] = scaler.fit_transform(df_ppo[features_list])

    return df_ppo
