# 檔案: train_trend_model_ppo.py

import pandas as pd
import numpy as np
import argparse
import warnings
import os
import json
import gym  # 用於定義強化學習環境
from gym import spaces  # 定義狀態和動作空間
from stable_baselines3 import PPO  # PPO 演算法 (需安裝 stable_baselines3)
from stable_baselines3.common.vec_env import DummyVecEnv  # 向量環境包裝
from stable_baselines3.common.callbacks import EvalCallback  # 評估回調
import matplotlib.pyplot as plt  # 用於繪圖

# --- 1. 引用「設定檔」和「共用工具箱」 ---
import config
from common_utils import fetch_data, create_features_trend

warnings.simplefilter(action='ignore', category=FutureWarning)

# --- 新增: 定義「中立區」門檻 ---
# 漲跌幅在 +/- 0.2% 以內，都視為「持有 (0)」
HOLD_THRESHOLD = 0.002

# --- PPO 訓練基礎參數 ---
PPO_PARAMS = {
    'policy': 'MlpPolicy',  # 使用多層感知器策略
    'learning_rate': 0.0003,  # 學習率
    'n_steps': 2048,  # 每批次步數
    'batch_size': 64,  # 批次大小
    'n_epochs': 10,  # 每更新迭代次數
    'gamma': 0.99,  # 折扣因子
    'gae_lambda': 0.95,  # GAE 參數
    'clip_range': 0.2,  # PPO 裁剪範圍
    'ent_coef': 0.01,  # 熵係數
    'verbose': 1,  # 顯示訓練資訊
}

# --- 定義交易環境 (Gym 環境) ---
class TradingEnv(gym.Env):
    """
    自訂交易環境:
    - 狀態: 特徵向量 (從 create_features_trend 計算)
    - 動作: 0=持有, 1=買入 (做多), 2=賣出 (做空)
    - 獎勵: 淨收益 (考慮手續費)
    - 觀察: 當前特徵
    """
    def __init__(self, df):
        super(TradingEnv, self).__init__()
        self.df = df.reset_index()  # 重置索引以便步進
        self.current_step = 0  # 當前步數
        self.action_space = spaces.Discrete(3)  # 動作空間: 0,1,2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(features_list),))  # 狀態空間: 特徵維度
        self.position = 0  # 當前持倉: 0=無, 1=多, -1=空
        self.balance = 1.0  # 初始淨值
        self.FEE_RATE = 0.001  # 手續費率

    def reset(self):
        self.current_step = 0
        self.position = 0
        self.balance = 1.0
        return self._get_observation()

    def step(self, action):
        # 計算實際報酬
        actual_return = (self.df['Close'].iloc[self.current_step + 1] - self.df['Close'].iloc[self.current_step]) / self.df['Close'].iloc[self.current_step]
        
        # 根據動作更新持倉
        prev_position = self.position
        if action == 1:  # 買入
            self.position = 1
        elif action == 2:  # 賣出
            self.position = -1
        else:  # 持有
            self.position = 0
        
        # 計算交易成本 (若持倉變化)
        trade_cost = self.FEE_RATE if prev_position != self.position else 0
        
        # 獎勵: 持倉 * 報酬 - 成本
        reward = self.position * actual_return - trade_cost
        self.balance += reward  # 更新淨值
        
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1  # 是否結束
        
        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        # 獲取當前特徵
        return self.df[features_list].iloc[self.current_step].values

def train_ppo(df_features, features_list):
    """
    訓練 PPO 模型 (強化學習代理)
    - 建立環境
    - 分割訓練/測試集 (80/20)
    - 訓練 PPO 代理
    - 評估測試集表現 (總獎勵/淨值)
    - 向量化回測: 產生 signal, 計算淨值曲線
    - 繪製淨值曲線 (需 --show_equity)
    - 返回模型及總獎勵
    """
    if df_features is None: return None, 0.0

    # 分割訓練/測試集
    split_index = int(len(df_features) * 0.8)
    train_df = df_features.iloc[:split_index]
    test_df = df_features.iloc[split_index:]

    # 建立向量環境
    train_env = DummyVecEnv([lambda: TradingEnv(train_df)])
    eval_env = DummyVecEnv([lambda: TradingEnv(test_df)])

    # 建立 PPO 模型
    model = PPO(env=train_env, **PPO_PARAMS)

    # 訓練 (使用評估回調)
    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/', log_path='./logs/', eval_freq=10000, deterministic=True, render=False)
    model.learn(total_timesteps=100000, callback=eval_callback)  # 調整步數以訓練時間

    # 測試集評估
    obs = eval_env.reset()
    done = False
    total_reward = 0
    signals = []
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, _ = eval_env.step(action)
        total_reward += reward
        signals.append(action[0])  # 記錄動作作為 signal

    print(f"測試集總獎勵: {total_reward:.4f}")

    # --- 向量化回測 (預設不顯示，需 --show_equity) ---
    if args.show_equity:
        print("\n--- PPO 策略回測 ---")
        df_test = test_df.copy()
        df_test['signal'] = [0 if s == 0 else 1 if s == 1 else -1 for s in signals]  # 轉換 signal
        df_test['actual_return'] = df_test['Close'].pct_change()
        df_test['strategy_return'] = df_test['signal'].shift(1) * df_test['actual_return']
        df_test['trades'] = df_test['signal'].diff().abs()
        FEE_RATE = 0.001
        df_test['transaction_costs'] = df_test['trades'] * FEE_RATE
        df_test['strategy_net_return'] = df_test['strategy_return'] - df_test['transaction_costs']
        df_test['strategy_gross_equity'] = (1 + df_test['strategy_return']).cumprod()
        df_test['strategy_net_equity'] = (1 + df_test['strategy_net_return']).cumprod()
        df_test['bh_equity'] = (1 + df_test['actual_return']).cumprod()
        
        # 繪製曲線
        plt.rc('font', family='MingLiu')
        plt.figure(figsize=(12, 6))
        plt.plot(df_test['bh_equity'], label='Buy & Hold', color='gray')
        plt.plot(df_test['strategy_gross_equity'], label='策略 (未扣費)', color='blue')
        plt.plot(df_test['strategy_net_equity'], label='策略 (扣費後)', color='red')
        plt.title('回測淨值曲線 (測試集)')
        plt.xlabel('時間步')
        plt.ylabel('淨值')
        plt.legend()
        plt.grid(True)
        plt.show()

    return model, total_reward

if __name__ == "__main__":

    # --- 參數解析器 (保持原樣) ---
    parser = argparse.ArgumentParser(description=f'訓練 {config.TREND_MODEL_TIMEFRAME} PPO 趨勢模型')

    parser.add_argument('-s', '--symbol', type=str, required=True, help='要訓練的交易對 (例如: ETH/USDT 或 BTC/USDT)')
    parser.add_argument('-tf', '--timeframe', type=str, required=True, help='要訓練的TimeFrame 例如:5m, 15m, 1h')
    parser.add_argument('-sd', '--start', type=str, help='回測起始日期 (YYYY-MM-DD)')
    parser.add_argument('-ed', '--end', type=str, help='回測結束日期 (YYYY-MM-DD)')
    parser.add_argument('-ns', '--no_search_params', action='store_true', help='關閉尋找模型最佳參數')
    parser.add_argument('-l', '--limit', type=int, help=f'K 線筆數限制')
    parser.add_argument('-v', '--version', type=str, default=config.TREND_MODEL_VERSION, help=f'要訓練的模型版本 (預設: {config.TREND_MODEL_VERSION})')
    parser.add_argument('--show_confidence', action='store_true', help='顯示高信心混淆矩陣 (預設不顯示)')
    parser.add_argument('--show_equity', action='store_true', help='顯示資金曲線 (預設不顯示)')
    parser.add_argument('--show_overfit', action='store_true', help='顯示過擬合檢測學習曲線 (預設不顯示)')

    args = parser.parse_args()

    # --- 執行訓練 ---
    print(f"--- 開始執行 PPO: {args.symbol} ({args.timeframe}), 資料量={args.limit} ---")

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    raw_df = fetch_data(symbol=args.symbol, start_date=args.start, end_date=args.end, timeframe=args.timeframe, total_limit=args.limit)

    # --- 計算特徵 ---
    df_features, features_list = create_features_trend(raw_df.copy())
    if df_features is None or features_list is None:
        print(f"特徵計算失敗，結束訓練。")
        exit()

    # --- 訓練 PPO ---
    model, total_reward = train_ppo(df_features, features_list)

    if total_reward <= 0 or model is None:
        print(f"訓練失敗 (總獎勵={total_reward:.4f})。")
        exit()

    print(f"訓練完成: 總獎勵={total_reward:.4f}")

    # --- 模型儲存 ---
    model_filename = config.get_trend_model_path(args.symbol, args.timeframe, args.version).replace('.json', '_ppo.zip')
    model.save(model_filename)
    print(f"模型儲存完畢！({model_filename})")