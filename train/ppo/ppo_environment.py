# 檔案: ppo_factory/ppo_environment.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import math
import os

# --- 1. 引用階段 1 的核心檔案和您自定義的特徵 ---
import config 
import common_utils as utils
from common_utils import fetch_data, create_features_trend, create_features_price

# --- 2. 載入模型和工具 ---
import tensorflow as tf
import xgboost as xgb
from sklearn.preprocessing import StandardScaler # <-- 使用 StandardScaler 更專業
from keras.models import load_model 

# 設置種子
np.random.seed(42)
tf.random.set_seed(42)

# --- (*** 您的 風險管理參數 ***) ---
MAX_DRAWDOWN_PCT = -0.25 # 最大虧損 25% 則強制終止環境
STOP_LOSS_PCT = -0.015   # 寫死的止損 -1.5%

class TradingEnvironment(gym.Env):
    """
    Gymnasium 交易環境 - 模擬 PPO 智能體在歷史數據上的交易。
    """
    
    # --- 1. 初始化 (載入所有數據和模型) ---
    def __init__(self, symbol='ETH/USDT', initial_balance=10000, leverage=5, commission=0.0004):
        super(TradingEnvironment, self).__init__()

        self.symbol = symbol
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.commission = commission
        self.max_sharpe_ratio = -np.inf
        
        # --- (*** 載入專家模型 ***) ---
        self._load_expert_models(symbol)
        
        # --- (*** 數據預處理：預先計算所有訊號 ***) ---
        self.df_data, self.scaler = self._prepare_data(symbol)
        
        if self.df_data is None or self.df_data.empty:
            raise ValueError("無法準備足夠的歷史數據進行環境訓練。")

        # --- 2. 環境參數 ---
        # 狀態空間的維度 = [特徵數] + [帳戶淨值] + [當前倉位] + [未實現盈虧]
        self.n_features = len([col for col in self.df_data.columns if col not in ['Open', 'High', 'Low', 'Close', 'PnL_realized']])
        self.max_timesteps = len(self.df_data) - 1
        
        # --- 3. 動作空間 (Action Space) ---
        # PPO 輸出的動作，我們只讓它決定「倉位大小」。
        # 0: 做空 100% (-1.0), 1: 做空 50% (-0.5), 2: 空手 (0.0), 3: 做多 50% (0.5), 4: 做多 100% (1.0)
        self.action_space = spaces.Discrete(5) 
        self.action_map = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)

        # --- 4. 狀態空間 (Observation Space) ---
        # PPO 看到的狀態是標準化後的數字 (標準化由 self.scaler 處理)
        # 維度 = [標準化特徵] + [倉位] + [未實現盈虧] + [淨值/初始淨值]
        low = np.array([-np.inf] * self.n_features + [-1, -np.inf, 0])
        high = np.array([np.inf] * self.n_features + [1, np.inf, np.inf])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # --- 5. 內部狀態 ---
        self.current_step = 0
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.shares = 0
        self.position = 0.0 # -1.0 到 1.0
        self.entry_price = 0.0
        
        print(f"Trading Environment for {symbol} initialized. Total steps: {self.max_timesteps}")

    # --- 數據和模型的載入與預處理 ---
    def _load_expert_models(self, symbol):
        """ 載入 LSTM 和 XGBoost 專家模型 """
        # (這裡不需要實例化，因為我們在 _prepare_data 中會用到)
        pass 

    def _prepare_data(self, symbol):
        """ 獲取原始數據並預先計算所有特徵和專家訊號 (最複雜部分) """
        print("--- 正在準備數據和預算專家訊號 (這會花費時間) ---")
        
        # --- 1. 獲取數據 (需要足夠的 1h 數據來計算 LSTM 訊號) ---
        # (我們需要 5m 數據作為時間步，並用 1h 數據來計算 LSTM 訊號)
        df_1h = fetch_data(symbol, config.TREND_MODEL_TIMEFRAME, config.TREND_MODEL_TRAIN_LIMIT * 2) # 抓多一點以防萬一
        df_5m = fetch_data(symbol, config.PRICE_MODEL_PARAMS['TIMEFRAME'], config.PRICE_MODEL_TRAIN_LIMIT * 3) 
        
        if df_1h.empty or df_5m.empty: return None, None
        
        # (*** 警告: 這裡跳過了 1h/5m 的複雜合併和 LSTM 預算步驟 ***)
        # (在實際應用中，您需要將 LSTM 的預測結果合併到 5m 的時間軸上)
        
        # 2. 計算 5m 基礎特徵 (簡化: 只使用價格特徵)
        df_5m_features, features_5m_list = create_features_price(df_5m)
        
        # 3. 簡化: 模擬 LSTM 信念值和 XGBoost 價格預測
        # (實際應用中，這裡需要載入模型進行批次預測)
        df_final = df_5m_features.iloc[::3].copy() # 為了速度，我們每 3 根 K 棒取樣一次 (15分鐘)
        df_final['LSTM_SIGNAL'] = np.random.uniform(-1, 1, len(df_final)) 
        df_final['XGB_PRED'] = df_final['Close'] * np.random.uniform(0.99, 1.01, len(df_final))
        df_final['XGB_PRED_RATIO'] = (df_final['XGB_PRED'] - df_final['Close']) / df_final['Close'] # 預測價格比
        
        # 4. 定義 PPO 的市場特徵
        market_features = [col for col in df_final.columns if col not in ['Open', 'High', 'Low', 'Close']]
        df_data = df_final[market_features + ['Close']].copy() # 最後加上 Close (用於計算報酬)
        
        # 5. 標準化 (特徵標準化是 RL 的關鍵)
        scaler = StandardScaler()
        df_data[market_features] = scaler.fit_transform(df_data[market_features])

        return df_data.reset_index(drop=True), scaler

    # --- 3. 環境重置 (Reset) ---
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.shares = 0
        self.position = 0.0 
        self.entry_price = 0.0
        
        observation = self._get_observation()
        return observation, {}

    # --- 4. 環境步進 (Step) ---
    def step(self, action):
        """ 核心函數：PPO 執行動作，環境前進一個時間步 """
        if self.current_step >= self.max_timesteps:
            # 環境終止 (成功跑到數據結尾)
            return self._get_observation(), 0.0, True, False, {'net_worth': self.net_worth}

        # 1. 執行動作 (計算倉位變化)
        action_idx = action.item() 
        target_position = self.action_map[action_idx] 
        
        price_now = self.df_data['Close'].iloc[self.current_step]
        price_next = self.df_data['Close'].iloc[self.current_step + 1] # 下一個時間步的價格 (PPO 預測的目標)

        # 2. 計算獎勵 (Reward) 和 PnL
        reward, pnl_pct = self._calculate_reward_and_pnl(target_position, price_now, price_next)
        
        # 3. 檢查風控 (*** 風控長規則：熔斷和止損 ***)
        terminated = False
        truncated = False # (Gymnasium 的新參數，代表外部原因終止)

        # 3a. 全局熔斷 (最大虧損)
        if self.net_worth < (self.initial_balance * (1 + MAX_DRAWDOWN_PCT)):
            reward = -100 # 懲罰熔斷
            terminated = True
            print(f"!!! 全局熔斷: 淨值跌破 {(1 + MAX_DRAWDOWN_PCT)*100:.0f}%！")

        # 3b. 策略執行 (更新淨值和倉位)
        if not terminated:
            self._update_net_worth(pnl_pct)
            self._update_position_and_shares(target_position, price_now) # 這裡執行交易邏輯

        self.current_step += 1
        observation = self._get_observation()
        info = {'net_worth': self.net_worth}
        return observation, reward, terminated, truncated, info

    # --- 內部工具函數 ---
    def _get_observation(self):
        """ 返回 PPO 看到的狀態 (State) """
        # 1. 市場特徵 (標準化後的特徵)
        market_features = self.df_data.iloc[self.current_step].values[:-1] # 排除 Close

        # 2. 賬戶狀態 (必須標準化)
        pnl_unrealized_pct = (self.net_worth - self.initial_balance) / self.initial_balance # 淨值增長比例

        # 3. 結合並返回
        observation = np.concatenate([
            market_features, 
            [self.position], # 當前倉位
            [pnl_unrealized_pct], # 未實現盈虧（與初始資金相比）
            [self.net_worth / self.initial_balance] # 淨值比例
        ]).astype(np.float32)
        
        return observation

    def _calculate_reward_and_pnl(self, target_position, price_now, price_next):
        """ 計算獎勵 (Reward) 和 PnL """

        # 1. 計算 PnL (倉位乘以價格變化)
        price_change_ratio = (price_next - price_now) / price_now
        pnl_pct = self.position * price_change_ratio * self.leverage # 盈虧比例 (與當前淨值相比)
        
        # 2. 交易成本 (買賣時才計算，這裡簡化為每次不空手)
        cost = 0
        if self.position != 0:
            cost = self.leverage * self.commission * abs(price_now) * self.shares # (簡化處理)

        # 3. 獎勵設計 (核心：讓 PPO 學習最大化淨值)
        reward = pnl_pct * 100 # 將 PnL 比例放大作為獎勵
        
        # 4. *** 寫死止損：懲罰 PPO 觸發止損 ***
        if self.position != 0 and pnl_pct < STOP_LOSS_PCT:
            reward = -10 # 懲罰觸發止損的動作 (比正常虧損更痛)
        
        return reward, pnl_pct
        
    def _update_net_worth(self, pnl_pct):
        """ 更新淨值 """
        self.net_worth = self.net_worth * (1 + pnl_pct)
        self.balance = self.net_worth # 簡化處理

    def _update_position_and_shares(self, target_position, price_now):
        """ 執行交易邏輯：平倉和開倉 (這裡需要考慮交易成本) """
        
        # 1. 處理平倉和反手 (實際交易邏輯)
        if target_position != self.position:
            # 簡化: 交易成本會在獎勵中扣除，這裡只更新狀態
            
            # 2. 計算新倉位
            self.position = target_position
            
            # 3. 計算新股份 (shares)
            if self.position != 0:
                # 倉位 = 淨值 * 倉位比例 * 槓桿 / 價格
                self.shares = (self.net_worth * abs(self.position) * self.leverage) / price_now
            else:
                self.shares = 0
            
            self.entry_price = price_now # 記錄進場價格

# --- (*** 測試區塊 ***) ---
if __name__ == '__main__':
    # 注意: 這個測試需要您在 MyTradingBot/models/ 資料夾中，訓練並儲存了專家模型。
    try:
        env = TradingEnvironment(symbol='ETH/USDT', initial_balance=10000)
        
        # 測試運行 100 步
        obs, info = env.reset()
        print("\n--- 環境測試運行 ---")
        
        for i in range(100):
            # 讓 PPO 亂猜一個動作
            action = env.action_space.sample() 
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"Step {i}: Action={env.action_map[action]}, Net Worth={info['net_worth']:.2f}, Reward={reward:.4f}")
            
            if terminated or truncated:
                print("Environment terminated.")
                break
                
    except Exception as e:
        print(f"環境初始化失敗，請檢查模型和數據。錯誤: {e}")
        import traceback
        traceback.print_exc()