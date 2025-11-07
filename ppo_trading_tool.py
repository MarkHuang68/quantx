# 檔案: ppo_trading_tool.py
# 目的: 根據使用者指定 symbol (e.g., 'ETH/USDT'), timeframe (e.g., '1m'), 時間區間 (start/end, e.g., '2023-01-01'/'2023-12-31')，
# 自動抓取資料並快取到 CSV。載入 3 個 XGBoost 分類模型 (短:1m, 中:5m, 長:15m) 預測漲跌作為特徵。
# 使用 PPO 訓練代理判斷買入/賣出及倉位大小，目標最大化獲利。
# 資料 80% 訓練 PPO，20% 回測 (逐步模擬，非向量)。
# 輸出統計: Sharpe Ratio (SR), Max Drawdown (MDD)。
# 顯示 PPO 回測資金曲線及 Buy&Hold 曲線。
# 依賴: pip install ccxt xgboost stable-baselines3 gym pandas numpy matplotlib torch

import ccxt  # 用於抓取加密幣資料
import pandas as pd
import numpy as np
import os
import xgboost as xgb
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecCheckNan  # nan 檢查
import matplotlib.pyplot as plt
from datetime import datetime
from common_utils import create_features_trend  # 導入特徵計算函數
import torch as th  # 新增: torch 偵測異常

# 啟用 torch nan 偵測
th.autograd.set_detect_anomaly(True)

# --- 1. 資料抓取與快取函數 ---
def fetch_data_with_cache(symbol, timeframe, start, end):
    """
    抓取指定 symbol/timeframe/時間區間 的 OHLCV 資料。
    - 先檢查本地快取 CSV，若存在則載入；否則用 ccxt 抓取並儲存。
    - 回傳 pd.DataFrame 含 'Open', 'High', 'Low', 'Close', 'Volume'，索引為 datetime。
    - 新增: 清洗 nan/inf 在抓取後
    """
    cache_file = f"data/{symbol.replace('/', '_')}_{timeframe}_{start}_{end}.csv"
    os.makedirs('data', exist_ok=True)  # 確保資料夾存在
    
    if os.path.exists(cache_file):
        print(f"從快取載入: {cache_file}")
        df = pd.read_csv(cache_file, index_col='Timestamp', parse_dates=True)
    else:
        print(f"抓取新資料: {symbol} {timeframe} from {start} to {end}")
        exchange = ccxt.binance()  # 以 Binance 為例，可換其他交易所
        since = int(datetime.fromisoformat(start).timestamp() * 1000)  # ms timestamp
        limit = None  # 無限筆數，ccxt 會分批抓取
        df = pd.DataFrame(exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit),
                          columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')  # 轉 datetime
        df.set_index('Timestamp', inplace=True)
        df = df[df.index <= pd.to_datetime(end)]  # 過濾結束時間
        df.to_csv(cache_file)  # 儲存快取
    
    # 清洗資料: ffill nan, 移除 inf
    df = df.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').dropna()
    return df

# --- 2. 載入 XGBoost 模型並計算預測作為特徵 ---
def add_xgb_predictions(df_main, symbol):
    """
    載入 3 個 XGBoost 分類模型 (短:1m, 中:5m, 長:15m)。
    - 對 df_main (主資料) 計算特徵。
    - 使用模型預測漲跌 (1:漲, 0:跌)，加入欄位 'short_pred', 'mid_pred', 'long_pred'。
    - 回傳更新後 df。
    - 新增: 預測後檢查 nan
    """
    # 載入指定模型檔案
    short_model = xgb.Booster(model_file='models/entry_model_XGB_ETH_USDT_1m_v1.0.json')  # 短:1m
    mid_model = xgb.Booster(model_file='models/entry_model_XGB_ETH_USDT_5m_v1.0.json')      # 中:5m
    long_model = xgb.Booster(model_file='models/entry_model_XGB_ETH_USDT_15m_v1.0.json')    # 長:15m
    
    # 計算特徵 (使用導入的 create_features_entry)
    df_features, features_list = create_features_trend(df_main.copy())
    
    X_dmatrix = xgb.DMatrix(df_features[features_list])
    
    # 預測並加入 (模型輸出概率，>0.5 為漲)
    df_features['short_pred'] = (short_model.predict(X_dmatrix) > 0.5).astype(int)
    df_features['mid_pred'] = (mid_model.predict(X_dmatrix) > 0.5).astype(int)
    df_features['long_pred'] = (long_model.predict(X_dmatrix) > 0.5).astype(int)
    
    # 檢查預測 nan
    df_features = df_features.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df_features

# --- 3. 自訂交易環境 (Gym Env) ---
class TradingEnv(gym.Env):
    """
    Gym 環境讓 PPO 學習買賣時機及倉位大小。
    - 狀態: 特徵 + 短中長預測 + 當前餘額 + 持倉比例。
    - 行動: Box(-1,1) -1:全賣/空倉, 0:持倉, 1:全買/多倉 (連續值決定倉位比例)。
    - 報酬: PnL - 手續費 + 罰項 (e.g., 持倉過久罰)。
    - 目標: 最大化累積報酬 (獲利)。
    - 修復: 防負 balance 導致 inf, 夾限 reward, 初始 entry_price 非0
    """
    def __init__(self, df, features_list):
        super(TradingEnv, self).__init__()
        self.df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)  # 移除 nan/inf，重置索引確保連續
        if self.df.empty:
            raise ValueError("DataFrame is empty after cleaning.")  # 檢查空資料
        self.features_list = features_list + ['short_pred', 'mid_pred', 'long_pred']  # 包含預測特徵
        self.current_step = 0
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,))  # 連續行動: -1~1 決定倉位調整
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.features_list) + 2,))  # 特徵 + 餘額 + 持倉
        self.initial_balance = 10000.0
        self.balance = self.initial_balance
        self.position = 0.0  # 持倉比例 (0~1:多, -1~0:空)
        self.entry_price = self.df['Close'].mean() or 1.0  # 初始非0，避免 division by zero
        self.fee = 0.00055  # 手續費

    def seed(self, seed=None):
        """ 設定隨機種子 (確定性環境) """
        import numpy as np
        np.random.seed(seed)
        return [seed]

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = self.df['Close'].mean() or 1.0  # 重置非0
        obs = self._get_observation()
        print(f"Debug reset obs: {obs}")  # 調試 print 檢查 obs nan
        return obs

    def step(self, action):
        action = action[0]  # 取出 scalar
        current_price = self.df['Close'].iloc[self.current_step]
        
        reward = -0.01  # 預設小罰 (避免無操作)
        
        # 調整倉位: action >0 買/加多, <0 賣/加空
        target_position = np.clip(action, -1, 1)  # 目標倉位比例
        delta_position = target_position - self.position  # 變化量
        
        if abs(delta_position) > 0:
            # 限於可用 balance，避免負 trade_amount
            delta_position = np.clip(delta_position, -self.balance / self.initial_balance, self.balance / self.initial_balance)
            trade_amount = abs(delta_position) * self.balance  # 交易金額
            cost = trade_amount * self.fee  # 手續費
            self.balance -= cost
            
            if delta_position > 0:  # 買/加多
                units = trade_amount / current_price
                self.position += units / (self.initial_balance / current_price)  # 正規化比例
                self.entry_price = current_price if self.position == delta_position else (self.entry_price * abs(self.position - delta_position) + current_price * abs(delta_position)) / abs(self.position)
            else:  # 賣/加空
                units = trade_amount / current_price
                self.position -= units / (self.initial_balance / current_price)
                pnl = units * (self.entry_price - current_price) if self.position < 0 else units * (current_price - self.entry_price)
                reward += pnl - cost
                self.balance += pnl

        # 持倉報酬 (浮動 PnL)
        if self.position != 0:
            if self.entry_price == 0:  # 防 zero
                pnl_pct = 0.0
            else:
                pnl_pct = (current_price - self.entry_price) / self.entry_price * np.sign(self.position)
            reward += pnl_pct * 0.05  # 小額鼓勵持盈

        # 檢查負 balance: 破產罰
        if self.balance <= 0:
            reward = -1e6  # 大負罰
            done = True
            self.balance = 0.0  # 設0避免 inf
        else:
            done = False

        self.current_step += 1
        done = done or (self.current_step >= len(self.df))  # 調整為 >= len(self.df)
        
        if done:
            # 強制平倉計算最終 reward
            if self.position != 0:
                pnl = self.position * (self.df['Close'].iloc[-1] - self.entry_price) * np.sign(self.position)
                self.balance += pnl - abs(pnl) * self.fee
                reward += pnl
            return np.zeros(self.observation_space.shape), np.clip(reward, -1e6, 1e6), done, {}  # 虛擬 obs, 夾限 reward

        obs = self._get_observation()
        print(f"Debug step obs: {obs}")  # 調試 print 檢查 obs nan
        return obs, np.clip(reward, -1e6, 1e6), done, {}  # 夾限 reward 防 inf/nan

    def _get_observation(self):
        step = min(self.current_step, len(self.df) - 1)  # 防越界
        features = np.nan_to_num(self.df[self.features_list].iloc[step].values, nan=0.0)  # 處理 nan
        extra = np.array([self.balance / self.initial_balance if self.initial_balance != 0 else 0.0, self.position])  # 防 nan
        extra = np.nan_to_num(extra, nan=0.0)  # 額外處理 nan
        return np.concatenate([features, extra])

# --- 4. 主函數: 訓練 PPO 並回測 ---
def run_ppo_tool(symbol, timeframe, start, end):
    """
    執行完整流程: 抓資料 -> 加 XGB 預測 -> 分割資料 -> 訓練 PPO -> 逐步回測 -> 計算統計 -> 畫曲線。
    """
    # 抓資料並加 XGB 預測
    df = fetch_data_with_cache(symbol, timeframe, start, end)
    df = add_xgb_predictions(df, symbol)  # 加預測特徵
    features_list = ['Open', 'High', 'Low', 'Close', 'Volume']  # 假設基本特徵，需擴展

    # 分割: 80% 訓練, 20% 測試
    split_index = int(len(df) * 0.8)
    df_train = df.iloc[:split_index]
    df_test = df.iloc[split_index:]

    # 訓練 PPO (新增 VecCheckNan 防護)
    env = make_vec_env(lambda: TradingEnv(df_train, features_list), n_envs=1)
    env = VecCheckNan(env, raise_exception=True)  # nan 檢查 wrapper
    model = PPO("MlpPolicy", env, verbose=1, clip_range=0.1, ent_coef=0.01)  # 調整參數增加穩定, verbose=1 看 log
    model.learn(total_timesteps=100000)  # 調整步數

    # 回測: 逐步模擬 on 測試資料
    test_env = TradingEnv(df_test, features_list)
    obs = test_env.reset()
    equity_curve = [test_env.initial_balance]  # PPO 資金曲線
    bh_curve = [test_env.initial_balance]  # Buy&Hold 曲線
    bh_position = test_env.initial_balance / df_test['Close'].iloc[0]  # 初始買入量
    
    while True:
        action, _ = model.predict(obs)
        obs, _, done, _ = test_env.step(action)
        equity_curve.append(test_env.balance)
        bh_net = bh_position * df_test['Close'].iloc[test_env.current_step - 1]
        bh_curve.append(bh_net)
        if done: break

    # 計算統計
    returns = pd.Series(equity_curve).pct_change().dropna()  # 日報酬
    sr = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0  # 年化 SR (假設 252 交易日)
    peak = np.maximum.accumulate(equity_curve)
    dd = (np.array(equity_curve) - peak) / peak
    mdd = dd.min() * 100  # MDD (%)

    print(f"SR: {sr:.2f}, MDD: {mdd:.2f}%")

    # 畫曲線
    plt.figure(figsize=(10, 5))
    plt.plot(equity_curve, label='PPO 資金曲線')
    plt.plot(bh_curve, label='Buy&Hold 資金曲線')
    plt.legend()
    plt.show()

# --- 5. 執行範例 (需實作 create_features_entry 及 XGB 模型檔案) ---
if __name__ == "__main__":
    run_ppo_tool('ETH/USDT', '1m', '2023-01-01', '2023-12-31')