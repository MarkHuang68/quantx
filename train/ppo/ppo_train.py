# 檔案: ppo_factory/ppo_train.py

import os
import time
import argparse
import sys
import numpy as np

# 確保可以引用到上層目錄的 config
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# --- 1. 引用核心工具 ---
import config
from ppo_environment import TradingEnvironment # <-- 引用我們剛剛創建的環境

# --- 2. 引用 Stable-Baselines3 (PPO) ---
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

# --- 3. 訓練參數設定 ---
PPO_HYPERPARAMS = {
    "n_steps": 2048,          # 收集數據的步數
    "batch_size": 64,         # 優化使用的數據量
    "gamma": 0.99,            # 長期獎勵折扣
    "learning_rate": 0.0003,
    "verbose": 1
}

def train_ppo_agent(symbol, total_timesteps=1_000_000, output_dir="ppo_models"):
    """
    載入交易環境並訓練 PPO 智能體。
    """
    symbol_str = symbol.replace('/', '_')
    run_name = f"ppo_agent_{symbol_str}_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir = os.path.join(output_dir, "logs", run_name)
    
    print(f"\n=======================================================")
    print(f"--- 階段 2: 啟動 PPO 決策工廠 - 訓練 {symbol} ---")
    print(f"=======================================================")

    # 1. 創建環境
    try:
        # PPO 要求環境必須是向量化的，即使只有一個環境
        env = DummyVecEnv([lambda: TradingEnvironment(
            symbol=symbol,
            initial_balance=10000,
            leverage=5, 
            commission=0.0004
        )])
    except Exception as e:
        print(f"🛑 錯誤：無法創建 TradingEnvironment。請檢查專家模型和數據。{e}")
        return

    # 2. 建立回調 (Callback) 以便保存中間檢查點
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,  # 每 10 萬步保存一次
        save_path=os.path.join(output_dir, "checkpoints"),
        name_prefix=f"ppo_checkpoint_{symbol_str}"
    )

    # 3. 建立 PPO 模型
    model = PPO("MlpPolicy", env, **PPO_HYPERPARAMS, seed=42, tensorboard_log=log_dir)

    # 4. 訓練智能體 (這將在歷史數據上模擬交易 100 萬次)
    print(f"--- PPO 智能體開始學習 ({total_timesteps} 步) ---")
    
    start_time = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        reset_num_timesteps=False
    )
    training_time = time.time() - start_time
    print(f"\n--- 訓練完成！總耗時: {training_time:.2f} 秒 ---")

    # 5. 儲存最終模型
    final_save_path = os.path.join(output_dir, f"ppo_agent_{symbol_str}_final.zip")
    os.makedirs(output_dir, exist_ok=True)
    model.save(final_save_path)
    print(f"✅ PPO 智能體儲存完畢：{final_save_path}")

    # 6. 最終性能測試 (在訓練數據上跑一次，查看最終淨值)
    obs, _ = env.reset()
    final_net_worth = 0
    for i in range(env.envs[0].max_timesteps):
        action, _states = model.predict(obs, deterministic=True) 
        obs, reward, done, info = env.step(action)
        if done[0]:
            final_net_worth = info[0]['net_worth']
            break
            
    print(f"\n--- PPO 最終性能測試 ---")
    print(f"初始資金: {env.envs[0].initial_balance}")
    print(f"最終淨值: {final_net_worth:.2f}")
    print(f"總報酬率: {((final_net_worth / env.envs[0].initial_balance) - 1) * 100:.2f}%")


if __name__ == '__main__':
    # --- 命令行參數 ---
    parser = argparse.ArgumentParser(description='PPO 智能體訓練工廠')
    parser.add_argument('-s', '--symbol', type=str, required=True, help='要訓練的交易對 (例如: ETH/USDT)')
    parser.add_argument('-t', '--timesteps', type=int, default=1_000_000, help='總模擬交易步數 (預設: 1,000,000)')
    
    args = parser.parse_args()
    
    # --- 執行多資產訓練 ---
    if args.symbol.upper() == 'ALL':
        for symbol in config.SYMBOLS_TO_TRADE:
            train_ppo_agent(symbol, args.timesteps)
    elif args.symbol in config.SYMBOLS_TO_TRADE:
        train_ppo_agent(args.symbol, args.timesteps)
    else:
        print(f"🛑 錯誤：請使用 'ALL' 或 config.py 中定義的交易對。")
        print(f"可用交易對: {config.SYMBOLS_TO_TRADE}")