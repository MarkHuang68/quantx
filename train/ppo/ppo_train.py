# 檔案: train/ppo/ppo_train.py

import os
import time
import argparse
import sys
import pandas as pd
from multiprocessing import cpu_count

# 確保可以引用到上層目錄
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.data_loader import load_csv_data
from train.ppo.ppo_environment import TradingEnvironment, prepare_data_for_ppo
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed

from settings import SYMBOLS_TO_TRADE

DATA_DIR = "data"

PPO_HYPERPARAMS = {
    "n_steps": 2048,
    "batch_size": 64,
    "gamma": 0.99,
    "learning_rate": 0.0003,
    "verbose": 1
}

def make_env(df, rank, seed=0):
    """
    SubprocVecEnv 的環境產生器。
    """
    def _init():
        env = TradingEnvironment(df)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

def train_ppo_agent(total_timesteps=1_000_000, output_dir="ppo_models"):
    """
    載入所有交易對的數據，準備統一的環境，並訓練一個共用的 PPO 智能體。
    """
    run_name = f"ppo_agent_UNIFIED_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir = os.path.join(output_dir, "logs", run_name)

    print(f"\n=======================================================")
    print(f"--- 開始訓練統一的 PPO 風險管理代理 ---")
    print(f"--- 交易對: {SYMBOLS_TO_TRADE} ---")
    print(f"=======================================================")

    # 1. 載入並準備所有交易對的數據
    all_symbols_data = []
    for symbol in SYMBOLS_TO_TRADE:
        # 假設數據檔案命名格式為: {DATA_DIR}/{SYMBOL_pair}_{timeframe}.csv, e.g., data/ETHUSDT_1m.csv
        # 這裡我們需要一個統一的數據檔案命名約定
        csv_path = os.path.join(DATA_DIR, f"{symbol.replace('/', '')}_1m.csv")

        print(f"\n--- 正在處理 {symbol} 的數據 ---")
        if not os.path.exists(csv_path):
            print(f"🛑 警告：找不到 {symbol} 的數據檔案：{csv_path}，已跳過。")
            continue

        raw_data = load_csv_data(csv_path, symbol=symbol)
        if raw_data is None:
            continue

        df_ppo = prepare_data_for_ppo(symbol, raw_data)
        if df_ppo is not None:
            all_symbols_data.append(df_ppo)

    if not all_symbols_data:
        print("🛑 錯誤：沒有任何數據可供訓練。請檢查數據檔案。")
        return

    # 2. 合併所有數據集
    print("\n--- 正在合併所有交易對的數據集 ---")
    unified_df = pd.concat(all_symbols_data, ignore_index=True)
    print(f"✅ 統一數據集創建完畢，總共 {len(unified_df)} 筆數據。")

    # 3. 創建多核心 PPO 環境
    try:
        num_cpu = cpu_count()
        print(f"--- 偵測到 {num_cpu} 個 CPU 核心，將用於平行化訓練 ---")
        env = SubprocVecEnv([make_env(unified_df, i) for i in range(num_cpu)])
    except Exception as e:
        print(f"🛑 錯誤：無法創建 SubprocVecEnv 環境。{e}")
        return

    # 4. 設定回調
    checkpoint_callback = CheckpointCallback(
        save_freq=max(100000 // num_cpu, 1),
        save_path=os.path.join(output_dir, "checkpoints"),
        name_prefix="ppo_checkpoint_UNIFIED"
    )

    # 5. 建立並訓練 PPO 模型
    model = PPO("MlpPolicy", env, **PPO_HYPERPARAMS, seed=42, tensorboard_log=log_dir)

    print(f"--- PPO 智能體開始學習 ({total_timesteps} 步) ---")
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    training_time = time.time() - start_time
    print(f"\n--- 訓練完成！總耗時: {training_time:.2f} 秒 ---")

    # 6. 儲存最終模型
    final_save_path = os.path.join(output_dir, "ppo_agent_UNIFIED_final.zip")
    os.makedirs(output_dir, exist_ok=True)
    model.save(final_save_path)
    print(f"✅ 統一 PPO 智能體儲存完畢：{final_save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='統一 PPO 智能體平行化訓練腳本')
    parser.add_argument('-t', '--timesteps', type=int, default=2_000_000, help='總訓練步數 (預設: 2,000,000)')
    args = parser.parse_args()

    train_ppo_agent(args.timesteps)
