# 檔案: train/ppo/ppo_train.py

import os
import time
import argparse
import sys
from multiprocessing import cpu_count

# 確保可以引用到上層目錄
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.common import fetch_data
from train.ppo.ppo_environment import TradingEnvironment, prepare_data_for_ppo
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from settings import SYMBOLS_TO_TRADE

PPO_HYPERPARAMS = {
    "n_steps": 2048,
    "batch_size": 64,
    "gamma": 0.99,
    "learning_rate": 0.0003,
    "verbose": 1
}

def make_env(df_ppo):
    """
    輔助函式，用於 SubprocVecEnv 序列化環境建立過程。
    """
    def _init():
        return TradingEnvironment(df_ppo)
    return _init

def train_unified_ppo_agent(timeframe, start_date, end_date, total_timesteps=2_000_000, output_dir="ppo_models"):
    """
    為 SYMBOLS_TO_TRADE 中的所有交易對載入數據，並行訓練一個統一的 PPO 模型。
    """
    run_name = f"ppo_agent_unified_{timeframe}_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir = os.path.join(output_dir, "logs", run_name)
    os.makedirs(log_dir, exist_ok=True)

    print(f"\n=======================================================")
    print(f"--- 開始為所有交易對訓練統一 PPO 模型 ({timeframe}) ---")
    print(f"--- 時間範圍: {start_date} to {end_date} ---")
    print(f"--- 交易對: {SYMBOLS_TO_TRADE} ---")
    print(f"=======================================================")

    # 1. 為每個交易對載入數據並建立環境
    env_makers = []
    for symbol in SYMBOLS_TO_TRADE:
        print(f"\n--- 正在處理交易對: {symbol} ---")

        # 使用新的時間參數來獲取數據
        raw_data = fetch_data(symbol=symbol, start_date=start_date, end_date=end_date, timeframe=timeframe)
        if raw_data is None or raw_data.empty:
            print(f"🛑 警告: {symbol} 在指定時間範圍內的數據無法載入，將跳過。")
            continue

        df_ppo = prepare_data_for_ppo(symbol, raw_data)
        if df_ppo is None:
            print(f"🛑 警告: {symbol} 的 PPO 數據準備失敗，將跳過。")
            continue

        env_makers.append(make_env(df_ppo))

    if not env_makers:
        print("🛑 錯誤: 沒有可用於訓練的環境。請檢查數據和 XGBoost 模型。")
        return

    # 2. 建立並行化的 PPO 向量環境
    num_cpu = min(cpu_count(), len(env_makers))
    print(f"\n--- 使用 {num_cpu} 個 CPU 核心進行並行訓練 ---")
    try:
        env = SubprocVecEnv(env_makers)
    except Exception as e:
        print(f"🛑 錯誤：無法創建 SubprocVecEnv。{e}")
        return

    # 3. 設定回調
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path=os.path.join(output_dir, "checkpoints"),
        name_prefix=f"ppo_checkpoint_unified_{timeframe}"
    )

    # 4. 建立並訓練統一的 PPO 模型
    model = PPO("MlpPolicy", env, **PPO_HYPERPARAMS, seed=42, tensorboard_log=log_dir)

    print(f"--- 統一 PPO 模型開始學習 ({total_timesteps} 步) ---")
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    training_time = time.time() - start_time
    print(f"\n--- 訓練完成！總耗時: {training_time:.2f} 秒 ---")

    # 5. 儲存最終的統一模型
    final_save_path = os.path.join(output_dir, f"ppo_agent_unified_{timeframe}_final.zip")
    model.save(final_save_path)
    print(f"✅ 統一 PPO 模型儲存完畢：{final_save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='統一 PPO 模型訓練腳本')
    parser.add_argument('-tf', '--timeframe', type=str, required=True, help='要訓練的時間週期 (例如: 5m, 1h)')
    parser.add_argument('-sd', '--start', type=str, required=True, help='訓練起始日期 (YYYY-MM-DD)')
    parser.add_argument('-ed', '--end', type=str, required=True, help='訓練結束日期 (YYYY-MM-DD)')
    parser.add_argument('-t', '--timesteps', type=int, default=2_000_000, help='總訓練步數 (預設: 2,000,000)')

    args = parser.parse_args()

    train_unified_ppo_agent(args.timeframe, args.start, args.end, args.timesteps)
