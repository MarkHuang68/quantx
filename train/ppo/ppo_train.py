# 檔案: train/ppo/ppo_train.py

import os
import time
import argparse
import sys
import numpy as np

# 確保可以引用到上層目錄
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.data_loader import load_csv_data
from train.ppo.ppo_environment import TradingEnvironment, prepare_data_for_ppo
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from settings import SYMBOLS_TO_TRADE

PPO_HYPERPARAMS = {
    "n_steps": 2048,
    "batch_size": 64,
    "gamma": 0.99,
    "learning_rate": 0.0003,
    "verbose": 1
}

def train_ppo_agent(symbol, csv_path, total_timesteps=1_000_000, output_dir="ppo_models"):
    """
    載入數據、準備環境並訓練 PPO 智能體。
    """
    symbol_str = symbol.replace('/', '_')
    run_name = f"ppo_agent_{symbol_str}_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir = os.path.join(output_dir, "logs", run_name)

    print(f"\n=======================================================")
    print(f"--- 開始訓練 PPO 代理: {symbol} ---")
    print(f"=======================================================")

    # 1. 載入並準備數據
    raw_data = load_csv_data(csv_path, symbol=symbol)
    if raw_data is None:
        return

    df_ppo = prepare_data_for_ppo(symbol, raw_data)
    if df_ppo is None:
        return

    # 2. 創建 PPO 環境
    try:
        env = DummyVecEnv([lambda: TradingEnvironment(df_ppo)])
    except Exception as e:
        print(f"🛑 錯誤：無法創建 TradingEnvironment。{e}")
        return

    # 3. 設定回調
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path=os.path.join(output_dir, "checkpoints"),
        name_prefix=f"ppo_checkpoint_{symbol_str}"
    )

    # 4. 建立並訓練 PPO 模型
    model = PPO("MlpPolicy", env, **PPO_HYPERPARAMS, seed=42, tensorboard_log=log_dir)

    print(f"--- PPO 智能體開始學習 ({total_timesteps} 步) ---")
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    training_time = time.time() - start_time
    print(f"\n--- 訓練完成！總耗時: {training_time:.2f} 秒 ---")

    # 5. 儲存最終模型
    final_save_path = os.path.join(output_dir, f"ppo_agent_{symbol_str}_final.zip")
    os.makedirs(output_dir, exist_ok=True)
    model.save(final_save_path)
    print(f"✅ PPO 智能體儲存完畢：{final_save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPO 智能體訓練腳本')
    parser.add_argument('-s', '--symbol', type=str, required=True, help='要訓練的交易對 (例如: ETH/USDT)')
    parser.add_argument('--csv', type=str, required=True, help='包含歷史 K 線數據的 CSV 檔案路徑')
    parser.add_argument('-t', '--timesteps', type=int, default=1_000_000, help='總訓練步數 (預設: 1,000,000)')

    args = parser.parse_args()

    if args.symbol in SYMBOLS_TO_TRADE:
        train_ppo_agent(args.symbol, args.csv, args.timesteps)
    else:
        print(f"🛑 錯誤：請使用 config.py 中定義的交易對。")
        print(f"可用交易對: {SYMBOLS_TO_TRADE}")
