# generate_test_data.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_random_walk_data(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """產生隨機漫步的 OHLCV 資料"""
    np.random.seed(seed)
    start_time = datetime(2023, 1, 1)
    times = [start_time + timedelta(hours=i) for i in range(n)]

    price = 20000.0
    data = []

    for t in times:
        open_p = price
        close_p = price + np.random.randn() * 50
        high_p = max(open_p, close_p) + np.random.rand() * 30
        low_p = min(open_p, close_p) - np.random.rand() * 30
        volume = np.random.randint(500, 2000)
        data.append([t, open_p, high_p, low_p, close_p, volume])
        price = close_p

    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    return df

def force_multiple_smc_signals(df: pd.DataFrame, num_signals: int = 20) -> pd.DataFrame:
    """
    強制插入多組 SMC 流程（看漲為例）
    每組包含：
    1. 下跌蠟燭 → 建立 OB
    2. 上漲蠟燭 + 突破前開盤 → 觸發 OB
    3. BOS 突破前高
    4. 隨機插入 FVG（增加 HQ-OB 機率）
    """
    np.random.seed(123)
    df = df.copy()

    for _ in range(num_signals):
        if len(df) < 150:
            continue
        i = np.random.randint(50, len(df) - 100)

        base_price = df['close'].iloc[i-1] if i > 0 else 20000

        # --- 1. 下跌蠟燭 (建立看漲 OB) ---
        df.loc[df.index[i], 'open'] = base_price + np.random.uniform(50, 150)
        df.loc[df.index[i], 'close'] = df.loc[df.index[i], 'open'] - np.random.uniform(80, 200)
        df.loc[df.index[i], 'high'] = df.loc[df.index[i], 'open'] + np.random.uniform(10, 40)
        df.loc[df.index[i], 'low'] = df.loc[df.index[i], 'close'] - np.random.uniform(10, 40)

        # --- 2. 上漲蠟燭 + 突破前開盤 → 觸發 OB ---
        ob_open = df.loc[df.index[i], 'open']
        df.loc[df.index[i+1], 'open'] = df.loc[df.index[i], 'close'] + np.random.uniform(10, 50)
        df.loc[df.index[i+1], 'close'] = ob_open + np.random.uniform(100, 250)  # 突破！
        df.loc[df.index[i+1], 'high'] = df.loc[df.index[i+1], 'close'] + np.random.uniform(20, 50)
        df.loc[df.index[i+1], 'low'] = df.loc[df.index[i+1], 'open'] - np.random.uniform(10, 30)

        # --- 3. BOS 突破前高 ---
        bos_idx = i + np.random.randint(4, 8)
        if bos_idx < len(df):
            prev_high = df['high'].iloc[i-5:i+1].max()
            df.loc[df.index[bos_idx], 'close'] = prev_high + np.random.uniform(80, 200)
            df.loc[df.index[bos_idx], 'high'] = df.loc[df.index[bos_idx], 'close'] + 20
            df.loc[df.index[bos_idx], 'bos_bullish'] = prev_high + 50  # 強制 BOS

        # --- 4. 隨機插入 FVG（增加 HQ-OB）---
        if np.random.rand() < 0.4:  # 40% 機率有 FVG
            fvg_idx = i + np.random.randint(-2, 2)
            if 0 < fvg_idx < len(df) - 1:
                df.loc[df.index[fvg_idx], 'high'] = df.loc[df.index[fvg_idx-1], 'low'] - 50
                df.loc[df.index[fvg_idx+1], 'low'] = df.loc[df.index[fvg_idx], 'high'] + 10

    return df

# ======================
# 主流程
# ======================
if __name__ == "__main__":
    print("正在產生模擬資料...")
    df = generate_random_walk_data(n=1000)

    print("正在強制插入 20 組 SMC 訊號...")
    df = force_multiple_smc_signals(df, num_signals=20)

    # 儲存
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/btc_1h.csv', index=False)
    print(f"資料已儲存至 data/btc_1h.csv，共 {len(df)} 筆")
    print(f"   - 平均價格: {df['close'].mean():.2f}")
    print(f"   - 強制 SMC 訊號: 20 組")