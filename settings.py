# 檔案: config/settings.py

import os

# --- 1. 儲存路徑 ---
MODEL_DIR = 'models'

# --- 2. 交易與訓練的資產列表 ---
# 用於線上交易的交易對
SYMBOLS_TO_TRADE = [
    'ETHUSDT',
    # 'BTCUSDT',
]

# 用於模型訓練的交易對
SYMBOLS_TO_TRAIN = [
    'ETHUSDT',
    'BTCUSDT',
]

DEFAULT_SYMBOL = SYMBOLS_TO_TRADE[0]

# --- 3. 模型版本控制 ---
TREND_MODEL_VERSION = "1.0"
FEE_RATE = 0.00055  # 交易手續費 (0.1%)

def get_trend_model_path(symbol, timeframe, version):
    """
    根據 symbol 和 version 生成模型的儲存路徑。
    'ETHUSDT' -> 'ETHUSDT'
    """
    # 確保 symbol 不包含斜線或冒號
    symbol_str = symbol.replace('/', '').replace(':', '')
    return os.path.join(MODEL_DIR, f"trend_model_XGB_{symbol_str}_{timeframe}_v{version}.json")

# --- 5. 機器人執行參數 ---
LEVERAGE = 5  # 全域槓桿設定
BOT_LOOP_SLEEP_SECONDS = 300
MAINTENANCE_MARGIN_RATE = 0.005 # 維持保證金率 (Bybit 預設為 0.5%)
