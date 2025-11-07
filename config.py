# 檔案: config.py

import os

# --- 1. 儲存路徑 ---
MODEL_DIR = 'models' # <-- *** 您的新要求 ***

# --- 2. 您要交易的所有資產 ---
SYMBOLS_TO_TRADE = [
    'ETH/USDT',
    'BTC/USDT',
    'SOL/USDT',
    'WLFI/USDT'
]
DEFAULT_SYMBOL = SYMBOLS_TO_TRADE[0] 

# --- 3. 模型 A (5m 進場模型) 的「版本控制」 ---
TREND_MODEL_TIMEFRAME = '5m'
TREND_MODEL_TRAIN_LIMIT = 10000
TREND_MODEL_RMSE_THRESHOLD = 8.00
TREND_MODEL_VERSION = "1.0" # <-- *** 版本號 ***
ABS_MAX_RMSE_PCT = 0.006

def get_trend_model_path(symbol, timeframe, version):
    """
    根據 symbol 和 version 生成「進場模型」的儲存路
    例如: 'models/model_entry_5m_XGB_ETH_USDT_v1.0.json'
    """
    symbol_str = symbol.replace('/', '_')
    # (os.path.join 確保路徑在 Windows/Linux 上都正確)
    return os.path.join(MODEL_DIR, f"trend_model_XGB_{symbol_str}_{timeframe}_v{version}.json")

# --- 5. 機器人執行參數 ---
BOT_LOOP_SLEEP_SECONDS = 300