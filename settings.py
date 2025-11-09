# 檔案: config/settings.py

import os

# --- 1. 儲存路徑 ---
MODEL_DIR = 'models'

# --- 2. 您要交易的所有資產 ---
SYMBOLS_TO_TRADE = [
    'ETH/USDT',
    'BTC/USDT',
    'SOL/USDT',
    'WLFI/USDT'
]
DEFAULT_SYMBOL = SYMBOLS_TO_TRADE[0]

# --- 3. 模型版本控制 ---
TREND_MODEL_VERSION = "1.0"

def get_trend_model_path(symbol, version):
    """
    根據 symbol 和 version 生成模型的儲存路徑。
    """
    symbol_str = symbol.replace('/', '_')
    return os.path.join(MODEL_DIR, f"trend_model_XGB_{symbol_str}_v{version}.json")

# --- 5. 機器人執行參數 ---
BOT_LOOP_SLEEP_SECONDS = 300
