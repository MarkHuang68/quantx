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
ENTRY_MODEL_TIMEFRAME = '5m'
ENTRY_MODEL_TRAIN_LIMIT = 300000
ENTRY_MODEL_RMSE_THRESHOLD = 8.00
ENTRY_MODEL_VERSION = "1.0" # <-- *** 版本號 ***
ABS_MAX_RMSE_PCT = 0.003

def get_entry_model_path(symbol, version):
    """
    根據 symbol 和 version 生成「進場模型」的儲存路
    例如: 'models/model_entry_5m_XGB_ETH_USDT_v1.0.json'
    """
    symbol_str = symbol.replace('/', '_')
    # (os.path.join 確保路徑在 Windows/Linux 上都正確)
    return os.path.join(MODEL_DIR, f"entry_model_XGB_{symbol_str}_v{version}.json")

# --- 4. 模型 B (1h 趨勢模型) 的「版本控制」 ---
TREND_MODEL_TIMEFRAME = '1h'
TREND_MODEL_TRAIN_LIMIT = 30000 
TREND_MODEL_VERSION = "1.0" # <-- *** 版本號 ***
ABS_MIN_ACCURACY = 0.6

# (LSTM 模型架構參數)
TREND_MODEL_PARAMS = {
    "LOOKBACK_WINDOW": 48, # 增加以適應更多特徵
    "FORECAST_HORIZON": 24,    
    "LSTM_UNITS_1": 128,
    "LSTM_UNITS_2": 64,
    "DENSE_UNITS": 50,
    "DROPOUT_RATE": 0.3, # 新增 dropout 防止過擬合
    "ATTENTION": True
}

def get_trend_model_path(symbol, version):
    """
    根據 symbol 和 version 生成「趨勢模型」的儲存路徑。
    例如: 'models/model_trend_1h_LSTM_ETH_USDT_v1.0.keras'
    """
    symbol_str = symbol.replace('/', '_')
    return os.path.join(MODEL_DIR, f"trend_model_LSTM_{symbol_str}_v{version}.keras")

# --- 5. 機器人執行參數 ---
BOT_LOOP_SLEEP_SECONDS = 300