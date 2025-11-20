# 檔案: config/settings.py
# 【已修改】: 新增「模型註冊表」 (Registry) 相關函數

import os
import json # 【新增】

# --- 1. 儲存路徑 ---
MODEL_DIR = 'models'
# 【新增】: 這是您所有最佳模型的「總清單」
REGISTRY_FILE = os.path.join(MODEL_DIR, 'best_model_registry.json')

# --- 2. 交易與訓練的資產列表 ---
# (保持不變)
SYMBOLS_TO_TRADE = [
    'ETHUSDT',
    'BTCUSDT',
]
SYMBOLS_TO_TRAIN = [
    'ETHUSDT',
    # 'BTCUSDT',
]
DEFAULT_SYMBOL = SYMBOLS_TO_TRADE[0]

# --- 3. 模型版本控制 ---
TREND_MODEL_VERSION = "1.0" # (這個現在由 Registry 自動管理)
FEE_RATE = 0.0006  # 交易手續費 (0.1%)

def get_trend_model_path(symbol, timeframe, version):
    """
    (舊函數，保留)
    根據 symbol 和 version 生成模型的儲存路徑。
    'ETHUSDT' -> 'ETHUSDT'
    """
    symbol_str = symbol.replace('/', '').replace(':', '')
    symbol_dir = os.path.join(MODEL_DIR, symbol_str)
    os.makedirs(symbol_dir, exist_ok=True) # 【修正】: 確保目錄存在
    return os.path.join(symbol_dir, f"trend_model_XGB_{symbol_str}_{timeframe}_v{version}.json")

# --- 【!!! 核心新增 !!!】 ---
def load_registry():
    """ (輔助函數) 載入 JSON 註冊表檔案 """
    if os.path.exists(REGISTRY_FILE):
        try:
            with open(REGISTRY_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"警告: 註冊表 {REGISTRY_FILE} 損壞，將建立新的。")
            return {}
    return {}

def save_registry(data):
    """ (輔助函數) 安全地儲存 JSON 註冊表檔案 """
    try:
        with open(REGISTRY_FILE, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"錯誤: 無法儲存註冊表 {REGISTRY_FILE}！ 錯誤: {e}")

def get_best_model_config(symbol, timeframe):
    """
    【!!! 您的新函數 !!!】
    自動從註冊表讀取 symbol/timeframe 對應的
    「最佳模型」、「最佳特徵」、和「最佳信心門檻」。
    
    返回: (dict) 包含所有參數, or None
    """
    symbol_safe = symbol.replace('/', '').replace(':', '')
    key = f"{symbol_safe}_{timeframe}" # e.g., "ETHUSDT_1m"
    
    registry = load_registry()
    
    if key not in registry:
        print(f"提示: 在 {REGISTRY_FILE} 中未找到 {key} 的紀錄。")
        return None
        
    config = registry[key]
    
    # 檢查檔案是否存在
    if not os.path.exists(config['model_file']) or not os.path.exists(config['config_file']):
        print(f"警告: 註冊表中的檔案路徑已失效 ( {config['model_file']} )。")
        return None
        
    print(f"✅ 成功載入 {key} 的最佳配置。")
    return config
# --- 【新增結束】 ---


# --- 5. 機器人執行參數 ---
# (保持不變)
LEVERAGE = 5  
BOT_LOOP_SLEEP_SECONDS = 300
MAINTENANCE_MARGIN_RATE = 0.005