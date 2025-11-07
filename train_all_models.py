# 檔案: train_all_models.py
# 這是「萬能訓練入口」

import subprocess
import config
import time
import os

def run_training_script(script_name, symbol, limit, version=None):
    """
    建立並執行訓練指令。
    """
    command = ['python', script_name, '--symbol', symbol, '--limit', str(limit)]
    if version:
        command.extend(['--version', version])
    
    print(f"\n--- 正在啟動 {symbol} 的 {script_name} 訓練 ---")
    
    # (執行子進程，並等待完成)
    try:
        # 使用 check_call 會在子進程失敗時拋出錯誤
        subprocess.check_call(command) 
        print(f"✅ {symbol} {script_name} 訓練成功完成。")
        return True
    except subprocess.CalledProcessError as e:
        print(f"🛑 錯誤：{symbol} 的 {script_name} 訓練失敗，請檢查錯誤日誌。")
        print(f"詳細錯誤碼: {e.returncode}")
        return False

def train_all_symbols():
    
    # --- 從 config.py 讀取所有 Symbols ---
    symbols = config.SYMBOLS_TO_TRADE
    
    print(f"==================================================")
    print(f"🚀 啟動萬能訓練入口，總共 {len(symbols)} 個資產。")
    print(f"==================================================")
    
    for symbol in symbols:
        
        # --- 1. 訓練趨勢模型 (LSTM) ---
        print(f"\n--- 開始處理資產: {symbol} ---")
        
        # 參數讀取
        trend_limit = config.TREND_MODEL_TRAIN_LIMIT
        trend_version = config.TREND_MODEL_VERSION
        entry_limit = config.TREND_MODEL_TRAIN_LIMIT # <-- 修改
        entry_version = config.TREND_MODEL_VERSION # <-- 修改

        # 執行 LSTM 趨勢訓練
        success = run_training_script(
            'train_trend_model.py', 
            symbol, 
            trend_limit, 
            trend_version
        )
        if not success:
            print(f"🛑 錯誤：{symbol} 的趨勢模型訓練失敗，跳過進場模型。") # <-- 修改
            continue 

        # --- 2. 訓練進場模型 (XGBoost) ---
        success = run_training_script(
            'train_entry_model.py', # <-- 修改
            symbol, 
            entry_limit,
            entry_version
        )
        if not success:
            print(f"🛑 錯誤：{symbol} 的進場模型訓練失敗。") # <-- 修改
            continue 

        print(f"🎉 {symbol} 兩項模型訓練皆成功完成！")

    print("\n==================================================")
    print("所有資產的訓練任務已完成。")
    print("==================================================")


if __name__ == '__main__':
    # (您必須確保這 4 個檔案都在同一個目錄)
    # config.py, common_utils.py, train_trend_model.py, train_entry_model.py
    train_all_symbols()
