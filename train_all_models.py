# 檔案: train_all_models.py
# 這是「萬能訓練入口」
# 新增功能: 添加 --overwrite 命令行參數。
#   - train_all_models.py 自行檢查模型檔案是否存在。
#   - 如果未指定 --overwrite，且模型檔案已存在，則略過訓練 (不呼叫子腳本)。
#   - 如果指定 --overwrite，則無視存在與否，直接執行訓練 (呼叫子腳本)。
#   - 模型路徑假設為 f"models/{symbol}_{timeframe}_trend_v{version}.pkl" (需確認實際路徑，若不同請調整)。
#   - 不需修改 train_trend_model.py，檢查邏輯全在本腳本。

import subprocess
import config
import time
import os
import argparse  # 用於解析命令行參數

def run_training_script(script_name, symbol, start, end, version=None, timeframe=None):
    """
    建立並執行訓練指令。
    - 無需傳遞 overwrite，因為檢查已在上層處理。
    """
    command = ['python', script_name, '--symbol', symbol, '-sd', start, '-ed', end]
    if timeframe:
        command.extend(['--timeframe', timeframe])
    if version:
        command.extend(['--version', version])
    
    print(f"\n--- 正在啟動 {symbol} ({timeframe}) 的 {script_name} 訓練 ---")
    
    # 執行子進程，並等待完成
    try:
        subprocess.check_call(command) 
        print(f"✅ {symbol} ({timeframe}) {script_name} 訓練成功完成。")
        return True
    except subprocess.CalledProcessError as e:
        print(f"🛑 錯誤：{symbol} ({timeframe}) 的 {script_name} 訓練失敗，請檢查錯誤日誌。")
        print(f"詳細錯誤碼: {e.returncode}")
        return False

def train_all_symbols(overwrite=False):
    """
    訓練所有符號的模型。
    - overwrite: 若 True，則強制執行訓練 (覆蓋)。
    """
    # 從 config.py 讀取所有 Symbols
    symbols = config.SYMBOLS_TO_TRADE
    
    print(f"==================================================")
    print(f"🚀 啟動萬能訓練入口，總共 {len(symbols)} 個資產 (overwrite={overwrite})。")
    print(f"==================================================")
    
    timeframes = ['1m', '5m', '15m']
    
    for symbol in symbols:
        
        # 訓練趨勢模型 (XGBoost)
        print(f"\n--- 開始處理資產: {symbol} ---")
        
        # 參數讀取
        trend_limit = config.TREND_MODEL_TRAIN_LIMIT
        trend_version = config.TREND_MODEL_VERSION

        for tf in timeframes:
            # 建構模型檔案路徑 (調整若實際不同)
            model_path = config.get_trend_model_path(symbol, tf, config.TREND_MODEL_VERSION)
            if not os.path.exists(config.MODEL_DIR):
                os.makedirs(config.MODEL_DIR)  # 若不存在，創建資料夾
            
            # 檢查是否存在
            if not overwrite and os.path.exists(model_path):
                print(f"📂 {symbol} ({tf}) 模型已存在，略過訓練。")
                continue
            
            # 若需訓練，則執行
            success = run_training_script(
                'train_trend_model.py', 
                symbol, 
                '2023-05-01',
                '2024-05-01', 
                trend_version,
                tf
            )
            if not success:
                print(f"🛑 錯誤：{symbol} 的 {tf} 趨勢模型訓練失敗。")
                continue 

        print(f"🎉 {symbol} 三項趨勢模型訓練皆成功完成！")

    print("\n==================================================")
    print("所有資產的訓練任務已完成。")
    print("==================================================")


if __name__ == '__main__':
    # 解析命令行參數
    parser = argparse.ArgumentParser(description="萬能訓練入口腳本")
    parser.add_argument('--overwrite', action='store_true', help='強制執行訓練並覆蓋已存在模型 (預設: False，若存在則略過)')
    args = parser.parse_args()
    
    # 您必須確保這 4 個檔案都在同一個目錄
    # config.py, common_utils.py, train_trend_model.py
    train_all_symbols(overwrite=args.overwrite)