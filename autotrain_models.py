
# 檔案: autotrain_models.py
# 【!!! 核心修改 !!!】: 
# 1. --force_all 現在會傳遞 --force_save 給 train_trend_model.py
# 2. 合併 --loop 參數
# 3. 【!!! NEW !!!】 修正 Retrain 模式下 -l (limit) 參數的優先權

import os
import sys
import argparse
import json
import pandas as pd 
import time

# --- 1. 設定環境路徑 (與 train_trend_model.py 相同) ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- 2. 載入您的 settings.py 模組 ---
try:
    import settings
except ImportError:
    print("❌ 錯誤: 找不到 settings.py。")
    print("請確保此腳本與 train_trend_model.py 位於同一目錄下。")
    sys.exit(1)

def main(args):
    print(f"--- 自動訓練腳本已啟動 (PID: {os.getpid()}) ---")

    # --- 3. 載入註冊表 ---
    registry = settings.load_registry()
    if not registry and not args.timeframes:
        print(f"❌ 錯誤: 註冊表 {settings.REGISTRY_FILE} 為空或不存在。")
        print("您必須先手動執行 train_trend_model.py 建立第一筆紀錄，或使用 -tf 參數來執行 'autotrain' (新訓練)。")
        return # 【修正】: 在循環模式下，這裡應該 return 而不是 exit

    # --- 4. 【!!! 核心邏輯：決定任務清單 !!!】 ---
    tasks_to_run = [] # 儲存 (key, symbol, tf, start_date, end_date)

    if args.timeframes:
        # --- 模式二: "autotrain" (訓練新模型) ---
        print(f"--- 模式: AUTOTRAIN (訓練新模型) ---")
        if not args.symbol or (not args.start and not args.limit) or (args.start and not args.end):
            print("❌ 錯誤: 'autotrain' 模式 (-tf) 必須同時指定 -s (符號),")
            print("         以及 (-sd 和 -ed) 或 (-l)。")
            sys.exit(1) # Autotrain 模式出錯應直接退出
        
        symbol_safe = args.symbol.replace('/', '')
        for tf in args.timeframes:
            key = f"{symbol_safe}_{tf}"
            tasks_to_run.append((key, args.symbol, tf, args.start, args.end))
        
        print(f"✅ 將為 {args.symbol} 建立/訓練 {len(args.timeframes)} 個新模型。")

    else:
        # --- 模式一: "retrain" (更新現有模型) ---
        print(f"--- 模式: RETRAIN (更新現有模型) ---")
        
        # 【!!! 核心修正 !!!】: Retrain 模式必須指定一個動作
        if not args.force_all and not args.min_sr:
            print("❌ 錯誤: 'Retrain' 模式必須指定一個動作 (因為您沒有使用 -tf)。")
            print("         請使用 --force_all (強制全部重訓)")
            print("         或   --min_sr [score] (重訓 SR 低於 N 分的模型)")
            return # 【修正】: 在循環模式下，這裡應該 return

        symbol_to_retrain = args.symbol.replace('/', '') if args.symbol else None

        for key, config in registry.items():
            symbol_in_key = key.split('_')[0]
            tf_in_key = key.split('_')[-1]

            # 1. 過濾符號
            if symbol_to_retrain and (symbol_in_key != symbol_to_retrain):
                continue # 跳過，這不是我們要訓練的符號
            
            # --- 【!!! 核心修正：檢查是否需要執行 !!!】 ---
            should_run_task = False
            reason = ""
            
            if args.force_all:
                # 2a. (功能2) 強制全部重訓
                should_run_task = True
                reason = "強制重訓 (--force_all)"
            
            elif args.min_sr:
                # 2b. (功能1) SR 門檻重訓
                current_sr = config.get('objective_sharpe_ratio', 0.0)
                if current_sr < args.min_sr:
                    should_run_task = True
                    reason = f"SR 低分 ({current_sr:.4f} < {args.min_sr:.4f})"
                else:
                    print(f"--- ⏩ 跳過 {key} (SR {current_sr:.4f} >= {args.min_sr:.4f}) ---")
            
            if not should_run_task:
                continue # 跳過此模型
            # --- 【修正結束】 ---

            # 3. 決定日期 (如果需要執行)
            
            # === 【!!! 核心修正：-l 優先於 registry 日期 !!!】 ===
            start_date_to_use = args.start
            end_date_to_use = args.end
            
            if args.limit:
                # 如果使用者提供了 -l (limit)，則必須強制清除所有日期
                # 這樣 train_trend_model.py 才會只看 -l
                print(f"--- 偵測到 -l 參數，將忽略 registry 中的日期 ---")
                start_date_to_use = None
                end_date_to_use = None
            
            elif not start_date_to_use: 
                # (原始邏輯) 
                # 只有在「沒有 -l」且「沒有 -sd」時，才使用 registry 的日期
                start_date_to_use = config.get('start_date')
                end_date_to_use = config.get('end_date')
            # === 【!!! 修正結束 !!!】 ===


            if not start_date_to_use and not end_date_to_use and not args.limit:
                print(f"❌ 錯誤: {key} 在註冊表中沒有儲存日期，您必須手動指定 -sd 和 -ed (或 -l)。跳過。")
                continue
            
            print(f"--- 📥 加入任務 {key} (理由: {reason}) ---")
            tasks_to_run.append((key, symbol_in_key, tf_in_key, start_date_to_use, end_date_to_use))

    if not tasks_to_run:
        print("✅ 根據您的參數，找不到任何需要執行的訓練任務。")
        return # (在循環模式下，這只是 "本輪沒事做")

    print(f"✅ 將執行 {len(tasks_to_run)} 項訓練任務...")

    # --- 5. 迴圈執行 train_trend_model.py ---
    for (key, symbol, tf, start_date, end_date) in tasks_to_run:
        print(f"\n{'='*50}")
        print(f"--- 正在執行: {symbol} @ {tf} ---")
        
        # 建立命令
        base_command = (
            f"python train_trend_model.py "
            f"-s {symbol} "
            f"--force_train "  # <-- 強制重新訓練 (觸發三階段尋參)
        )
        
        # 處理日期或 K 棒限制
        if args.limit:
             base_command += f"-l {args.limit} "
             print(f"--- 使用 K 棒限制: {args.limit} ---")
        elif start_date and end_date:
             base_command += f"-sd {start_date} -ed {end_date} "
             print(f"--- 使用日期區間: {start_date} 到 {end_date} ---")
        
        # 傳遞其他可選參數
        if args.no_search_params:
            base_command += "--no_search_params "
        if args.no_search_conf:
            base_command += "--no_search_conf "
        if args.no_search_model:
            base_command += "--no_search_model "
        
        # --- 【!!! 核心修正 !!!】 ---
        # 如果 autotrain 模式是 --force_all，則傳遞 --force_save
        if args.force_all:
            base_command += "--force_save "
        # --- 【!!! 修正結束 !!!】 ---

        # 組合最終命令
        retrain_command = f"{base_command} -tf {tf}"
        print(f"執行命令: {retrain_command}")
        
        # 執行命令 (呼叫您現有的檔案)
        status = os.system(retrain_command)
        
        if status == 0:
            print(f"--- ✅ 完成: {symbol} @ {tf} ---")
        else:
            print(f"--- ❌ 錯誤: {symbol} @ {tf} 訓練失敗 (狀態碼: {status}) ---")
            print("請檢查 train_trend_model.py 的錯誤輸出。")
            
    print(f"\n{'='*50}")
    print("--- 🔔 本輪自動訓練任務已完成 ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='自動訓練 (Autotrain) / 重新訓練 (Retrain) 註冊表中的所有模型')
    
    # --- 模式參數 (決定 Autotrain 或 Retrain) ---
    parser.add_argument('-s', '--symbol', type=str, help='(Autotrain 必填 / Retrain 可選) 要訓練的符號')
    parser.add_argument('-tf', '--timeframes', nargs='+', help='(Autotrain 模式) 觸發 "autotrain" (新訓練) 模式，並指定時間框架 (例如: 1m 5m 1h)')

    # --- Retrain 模式的觸發器 (功能 1 & 2) ---
    parser.add_argument('--force_all', action='store_true', help='(Retrain 模式) 強制重新訓練「所有」模型 (並強制覆蓋儲存)')
    parser.add_argument('--min_sr', type=float, help='(Retrain 模式) 僅重新訓練 objective_sharpe_ratio 低於此分數的模型')

    # --- 【!!! 核心修正：合併 --loop 和 --loop_count !!!】 ---
    parser.add_argument(
        '--loop', 
        nargs='?', 
        const=-1, 
        default=None, 
        type=int, 
        help='(Loop 模式) 循環執行。'
             '不加: 執行一次。 '
             '--loop: 無限循環。 '
             '--loop 5: 循環 5 次。'
    )
    parser.add_argument('--loop_delay', type=int, default=60, help='(Loop 模式) 循環之間的延遲秒數 (預設: 3600s = 1hr)')
    # --- 【修正結束】 ---

    # --- 日期參數 ---
    parser.add_argument('-sd', '--start', type=str, help='(Autotrain 必填 / Retrain 可選) 資料起始日期 (YYYY-MM-DD)')
    parser.add_argument('-ed', '--end', type=str, help='(Autotrain 必填 / Retrain 可選) 資料結束日期 (YYYY-MM-DD)')
    
    # --- 傳遞參數 ---
    parser.add_argument('-l', '--limit', type=int, help='(傳遞) K 線筆數限制 (可選)')
    parser.add_argument('-nsp', '--no_search_params', action='store_true', help='(傳遞) 關閉「階段一」的特徵參數調校')
    parser.add_argument('-nsm', '--no_search_model', action='store_true', help='(傳遞) 關閉「階段二」的模型參數調校')
    parser.add_argument('-nsc', '--no_search_conf', action='store_true', help='(傳遞) 關閉「階段三」的信心門檻調校')
    
    parsed_args = parser.parse_args()
    
    # --- 邏輯檢查 ---
    if (parsed_args.start and not parsed_args.end) or (not parsed_args.start and parsed_args.end):
        print("❌ 錯誤: -sd (起始日期) 和 -ed (結束日期) 必須同時提供，或者都不提供。")
        sys.exit(1)
        
    if parsed_args.timeframes and (parsed_args.min_sr or parsed_args.force_all):
        print("❌ 錯誤: 'Autotrain' 模式 (-tf) 不能與 'Retrain' 模式的觸發器 (--min_sr, --force_all) 同時使用。")
        sys.exit(1)
        
    if parsed_args.timeframes and parsed_args.loop is not None: # 【修正】
        print("❌ 錯誤: 'Autotrain' 模式 (-tf) (新訓練) 不能與 --loop (循環) 同時使用。")
        sys.exit(1)
        
    if parsed_args.min_sr and parsed_args.force_all:
        print("❌ 錯誤: --min_sr 和 --force_all 是互斥的，請只選一個。")
        sys.exit(1)

    if not os.path.exists("train_trend_model.py"):
         print("❌ 錯誤: 找不到 train_trend_model.py 檔案。")
         print("請確保此腳本 (autotrain_models.py) 與 train_trend_model.py 位於同一目錄下。")
         sys.exit(1)

    # --- 【!!! 核心修正：循環邏輯 !!!】 ---
    if parsed_args.loop is not None: # 檢查 --loop 是否被啟用 (無論是 -1 還是 N)
        print(f"--- ♾️ 循環模式已啟動 ---")
        
        current_loop = 0 # 循環計數器
        target_loops = parsed_args.loop # 目標次數 (-1 或 N)
        
        while True:
            try:
                loop_display = f"{current_loop + 1} / {target_loops if target_loops != -1 else '∞'}"
                
                print(f"\n{'='*60}")
                print(f"--- 🔁 (時間: {pd.Timestamp.now()}) 循環 {loop_display} 開始，執行 main()... ---")
                main(parsed_args)
                
                current_loop += 1
                
                # 檢查是否達到目標次數
                if target_loops != -1 and current_loop >= target_loops:
                    print(f"--- ✅ 已達到目標循環次數 ({target_loops})，停止。 ---")
                    break # 退出 while True 循環
                    
                print(f"--- 🔁 循環 {current_loop} 完成，將休眠 {parsed_args.loop_delay} 秒... (按 Ctrl+C 停止) ---")
                time.sleep(parsed_args.loop_delay)
                
            except KeyboardInterrupt:
                print("\n--- 🛑 偵測到 Ctrl+C，循環停止。 ---")
                sys.exit(0)
            except Exception as e:
                print(f"--- ❌ 循環中發生嚴重錯誤: {e} ---")
                print(f"--- 將休眠 60 秒後重試... ---")
                time.sleep(60)
    else:
        # (原始行為: 只執行一次)
        main(parsed_args)