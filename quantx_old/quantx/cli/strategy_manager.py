# 檔案: quantx/cli/strategy_manager.py
# 版本: v2 (DI 重構)
# 說明:
# - 重構以使用 AppContainer 進行依賴注入，移除 get_runtime() 調用。
# - runtime 和 candidate_store 實例由容器創建，並作為參數傳遞。

import argparse
import json
from pathlib import Path
from typing import List

from quantx.core.config import Config
from quantx.containers import AppContainer
from quantx.core.runtime import Runtime
from quantx.core.policy.candidate_store import CandidateStore

def remove_strategy(runtime: Runtime, store: CandidateStore, strategy_name: str):
    """
    執行移除策略的完整自動化流程。

    Args:
        runtime (Runtime): 核心運行環境實例。
        store (CandidateStore): 候選池儲存服務實例。
        strategy_name (str): 要移除的策略名稱。
    """
    log = runtime.log
    log.info(f"=========== 開始執行策略移除流程: '{strategy_name}' ===========")

    # --- 步驟 1: 全局掃描，找出正在運行的策略實例 ---
    status_file = Path("results") / "live_status.json"
    symbols_to_close: List[str] = []
    
    if not status_file.exists():
        log.warning(f"找不到狀態檔案 {status_file}，無法自動偵測運行中的策略。")
    else:
        try:
            with open(status_file, "r", encoding="utf-8") as f:
                status_data = json.load(f)
            
            for instance in status_data.get("strategy_status", []):
                if instance.get("strategy") == strategy_name:
                    symbol = instance.get("symbol", "").split('-')[0]
                    if symbol and symbol not in symbols_to_close:
                        symbols_to_close.append(symbol)
            
            if symbols_to_close:
                log.info(f"找到 {len(symbols_to_close)} 個由 '{strategy_name}' 管理的倉位: {', '.join(symbols_to_close)}")
            else:
                log.info(f"未找到由 '{strategy_name}' 管理的運行中倉位。")
        except Exception as e:
            log.error(f"讀取或解析 {status_file} 失敗: {e}，請手動檢查倉位。")

    # --- 步驟 2: 精準平倉 ---
    if symbols_to_close:
        provider = runtime.provider
        try:
            online_positions_raw = provider.get_positions()
            online_positions = {p['symbol'].replace('/', ''): p for p in online_positions_raw if 'symbol' in p}
        except Exception as e:
            log.error(f"從交易所獲取倉位失敗: {e}", exc_info=True)
            online_positions = {}

        for symbol in symbols_to_close:
            if symbol in online_positions:
                pos_data = online_positions[symbol]
                pos_size = float(pos_data.get('contracts', 0))
                side = pos_data.get('side')
                
                if pos_size > 0:
                    log.info(f"正在為 {symbol} 提交市價平倉指令...")
                    try:
                        close_side = 'sell' if side == 'long' else 'buy'
                        provider.submit_order(
                            symbol=symbol,
                            side=close_side,
                            qty=pos_size,
                            reduce_only=True,
                            order_type="market"
                        )
                        log.info(f"✅ {symbol} 平倉指令已成功提交。")
                    except Exception as e:
                        log.error(f"❌ {symbol} 平倉失敗: {e}", exc_info=True)
            else:
                log.info(f"在交易所查詢不到 {symbol} 的倉位資訊，跳過平倉。")
    
    # --- 步驟 3: 徹底清理候選池 ---
    log.info(f"正在從候選池中清理 '{strategy_name}' 的所有歷史紀錄...")
    candidate_base_dir = Path("results") / runtime.scope / "candidates"
    cleaned_files_count = 0
    if not candidate_base_dir.exists():
        log.warning(f"候選池目錄不存在: {candidate_base_dir}。")
    else:
        for json_file in candidate_base_dir.glob("*.json"):
            if json_file.name == "candidates_index.json": continue
            original_candidates = store._load_json(json_file)
            if not original_candidates: continue
            
            cleaned_candidates = [cand for cand in original_candidates if cand.get("name") != strategy_name]
            
            if len(cleaned_candidates) < len(original_candidates):
                log.info(f"  - 正在清理 {json_file.name}...")
                store._save_json(json_file, cleaned_candidates)
                cleaned_files_count += 1
    
    log.info(f"候選池清理完成，共更新了 {cleaned_files_count} 個檔案。")

    # --- 步驟 4: 清晰回報 ---
    log.info("===================================================================")
    log.info(f"✅ 策略 '{strategy_name}' 已成功從系統中移除。")
    log.info("下一步操作建議:")
    log.info("1. 手動從 'conf/train.yaml' 中移除此策略的訓練設定。")
    log.info("2. 重新啟動 launch.py 和 train_daemon.py 服務。")
    log.info("===================================================================")

def main():
    """策略管理器主程式入口。"""
    parser = argparse.ArgumentParser(description="QuantX 策略維運管理工具")
    subparsers = parser.add_subparsers(dest="command", help="可用的指令", required=True)

    parser_remove = subparsers.add_parser("remove", help="安全地移除一個策略。")
    parser_remove.add_argument("--strategy", type=str, required=True, help="要移除的策略名稱 (e.g., z_score_reverse)")

    args = parser.parse_args()

    # 初始化容器並獲取服務
    cfg = Config()
    container = AppContainer(cfg)
    runtime = container.runtime
    store = container.candidate_store

    if args.command == "remove":
        remove_strategy(runtime, store, args.strategy)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()