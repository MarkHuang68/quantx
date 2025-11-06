# 檔案: search_backtest.py
# 目的: 使用 SearchIterator 產生參數組合，運行 backtest.py 尋找最佳止損/止盈/進場門檻參數。
# 修改說明:
# - 原檔名 hyperparameter_search_backtest.py 改為 search_backtest.py。
# - 新增 argparse --start 和 --end 參數（預設 None），傳給 backtest.py 以啟用快取邏輯（當皆設值時，backtest.py 使用 data/ 快取 CSV）。
# - SEARCH_SPACE 定義離散參數範圍。
# - 使用 random 模式測試 20 組，避免過多計算。
# - 每組呼叫 backtest.py (傳 --stop_loss 等及 --start/--end/--no_plot)，讀 pnl.json 取 total_pnl 更新最佳。
# - 輸出最佳參數及 PnL。
# 使用方式: python search_backtest.py --start 2023-01-01 --end 2024-01-01

import numpy as np
import itertools
import random
import subprocess  # 用於呼叫 backtest.py 命令列
import json  # 用於讀取 pnl.json 結果
import argparse  # 用於解析命令列參數

# 引用 SearchIterator (假設 hyperparameter_search.py 已存在並 import)
from hyperparameter_search import SearchIterator

# 定義尋參空間 (離散值列表，根據需求調整範圍)
SEARCH_SPACE = {
    'stop_loss_pct': [0.01, 0.015, 0.02, 0.025],  # 止損百分比範圍
    'take_profit_pct': [0.02, 0.03, 0.04, 0.05],  # 止盈百分比範圍
    'entry_threshold': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]  # 進場門檻範圍
}

FORMAT_TYPES = {
    'stop_loss_pct': 'discrete',  # 離散類型
    'take_profit_pct': 'discrete',
    'entry_threshold': 'discrete'
}

if __name__ == "__main__":
    # 解析命令列參數（--start 和 --end 傳給 backtest.py 以啟用快取）
    parser = argparse.ArgumentParser(description='執行回測參數尋優')
    parser.add_argument('-s', '--symbol', type=str, default=None, help='Symbol，預設 None')
    parser.add_argument('--start', type=str, default=None, help='回測起始日期 (YYYY-MM-DD)，預設 None')
    parser.add_argument('--end', type=str, default=None, help='回測結束日期 (YYYY-MM-DD)，預設 None')
    args = parser.parse_args()

    # 初始化 iterator (random 模式，隨機測試 20 組組合；可改 'grid' 測試全部)
    iterator = SearchIterator(SEARCH_SPACE, search_type='random', n_iter=20, format_types=FORMAT_TYPES)

    best_pnl = -np.inf  # 初始化最佳 PnL (負無窮，尋找最大值)
    best_params = None  # 初始化最佳參數字典

    # 迴圈測試每個組合
    for i, config in enumerate(iterator):
        print(f"Run {i+1}: 測試參數 {config}")
        
        # 建構 backtest.py 命令列（傳 symbol、參數、--start/--end、--no_plot 跳過圖形顯示）
        cmd = [
            'python', 'backtest.py', '-s', str(args.symbol),  # 固定 symbol，可調整
            '--stop_loss', str(config['stop_loss_pct']),
            '--take_profit', str(config['take_profit_pct']),
            '--entry_threshold', str(config['entry_threshold']),
            '--no_plot'  # 跳過顯示圖形，避免尋參時彈窗干擾
        ]
        
        # 若 --start 和 --end 有值，添加至 cmd 以啟用快取
        if args.start:
            cmd.extend(['--start', args.start])
        if args.end:
            cmd.extend(['--end', args.end])
        
        subprocess.run(cmd)  # 運行 backtest.py
        
        # 讀 pnl.json (從 backtest.py 輸出取 total_pnl)
        with open('pnl.json', 'r') as f:
            result = json.load(f)
            pnl = result['total_pnl']  # 提取 total_pnl 作為評估指標
        
        # 更新最佳組合（若 pnl 更好）
        if pnl > best_pnl:
            best_pnl = pnl
            best_params = config

    # 輸出最終結果
    print(f"最佳參數: {best_params}")
    print(f"最佳 PnL: {best_pnl}")