# 檔案: backtest.py

import os
import argparse
import matplotlib.pyplot as plt

from core.context import Context
from core.exchange import PaperExchange
from core.portfolio import Portfolio
from strategies.xgboost_trend_strategy import XGBoostTrendStrategy
from core.backtest_engine import BacktestEngine
from settings import SYMBOLS_TO_TRADE
from utils.common import fetch_data

# ... (plot_results 函數保持不變) ...

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='向量化回測腳本')
    parser.add_argument('-sd', '--start', type=str, help='回測起始日期 (YYYY-MM-DD)')
    parser.add_argument('-ed', '--end', type=str, help='回測結束日期 (YYYY-MM-DD)')
    parser.add_argument('-tf', '--timeframe', type=str, default='1h', help='K 線時間週期 (例如: 5m, 15m, 1h)')
    parser.add_argument('--strategy', type=str, default='xgboost_trend', help='要執行的策略')
    parser.add_argument('--use-ppo', action='store_true', help='使用 PPO 進行倉位管理')
    parser.add_argument('--ppo-model', type=str, help='PPO 模型檔案的路徑')
    
    args = parser.parse_args()

    # --- 核心物件初始化流程重構 ---
    # 1. 初始化 Portfolio
    portfolio = Portfolio(10000)

    # 2. 初始化 PaperExchange (注入 Portfolio)
    exchange = PaperExchange(portfolio)

    # 3. 初始化 Context (使用新的、更严格的建构函式)
    context = Context(exchange=exchange, portfolio=portfolio, initial_capital=10000)

    # 4. 載入數據
    print("--- 開始載入數據 ---")
    data = {}
    symbols_to_run = SYMBOLS_TO_TRADE
    for symbol in symbols_to_run:
        print(f"正在獲取 {symbol} 的數據...")
        raw_df = fetch_data(symbol=symbol, start_date=args.start, end_date=args.end, timeframe=args.timeframe)
        if raw_df is not None and not raw_df.empty:
            data[symbol] = raw_df
            print(f"✅ {symbol} 數據載入成功，共 {len(raw_df)} 行。")
        else:
            print(f"警告：找不到 {symbol} 在指定時間範圍內的數據，將跳過。")

    if not data:
        print("錯誤：找不到任何有效的數據。")
        exit()
    print("--- 數據載入完成 ---")

    # 5. 初始化策略
    strategy = XGBoostTrendStrategy(
        context,
        symbols=list(data.keys()),
        use_ppo=args.use_ppo,
        ppo_model_path=args.ppo_model
    )

    # 6. 執行回測
    engine = BacktestEngine(context, strategy, data)
    results = engine.run()

    # 7. 顯示結果
    # ... (顯示結果邏輯保持不變) ...
