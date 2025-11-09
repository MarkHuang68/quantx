# 檔案: backtest.py

import os
import sys
import argparse
import matplotlib.pyplot as plt

from core.context import Context
from core.exchange import PaperExchange
from core.portfolio import Portfolio
from strategies.xgboost_trend_strategy import XGBoostTrendStrategy
from core.backtest_engine import BacktestEngine
from settings import SYMBOLS_TO_TRADE
from utils.common import fetch_data

def plot_results(results, symbol):
    """
    繪製回測結果。
    """
    history = results['history']
    
    plt.figure(figsize=(12, 6))
    plt.plot(history.index, history['total_value'], label='Strategy Equity')
    
    plt.title(f'Backtest Results for {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (USDT)')
    plt.legend()
    plt.grid(True)
    # 在非 GUI 環境下，我们不调用 plt.show()
    # plt.show()
    plt.savefig('backtest_results.png')
    print("回測結果圖表已保存為 backtest_results.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='向量化回測腳本')
    parser.add_argument('-sd', '--start', type=str, help='回測起始日期 (YYYY-MM-DD)')
    parser.add_argument('-ed', '--end', type=str, help='回測結束日期 (YYYY-MM-DD)')
    parser.add_argument('-tf', '--timeframe', type=str, default='1h', help='K 線時間週期 (例如: 5m, 15m, 1h)')
    parser.add_argument('--strategy', type=str, default='xgboost_trend', help='要執行的策略')
    parser.add_argument('--use-ppo', action='store_true', help='使用 PPO 進行倉位管理')
    parser.add_argument('--ppo-model', type=str, help='PPO 模型檔案的路徑')
    
    args = parser.parse_args()

    # 1. 初始化回測環境
    context = Context()
    context.exchange = PaperExchange()
    context.portfolio = Portfolio(context.initial_capital, context.exchange)

    # 2. 載入數據
    print("--- 開始載入數據 ---")
    data = {}
    # (為了測試，我们暂时只用一个 symbol)
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

    # 3. 選擇並初始化策略
    if args.strategy == 'xgboost_trend':
        strategy = XGBoostTrendStrategy(
            context,
            symbols=list(data.keys()),
            use_ppo=args.use_ppo,
            ppo_model_path=args.ppo_model
        )
    else:
        raise ValueError(f"未知的策略: {args.strategy}")

    # 4. 執行回測
    engine = BacktestEngine(context, strategy, data)
    results = engine.run()

    # 5. 顯示結果
    if results:
        print("\n--- 回測績效 ---")
        print(f"總報酬率: {results['total_return']:.2%}")
        print(f"夏普比率: {results['sharpe_ratio']:.2f}")
        print(f"最大回撤: {results['max_drawdown']:.2%}")

        # 繪製圖表
        main_symbol = list(data.keys())[0]
        # plot_results(results, main_symbol)
    else:
        print("回測未產生任何結果。")
