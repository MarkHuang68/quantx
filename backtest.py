# 檔案: backtest.py

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import matplotlib.pyplot as plt

from core.context import Context
from core.exchange import PaperExchange
from core.portfolio import Portfolio
from core.data_loader import load_csv_data
from strategies.xgboost_trend_strategy import XGBoostTrendStrategy
from backtest.engine import BacktestEngine

def plot_results(results, symbol):
    """
    繪製回測結果。
    """
    history = results['history']
    
    plt.figure(figsize=(12, 6))
    plt.plot(history.index, history['total_value'], label='Strategy Equity')
    
    # 計算 Buy & Hold
    # buy_hold_returns = data[symbol]['close'].pct_change().dropna()
    # buy_hold_equity = (1 + buy_hold_returns).cumprod() * context.initial_capital
    # plt.plot(buy_hold_equity.index, buy_hold_equity, label='Buy & Hold')
    
    plt.title(f'Backtest Results for {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (USDT)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='向量化回測腳本')
    parser.add_argument('--data-dir', type=str, required=True, help='包含 CSV 數據檔案的目錄')
    parser.add_argument('--strategy', type=str, default='xgboost_trend', help='要執行的策略')
    parser.add_argument('--use-ppo', action='store_true', help='使用 PPO 進行倉位管理')
    parser.add_argument('--ppo-model', type=str, help='PPO 模型檔案的路徑')
    
    args = parser.parse_args()

    # 1. 初始化回測環境
    context = Context()
    context.exchange = PaperExchange()
    context.portfolio = Portfolio(context.initial_capital, context.exchange)

    # 2. 讀取交易對設定
    from config.settings import SYMBOLS_TO_TRADE

    # 3. 載入數據
    data = {}
    for symbol in SYMBOLS_TO_TRADE:
        filepath = os.path.join(args.data_dir, f"{symbol.replace('/', '_')}.csv")
        data[symbol] = load_csv_data(filepath, symbol=symbol)
        if data[symbol] is None:
            print(f"警告：找不到 {symbol} 的數據檔案，將跳過。")

    if not data:
        print("錯誤：找不到任何有效的數據檔案。")
        exit()

    # 4. 選擇並初始化策略
    if args.strategy == 'xgboost_trend':
        strategy = XGBoostTrendStrategy(
            context,
            symbols=list(data.keys()),
            use_ppo=args.use_ppo,
            ppo_model_path=args.ppo_model
        )
    else:
        raise ValueError(f"未知的策略: {args.strategy}")

    # 5. 執行回測
    engine = BacktestEngine(context, strategy, data)
    results = engine.run()

    # 5. 顯示結果
    print("\n--- 回測績效 ---")
    print(f"總報酬率: {results['total_return']:.2%}")
    print(f"夏普比率: {results['sharpe_ratio']:.2f}")
    print(f"最大回撤: {results['max_drawdown']:.2%}")
    
    # 6. 繪製圖表
    # 在無 GUI 環境下，我們無法顯示圖表，所以將其註解掉
    # main_symbol = list(data.keys())[0]
    # plot_results(results, main_symbol)
