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

def plot_results(results, symbol, data, timeframe, start_date, end_date):
    """
    繪製回測結果，包含策略表現與 Buy & Hold 的比較。
    """
    history = results['history']
    
    # 創建 Buy & Hold 曲線
    main_symbol_data = data[symbol].copy()
    initial_price = main_symbol_data['Close'].iloc[0]
    initial_capital = history['total_value'].iloc[0]
    main_symbol_data['bh_equity'] = initial_capital * (main_symbol_data['Close'] / initial_price)

    plt.figure(figsize=(14, 7))

    # 繪製策略曲線 (藍色)
    plt.plot(history.index, history['total_value'], label='Strategy Equity', color='blue')

    # 繪製 Buy & Hold 曲線 (灰色)
    plt.plot(main_symbol_data.index, main_symbol_data['bh_equity'], label=f'Buy & Hold {symbol}', color='grey', linestyle='--')

    plt.title(f'Backtest Results: {symbol} ({timeframe})')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (USDT)')
    plt.legend()
    plt.grid(True)

    # 建立目錄與檔名
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)

    safe_symbol = symbol.replace('/', '-')
    filename = f"equity_curve_{safe_symbol}_{timeframe}_{start_date}_{end_date}.png"
    filepath = os.path.join(output_dir, filename)

    # 儲存圖表
    plt.savefig(filepath)
    print(f"資金曲線圖已保存為 {filepath}")


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
            timeframe='1m',
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
        plot_results(results, main_symbol, data, args.timeframe, args.start, args.end)
    else:
        print("回測未產生任何結果。")
