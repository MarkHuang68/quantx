# 檔案: main.py

import os
import sys
import time
import argparse
import pandas as pd
from dotenv import load_dotenv

from core.context import Context
from core.exchange import BinanceExchange, CoinbaseExchange, PaperExchange
from core.data_loader import load_csv_data
from strategies.xgboost_trend_strategy import XGBoostTrendStrategy
from core.portfolio import Portfolio

def run_live(context, strategy):
    """
    執行實盤交易。
    """
    print("--- 啟動實盤交易模式 ---")
    while True:
        try:
            # 獲取當前時間並觸發策略
            current_dt = pd.Timestamp.now(tz='UTC')
            strategy.on_bar(current_dt)

            # 更新投資組合
            context.portfolio.update(current_dt)
            print(f"目前總資產: {context.portfolio.get_total_value():.2f} USDT")

            time.sleep(300) # 每 5 分鐘執行一次

        except KeyboardInterrupt:
            print("--- 交易機器人已手動停止 ---")
            break
        except Exception as e:
            print(f"發生錯誤: {e}")
            time.sleep(60)

def run_paper(context, strategy, data):
    """
    執行模擬交易。
    """
    print("--- 啟動模擬交易模式 ---")

    # 將數據載入到模擬交易所
    for symbol, df in data.items():
        context.exchange.set_kline_data(symbol, df)

    # 遍歷數據並觸發策略
    # 假設所有數據的時間戳是對齊的
    main_symbol = list(data.keys())[0]
    for dt in data[main_symbol].index:
        context.exchange.set_current_dt(dt)
        strategy.on_bar(dt)

        # 模擬更新投資組合
        # (在模擬交易中，我們可能需要一個更詳細的 Portfolio 更新機制)
        # print(f"[{dt}] 模擬總資產: {context.portfolio.get_total_value():.2f} USDT")

    print("--- 模擬交易結束 ---")

if __name__ == '__main__':
    load_dotenv() # 載入 .env 檔案中的環境變數

    parser = argparse.ArgumentParser(description='交易機器人主程式')
    parser.add_argument('--mode', type=str, choices=['live', 'paper'], required=True, help='執行模式 (live: 實盤, paper: 模擬)')
    parser.add_argument('--exchange', type=str, choices=['binance', 'coinbase'], default='binance', help='交易所 (僅在 live 模式下有效)')
    parser.add_argument('--data-dir', type=str, help='包含 CSV 數據檔案的目錄 (僅在 paper 模式下需要)')
    parser.add_argument('--use-ppo', action='store_true', help='使用 PPO 進行倉位管理')
    parser.add_argument('--ppo-model', type=str, help='PPO 模型檔案的路徑')

    args = parser.parse_args()

    # 1. 初始化 Context
    context = Context()

    # 2. 初始化交易所
    if args.mode == 'live':
        api_key = os.getenv(f"{args.exchange.upper()}_API_KEY")
        api_secret = os.getenv(f"{args.exchange.upper()}_API_SECRET")
        if not api_key or not api_secret:
            raise ValueError(f"請在 .env 檔案中設定 {args.exchange.upper()}_API_KEY 和 {args.exchange.upper()}_API_SECRET")

        if args.exchange == 'binance':
            context.exchange = BinanceExchange(api_key, api_secret)
        elif args.exchange == 'coinbase':
            context.exchange = CoinbaseExchange(api_key, api_secret)

    elif args.mode == 'paper':
        if not args.data_dir:
            raise ValueError("在 paper 模式下，必須透過 --data-dir 提供數據目錄")
        context.exchange = PaperExchange()

    # 3. 初始化 Portfolio
    context.portfolio = Portfolio(context.initial_capital, context.exchange)

    # 4. 讀取交易對設定
    from config.settings import SYMBOLS_TO_TRADE

    # 5. 初始化策略
    strategy = XGBoostTrendStrategy(
        context,
        symbols=SYMBOLS_TO_TRADE,
        use_ppo=args.use_ppo,
        ppo_model_path=args.ppo_model
    )

    # 6. 執行
    if args.mode == 'live':
        context.exchange.sync_positions(context.portfolio)
        run_live(context, strategy)
    elif args.mode == 'paper':
        data = {}
        for symbol in SYMBOLS_TO_TRADE:
            filepath = os.path.join(args.data_dir, f"{symbol.replace('/', '_')}.csv")
            data[symbol] = load_csv_data(filepath, symbol=symbol)
            if data[symbol] is None:
                print(f"警告：找不到 {symbol} 的數據檔案，將跳過。")

        if data:
            run_paper(context, strategy, data)
        else:
            print("錯誤：找不到任何有效的數據檔案。")
