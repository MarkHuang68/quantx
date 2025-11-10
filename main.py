# 檔案: main.py

import os
import sys
import time
import argparse
import pandas as pd
from dotenv import load_dotenv
import json
import atexit

from core.context import Context
from core.exchange import BinanceExchange, PaperExchange
from core.portfolio import Portfolio
from strategies.xgboost_trend_strategy import XGBoostTrendStrategy
from utils.common import fetch_data, create_features_trend

PRODUCTION_MODELS_FILE = "production_models.json"
PERFORMANCE_FILE = "performance.json"
CIRCUIT_BREAKER_CRITERIA = { "max_drawdown_from_peak": 0.30, "max_loss_from_start": 0.15 }

def run_live(context, strategy, symbols, timeframe):
    print("--- 啟動實盤交易模式 ---")
    atexit.register(context.portfolio.save_performance)

    while True:
        try:
            current_dt = pd.Timestamp.now(tz='UTC')
            print(f"\n--- [{current_dt.strftime('%Y-%m-%d %H:%M:%S')}] ---")

            # 1. 獲取所有資產的最新價格 (一次性獲取，以供後續使用)
            current_prices = {}
            for symbol in symbols:
                current_prices[symbol] = context.exchange.get_latest_price(symbol)

            # 2. 績效監控與熔斷檢查
            total_value = context.portfolio.get_total_value(current_prices)
            # ... (熔斷邏輯保持不變) ...

            # 3. 同步倉位
            context.exchange.sync_positions()

            # 4. 獲取特徵並執行策略
            current_features = {}
            # ... (獲取數據和計算特徵的邏輯保持不變) ...

            if current_features:
                strategy.on_bar(current_dt, current_features)

            # 5. 更新 Portfolio 狀態
            context.portfolio.update(current_dt, current_prices)

            print(f"目前總資產: {context.portfolio.get_total_value(current_prices):.2f} USDT")
            # ... (日誌打印邏輯保持不變) ...

            time.sleep(300)

        except KeyboardInterrupt:
            # ... (退出邏輯保持不變) ...
            pass
        except Exception as e:
            # ... (錯誤處理邏輯保持不變) ...
            pass

if __name__ == '__main__':
    # ... (參數解析和模型載入邏輯保持不變) ...

    # --- 核心物件初始化流程重構 ---

    # 1. 初始化 Portfolio (唯一的狀態管理器)
    portfolio = Portfolio(10000, performance_file=PERFORMANCE_FILE)

    # 2. 初始化交易所 (將 Portfolio 注入)
    if args.mode == 'live':
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")
        exchange = BinanceExchange(api_key, api_secret, portfolio)
    else:
        exchange = PaperExchange(portfolio)

    # 3. 初始化 Context (現在包含已關聯的 portfolio 和 exchange)
    context = Context(exchange, portfolio, 10000)

    # ... (策略初始化和模型設定邏輯保持不變) ...

    # 執行
    if args.mode == 'live':
        run_live(context, strategy, symbols_to_trade, args.timeframe)
    else:
        # ... (paper 模式邏輯需要類似的重構)
        pass
