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
from core.data_loader import load_csv_data
from strategies.xgboost_trend_strategy import XGBoostTrendStrategy
from core.portfolio import Portfolio

from utils.common import fetch_data, create_features_trend

PRODUCTION_MODELS_FILE = "production_models.json"
PERFORMANCE_FILE = "performance.json"

# --- 新增：熔斷機制標準 ---
CIRCUIT_BREAKER_CRITERIA = {
    "max_drawdown_from_peak": 0.30, # 從組合最高淨值回撤 30%
    "max_loss_from_start": 0.15   # 從初始資金虧損 15%
}

def run_live(context, strategy, symbols, timeframe):
    """
    執行實盤交易，並整合績效監控與熔斷機制。
    """
    print("--- 啟動實盤交易模式 ---")
    print(f"交易對: {symbols}, K線週期: {timeframe}")

    # 註冊一個退出處理函數，確保程式終止時能儲存績效
    atexit.register(context.portfolio.save_performance)

    while True:
        try:
            current_dt = pd.Timestamp.now(tz='UTC')
            print(f"\n--- [{current_dt.strftime('%Y-%m-%d %H:%M:%S')}] ---")

            # --- 1. 績效監控與熔斷檢查 ---
            total_value = context.portfolio.get_total_value()
            perf_data = context.portfolio.performance_tracking

            # 初始化全局績效追蹤
            if "GLOBAL" not in perf_data:
                perf_data["GLOBAL"] = {"peak_net_worth": context.initial_capital, "status": "ACTIVE"}

            global_perf = perf_data["GLOBAL"]

            # 更新最高淨值
            global_perf["peak_net_worth"] = max(global_perf["peak_net_worth"], total_value)

            # 檢查熔斷條件
            drawdown_from_peak = (global_perf["peak_net_worth"] - total_value) / global_perf["peak_net_worth"]
            loss_from_start = (context.initial_capital - total_value) / context.initial_capital

            if drawdown_from_peak > CIRCUIT_BREAKER_CRITERIA["max_drawdown_from_peak"] or \
               loss_from_start > CIRCUIT_BREAKER_CRITERIA["max_loss_from_start"]:

                global_perf["status"] = "HALTED"
                print("🛑🛑🛑 熔斷機制觸發！🛑🛑🛑")
                print(f"--- 從最高淨值回撤: {drawdown_from_peak:.2%}")
                print(f"--- 從初始資金虧損: {loss_from_start:.2%}")
                print("--- 系統將平掉所有倉位並終止交易。 ---")

                context.exchange.close_all_positions(context.portfolio)
                context.portfolio.save_performance()
                sys.exit(1) # 終止程式

            print("--- 正在同步倉位... ---")
            context.exchange.sync_positions(context.portfolio)

            current_features = {}
            active_symbols = [s for s in symbols] # 未來可以整合單一幣種的熔斷

            for symbol in active_symbols:
                # ... (獲取數據和計算特徵的邏輯保持不變) ...
                ohlcv = fetch_data(symbol=symbol, timeframe=timeframe, limit=200)
                if ohlcv is None or ohlcv.empty: continue
                df_with_features, _ = create_features_trend(ohlcv)
                if df_with_features is not None and not df_with_features.empty:
                    current_features[symbol] = df_with_features.iloc[-1]

            if current_features:
                strategy.on_bar(current_dt, current_features)

            context.portfolio.update(current_dt)
            print(f"目前總資產: {context.portfolio.get_total_value():.2f} USDT")
            print(f"歷史最高資產: {global_perf['peak_net_worth']:.2f} USDT")
            print(f"目前倉位: {context.portfolio.get_positions()}")

            time.sleep(300)

        except KeyboardInterrupt:
            print("\n--- 交易機器人已手動停止 ---")
            context.portfolio.save_performance()
            break
        except Exception as e:
            print(f"發生嚴重錯誤: {e}")
            context.portfolio.save_performance()
            time.sleep(60)

# ... (run_paper 函數保持不變)

if __name__ == '__main__':
    # ... (參數解析和模型載入邏輯保持不變) ...
    load_dotenv()
    parser = argparse.ArgumentParser(description='交易機器人主程式')
    # ...

    # --- 整合「上線關卡」邏輯 ---
    # ... (這部分邏輯保持不變)

    # 1. 初始化 Context 和 Portfolio (現在會自動載入績效)
    context = Context()
    context.portfolio = Portfolio(context.initial_capital, performance_file=PERFORMANCE_FILE)

    # ... (交易所和策略初始化邏輯保持不變) ...

    # 執行
    if args.mode == 'live':
        run_live(context, strategy, symbols_to_trade, args.timeframe)
    else:
        # ... (paper 模式邏輯保持不變)
        pass
