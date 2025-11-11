# 檔案: main.py

import os
import sys
import time
import argparse
import pandas as pd
import asyncio
import ccxt.pro
from dotenv import load_dotenv
from datetime import datetime, timezone

from core.context import Context
from core.exchange import BinanceExchange, CoinbaseExchange, PaperExchange, BybitExchange
from core.data_loader import load_csv_data
from strategies.xgboost_trend_strategy import XGBoostTrendStrategy
from core.portfolio import Portfolio
from utils.common import fetch_data, create_features_trend

def convert_symbol_to_ccxt(symbol: str) -> str:
    """將 'ETHUSDT' 轉換為 'ETH/USDT:USDT'"""
    if '/' in symbol: return symbol
    base = symbol.replace('USDT', '')
    return f"{base}/USDT:USDT"

def convert_symbol_from_ccxt(ccxt_symbol: str) -> str:
    """將 'ETH/USDT:USDT' 轉換為 'ETHUSDT'"""
    return ccxt_symbol.split('/')[0]

async def warm_up(context, symbols, timeframe):
    print("--- 數據預熱階段開始 ---")
    initial_features = {}
    for symbol in symbols:
        ccxt_symbol = convert_symbol_to_ccxt(symbol)
        print(f"正在為 {ccxt_symbol} 預載歷史數據...")
        try:
            ohlcv = await context.exchange.get_ohlcv(symbol=ccxt_symbol, timeframe=timeframe, limit=200)
            if ohlcv is None or ohlcv.empty or len(ohlcv) < 200:
                print(f"警告：為 {ccxt_symbol} 預載的數據不足...")
                continue
            print(f"正在為 {symbol} 計算初始特徵...")
            df_with_features, _ = create_features_trend(ohlcv)
            if df_with_features is not None and not df_with_features.empty:
                initial_features[symbol] = df_with_features.iloc[-1]
                print(f"✅ {symbol} 預熱完成。")
            else:
                print(f"警告：無法為 {symbol} 計算初始特徵。")
        except Exception as e:
            print(f"🛑 為 {symbol} 預熱數據時發生錯誤: {e}")
    print("--- 數據預熱階段完成 ---")
    return initial_features

async def run_live(context, strategy, symbols, timeframe):
    print("--- 啟動實盤交易模式 (WebSocket) ---")
    ccxt_symbols = [convert_symbol_to_ccxt(s) for s in symbols]
    print(f"交易對: {ccxt_symbols}, K線週期: {timeframe}")

    subscription_topics = [[symbol, timeframe] for symbol in ccxt_symbols]
    last_kline_timestamp = {}

    while True:
        try:
            print("正在連接/重連 WebSocket...")
            while True:
                ohlcv_by_symbol = await context.exchange.exchange.watch_ohlcv_for_symbols(subscription_topics)
                for ccxt_symbol, tf_data in ohlcv_by_symbol.items():
                    if timeframe not in tf_data: continue
                    latest_kline = tf_data[timeframe][-1]
                    kline_timestamp = latest_kline[0]
                    internal_symbol = convert_symbol_from_ccxt(ccxt_symbol)

                    if kline_timestamp > last_kline_timestamp.get(internal_symbol, 0):
                        last_kline_timestamp[internal_symbol] = kline_timestamp
                        current_dt = pd.to_datetime(kline_timestamp, unit='ms', utc=True)
                        print(f"\n--- [{current_dt.strftime('%Y-%m-%d %H:%M:%S')}] 發現 {internal_symbol} 新 K 棒 ---")

                        await context.exchange.sync_positions(context.portfolio)
                        ohlcv = await context.exchange.get_ohlcv(symbol=ccxt_symbol, timeframe=timeframe, limit=200)
                        if ohlcv is None or ohlcv.empty:
                            print(f"警告：無法獲取 {ccxt_symbol} 的歷史數據，跳過。")
                            continue
                        df_with_features, _ = create_features_trend(ohlcv)
                        if df_with_features is not None and not df_with_features.empty:
                            latest_features = {internal_symbol: df_with_features.iloc[-1]}
                            await strategy.on_bar(current_dt, latest_features)
                        else:
                            print(f"警告：無法為 {internal_symbol} 計算特徵。")
                        await context.portfolio.update(current_dt)
                        print(f"目前總資產: {context.portfolio.get_total_value():.2f} USDT")
                        print(context.portfolio.get_positions_summary())
        except asyncio.CancelledError:
            print("\n--- WebSocket 處理器被取消，安全退出 ---")
            break
        except Exception as e:
            print(f"發生未知嚴重錯誤: {e}")
            await asyncio.sleep(15)
        finally:
            await context.exchange.close()

def run_paper(context, strategy, data):
    print("--- 啟動模擬交易模式 ---")
    features_data = {}
    for symbol, df in data.items():
        print(f"正在為 {symbol} 預計算特徵...")
        df_with_features, _ = create_features_trend(df)
        if df_with_features is not None:
            features_data[symbol] = df_with_features
        # 確保傳遞給 PaperExchange 的是 ccxt 格式
        context.exchange.set_kline_data(convert_symbol_to_ccxt(symbol), df)

    try:
        main_symbol = list(data.keys())[0]
    except IndexError:
        print("錯誤：數據字典為空，無法開始回測。")
        return

    print(f"--- 開始模擬回放 (主時間軸: {main_symbol}) ---")
    async def run_paper_async():
        for dt in data[main_symbol].index:
            context.exchange.set_current_dt(dt)
            current_features = {}
            for symbol, df_features in features_data.items():
                if dt in df_features.index:
                    current_features[symbol] = df_features.loc[dt]
            if current_features:
                await strategy.on_bar(dt, current_features)
            await context.portfolio.update(dt)
    asyncio.run(run_paper_async())
    print("--- 模擬交易結束 ---")
    final_value = context.portfolio.get_total_value()
    initial_capital = context.initial_capital
    total_return = (final_value / initial_capital - 1) * 100
    print(f"初始資金: {initial_capital:.2f} USDT")
    print(f"最終資產: {final_value:.2f} USDT")
    print(f"總報酬率: {total_return:.2%}")

if __name__ == '__main__':
    load_dotenv()
    parser = argparse.ArgumentParser(description='交易機器人主程式')
    parser.add_argument('--mode', type=str, choices=['live', 'paper'], required=True)
    parser.add_argument('--exchange', type=str, choices=['binance', 'coinbase', 'bybit'], default='bybit')
    parser.add_argument('--timeframe', type=str, default='5m')
    parser.add_argument('--testnet', action='store_true')
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--use-ppo', action='store_true')
    parser.add_argument('--ppo-model', type=str)
    args = parser.parse_args()

    context = Context()

    if args.mode == 'live':
        env_prefix = f"{args.exchange.upper()}"
        if args.testnet: env_prefix = f"{env_prefix}_TESTNET"
        api_key = os.getenv(f"{env_prefix}_API_KEY")
        api_secret = os.getenv(f"{env_prefix}_API_SECRET")
        if not api_key or not api_secret:
            raise ValueError(f"請在 .env 中設定 {env_prefix}_API_KEY 和 {env_prefix}_API_SECRET")
        if args.exchange == 'bybit':
            context.exchange = BybitExchange(api_key, api_secret, is_testnet=args.testnet)
        elif args.exchange == 'binance':
            context.exchange = BinanceExchange(api_key, api_secret)
        elif args.exchange == 'coinbase':
            context.exchange = CoinbaseExchange(api_key, api_secret)
    elif args.mode == 'paper':
        if not args.data_dir: raise ValueError("Paper 模式下必須提供 --data-dir")
        context.exchange = PaperExchange()

    context.portfolio = Portfolio(context.initial_capital, context.exchange)
    from settings import SYMBOLS_TO_TRADE
    strategy = XGBoostTrendStrategy(context, symbols=SYMBOLS_TO_TRADE, use_ppo=args.use_ppo, ppo_model_path=args.ppo_model)

    if args.mode == 'live':
        async def main_live():
            try:
                await context.exchange.connect()
                from settings import LEVERAGE
                for symbol in SYMBOLS_TO_TRADE:
                    await context.exchange.set_leverage(convert_symbol_to_ccxt(symbol), LEVERAGE)
                await warm_up(context, SYMBOLS_TO_TRADE, args.timeframe)
                await run_live(context, strategy, SYMBOLS_TO_TRADE, args.timeframe)
            except KeyboardInterrupt:
                print("\n--- 交易機器人已手動停止 ---")
            finally:
                print("--- 正在關閉交易所連線 ---")
                await context.exchange.close()
        asyncio.run(main_live())
    elif args.mode == 'paper':
        data = {}
        if os.path.isfile(args.data_dir) and args.data_dir.endswith('.csv'):
            symbol_part = os.path.basename(args.data_dir).split('_')[0]
            if symbol_part in SYMBOLS_TO_TRADE:
                data[symbol_part] = load_csv_data(args.data_dir, symbol=symbol_part)
        else:
            for symbol in SYMBOLS_TO_TRADE:
                filename_symbol = symbol.replace('/', '').replace(':', '')
                filepath = os.path.join(args.data_dir, f"{filename_symbol}.csv")
                if os.path.exists(filepath):
                    data[symbol] = load_csv_data(filepath, symbol=symbol)
                else:
                    data[symbol] = None
        valid_data = {s: d for s, d in data.items() if d is not None and not d.empty}
        if valid_data:
            run_paper(context, strategy, valid_data)
        else:
            print("錯誤：找不到任何有效的數據檔案來執行回測。")
