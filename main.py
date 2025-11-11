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

async def warm_up(context, symbols, timeframe):
    """
    在啟動時預載數據並計算初始特徵，但不執行交易。
    """
    print("--- 數據預熱階段開始 ---")
    initial_features = {}
    for symbol in symbols:
        print(f"正在為 {symbol} 預載歷史數據...")
        try:
            ohlcv = await context.exchange.get_ohlcv(symbol=symbol, timeframe=timeframe, limit=200)
            if ohlcv is None or ohlcv.empty or len(ohlcv) < 200:
                print(f"警告：為 {symbol} 預載的數據不足 (共 {len(ohlcv) if ohlcv is not None else 0} 根)，啟動時的指標可能不準確。")
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
    """
    執行實盤交易 (WebSocket 版本)。
    """
    print("--- 啟動實盤交易模式 (WebSocket) ---")
    print(f"交易對: {symbols}, K線週期: {timeframe}")

    subscription_topics = [[symbol, timeframe] for symbol in symbols]

    # 在開始交易循環前，先為所有交易對設定槓桿
    from settings import LEVERAGE
    for symbol in symbols:
        await context.exchange.set_leverage(symbol, LEVERAGE)

    while True:
        try:
            print("正在連接/重連 WebSocket...")
            # await context.exchange.connect() # connect 裡面只有 set_hedge_mode，現在可以移到啟動時

            while True:
                ohlcv_stream = await context.exchange.exchange.watch_ohlcv_for_symbols(subscription_topics)

                current_dt = pd.Timestamp.now(tz='UTC')
                print(f"\n--- [{current_dt.strftime('%Y-%m-%d %H:%M:%S')}] 收到數據 ---")

                print("正在同步倉位...")
                await context.exchange.sync_positions(context.portfolio)

                current_features = {}
                for symbol in symbols:
                    print(f"正在為 {symbol} 準備數據...")
                    ohlcv = await context.exchange.get_ohlcv(symbol=symbol, timeframe=timeframe, limit=200)
                    if ohlcv is None or ohlcv.empty:
                        print(f"警告：無法獲取 {symbol} 的歷史數據，跳過此輪。")
                        continue

                    print(f"正在計算 {symbol} 的特徵...")
                    df_with_features, _ = create_features_trend(ohlcv)

                    if df_with_features is not None and not df_with_features.empty:
                        latest_features = df_with_features.iloc[-1]
                        current_features[symbol] = latest_features
                    else:
                        print(f"警告：無法為 {symbol} 計算特徵。")

                if current_features:
                    print("觸發策略決策...")
                    await strategy.on_bar(current_dt, current_features)
                else:
                    print("沒有足夠的數據來觸發策略決責。")

                print("正在更新投資組合...")
                await context.portfolio.update(current_dt)
                print(f"目前總資產: {context.portfolio.get_total_value():.2f} USDT")
                print(context.portfolio.get_positions_summary())

        except asyncio.CancelledError:
            print("\n--- WebSocket 處理器被取消，安全退出 ---")
            break
        except ccxt.NetworkError as e:
            print(f"網絡錯誤: {e}。將在 10 秒後重試...")
            await asyncio.sleep(10)
        except ccxt.ExchangeError as e:
            print(f"交易所錯誤: {e}。將在 15 秒後重試...")
            await asyncio.sleep(15)
        except Exception as e:
            print(f"發生未知嚴重錯誤: {e}")
            print("將在 15 秒後重試...")
            await asyncio.sleep(15)
        finally:
            await context.exchange.close()


def run_paper(context, strategy, data):
    """
    執行模擬交易（修正版）。
    """
    print("--- 啟動模擬交易模式 ---")

    features_data = {}
    for symbol, df in data.items():
        print(f"正在為 {symbol} 預計算特徵...")
        df_with_features, _ = create_features_trend(df)
        if df_with_features is not None:
            features_data[symbol] = df_with_features
        context.exchange.set_kline_data(symbol, df)

    # 確保 main_symbol 是一個有效的、已成功載入數據的交易對
    try:
        main_symbol = list(data.keys())[0]
    except IndexError:
        print("錯誤：數據字典為空，無法開始回測。")
        return

    print(f"--- 開始模擬回放 (主時間軸: {main_symbol}) ---")

    # 注意：Paper trade 的 on_bar 和 update 是 async 的
    async def run_paper_async():
        for dt in data[main_symbol].index:
            context.exchange.set_current_dt(dt)
            current_features = {}
            for symbol, df_features in features_data.items():
                if dt in df_features.index:
                    current_features[symbol] = df_features.loc[dt]
            if current_features:
                await strategy.on_bar(dt, current_features) # on_bar 現在是 async
            await context.portfolio.update(dt) # update 現在是 async

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
    parser.add_argument('--mode', type=str, choices=['live', 'paper'], required=True, help='執行模式')
    parser.add_argument('--exchange', type=str, choices=['binance', 'coinbase', 'bybit'], default='bybit', help='交易所')
    parser.add_argument('--timeframe', type=str, default='5m', help='K 線週期')
    parser.add_argument('--testnet', action='store_true', help='使用測試網')
    parser.add_argument('--data-dir', type=str, help='Paper模式的數據目錄')
    parser.add_argument('--use-ppo', action='store_true', help='使用PPO')
    parser.add_argument('--ppo-model', type=str, help='PPO模型路徑')
    args = parser.parse_args()

    context = Context()

    if args.mode == 'live':
        env_prefix = f"{args.exchange.upper()}"
        if args.testnet:
            env_prefix = f"{env_prefix}_TESTNET"
        api_key = os.getenv(f"{env_prefix}_API_KEY")
        api_secret = os.getenv(f"{env_prefix}_API_SECRET")
        if not api_key or not api_secret:
            raise ValueError(f"請在 .env 中設定 {env_prefix}_API_KEY 和 {env_prefix}_API_SECRET")

        if args.exchange == 'bybit':
            context.exchange = BybitExchange(api_key, api_secret, is_testnet=args.testnet)
        else:
            # 提醒用戶 Binance 和 Coinbase 尚未 async 化
            print(f"警告：{args.exchange} 尚未完全支援目前的 async 架構。")
            # 這裡可以選擇拋出錯誤或使用舊的同步 Exchange
            if args.exchange == 'binance':
                context.exchange = BinanceExchange(api_key, api_secret)
            elif args.exchange == 'coinbase':
                context.exchange = CoinbaseExchange(api_key, api_secret)

    elif args.mode == 'paper':
        if not args.data_dir:
            raise ValueError("Paper 模式下必須提供 --data-dir")
        context.exchange = PaperExchange()

    context.portfolio = Portfolio(context.initial_capital, context.exchange)
    from settings import SYMBOLS_TO_TRADE
    strategy = XGBoostTrendStrategy(
        context,
        symbols=SYMBOLS_TO_TRADE,
        use_ppo=args.use_ppo,
        ppo_model_path=args.ppo_model
    )

    if args.mode == 'live':
        async def main_live():
            try:
                # 1. 執行非同步初始化：設定模式和槓桿
                await context.exchange.connect()
                from settings import LEVERAGE
                for symbol in SYMBOLS_TO_TRADE:
                    await context.exchange.set_leverage(symbol, LEVERAGE)

                # 2. 執行數據預熱
                await warm_up(context, SYMBOLS_TO_TRADE, args.timeframe)

                # 3. 開始主交易循環
                await run_live(context, strategy, SYMBOLS_TO_TRADE, args.timeframe)
            except KeyboardInterrupt:
                print("\n--- 交易機器人已手動停止 ---")
            finally:
                print("--- 正在關閉交易所連線 ---")
                await context.exchange.close()

        asyncio.run(main_live())

    elif args.mode == 'paper':
        data = {}
        # 判斷 --data-dir 是檔案還是目錄
        if os.path.isfile(args.data_dir) and args.data_dir.endswith('.csv'):
            print(f"--- 偵測到單一檔案模式 ---")
            # 從檔名推斷 symbol
            filename = os.path.basename(args.data_dir)
            # 假設檔名格式為 'ETHUSDT_1h_...' or 'ETH_USDT_USDT_1h_...'
            symbol_part = filename.split('_')[0] + '/' + filename.split('_')[1]
            if ':USDT' in SYMBOLS_TO_TRADE[0]: # 檢查是否為永續合約格式
                 symbol_part += ':USDT'

            # 確保推斷出的 symbol 在我們的交易列表中
            if symbol_part in SYMBOLS_TO_TRADE:
                print(f"從檔名推斷出交易對：{symbol_part}")
                data[symbol_part] = load_csv_data(args.data_dir, symbol=symbol_part)
            else:
                print(f"警告：從檔名 {filename} 推斷出的交易對 {symbol_part} 不在 settings.py 的交易列表中。")

        else: # 目錄模式
            print(f"--- 偵測到目錄模式 ---")
            for symbol in SYMBOLS_TO_TRADE:
                # 將 'BTC/USDT:USDT' 轉換為 'BTC_USDT_USDT'
                filename_symbol = symbol.replace('/', '_').replace(':', '_')
                filepath = os.path.join(args.data_dir, f"{filename_symbol}.csv")

                # 為了向後兼容，也檢查舊的格式
                if not os.path.exists(filepath):
                    old_filename_symbol = symbol.replace('/', '_').split(':')[0]
                    filepath = os.path.join(args.data_dir, f"{old_filename_symbol}.csv")

                if os.path.exists(filepath):
                    data[symbol] = load_csv_data(filepath, symbol=symbol)
                else:
                    data[symbol] = None

        # 過濾掉載入失敗的數據
        valid_data = {s: d for s, d in data.items() if d is not None and not d.empty}

        if valid_data:
            run_paper(context, strategy, valid_data)
        else:
            print("錯誤：找不到任何有效的數據檔案來執行回測。")
