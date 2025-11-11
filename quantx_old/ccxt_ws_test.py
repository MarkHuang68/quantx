# ccxt_ws_test.py
# -*- coding: utf-8 -*-
"""
獨立 WebSocket K 線數據流測試腳本

用途：診斷量化平台中 ccxt.pro 訂閱 K 線數據流是否能正常運行。
      請確保您已設定 .env 檔案中的 API Key 和 Secret。
"""
import asyncio
import ccxt.pro
import os
import time
import logging
from datetime import datetime, timezone
from typing import List, Dict

# --- 1. 環境變數和日誌設定 ---
# 警告：此處使用 os.environ.get 讀取 API Key，請確保您已在執行環境（例如 .env 檔案）中正確配置。

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

EXCHANGE_ID = 'bybit'
# 預設使用 testnet 模式，可根據 .env 覆蓋
MODE = os.environ.get('mode', 'testnet').lower() 
API_KEY = os.environ.get(f'{MODE}_api_key', '')
API_SECRET = os.environ.get(f'{MODE}_api_secret', '')

# 訂閱的交易對與時間框 (可根據 conf/symbol.yaml 調整)
# 使用 BTCUSDT-1m 進行高頻率測試
TARGET_SYMBOLS_TFS = [['BTC/USDT', '1h']] 

async def watch_ohlcv_stream_test():
    """
    使用 ccxt.pro 訂閱 OHLCV 數據流的獨立測試函式。
    - 包含連線、重連和錯誤處理邏輯。
    """
    
    config = {
        'apiKey': API_KEY, 
        'secret': API_SECRET,
        'options': {'defaultType': 'swap', 'ws': {'pingInterval': 20000}},
        'timeout': 30000,
    }

    # 實例化 ccxt.pro 交易所
    try:
        exchange = getattr(ccxt.pro, EXCHANGE_ID)(config)
    except AttributeError:
        logging.critical(f"❌ 無法載入 {EXCHANGE_ID}，請確認 ccxt.pro 是否已正確安裝並為最新版本。")
        return
    
    if MODE == 'testnet':
        exchange.set_sandbox_mode(True)
        logging.info(f"設定為 {EXCHANGE_ID} Testnet (沙盒) 模式")
    else:
        logging.info(f"設定為 {EXCHANGE_ID} Live (實盤) 模式")

    logging.info(f"開始連接 WebSocket，訂閱主題: {TARGET_SYMBOLS_TFS}")
    
    # 外部迴圈：處理連線中斷和重連
    while True:
        try:
            logging.info("正在連接/重連 WebSocket...")
            
            # 內部迴圈：處理數據流的連續接收
            while True:
                # watch_ohlcv_for_symbols 會等待下一批 K 線更新
                ohlcv_stream = await exchange.watch_ohlcv_for_symbols(TARGET_SYMBOLS_TFS)
                
                if not ohlcv_stream:
                    logging.debug("收到空數據流，等待下一批更新...")
                    await asyncio.sleep(1)
                    continue

                for symbol, tf_data in ohlcv_stream.items():
                    for timeframe, ohlcv_list in tf_data.items():
                        if ohlcv_list:
                            # ohlcv_list 是一個包含多個 K 棒列表的列表
                            for bar in ohlcv_list:
                                timestamp_ms = bar[0]
                                # 將時間戳轉換為 UTC 時間
                                dt_utc = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
                                
                                logging.info(
                                    f"🟢 收到新 K 棒: {symbol}/{timeframe} @ {dt_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC, "
                                    f"Close={bar[4]}, Volume={bar[5]}"
                                )
                                # 這裡就是您程式碼中 on_bar 回呼應該被觸發的地方

        except asyncio.CancelledError:
            logging.info("WS 處理器被外部取消，安全退出。")
            break
        except ccxt.NetworkError as e:
            logging.error(f"網絡錯誤: {e}。將在 10 秒後重試...")
            await asyncio.sleep(10)
        except ccxt.ExchangeError as e:
            error_msg = str(e)
            if 'AuthenticationError' in error_msg or 'API Key' in error_msg:
                 logging.critical(f"❌ 交易所錯誤: API Key 或 Secret 無效。請檢查配置。錯誤: {error_msg}")
                 break
            logging.error(f"交易所錯誤: {error_msg}。將在 15 秒後重試...")
            await asyncio.sleep(15)
        except Exception as e:
            logging.error(f"發生未知錯誤: {e}")
            logging.info("將在 15 秒後重試...")
            await asyncio.sleep(15)
        finally:
            if 'exchange' in locals() and exchange.opened:
                 logging.info("關閉 WebSocket 連線...")
                 await exchange.close()


def main():
    """主程式入口：檢查環境變數並啟動異步迴圈。"""
    # 警告：檢查 API Key 是否設置
    if not API_KEY or not API_SECRET:
        logging.critical("=========================================================")
        logging.critical("🚨 警告：API Key 或 Secret 未設定。請在 .env 檔案中設定。")
        logging.critical(f"請檢查 {MODE}_api_key 和 {MODE}_api_secret。")
        logging.critical("=========================================================")

    try:
        asyncio.run(watch_ohlcv_stream_test())
    except KeyboardInterrupt:
        logging.info("程式被用戶中斷。")

if __name__ == "__main__":
    main()