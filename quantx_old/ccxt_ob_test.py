# ccxt_ob_test.py
# -*- coding: utf-8 -*-
"""
獨立 WebSocket 訂單簿數據流測試腳本

用途：診斷 Level 2 訂單簿 (Order Book) 的原始數據結構，確認其 bids 和 asks 格式。
"""
import asyncio
import ccxt.pro
import logging
import os
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

EXCHANGE_ID = 'bybit'
MODE = os.environ.get('mode', 'live').lower() 
API_KEY = os.environ.get(f'{MODE}_api_key', '')
API_SECRET = os.environ.get(f'{MODE}_api_secret', '')

# 訂閱的交易對 (請根據您的需求修改)
TARGET_SYMBOL = 'BTC/USDT' 

def process_orderbook_update(symbol: str, orderbook: Dict[str, Any]):
    """
    處理收到的訂單簿快照，並打印關鍵結構。
    """
    # 獲取 Bids 和 Asks 列表
    bids: List[List[float]] = orderbook.get('bids', [])
    asks: List[List[float]] = orderbook.get('asks', [])

    # 檢查數據結構
    if not bids or not asks:
        logging.warning(f"⚠️ {symbol} 收到不完整/空的訂單簿數據。")
        logging.info(f"   - 原始數據 keys: {orderbook.keys()}")
        return

    # 獲取最佳買價 (Best Bid) 和最佳賣價 (Best Ask)
    # 這裡假設 bids/asks 至少有第一檔，且第一檔包含價格
    best_bid = bids[0][0] if bids[0] and len(bids[0]) > 0 else 'N/A'
    best_ask = asks[0][0] if asks[0] and len(asks[0]) > 0 else 'N/A'
    
    logging.info(f"🟢 {symbol} 收到訂單簿更新。時間: {orderbook.get('datetime', 'N/A')}")
    logging.info(f"   - 快照深度: Bids={len(bids)}, Asks={len(asks)}")
    logging.info(f"   - 最佳報價 (BBO): Bid={best_bid}, Ask={best_ask}")
    
    # 打印原始數據結構的前幾層，以供用戶確認
    logging.info("--- 原始數據結構範例 (前 3 檔) ---")
    logging.info(f"   Bids: {bids[:10]}  # 格式應為 [[價格, 數量], [價格, 數量], ...]")
    logging.info(f"   Asks: {asks[:10]}  # 格式應為 [[價格, 數量], [價格, 數量], ...]")
    logging.info("--------------------------------")


async def watch_orderbook_stream_test():
    """
    使用 ccxt.pro 訂閱 Level 2 訂單簿的獨立測試函式。
    """
    config = {
        'apiKey': API_KEY, 
        'secret': API_SECRET,
        'options': {'defaultType': 'swap', 'ws': {'pingInterval': 20000}},
        'timeout': 30000,
    }

    try:
        exchange = getattr(ccxt.pro, EXCHANGE_ID)(config)
    except AttributeError:
        logging.critical(f"❌ 無法載入 {EXCHANGE_ID}，請確認 ccxt.pro 是否已正確安裝。")
        return
    
    if MODE == 'testnet':
        exchange.set_sandbox_mode(True)

    # 外部迴圈：處理連線中斷和重連
    while True:
        try:
            logging.info("正在連接/重連 Order Book WebSocket...")
            
            # 內部迴圈：處理數據流的連續接收
            while True:
                # 訂閱單一 Symbol 的訂單簿
                # watch_order_book 會等待並返回下一個更新的 orderbook 快照
                orderbook = await exchange.watch_order_book(TARGET_SYMBOL)
                
                if orderbook:
                    process_orderbook_update(TARGET_SYMBOL, orderbook)
                
                # 為了避免在數據流極快時過度消耗 CPU，可以加入短暫的 sleep
                await asyncio.sleep(0.01)

        except asyncio.CancelledError:
            break
        except Exception as e:
            logging.error(f"Order Book WS 發生錯誤: {e}")
            logging.info("將在 5 秒後重試連線...")
            await asyncio.sleep(5)
        finally:
            # 🟢 修正：安全關閉連線，不再檢查 .opened 屬性
            if 'exchange' in locals():
                 logging.info("正在關閉 WebSocket 連線...")
                 await exchange.close()


def main():
    """主程式入口：檢查環境變數並啟動異步迴圈。"""
    if not API_KEY or not API_SECRET:
        logging.critical("🚨 警告：API Key 或 Secret 未設定。請在 .env 檔案中設定。")

    try:
        # 運行測試 15 秒
        loop = asyncio.get_event_loop()
        task = loop.create_task(watch_orderbook_stream_test())
        
        # 設置計時器，15 秒後取消任務
        loop.call_later(15, task.cancel) 
        loop.run_until_complete(task)
        
    except KeyboardInterrupt:
        pass
    except asyncio.CancelledError:
        logging.info("程式已按時序完成測試。")

if __name__ == "__main__":
    main()