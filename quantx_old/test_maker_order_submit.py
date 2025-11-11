# test_maker_order_submit.py
# -*- coding: utf-8 -*-
"""
Maker 訂單提交測試腳本 (LIMIT Order)

用途：
1. 確認 ccxt.pro 是否能成功抓取 Level 2 訂單簿數據。
2. 驗證 calculate_optimal_maker_price 函式在動態精度下的正確性。
3. 測試 BybitProvider 提交一筆符合 Maker 策略的訂單。
"""
import asyncio
import ccxt.pro
import os
import logging
from typing import Dict, Any, List, Optional, Tuple
import math
import numpy as np
import os
from dotenv import load_dotenv
import ccxt
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("MAKER_TEST")

# --- 模擬環境配置 (請確保 .env 檔案中的 testnet 參數已設定) ---
EXCHANGE_ID = 'bybit'
MODE = 'testnet' 
API_KEY = os.environ.get(f'{MODE}_api_key', '8BxkMF4Cms1uJryKgh')
API_SECRET = os.environ.get(f'{MODE}_api_secret', '6Pa1VfDluGx9zqXXHx6XiRijOgeaKHUz47Uu')

# 測試參數
TEST_SYMBOL_CCXT = 'BTC/USDT'     # CCXT 標準格式 (用於 WS)
TEST_SYMBOL_INTERNAL = 'BTCUSDT' # 內部應用程式格式 (用於 Provider 提交)
TEST_QTY = 0.007064                 # 測試數量 (確保大於 0.001 的最小限制)

# --- 模擬核心函式 (基於 quantx/core/orderbook_utils.py) ---

def _get_instrument_precision(symbol: str) -> float:
    """模擬 LiveTradeManager 獲取價格精度。"""
    base_symbol = symbol.replace('/USDT', '').upper()
    if 'BTC' in base_symbol:
        return 0.1  # 實際 Tick Size
    elif 'ETH' in base_symbol:
        return 0.01
    else:
        return 0.0001 # 山寨幣安全回退

def calculate_optimal_maker_price(
    orderbook_snapshot: Optional[Dict[str, List[List[float]]]],
    side: str,
    instrument_precision: float
) -> Optional[float]:
    """計算最佳的 Maker 掛單價格（BBO 之外一檔）。"""
    if not orderbook_snapshot or not orderbook_snapshot.get('bids') or not orderbook_snapshot.get('asks'):
        return None
    
    bids, asks = orderbook_snapshot['bids'], orderbook_snapshot['asks']
    
    best_bid = bids[0][0] if bids[0] and len(bids[0]) > 0 else 0.0
    best_ask = asks[0][0] if asks[0] and len(asks[0]) > 0 else 0.0

    if best_bid <= 0.0 or best_ask <= 0.0 or best_ask <= best_bid:
        return None

    precision = max(instrument_precision, 1e-8)
    
    if side == 'buy':
        # Maker Buy: 掛在 Best Bid 下方一檔 (Buy Low)
        optimal_price = best_bid - precision
    elif side == 'sell':
        # Maker Sell: 掛在 Best Ask 上方一檔 (Sell High)
        optimal_price = best_ask + precision
    else:
        return None

    if optimal_price <= 0:
        return None

    # 修正：確保精度捨入正確
    decimal_places = max(0, -int(np.floor(np.log10(precision)))) if precision < 1.0 else 0
    return round(optimal_price, decimal_places)


async def submit_test_order(exchange: ccxt.pro.bybit, provider: Any):
    """
    執行提交 Maker 訂單的流程。
    """
    precision = _get_instrument_precision(TEST_SYMBOL_INTERNAL)
    logger.info(f"--- 1. 測試參數 ---")
    logger.info(f"目標: {TEST_SYMBOL_CCXT}, 精度 (Tick Size): {precision}")
    logger.info(f"測試數量: {TEST_QTY}")
    
    # 步驟 1: 獲取最新的訂單簿快照
    logger.info(f"--- 2. 獲取訂單簿並計算 Maker 價格 ---")
    try:
        # 使用 REST API 的同步方法獲取 Order Book (為了簡化測試，只抓一次)
        orderbook = exchange.fetch_order_book(TEST_SYMBOL_INTERNAL, limit=1)
        best_bid = orderbook['bids'][0][0]
        best_ask = orderbook['asks'][0][0]
    except Exception as e:
        logger.error(f"❌ 無法獲取 Order Book，跳過測試。錯誤: {e}")
        return

    # 測試 Maker Buy 意圖
    maker_price = calculate_optimal_maker_price(orderbook, 'buy', precision)

    if maker_price is None:
        logger.error(f"❌ Maker 價格計算失敗。BBO: {best_bid}/{best_ask}")
        return

    logger.info(f"最佳報價 (BBO): {best_bid} / {best_ask}")
    logger.info(f"Maker Buy Limit Price: {maker_price} (應為 {best_bid} - {precision})")

    # 步驟 3: 提交 Maker 訂單 (LIMIT Order)
    try:
        order_type = "limit"
        logger.info(f"--- 3. 提交 {order_type.upper()} Maker 訂單... ---")
        
        # 這裡呼叫您的 BybitProvider.submit_order
        market = provider.get_market_params("BTCUSDT")
        print(market)
        result = provider.submit_order(
            symbol=TEST_SYMBOL_INTERNAL, 
            side='buy', 
            qty=TEST_QTY, 
            order_type=order_type,
            price=maker_price,
        )

        # 步驟 4: 檢查結果
        if result.get('dry_run', False):
            logger.info(f"✅ 測試成功：訂單被 LiveTradeManager 攔截 (Paper Trading)。")
            logger.info(f"   類型: {result['type']}, 價格: {result['price']}, 數量: {result['qty']}")
            logger.info(f"   請手動確認價格 {maker_price} 是否低於當前最佳買價 {best_bid}。")
        else:
            logger.info(f"✅ 提交成功！請檢查交易所。訂單 ID: {result.get('resp', {}).get('id', 'N/A')}")

    except Exception as e:
        logger.error(f"❌ 訂單提交失敗，請檢查 API Key 權限或最小交易量限制。")
        print(f"\n--- 原始交易所錯誤 ---")
        print(e)
        print(f"----------------------\n")


def main():
    """主程式入口，負責初始化 Provider 並執行異步測試。"""
    if not API_KEY or not API_SECRET:
        logger.critical("🚨 警告：API Key 或 Secret 未設定。請在 .env 檔案中設定。")
        return

    try:
        # 模擬 LiveRuntime 初始化 BybitProvider 的過程
        from quantx.market.provider.bybit import BybitProvider
        provider = BybitProvider(mode=MODE, test_run=False) # 使用紙上交易模式
        
        # 獲取同步交易所實例 (用於 fetch_order_book)
        exchange = provider.exchange 
    except Exception as e:
        logger.critical(f"無法初始化 BybitProvider: {e}")
        return

    try:
        # 由於 submit_test_order 包含同步的 fetch_ticker，需要一個異步環境
        asyncio.run(submit_test_order(exchange, provider)) 
    except KeyboardInterrupt:
        logger.info("程式被用戶中斷。")

if __name__ == "__main__":
    main()