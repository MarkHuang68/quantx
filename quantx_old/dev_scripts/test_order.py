# test_order.py
import os
import asyncio
from pathlib import Path
import logging
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s')
log = logging.getLogger("OrderTest")

def _load_env():
    """從 .env 檔案載入環境變數。"""
    p = Path(".env")
    if not p.exists():
        log.warning(".env file not found.")
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        os.environ[k.strip()] = v.strip()
    log.info(".env file loaded successfully.")

async def main():
    """
    獨立的下單邏輯測試腳本。
    """
    log.info("--- Starting Order Submission Test ---")
    _load_env()

    try:
        from quantx.containers import AppContainer
        from quantx.core.context import LiveContext
        from quantx.core.executor.base import BaseExecutor
    except ImportError as e:
        log.error(f"Failed to import components. Error: {e}")
        return

    # --- Setup ---
    cfg = MagicMock()
    cfg.load_risk.return_value = {
        "risk_management": {
            "size_mode": "fixed_qty",
            "risk_pct": 0.001,
            "leverage": 1.0
        },
        "reporting": { "report_status_file": False }
    }
    container = AppContainer(cfg)
    runtime = container.runtime

    class DummyExecutor(BaseExecutor):
        def on_bar(self, ctx): pass

    executor = DummyExecutor()
    ctx = LiveContext(runtime, symbol="DUMMY", tf="1m", executor=executor)

    runtime.loader.provider.test_run = False
    log.info("Paper trading mode disabled. REAL orders will be sent to the exchange.")

    # --- Test Cases ---
    try:
        symbol_to_test = "BTCUSDT"
        log.info(f"Fetching current price for {symbol_to_test}...")
        ticker = runtime.loader.provider.exchange.fetch_ticker(symbol_to_test.replace("USDT", "/USDT"))
        current_price = ticker['last']
        log.info(f"Current price for {symbol_to_test} is {current_price}")

        provider = runtime.loader.provider
        ccxt_symbol = provider.exchange.symbol(symbol_to_test)
        trade_manager = ctx.trade_manager
        ts_now = datetime.now(timezone.utc)

        async def check_state(step_name: str):
            log.info(f"--- Checking state after: {step_name} ---")
            await asyncio.sleep(3)
            try:
                positions = provider.get_positions()
                open_orders = provider.exchange.fetch_open_orders(ccxt_symbol)
                log.info(f"  - Positions: {positions if positions else 'None'}")
                log.info(f"  - Open Orders: {open_orders if open_orders else 'None'}")
            except Exception as e:
                log.error(f"  - Failed to check state: {e}")
            log.info("------------------------------------")

        # --- Initial State ---
        log.info("\n--- [Initial State] Cancelling all orders and closing all positions ---")
        try:
            provider.exchange.cancel_all_orders(ccxt_symbol)
            log.info("  - All open orders cancelled.")
            await asyncio.sleep(2)
            intent_cleanup = [{"action": "close", "symbol": symbol_to_test, "qty": None, "order_type": "market", "price": None, "params": {}}]
            trade_manager.execute_commands(intent_cleanup, current_price, ts_now)
            await asyncio.sleep(2)
            log.info("  - Close command sent.")
            await check_state("Initial Cleanup")
        except Exception as e:
            log.error(f"  - Initial cleanup failed: {e}")


        # --- Test Case 1: Market Buy (Taker) ---
        log.info("\n--- [Test Case 1] Submitting a small Market Buy (Taker) order ---")
        # 獲取當前訂單簿的最佳賣價，以模擬更穩健的市價單
        orderbook = provider.exchange.fetch_order_book(ccxt_symbol)
        best_ask_price = orderbook['asks'][0][0] if orderbook['asks'] else current_price * 1.001
        log.info(f"Using Best Ask Price {best_ask_price} for market buy.")

        intent1 = [{"action": "open_long", "symbol": symbol_to_test, "qty": 0.001, "order_type": "limit", "price": best_ask_price, "params": {}}]
        trade_manager.execute_commands(intent1, current_price, ts_now)
        await check_state("Market Buy")

        # --- Test Case 2: Limit Sell (Maker, Post-Only) ---
        limit_price = round(current_price * 1.05, 1)
        log.info(f"\n--- [Test Case 2] Submitting a Limit Sell (Maker) order at {limit_price} (Post-Only) ---")
        intent2 = [{"action": "open_short", "symbol": symbol_to_test, "qty": 0.001, "order_type": "limit", "price": limit_price, "params": {"postOnly": True}}]
        trade_manager.execute_commands(intent2, current_price, ts_now)
        await check_state("Limit Sell (Short)")

        # --- Final Cleanup ---
        log.info("\n--- [Final Cleanup] Closing all positions and cancelling orders ---")
        try:
            provider.exchange.cancel_all_orders(ccxt_symbol)
            log.info("  - All open orders cancelled.")
            await asyncio.sleep(2)
            intent_cleanup = [{"action": "close", "symbol": symbol_to_test, "qty": None, "order_type": "market", "price": None, "params": {}}]
            trade_manager.execute_commands(intent_cleanup, current_price, ts_now)
            await asyncio.sleep(2)
            log.info("  - Close command sent.")
            await check_state("Final Cleanup")
        except Exception as e:
            log.error(f"  - Final cleanup failed: {e}")

        log.info("\n✅✅✅ Order Test Script Finished Successfully ✅✅✅")

    except Exception as e:
        log.error("An error occurred during the order submission test.", exc_info=True)
        log.error("❌❌❌ TEST FAILED ❌❌❌")
    finally:
        if hasattr(runtime.loader.provider, 'close_ws'):
            await runtime.loader.provider.close_ws()
            log.info("WebSocket connection closed.")

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Test interrupted by user.")