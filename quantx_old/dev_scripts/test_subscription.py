# test_subscription.py
import os
import asyncio
from pathlib import Path
import logging
from unittest.mock import MagicMock, AsyncMock

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s')
log = logging.getLogger("SubscriptionTest")

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
    獨立的數據訂閱機制測試腳本。
    """
    log.info("--- Starting Data Subscription Test ---")
    _load_env()

    try:
        from quantx.containers import AppContainer
        from quantx.core.data.datafeed import DataFeed
    except ImportError as e:
        log.error(f"Failed to import components. Make sure quantx is in your PYTHONPATH. Error: {e}")
        return

    # --- Mocking a few components to isolate the test ---
    cfg = MagicMock()
    cfg.load_symbol.return_value = [("BTCUSDT", "1h"), ("ETHUSDT", "4h")]

    container = AppContainer(cfg)
    runtime = container.runtime
    datafeed = runtime.live.datafeed

    # --- Test Parameters ---
    targets = [
        ("BTC/USDT:USDT", "1h"),
        ("ETH/USDT:USDT", "4h"),
        ("SOL/USDT:USDT", "1h")
    ]
    main_symbols = [
        ("BTC/USDT:USDT", "1h"),
        ("ETH/USDT:USDT", "4h")
    ]

    log.info(f"Test targets: {len(targets)} symbols/tfs")
    log.info(f"Main triggers: {len(main_symbols)} symbols/tfs")

    try:
        log.info("Executing datafeed.subscribe_bulk()...")
        await datafeed.subscribe_bulk(targets, main_symbols)
        log.info("subscribe_bulk() finished.")

        # --- Verification Step ---
        log.info("--- Verifying DataFeed State ---")
        has_error = False

        expected_ohlcv_topics = {("BTC/USDT:USDT", "1h"), ("ETH/USDT:USDT", "4h"), ("SOL/USDT:USDT", "1h")}
        if datafeed.subscribed_ohlcv_topics == expected_ohlcv_topics:
            log.info(f"✅ OK: subscribed_ohlcv_topics match expected. ({len(datafeed.subscribed_ohlcv_topics)})")
        else:
            log.error(f"❌ FAIL: subscribed_ohlcv_topics mismatch!")
            has_error = True

        expected_ob_symbols = {"BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"}
        actual_ob_symbols = datafeed.subscribed_orderbook_symbols
        if actual_ob_symbols == {s.split(':')[0] for s in expected_ob_symbols}:
             log.info(f"✅ OK: subscribed_orderbook_symbols match expected. ({len(actual_ob_symbols)})")
        else:
            log.error(f"❌ FAIL: subscribed_orderbook_symbols mismatch!")
            has_error = True

        expected_buffer_keys = {"BTC/USDT:USDT-1h", "ETH/USDT:USDT-4h", "SOL/USDT:USDT-1h"}
        actual_buffer_keys = set(datafeed.ohlcv_buffers.keys())
        if actual_buffer_keys == expected_buffer_keys:
            log.info(f"✅ OK: ohlcv_buffers created for all targets. ({len(actual_buffer_keys)})")
            btc_buffer_len = len(datafeed.ohlcv_buffers.get("BTC/USDT:USDT-1h", []))
            if btc_buffer_len > 0:
                log.info(f"✅ OK: BTC/USDT-1h buffer contains {btc_buffer_len} backfilled records.")
            else:
                log.error("❌ FAIL: BTC/USDT-1h buffer is empty after backfill.")
                has_error = True
        else:
            log.error(f"❌ FAIL: ohlcv_buffers mismatch!")
            has_error = True

        expected_main_symbols = {"BTC/USDT:USDT-1h", "ETH/USDT:USDT-4h"}
        if datafeed.main_symbols == expected_main_symbols:
            log.info(f"✅ OK: main_symbols correctly set. ({len(datafeed.main_symbols)})")
        else:
            log.error(f"❌ FAIL: main_symbols mismatch!")
            has_error = True

        log.info("--- Verification Finished ---")
        if has_error:
            log.error("❌ TEST FAILED with errors.")
        else:
            log.info("✅✅✅ TEST PASSED SUCCESSFULLY!")

    except Exception as e:
        log.error("An unexpected error occurred during the test.", exc_info=True)
        log.error("❌ TEST FAILED with an exception.")

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Test interrupted by user.")