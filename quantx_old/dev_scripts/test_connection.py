# test_connection.py
import os
import asyncio
from pathlib import Path
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s')
log = logging.getLogger("ConnectionTest")

def _load_env():
    """從 .env 檔案載入環境變數。"""
    p = Path(".env")
    if not p.exists():
        log.warning(".env file not found. Make sure it exists in the root directory.")
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
    獨立的 API 連線測試腳本。
    """
    log.info("--- Starting API Connection Test ---")
    _load_env()

    try:
        from quantx.market.provider.bybit import BybitProvider
    except ImportError as e:
        log.error(f"Failed to import BybitProvider. Make sure quantx is in your PYTHONPATH. Error: {e}")
        return

    provider = None
    try:
        log.info("Initializing BybitProvider...")
        mode = os.environ.get("mode", "testnet").lower()
        provider = BybitProvider(mode=mode)
        log.info(f"BybitProvider initialized in '{provider.mode}' mode.")

        log.info("Attempting to fetch balance for USDT...")
        balance = provider.fetch_balance(currency='USDT')

        log.info("==========================================")
        log.info(f"✅ Connection Successful!")
        log.info(f"💰 Account Balance: {balance:.4f} USDT")
        log.info("==========================================")

    except Exception as e:
        log.error("==========================================")
        log.error(f"❌ Connection Failed!")
        log.error(f"An error occurred: {e}", exc_info=True)
        log.error("==========================================")
        log.error("Troubleshooting tips:")
        log.error("1. Verify that your API_KEY and API_SECRET in the .env file are correct and not expired.")
        log.error("2. Check if Bybit's Testnet services are operational.")
        log.error("3. Ensure you have the necessary permissions (e.g., Unified Trading Account) on your Bybit account.")

    finally:
        if provider and hasattr(provider, 'close_ws'):
            await provider.close_ws()
            log.info("WebSocket connection closed.")
        log.info("--- API Connection Test Finished ---")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Test interrupted by user.")