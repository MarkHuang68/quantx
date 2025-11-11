# 檔案: core/exchange.py

import ccxt
import ccxt.pro
import pandas as pd
from abc import ABC, abstractmethod
import settings

class Exchange(ABC):
    @abstractmethod
    async def get_ohlcv(self, symbol, timeframe, limit):
        pass

    @abstractmethod
    async def create_order(self, symbol, type, side, amount, price=None, params={}):
        pass

    @abstractmethod
    async def get_balance(self):
        pass

    @abstractmethod
    async def get_positions(self):
        pass

    @abstractmethod
    async def get_latest_price(self, symbol):
        pass

    async def close(self):
        pass


class BybitExchange(Exchange):
    def __init__(self, api_key, api_secret, is_testnet=False):
        config = {
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'swap',
                'ws': {'pingInterval': 20000},
            },
        }
        self.exchange = getattr(ccxt.pro, 'bybit')(config)

        if is_testnet:
            self.exchange.set_sandbox_mode(True)
            print("--- Bybit 已設定為 Testnet (沙盒) 模式 ---")
        else:
            print("--- Bybit 已設定為 Live (實盤) 模式 ---")

    async def connect(self):
        """執行非同步的初始化操作。"""
        await self.set_hedge_mode()

    async def set_hedge_mode(self):
        try:
            await self.exchange.set_position_mode(hedged=True)
            print("--- Bybit 已成功設定為雙向持倉模式 ---")
        except ccxt.ExchangeError as e:
            if 'position mode not modified' in str(e):
                print("--- 倉位模式無需修改，已是雙向持倉 ---")
            else:
                print(f"警告：設定雙向持倉模式失敗: {e}。請手動到 Bybit 網站確認設定。")

    async def get_ohlcv(self, symbol, timeframe='1m', limit=100):
        ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df

    async def create_order(self, symbol, type, side, amount, price=None, params={}):
        return await self.exchange.create_order(symbol, type, side, amount, price, params)

    async def get_balance(self):
        balance = await self.exchange.fetch_balance(params={'type': 'swap'})
        return balance.get('USDT', {})

    async def get_positions(self):
        positions = await self.exchange.fetch_positions(params={'type': 'swap'})
        active_positions = [p for p in positions if p.get('contracts') and float(p['contracts']) != 0]
        return active_positions

    async def sync_positions(self, portfolio):
        print("--- 正在同步 Bybit 倉位 ---")
        positions = await self.get_positions()
        portfolio.sync_with_exchange(positions)
        print(f"倉位同步完成")

    async def get_latest_price(self, symbol):
        ticker = await self.exchange.fetch_ticker(symbol)
        return ticker['last']

    async def close(self):
        if self.exchange.clients: # ccxt.pro uses a dictionary of clients
            await self.exchange.close()
            print("--- Bybit WebSocket 連線已關閉 ---")


# 注意：BinanceExchange 和 CoinbaseExchange 尚未更新為 async，
# 在當前的 WebSocket 架構下將無法使用。
# 為了完成任務，我們專注於修復 BybitExchange。

class BinanceExchange(Exchange):
    # ... (原有程式碼) ...
    pass

class CoinbaseExchange(Exchange):
    # ... (原有程式碼) ...
    pass

class PaperExchange(Exchange):
    # ... (原有程式碼) ...
    pass
