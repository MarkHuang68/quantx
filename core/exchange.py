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
            'options': {'defaultType': 'swap', 'ws': {'pingInterval': 20000}},
            'timeout': 30000,
        }
        self.exchange = getattr(ccxt.pro, 'bybit')(config)
        if is_testnet:
            self.exchange.set_sandbox_mode(True)
            print("--- Bybit 已設定為 Testnet (沙盒) 模式 ---")
        else:
            print("--- Bybit 已設定為 Live (實盤) 模式 ---")

    async def connect(self):
        await self.set_hedge_mode()

    async def set_hedge_mode(self):
        try:
            await self.exchange.set_position_mode(hedged=True)
            print("--- Bybit 已成功設定為雙向持倉模式 ---")
        except ccxt.ExchangeError as e:
            if 'position mode not modified' in str(e):
                print("--- 倉位模式無需修改，已是雙向持倉 ---")
            else:
                print(f"警告：設定雙向持倉模式失敗: {e}")

    async def set_leverage(self, symbol, leverage):
        try:
            await self.exchange.set_leverage(leverage, symbol)
            print(f"--- {symbol} 的槓桿已成功設定為 {leverage}x ---")
        except ccxt.ExchangeError as e:
            if 'leverage not modified' in str(e):
                print(f"--- {symbol} 的槓桿無需修改，已是 {leverage}x ---")
            else:
                print(f"警告：為 {symbol} 設定槓桿失敗: {e}")

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
        if self.exchange.clients:
            await self.exchange.close()
            print("--- Bybit WebSocket 連線已關閉 ---")

class BinanceExchange(Exchange):
    def __init__(self, api_key, api_secret):
        # 注意：ccxtpro 不一定支援所有交易所的 async 方法
        self.exchange = ccxt.pro.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
        })

    async def get_ohlcv(self, symbol, timeframe='1m', limit=100):
        ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df

    async def create_order(self, symbol, type, side, amount, price=None, params={}):
        return await self.exchange.create_order(symbol, type, side, amount, price, params)

    async def get_balance(self):
        return await self.exchange.fetch_balance()

    async def get_positions(self):
        if self.exchange.has['fetchPositions']:
            return await self.exchange.fetch_positions()
        else:
            balance = await self.get_balance()
            positions = {
                asset['asset']: asset['free']
                for asset in balance['info']['balances']
                if float(asset['free']) > 0
            }
            return positions

    async def sync_positions(self, portfolio):
        print("--- 正在同步幣安倉位 ---")
        raw_positions = await self.get_positions()
        portfolio.sync_spot_positions(raw_positions)
        print(f"倉位同步完成")

    async def get_latest_price(self, symbol):
        ticker = await self.exchange.fetch_ticker(symbol)
        return ticker['last']

class CoinbaseExchange(Exchange):
    def __init__(self, api_key, api_secret):
        self.exchange = ccxt.pro.coinbase({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
        })

    async def get_ohlcv(self, symbol, timeframe='1m', limit=100):
        ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df

    async def create_order(self, symbol, type, side, amount, price=None, params={}):
        return await self.exchange.create_order(symbol, type, side, amount, price, params)

    async def get_balance(self):
        return await self.exchange.fetch_balance()

    async def get_positions(self):
        balance = await self.get_balance()
        positions = {
            item['currency']: float(item['balance'])
            for item in balance.values()
            if isinstance(item, dict) and 'currency' in item and 'balance' in item and float(item['balance']) > 0
        }
        return positions

    async def sync_positions(self, portfolio):
        print("--- 正在同步 Coinbase 倉位 ---")
        raw_positions = await self.get_positions()
        portfolio.sync_spot_positions(raw_positions)
        print(f"倉位同步完成")

    async def get_latest_price(self, symbol):
        ticker = await self.exchange.fetch_ticker(symbol)
        return ticker['last']

class PaperExchange(Exchange):
    def __init__(self, initial_balance=100000):
        self._balance = {'USDT': {'free': initial_balance, 'total': initial_balance}}
        self._positions = {}
        self._current_dt = None
        self._kline_data = {}

    def set_kline_data(self, symbol, df):
        self._kline_data[symbol] = df

    def set_current_dt(self, dt):
        self._current_dt = dt

    async def get_ohlcv(self, symbol, timeframe='1m', limit=100):
        if symbol not in self._kline_data:
            raise ValueError(f"沒有 {symbol} 的 K 線資料")
        df = self._kline_data[symbol]
        try:
            end_idx = df.index.get_loc(self._current_dt)
        except KeyError:
            end_idx = df.index.searchsorted(self._current_dt, side='right') - 1
        if end_idx < 0:
            return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
        start_idx = max(0, end_idx - limit + 1)
        return df.iloc[start_idx:end_idx + 1]

    async def create_order(self, symbol, type, side, amount, price=None, params={}):
        if price is None:
            price = await self.get_latest_price(symbol)
        base_currency, quote_currency = symbol.split('/')
        trade_value = amount * price
        fee = trade_value * settings.FEE_RATE
        if side == 'buy':
            cost = trade_value
            if self._balance[quote_currency]['free'] < cost + fee:
                raise ValueError(f"資金不足")
            self._balance[quote_currency]['free'] -= (cost + fee)
            self._positions.setdefault(base_currency, 0)
            self._positions[base_currency] += amount
        elif side == 'sell':
            self._positions.setdefault(base_currency, 0)
            self._positions[base_currency] -= amount
            revenue = trade_value
            self._balance[quote_currency]['free'] += (revenue - fee)
        return {'info': {}, 'id': str(pd.Timestamp.now().timestamp()), 'timestamp': self._current_dt, 'status': 'closed', 'symbol': symbol, 'type': type, 'side': side, 'amount': amount, 'filled': amount, 'price': price}

    async def get_balance(self):
        return self._balance

    async def get_positions(self):
        return self._positions

    async def get_latest_price(self, symbol):
        if symbol in self._kline_data and self._current_dt is not None:
            try:
                if self._current_dt in self._kline_data[symbol].index:
                    return self._kline_data[symbol].loc[self._current_dt]['Close']
                else:
                    price_series = self._kline_data[symbol]['Close']
                    latest_price = price_series.asof(self._current_dt)
                    if pd.notna(latest_price):
                        return latest_price
            except Exception: pass
        if symbol in self._kline_data and not self._kline_data[symbol].empty:
            return self._kline_data[symbol]['Close'].iloc[-1]
        return None
