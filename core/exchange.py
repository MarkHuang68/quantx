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
    def __init__(self, portfolio):
        self.portfolio = portfolio
        self._current_dt = None
        self._kline_data = {}

    def set_kline_data(self, symbol, df):
        self._kline_data[symbol] = df

    async def set_current_dt(self, dt):
        self._current_dt = dt
        await self._check_for_liquidations()

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

    async def _check_for_liquidations(self):
        """檢查並處理爆倉。"""
        positions_to_liquidate = []

        # 使用 .copy() 來避免在迭代過程中修改字典
        for symbol, sides in self.portfolio.get_positions().copy().items():
            latest_price = await self.get_latest_price(symbol)
            if latest_price is None:
                continue

            # 檢查多頭倉位
            long_pos = sides['long']
            if long_pos['contracts'] > 0 and long_pos['liquidation_price'] > 0:
                if latest_price <= long_pos['liquidation_price']:
                    positions_to_liquidate.append((symbol, 'long', long_pos, latest_price))

            # 檢查空頭倉位
            short_pos = sides['short']
            if short_pos['contracts'] > 0 and short_pos['liquidation_price'] > 0:
                if latest_price >= short_pos['liquidation_price']:
                    positions_to_liquidate.append((symbol, 'short', short_pos, latest_price))

        for symbol, side, position, liquidation_price in positions_to_liquidate:
            print(f"!!! 爆倉警告 !!!")
            print(f"時間: {self._current_dt}")
            print(f"標的: {symbol} [{side}]")
            print(f"市價: {liquidation_price:.2f} 觸及爆倉價: {position['liquidation_price']:.2f}")

            # 計算虧損 (簡化模型：損失全部保證金)
            margin_lost = (position['contracts'] * position['entry_price']) / position['leverage']
            self.portfolio.cash -= margin_lost

            print(f"損失保證金: {margin_lost:.2f} USDT")
            print(f"剩餘現金: {self.portfolio.cash:.2f} USDT")

            # 從 portfolio 中移除倉位
            self.portfolio.close_position(symbol, side)

    async def create_order(self, symbol, type, side, amount, price=None, params={}):
        if price is None:
            price = await self.get_latest_price(symbol)

        leverage = params.get('leverage', settings.LEVERAGE)
        position_side = 'long' if side == 'buy' else 'short'

        # 模擬保證金計算
        margin_required = (amount * price) / leverage
        fee = (amount * price) * settings.FEE_RATE

        if self.portfolio.cash < margin_required + fee:
            raise ValueError(f"資金不足 (需要 {margin_required:.2f} USDT, 只有 {self.portfolio.cash:.2f} USDT)")

        self.portfolio.cash -= (margin_required + fee)

        # 更新倉位並計算爆倉價格
        self.portfolio.update_position(symbol, position_side, amount, price, leverage)

        return {
            'info': {}, 'id': str(pd.Timestamp.now().timestamp()), 'timestamp': self._current_dt,
            'status': 'closed', 'symbol': symbol, 'type': type, 'side': side,
            'amount': amount, 'filled': amount, 'price': price,
            'cost': margin_required, 'fee': {'cost': fee}
        }

    async def get_balance(self):
        # 模擬從 portfolio 獲取餘額
        return {'USDT': {'free': self.portfolio.cash, 'total': self.portfolio.get_total_value()}}

    async def get_positions(self):
        # 直接從 portfolio 獲取倉位
        return self.portfolio.get_positions()

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
