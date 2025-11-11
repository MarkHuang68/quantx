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
    # ... (其他抽象方法) ...

# ... (BybitExchange, BinanceExchange, CoinbaseExchange 程式碼) ...

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
            if symbol in self._kline_data and self._current_dt in self._kline_data[symbol].index:
                 price = self._kline_data[symbol].loc[self._current_dt]['Close']
            else:
                try:
                    price_series = self._kline_data[symbol]['Close']
                    price = price_series.asof(self._current_dt)
                except Exception as e:
                     raise ValueError(f"無法在 {self._current_dt} 找到 {symbol} 的價格: {e}")

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
            except Exception:
                pass
        if symbol in self._kline_data and not self._kline_data[symbol].empty:
            return self._kline_data[symbol]['Close'].iloc[-1]
        return None
