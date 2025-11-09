# 檔案: core/exchange.py

import ccxt
import pandas as pd
from abc import ABC, abstractmethod
import settings

class Exchange(ABC):
    @abstractmethod
    def get_ohlcv(self, symbol, timeframe, limit):
        pass

    @abstractmethod
    def create_order(self, symbol, type, side, amount, price=None, params={}):
        pass

    @abstractmethod
    def get_balance(self):
        pass

    @abstractmethod
    def get_positions(self):
        pass

    @abstractmethod
    def get_latest_price(self, symbol):
        pass


class BinanceExchange(Exchange):
    def __init__(self, api_key, api_secret):
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
        })

    def get_ohlcv(self, symbol, timeframe='1m', limit=100):
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df

    def create_order(self, symbol, type, side, amount, price=None, params={}):
        return self.exchange.create_order(symbol, type, side, amount, price, params)

    def get_balance(self):
        return self.exchange.fetch_balance()

    def get_positions(self):
        # 注意：ccxt 的 `fetch_positions` 主要用於期貨和衍生品。
        # 對於現貨交易，您需要從 `fetch_balance` 中推斷持倉。
        if self.exchange.has['fetchPositions']:
            return self.exchange.fetch_positions()
        else:
            balance = self.get_balance()
            positions = {
                asset['asset']: asset['free']
                for asset in balance['info']['balances']
                if float(asset['free']) > 0
            }
            return positions

    def sync_positions(self, portfolio):
        print("--- 正在同步幣安倉位 ---")
        positions = self.get_positions()
        portfolio.positions = positions
        print(f"倉位同步完成: {positions}")

    def get_latest_price(self, symbol):
        ticker = self.exchange.fetch_ticker(symbol)
        return ticker['last']


class CoinbaseExchange(Exchange):
    def __init__(self, api_key, api_secret):
        self.exchange = ccxt.coinbase({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
        })

    def get_ohlcv(self, symbol, timeframe='1m', limit=100):
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df

    def create_order(self, symbol, type, side, amount, price=None, params={}):
        return self.exchange.create_order(symbol, type, side, amount, price, params)

    def get_balance(self):
        return self.exchange.fetch_balance()

    def get_positions(self):
        balance = self.get_balance()
        positions = {
            item['currency']: float(item['balance'])
            for item in balance.values()
            if isinstance(item, dict) and 'currency' in item and 'balance' in item and float(item['balance']) > 0
        }
        return positions

    def sync_positions(self, portfolio):
        print("--- 正在同步 Coinbase 倉位 ---")
        positions = self.get_positions()
        portfolio.positions = positions
        print(f"倉位同步完成: {positions}")

    def get_latest_price(self, symbol):
        ticker = self.exchange.fetch_ticker(symbol)
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

    def get_ohlcv(self, symbol, timeframe='1m', limit=100):
        if symbol not in self._kline_data:
            raise ValueError(f"沒有 {symbol} 的 K 線資料")

        df = self._kline_data[symbol]
        # Find the index location for the current datetime
        try:
            end_idx = df.index.get_loc(self._current_dt)
        except KeyError:
             # If the exact time is not found, find the closest one before it
            end_idx = df.index.searchsorted(self._current_dt, side='right') - 1

        if end_idx < 0:
            return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])

        start_idx = max(0, end_idx - limit + 1)
        return df.iloc[start_idx:end_idx + 1]

    def create_order(self, symbol, type, side, amount, price=None, params={}):
        # 模擬訂單執行
        if price is None:
            # 模擬市價單，使用當前 K 線的收盤價
            if symbol in self._kline_data and self._current_dt in self._kline_data[symbol].index:
                 price = self._kline_data[symbol].loc[self._current_dt]['Close']
            else:
                 # If exact timestamp not found, use the latest available price
                try:
                    price_series = self._kline_data[symbol]['Close']
                    price = price_series.asof(self._current_dt)
                except Exception as e:
                     raise ValueError(f"無法在 {self._current_dt} 找到 {symbol} 的價格: {e}")

        base_currency, quote_currency = symbol.split('/')

        # 計算交易總額和手續費
        trade_value = amount * price
        fee = trade_value * settings.FEE_RATE

        if side == 'buy':
            cost = trade_value
            # 檢查包含手續費在內的總資金是否足夠
            if self._balance[quote_currency]['free'] < cost + fee:
                raise ValueError(f"資金不足，需要 {cost + fee:.2f} {quote_currency} (含手續費)，但只有 {self._balance[quote_currency]['free']:.2f}")

            self._balance[quote_currency]['free'] -= (cost + fee)
            self._positions.setdefault(base_currency, 0)
            self._positions[base_currency] += amount

        elif side == 'sell':
            # 允許做空，移除持倉檢查
            self._positions.setdefault(base_currency, 0)
            self._positions[base_currency] -= amount
            revenue = trade_value
            self._balance[quote_currency]['free'] += (revenue - fee)

        return {
            'info': {'symbol': symbol, 'side': side, 'type': type, 'executedQty': amount, 'avgPrice': price, 'fee': fee},
            'id': str(pd.Timestamp.now().timestamp()),
            'timestamp': self._current_dt,
            'datetime': self._current_dt.isoformat(),
            'status': 'closed',
            'symbol': symbol,
            'type': type,
            'side': side,
            'amount': amount,
            'filled': amount,
            'price': price,
            'cost': amount * price if side == 'buy' else 0,
        }

    def get_balance(self):
        return self._balance

    def get_positions(self):
        return self._positions

    def get_latest_price(self, symbol):
        if symbol in self._kline_data and self._current_dt is not None:
            try:
                # 首先尝试直接定位
                if self._current_dt in self._kline_data[symbol].index:
                    return self._kline_data[symbol].loc[self._current_dt]['Close']
                # 如果找不到，使用 asof 找到最新的有效价格
                else:
                    price_series = self._kline_data[symbol]['Close']
                    latest_price = price_series.asof(self._current_dt)
                    if pd.notna(latest_price):
                        return latest_price
            except Exception:
                pass # 如果出錯，就使用下面的 fallback

        # Fallback: 如果上面的方法都失敗，就返回數據中的最後一個價格
        if symbol in self._kline_data and not self._kline_data[symbol].empty:
            return self._kline_data[symbol]['Close'].iloc[-1]

        return None # 如果完全沒有數據，返回 None
