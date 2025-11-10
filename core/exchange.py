# 檔案: core/exchange.py

import ccxt
import pandas as pd
from abc import ABC, abstractmethod
import settings

class Exchange(ABC):
    @abstractmethod
    def get_ohlcv(self, symbol, timeframe, limit): pass
    @abstractmethod
    def create_order(self, symbol, type, side, amount, price=None, params={}): pass
    @abstractmethod
    def get_balance(self): pass
    @abstractmethod
    def get_positions(self): pass
    @abstractmethod
    def get_latest_price(self, symbol): pass
    @abstractmethod
    def close_all_positions(self): pass
    @abstractmethod
    def sync_positions(self): pass

class BinanceExchange(Exchange):
    def __init__(self, api_key, api_secret, portfolio):
        self.fee_rate = 0.001
        self.portfolio = portfolio
        self.exchange = ccxt.binance({'apiKey': api_key, 'secret': api_secret, 'enableRateLimit': True})

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
        return { asset['asset']: float(asset['free']) for asset in balance['info']['balances'] if float(asset['free']) > 0 }

    def sync_positions(self):
        print("--- 正在同步幣安倉位 ---")
        balance = self.get_balance()
        self.portfolio.cash = balance.get('USDT', {}).get('free', 0)
        self.portfolio.positions.clear()
        for currency, data in balance.items():
            if data.get('total', 0) > 0 and currency != 'USDT':
                 self.portfolio.positions[currency] = data['total']
        print(f"倉位同步完成: {self.portfolio.get_positions()}")

    def get_latest_price(self, symbol):
        return self.exchange.fetch_ticker(symbol)['last']

    def close_all_positions(self):
        print("--- 正在平掉所有幣安倉位... ---")
        positions = self.get_positions()
        for symbol_base, amount in positions.items():
            if symbol_base == 'USDT' or amount <= 0: continue
            try:
                self.create_order(f"{symbol_base}/USDT", 'market', 'sell', amount)
            except Exception as e:
                print(f"平倉 {symbol_base}/USDT 時發生錯誤: {e}")

class PaperExchange(Exchange):
    def __init__(self, portfolio):
        self.fee_rate = 0.001
        self.portfolio = portfolio
        self._current_dt = None
        self._kline_data = {}

    def set_kline_data(self, symbol, df): self._kline_data[symbol] = df
    def set_current_dt(self, dt): self._current_dt = dt

    def get_ohlcv(self, symbol, timeframe='1m', limit=100):
        if symbol not in self._kline_data: return pd.DataFrame()
        df = self._kline_data[symbol]
        end_idx = df.index.searchsorted(self._current_dt, side='right') - 1
        if end_idx < 0: return pd.DataFrame()
        start_idx = max(0, end_idx - limit + 1)
        return df.iloc[start_idx:end_idx + 1]

    def create_order(self, symbol, type, side, amount, price=None, params={}):
        if price is None: price = self.get_latest_price(symbol)
        if price is None: raise ValueError(f"無法在 {self._current_dt} 找到 {symbol} 的價格")

        trade_value = amount * price
        fee = trade_value * self.fee_rate
        if side == 'buy' and self.portfolio.cash < trade_value + fee:
            raise ValueError(f"資金不足，需要 {trade_value + fee:.2f} USDT，但只有 {self.portfolio.cash:.2f}")

        return {'status': 'closed', 'symbol': symbol, 'side': side, 'amount': amount, 'price': price, 'fee': fee}

    def get_balance(self):
        return {'USDT': {'free': self.portfolio.cash, 'total': self.portfolio.cash}}

    def get_positions(self):
        return self.portfolio.get_positions()

    def get_latest_price(self, symbol):
        if symbol in self._kline_data and self._current_dt is not None:
            price_series = self._kline_data[symbol]['Close']
            latest_price = price_series.asof(self._current_dt)
            if pd.notna(latest_price): return latest_price
        return None

    def close_all_positions(self):
        print("--- 正在平掉所有模擬倉位... ---")
        for symbol_base, amount in dict(self.portfolio.positions).items():
            if amount == 0: continue
            side = 'sell' if amount > 0 else 'buy'
            try: self.create_order(f"{symbol_base}/USDT", 'market', side, abs(amount))
            except Exception as e: print(f"平倉 {symbol_base}/USDT 時發生錯誤: {e}")

    def sync_positions(self): pass
