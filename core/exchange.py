# 檔案: core/exchange.py

import ccxt
import pandas as pd
from abc import ABC, abstractmethod
import settings

class Exchange(ABC):
    # ... (抽象方法定义保持不变) ...
    pass

class BinanceExchange(Exchange):
    # ... (BinanceExchange 逻辑暂时保持不变，因为它处理的是真实交易) ...
    pass

class PaperExchange(Exchange):
    def __init__(self, portfolio):
        self.fee_rate = 0.001
        self.portfolio = portfolio
        self._current_dt = None
        self._kline_data = {}

    def set_kline_data(self, symbol, df):
        self._kline_data[symbol] = df

    def set_current_dt(self, dt):
        self._current_dt = dt

    def get_ohlcv(self, symbol, timeframe='1m', limit=100):
        # ... (此函數邏輯不變) ...
        pass

    def create_order(self, symbol, type, side, amount, price=None, params={}):
        """
        新版職責：只檢查規則，成功則返回成交回報，不修改 Portfolio。
        """
        if price is None:
            price = self.get_latest_price(symbol)
            if price is None:
                 raise ValueError(f"無法在 {self._current_dt} 找到 {symbol} 的價格")

        base_currency, quote_currency = symbol.split('/')
        trade_value = amount * price
        fee = trade_value * self.fee_rate

        if side == 'buy':
            cost = trade_value + fee
            if self.portfolio.cash < cost:
                raise ValueError(f"資金不足，需要 {cost:.2f} {quote_currency}，但只有 {self.portfolio.cash:.2f}")

        # 對於賣出，我們允許做空，所以不需要檢查持倉是否足夠。

        # 檢查通過，返回一個包含所有成交細節的 "成交回報" 字典
        return {
            'status': 'closed',
            'symbol': symbol,
            'side': side,
            'amount': amount,
            'price': price,
            'fee': fee
        }

    def get_balance(self):
        return {'USDT': {'free': self.portfolio.cash, 'total': self.portfolio.cash}}

    def get_positions(self):
        return self.portfolio.get_positions()

    def get_latest_price(self, symbol):
        # ... (此函數邏輯不變) ...
        pass

    def close_all_positions(self):
        # ... (此函數邏輯不變) ...
        pass

    def sync_positions(self):
        pass # 模擬盤不需要同步
