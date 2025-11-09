# 檔案: core/portfolio.py

import pandas as pd
from core.exchange import PaperExchange

class Portfolio:
    def __init__(self, initial_capital, exchange):
        self.initial_capital = initial_capital
        self.exchange = exchange
        self.positions = {}
        self.history = []

    def update(self, dt):
        balance = self.exchange.get_balance()

        # 在模擬交易中，我們需要手動更新持倉
        if isinstance(self.exchange, PaperExchange):
            positions = self.exchange.get_positions()
        else:
            positions = self.exchange.get_positions()

        # 簡化處理：假設我們的報價貨幣是 USDT
        total_value = balance.get('USDT', {}).get('total', self.initial_capital)

        # 將持倉轉換為 USDT 價值
        for asset, amount in positions.items():
            if asset != 'USDT' and amount > 0:
                try:
                    # 獲取最新價格來計算持倉價值
                    ticker = self.exchange.exchange.fetch_ticker(f'{asset}/USDT')
                    total_value += amount * ticker['last']
                except Exception as e:
                    # 在回測中，我們可以使用當前的收盤價
                    if isinstance(self.exchange, __import__('core.exchange', fromlist=['PaperExchange']).PaperExchange):
                        price = self.exchange._kline_data[f'{asset}/USDT'].loc[dt]['close']
                        total_value += amount * price
                    else:
                        print(f"無法獲取 {asset}/USDT 的價格: {e}")

        self.positions = positions
        self.history.append({'timestamp': dt, 'total_value': total_value})

    def get_positions(self):
        return self.positions

    def get_total_value(self):
        if not self.history:
            return self.initial_capital
        return self.history[-1]['total_value']
