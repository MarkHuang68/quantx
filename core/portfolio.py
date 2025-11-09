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
        """
        更新投資組合的總價值。
        正確的計算方式應該是將所有資產（包含現金）的價值相加。
        """
        # 1. 從交易所獲取當前所有餘額和持倉
        balance = self.exchange.get_balance()
        positions = self.exchange.get_positions()

        # 2. 初始化總價值為報價貨幣 (USDT) 的可用餘額
        # 我們使用 'free' 而不是 'total'，因為 'total' 在 PaperExchange 中沒有被正確更新
        total_value = balance.get('USDT', {}).get('free', 0)

        # 3. 將所有非 USDT 的持倉轉換為 USDT 價值並加總
        # 合併 balance 和 positions 以處理所有可能的資產
        all_assets = {**{k: v.get('free', 0) for k, v in balance.items()}, **positions}

        for asset, amount in all_assets.items():
            if asset == 'USDT' or amount == 0:
                continue

            symbol = f'{asset}/USDT'
            price = 0
            try:
                # 在回測中，我們從預載的 K 線數據中獲取當前價格
                if isinstance(self.exchange, PaperExchange):
                    if symbol in self.exchange._kline_data and dt in self.exchange._kline_data[symbol].index:
                        price = self.exchange._kline_data[symbol].loc[dt]['Close']
                    else: # Fallback for live trading or missing data points
                        price = self.exchange.get_latest_price(symbol)
                else: # 實盤交易
                    price = self.exchange.get_latest_price(symbol)

                if price and price > 0:
                    total_value += amount * price

            except Exception as e:
                print(f"警告：在 {dt} 無法獲取 {symbol} 的價格來更新投資組合價值: {e}")

        # 4. 更新 portfolio 的內部狀態
        self.positions = positions
        self.history.append({'timestamp': dt, 'total_value': total_value})

    def get_positions(self):
        return self.positions

    def get_total_value(self):
        if not self.history:
            return self.initial_capital
        return self.history[-1]['total_value']
