# 檔案: core/portfolio.py

import pandas as pd
from collections import defaultdict
import json

class Portfolio:
    def __init__(self, initial_capital, performance_file="performance.json"):
        self.initial_capital = initial_capital
        self.performance_file = performance_file

        self.cash = initial_capital
        self.positions = defaultdict(float)
        self.history = []

        self.performance_tracking = self._load_performance()

    def _load_performance(self):
        try:
            with open(self.performance_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def save_performance(self):
        with open(self.performance_file, 'w') as f:
            json.dump(self.performance_tracking, f, indent=4)

    def update_position(self, symbol, amount, price):
        """
        純粹的記帳功能：更新倉位和現金，不考慮手續費。
        amount > 0 為買入, amount < 0 為賣出。
        """
        base_currency = symbol.split('/')[0]
        cost = amount * price

        self.positions[base_currency] += amount
        self.cash -= cost

        if abs(self.positions[base_currency]) < 1e-9:
            self.positions.pop(base_currency, None)

    def charge_fee(self, fee_amount):
        """從現金中扣除手續費。"""
        self.cash -= fee_amount

    def get_positions(self):
        return dict(self.positions)

    def get_total_value(self, current_prices):
        total_value = self.cash
        for symbol_base, amount in self.positions.items():
            symbol_pair = f"{symbol_base}/USDT"
            if symbol_pair in current_prices:
                total_value += amount * current_prices[symbol_pair]
        return total_value

    def update(self, dt, current_prices):
        total_value = self.get_total_value(current_prices)
        self.history.append({'timestamp': dt, 'total_value': total_value})

    def generate_performance_report(self):
        # ... (此函數保持不變) ...
        pass
