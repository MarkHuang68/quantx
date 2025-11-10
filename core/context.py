# 檔案: core/context.py

class Context:
    def __init__(self, exchange, portfolio, initial_capital=100000):
        if not exchange or not portfolio:
            raise ValueError("Context 需要有效的 exchange 和 portfolio 物件。")

        self.initial_capital = initial_capital
        self.exchange = exchange
        self.portfolio = portfolio
        self.current_dt = None
