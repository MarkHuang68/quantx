# 檔案: core/context.py

class Context:
    def __init__(self, start_date=None, end_date=None, initial_capital=100000):
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.current_dt = None
        self.portfolio = None
        self.exchange = None
