# 檔案: backtest/engine.py

import pandas as pd
from tqdm import tqdm
from core.exchange import PaperExchange

class BacktestEngine:
    def __init__(self, context, strategy, data):
        self.context = context
        self.strategy = strategy
        self.data = data
        self.results = {}

    def run(self):
        print("--- 開始回測 ---")

        # 將數據載入到模擬交易所
        for symbol, df in self.data.items():
            if isinstance(self.context.exchange, PaperExchange):
                self.context.exchange.set_kline_data(symbol, df)

        # 遍歷數據並觸發策略
        # 選擇第一個有效的數據作為主要時間戳
        main_symbol = next((symbol for symbol, df in self.data.items() if df is not None), None)

        if not main_symbol:
            print("錯誤：找不到任何有效的數據來進行回測。")
            return {}

        # 使用 tqdm 顯示進度條
        for dt in tqdm(self.data[main_symbol].index, desc="回測進度"):
            self.context.current_dt = dt
            if isinstance(self.context.exchange, PaperExchange):
                self.context.exchange.set_current_dt(dt)

            self.strategy.on_bar(dt)

            # 更新投資組合
            self.context.portfolio.update(dt)

        self.results = self._calculate_performance()
        print("--- 回測結束 ---")
        return self.results

    def _calculate_performance(self):
        history = pd.DataFrame(self.context.portfolio.history)
        history.set_index('timestamp', inplace=True)

        # 計算績效指標
        total_return = (history['total_value'].iloc[-1] / history['total_value'].iloc[0]) - 1

        returns = history['total_value'].pct_change().dropna()

        # 動態計算年化因子
        time_delta = history.index.to_series().diff().median()
        periods_per_day = pd.Timedelta(days=1) / time_delta
        annualization_factor = (periods_per_day * 252)**0.5 # 假設一年 252 個交易日

        sharpe_ratio = (returns.mean() / returns.std()) * annualization_factor if returns.std() != 0 else 0

        rolling_max = history['total_value'].cummax()
        daily_drawdown = history['total_value'] / rolling_max - 1.0
        max_drawdown = daily_drawdown.min()

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'history': history
        }
