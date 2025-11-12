# 檔案: core/backtest_engine.py

import pandas as pd
from tqdm import tqdm
from core.exchange import PaperExchange
from utils.common import create_features_trend

class BacktestEngine:
    def __init__(self, context, strategy, data):
        self.context = context
        self.strategy = strategy
        self.data = data
        self.results = {}
        self.features_data = {} # 用於存儲帶有特徵的數據

    def _precompute_features(self):
        """
        在回測開始前，為所有數據預先計算特徵。
        """
        print("--- 正在預先計算所有技術指標... ---")
        for symbol, df in self.data.items():
            df_with_features, _ = create_features_trend(df)
            self.features_data[symbol] = df_with_features

    async def run(self):
        print("--- 開始回測 ---")

        self._precompute_features()

        for symbol, df in self.data.items():
            if isinstance(self.context.exchange, PaperExchange):
                self.context.exchange.set_kline_data(symbol, df)

        main_symbol = next((symbol for symbol, df in self.data.items() if df is not None), None)

        if not main_symbol:
            print("錯誤：找不到任何有效的數據來進行回測。")
            return {}

        for dt in tqdm(self.data[main_symbol].index, desc="回測進度"):
            self.context.current_dt = dt
            if isinstance(self.context.exchange, PaperExchange):
                await self.context.exchange.set_current_dt(dt)

            current_features = {}
            for symbol, df_features in self.features_data.items():
                if dt in df_features.index:
                    current_features[symbol] = df_features.loc[dt]

            if current_features:
                await self.strategy.on_bar(dt, current_features)

            await self.context.portfolio.update(dt)

        self.results = self._calculate_performance()
        print("--- 回測結束 ---")
        return self.results

    def _calculate_performance(self):
        history = pd.DataFrame(self.context.portfolio.history)
        history.set_index('timestamp', inplace=True)

        if history.empty:
            print("警告: 投資組合歷史為空，無法計算績效。")
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'history': pd.DataFrame()
            }

        total_return = (history['total_value'].iloc[-1] / history['total_value'].iloc[0]) - 1
        returns = history['total_value'].pct_change().dropna()

        if returns.empty or returns.std() == 0:
            sharpe_ratio = 0.0
        else:
            time_delta = history.index.to_series().diff().median()
            periods_per_day = pd.Timedelta(days=1) / time_delta
            annualization_factor = (periods_per_day * 252)**0.5
            sharpe_ratio = (returns.mean() / returns.std()) * annualization_factor

        rolling_max = history['total_value'].cummax()
        daily_drawdown = history['total_value'] / rolling_max - 1.0
        max_drawdown = daily_drawdown.min()

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'history': history
        }
