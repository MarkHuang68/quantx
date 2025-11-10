# 檔案: core/portfolio.py

import pandas as pd
from collections import defaultdict
import json

class Portfolio:
    def __init__(self, initial_capital, exchange=None, performance_file="performance.json"):
        self.initial_capital = initial_capital
        self.exchange = exchange
        self.performance_file = performance_file

        self.cash = initial_capital
        self.positions = defaultdict(float) # key: 幣種 (e.g., 'ETH'), value: 數量
        self.history = []

        # --- 新增：精細化的績效追蹤 ---
        self.performance_tracking = self._load_performance()

    def _load_performance(self):
        """從檔案載入歷史績效，如果檔案不存在則初始化。"""
        try:
            with open(self.performance_file, 'r') as f:
                data = json.load(f)
                print("--- 成功載入歷史績效紀錄 ---")
                return data
        except FileNotFoundError:
            print("--- 未找到歷史績效紀錄，將創建新的。 ---")
            return {}

    def save_performance(self):
        """將當前的績效數據儲存到檔案。"""
        with open(self.performance_file, 'w') as f:
            json.dump(self.performance_tracking, f, indent=4)
        print(f"--- 績效紀錄已儲存至 {self.performance_file} ---")

    def update_position(self, symbol, amount, price):
        """更新指定幣種的倉位和現金。"""
        base_currency = symbol.split('/')[0]
        cost = amount * price

        self.positions[base_currency] += amount
        self.cash -= cost

        # 如果倉位接近於零，則直接清零
        if abs(self.positions[base_currency]) < 1e-9:
            self.positions.pop(base_currency)

    def get_positions(self):
        return dict(self.positions)

    def get_total_value(self):
        """計算總資產價值 (現金 + 持倉市值)。"""
        total_value = self.cash
        if self.exchange:
            for symbol_base, amount in self.positions.items():
                # 假設計價貨幣是 USDT
                symbol_pair = f"{symbol_base}/USDT"
                price = self.exchange.get_latest_price(symbol_pair)
                if price:
                    total_value += amount * price
        return total_value

    def update(self, dt):
        """
        每個時間點更新投資組合的歷史紀錄和每個幣種的 PnL。
        """
        total_value = self.get_total_value()
        self.history.append({'timestamp': dt, 'total_value': total_value})

        # --- 更新每個交易對的 PnL ---
        current_pnl = total_value - self.initial_capital

        # 這裡簡化處理：假設 PnL 平均分配到每個持倉中
        # 一個更精確的系統需要追蹤每個交易的成本基礎 (cost basis)
        # 但對於熔斷機制，追蹤總 PnL 通常足夠
        # 我們將在 main.py 中處理單一幣種的邏輯
        pass

    def update_symbol_performance(self, symbol, current_total_value):
        """
        由外部 (main.py) 呼叫，用來更新特定幣種的績效追蹤。
        這是一個簡化的實現，真實 PnL 應基於交易紀錄。
        """
        if symbol not in self.performance_tracking:
            self.performance_tracking[symbol] = {
                "total_pnl": 0,
                "peak_pnl": 0,
                "status": "active"
            }

        # 這是一個粗略的估算，真實 PnL 需要更複雜的計算
        # 暫時以總價值的變化作為代理
        pnl_change = current_total_value - self.initial_capital - sum(s['total_pnl'] for s in self.performance_tracking.values())

        perf = self.performance_tracking[symbol]
        perf['total_pnl'] += pnl_change # 這裡需要一個更好的 PnL 分配邏輯
        perf['peak_pnl'] = max(perf['peak_pnl'], perf['total_pnl'])

    def generate_performance_report(self):
        """生成詳細的回測績效報告。"""
        df = pd.DataFrame(self.history)
        if df.empty:
            return {}

        df.set_index('timestamp', inplace=True)

        # 計算總報酬率
        total_return = (df['total_value'].iloc[-1] / self.initial_capital) - 1

        # 計算夏普比率 (年化)
        returns = df['total_value'].pct_change().dropna()
        if returns.empty or returns.std() == 0:
            sharpe_ratio = 0.0
        else:
            time_delta = df.index.to_series().diff().median()
            periods_per_year = pd.Timedelta(days=365) / time_delta
            sharpe_ratio = (returns.mean() / returns.std()) * (periods_per_year ** 0.5) if periods_per_year > 0 else 0

        # 計算最大回撤
        rolling_max = df['total_value'].cummax()
        drawdown = (df['total_value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # 計算盈虧比
        wins = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        profit_factor = wins / losses if losses > 0 else float('inf')

        return {
            "total_pnl_pct": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": abs(max_drawdown),
            "profit_factor": profit_factor,
        }
