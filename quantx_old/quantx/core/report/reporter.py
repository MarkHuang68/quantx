# 檔案: quantx/core/report/reporter.py
# 版本: v11 (路徑修正與繪圖穩定)
# 說明:
# - 修正了 result_dir 的結構以符合用戶要求：results/scope/mode/symbol_tf/strategy_name/
# - 確保 trades.csv 包含 PnL 和 Maker 狀態。
# - 包含繪圖的最終修正，解決散點圖長度不匹配問題。

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from typing import List, Dict, Any
from datetime import datetime
import numpy as np

from ..eval.metrics import compute_kpis
from ..utils import sanitize

class ReportGenerator:
    def __init__(self, runtime, symbol: str, tf: str, strategy_name: str, mode: str = "backtest"):
        self.runtime = runtime
        self.log = runtime.log
        self.symbol = symbol
        self.tf = tf
        self.strategy_name = strategy_name
        self.mode = mode
        
        # 🟢 核心修正：更改報表目錄結構
        # 結構: results/scope/mode/symbol_tf/strategy_name/
        self.result_dir = os.path.join(
            "results", 
            self.runtime.scope, 
            self.mode, 
            f"{str(symbol)}_{str(tf)}", 
            strategy_name
        )
        
        os.makedirs(self.result_dir, exist_ok=True)
        self.log.info(f"[ReportGenerator] 報表將輸出至: {self.result_dir}")

    def generate(self, ohlcv: pd.DataFrame, equity_curve: pd.Series, trades: List[Dict[str, Any]], strategy_params: dict):
        """
        生成完整報表的主入口。
        """
        try:
            trades_df = self._process_trades(trades)
            self.save_trades(trades_df)
            self.save_summary(equity_curve, trades_df, strategy_params)
            
            if equity_curve is not None and not equity_curve.empty:
                self.plot_equity_curve(equity_curve)
                self.plot_drawdown(equity_curve)
                
            if ohlcv is not None and not ohlcv.empty:
                self.plot_chart(ohlcv, trades_df)
                
            self.log.info(f"[ReportGenerator] {self.symbol}-{self.tf} 報表已成功生成。")
        except Exception as e:
            self.log.error(f"[ReportGenerator] 生成報表時發生錯誤: {e}", exc_info=True)

    def _process_trades(self, trades: List[Dict[str, Any]]) -> pd.DataFrame:
        """將交易列表轉換為 DataFrame。"""
        # 🟢 定義標準欄位
        standard_cols = ['ts', 'side', 'price', 'qty', 'pnl', 'fee', 'maker']
        
        if not trades: 
            # 🟢 修正：若無交易，返回帶有標準欄位的空 DataFrame
            return pd.DataFrame(columns=standard_cols)
        
        df = pd.DataFrame(trades)
        
        if 'ts' in df.columns:
            # 🟢 核心修正：確保 ts 欄位被正確轉換為帶有 UTC 時區的 DatetimeIndex
            df['ts'] = pd.to_datetime(df['ts'], utc=True, errors='coerce')
            df['ts'] = df['ts'].dt.tz_convert('UTC')
            df = df.dropna(subset=['ts'])
        
        # 填充缺失的標準欄位
        for col in standard_cols:
             if col not in df.columns:
                 df[col] = 0.0 if col not in ['ts', 'side'] else (None if col == 'ts' else '')
        
        return df[standard_cols] # 確保只包含標準欄位

    def save_trades(self, trades_df: pd.DataFrame):
        """
        儲存交易明細到 CSV 檔案。
        """
        if trades_df.empty: return
        
        # 🟢 修正：現在 _process_trades 已確保欄位存在
        cols_order = ['ts', 'side', 'price', 'qty', 'pnl', 'fee', 'maker']
        trades_df.to_csv(os.path.join(self.result_dir, "trades.csv"), columns=cols_order, index=False, encoding='utf-8')


    def save_summary(self, equity_curve: pd.Series, trades_df: pd.DataFrame, strategy_params: dict):
        """計算 KPI 並儲存摘要到 JSON 檔案。"""
        kpis = compute_kpis(equity_curve, self.tf) if (equity_curve is not None and not equity_curve.empty) else {}
        
        # 🟢 修正：安全地計算 total_trades
        total_trades = 0
        if not trades_df.empty:
            total_trades = len(trades_df[trades_df['side'].isin(['buy', 'sell'])])

        summary = {
            "symbol": self.symbol, "tf": self.tf, "strategy": self.strategy_name,
            "params": sanitize(strategy_params), 
            "mode": self.mode,
            "total_trades": total_trades, # 使用修正後的 total_trades
            **kpis,
        }
        with open(os.path.join(self.result_dir, "summary.json"), "w", encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    def plot_equity_curve(self, equity_curve: pd.Series):
        """繪製資金曲線圖。"""
        plt.style.use('dark_background'); fig, ax = plt.subplots(figsize=(12, 6))
        equity_curve.plot(ax=ax, title=f"{self.symbol}-{self.tf} Equity Curve", grid=True)
        ax.set_ylabel("Equity"); ax.set_xlabel("Date"); fig.tight_layout()
        plt.savefig(os.path.join(self.result_dir, "equity_curve.png")); plt.close(fig)

    def plot_drawdown(self, equity_curve: pd.Series):
        """繪製最大回撤圖。"""
        peak = equity_curve.cummax(); dd = (equity_curve - peak) / peak
        plt.style.use('dark_background'); fig, ax = plt.subplots(figsize=(12, 6))
        dd.plot(ax=ax, kind='area', color='red', alpha=0.3, title=f"{self.symbol}-{self.tf} Drawdown", grid=True)
        ax.set_ylabel("Drawdown"); ax.set_xlabel("Date"); ax.fill_between(dd.index, dd.values, color='red'); fig.tight_layout()
        plt.savefig(os.path.join(self.result_dir, "drawdown.png")); plt.close(fig)

    def plot_chart(self, ohlcv: pd.DataFrame, trades_df: pd.DataFrame):
        """
        繪製包含 K 線、買賣點與持倉區間的詳細圖表。
        """
        df_chart = ohlcv.copy()
        
        # 1. 🟢 強制 df_chart 的索引為 UTC DatetimeIndex
        if 'timestamp' in df_chart.columns:
            df_chart['timestamp'] = pd.to_datetime(df_chart['timestamp'], unit='s', utc=True)
            df_chart = df_chart.set_index('timestamp')
        
        if not isinstance(df_chart.index, pd.DatetimeIndex):
            # Fallback and force to UTC
            df_chart.index = pd.to_datetime(df_chart.index, errors='coerce', utc=True)
            
        # 確保索引為 UTC 且非空
        df_chart.index = df_chart.index.tz_convert('UTC').rename('time')
        df_chart = df_chart.dropna()
        
        if df_chart.empty:
             self.log.warning("[PlotChart] OHLCV 數據為空，跳過繪圖。")
             return
        
        add_plots = []
        if not trades_df.empty:
            
            # 2. 🟢 設置 ts 為索引
            if 'ts' not in trades_df.columns:
                 self.log.warning("[PlotChart] Trades 數據缺少 'ts' 欄位，跳過繪圖。")
                 return
                 
            trades_df = trades_df.set_index('ts')
            trades_df.index = trades_df.index.tz_convert('UTC') # 確保與 df_chart 時區一致

            # 3. 創建完美對齊的價格 Series (長度 = len(df_chart))
            buy_prices_aligned = pd.Series(np.nan, index=df_chart.index)
            sell_prices_aligned = pd.Series(np.nan, index=df_chart.index)
            
            # 4. 過濾交易並填充 Series
            for index, row in trades_df.iterrows():
                # 僅在 K 線索引中存在的時間點進行填充 (解決時間精度差異)
                if index in df_chart.index:
                    if row['side'] in ['buy', 'close_short']:
                        buy_prices_aligned.loc[index] = row['price']
                    elif row['side'] in ['sell', 'close_long']:
                        sell_prices_aligned.loc[index] = row['price']
            
            # 5. 繪製
            if not buy_prices_aligned.dropna().empty:
                buy_plot = mpf.make_addplot(buy_prices_aligned, type='scatter', marker='^', color='lime', markersize=100)
                add_plots.append(buy_plot)
            
            if not sell_prices_aligned.dropna().empty:
                sell_plot = mpf.make_addplot(sell_prices_aligned, type='scatter', marker='v', color='red', markersize=100)
                add_plots.append(sell_plot)
        
        # 核心修正：使用條件式參數傳遞
        plot_kwargs = {
            'type': 'candle',
            'style': 'yahoo',
            'title': f"{self.symbol} - {self.tf} - {self.strategy_name}",
            'ylabel': 'Price',
            'returnfig': True,
            'figsize': (16, 8),
            'warn_too_much_data': len(df_chart) + 1 # 關閉數據量過多的警告
        }
        if add_plots:
            plot_kwargs['addplot'] = add_plots

        try:
             fig, axlist = mpf.plot(df_chart, **plot_kwargs)
             fig.savefig(os.path.join(self.result_dir, "chart.png")); plt.close(fig)
        except Exception as e:
             self.log.error(f"[ReportGenerator] 繪製 K 線圖失敗: {e}", exc_info=True)