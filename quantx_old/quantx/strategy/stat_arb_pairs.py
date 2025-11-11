# 檔案: quantx/strategy/stat_arb_pairs_trade.py
# 版本: v1 (新架構 - 雙向持倉版)
# 功能: 雙幣統計套利策略，展示如何在v22+框架下進行多標的交易。
# 相容架構: QuantX v22+ (BaseExecutor / ContextBase)

from __future__ import annotations
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from quantx.core.context import ContextBase
from quantx.core.executor.base import BaseExecutor


class stat_arb_pairs_trade(BaseExecutor):
    """
    統計套利策略 (Statistical Arbitrage / Pairs Trading)
    - 透過共整合檢定找出兩幣的穩定價差關係
    - 使用 Z-score 判斷價差偏離程度
    - 超出閾值時進場套利，回歸時平倉
    """

    params = {
        "symbol_a": "BTCUSDT",
        "symbol_b": "ETHUSDT",
        "z_entry": 2.0,
        "z_exit": 0.5,
        "window": 100,
        "min_pvalue": 0.05,
        "qty_a": 0.01,  # A 幣種的下單數量
        "qty_b": 0.1,   # B 幣種的下單數量
    }

    def on_bar(self, ctx: ContextBase):
        symbol_a = self.params["symbol_a"]
        symbol_b = self.params["symbol_b"]
        qty_a = float(self.params["qty_a"])
        qty_b = float(self.params["qty_b"])

        # 嘗試載入兩個幣的資料
        # 注意：多標的策略需要確保 launch.py 啟動時監控了所有需要的幣種
        data = ctx.data([symbol_a, symbol_b])
        df_a = data.get(symbol_a)
        df_b = data.get(symbol_b)
        
        if df_a is None or df_b is None or df_a.empty or df_b.empty:
            ctx.log(f"策略 {self.__class__.__name__}：缺少 {symbol_a} 或 {symbol_b} 的數據，跳過此次 on_bar。請確保回測環境提供了所有需要的數據。")
            return

        # 對齊時間並取最近 N 根
        df = pd.DataFrame({
            "A": df_a["close"],
            "B": df_b["close"]
        }).dropna().tail(int(self.params["window"]))
        
        if len(df) < int(self.params["window"]) // 2:
            return

        # 共整合檢定
        score, pvalue, _ = coint(df["A"], df["B"])
        if pvalue > float(self.params["min_pvalue"]):
            # ctx.log.debug(f"共整合關係不顯著 (p-value: {pvalue:.4f})，暫停套利。")
            return

        # 計算價差與 Z-score
        beta = np.polyfit(df["B"], df["A"], 1)[0]
        spread = df["A"] - beta * df["B"]
        zscore = (spread - spread.mean()) / spread.std()
        z = zscore.iloc[-1]

        # 🟢 核心修改：使用新的 get_position API 獲取雙邊倉位
        pos_a = ctx.get_position(symbol_a)
        pos_b = ctx.get_position(symbol_b)

        z_entry = float(self.params["z_entry"])
        z_exit = float(self.params["z_exit"])

        # === 進出場邏輯 ===
        # 確保雙邊都為空倉時，才考慮進場
        if pos_a.is_flat() and pos_b.is_flat():
            # 價差過高 → short A / long B
            if z > z_entry:
                ctx.log.info(f"🟩 價差擴大 (Z={z:.2f})，執行套利：Short {symbol_a}, Long {symbol_b}")
                ctx.open_short(symbol=symbol_a, qty=qty_a)
                ctx.open_long(symbol=symbol_b, qty=qty_b)
            # 價差過低 → long A / short B
            elif z < -z_entry:
                ctx.log.info(f"🟥 價差縮小 (Z={z:.2f})，執行套利：Long {symbol_a}, Short {symbol_b}")
                ctx.open_long(symbol=symbol_a, qty=qty_a)
                ctx.open_short(symbol=symbol_b, qty=qty_b)

        # 只要有一邊有倉位，就進入平倉監控邏輯
        elif not pos_a.is_flat() or not pos_b.is_flat():
            # 價差回歸中軸 → 平掉所有倉位
            if abs(z) < z_exit:
                ctx.log.info(f"⚪ 價差回歸 (Z={z:.2f})，全部平倉。")
                # 使用不帶 qty 參數的 close，代表全平
                if pos_a.is_short():
                    ctx.close_short(symbol=symbol_a)
                if pos_a.is_long():
                    ctx.close_long(symbol=symbol_a)
                if pos_b.is_short():
                    ctx.close_short(symbol=symbol_b)
                if pos_b.is_long():
                    ctx.close_long(symbol=symbol_b)