# 檔案: quantx/strategy/ema_cross.py
# 版本: v2 (多空換倉優化版)
# 說明:
# - 策略本身已是多空雙向 (黃金交叉做多，死亡交叉做空)。
# - 優化換倉邏輯：在開新倉前，先確保反向倉位已被平倉。

from __future__ import annotations
from quantx.core.context import ContextBase
from quantx.core.executor.base import BaseExecutor
from quantx.ta import indicators as ta # 引入ta指標

class ema_cross(BaseExecutor):
    """
    EMA 均線交叉策略 (多空雙向)
    - 快線上穿慢線 (黃金交叉) -> 做多
    - 快線下穿慢線 (死亡交叉) -> 做空
    """

    params = {
        "fast": 12,
        "slow": 26,
    }

    def on_bar(self, ctx: ContextBase):
        df = ctx.data(ctx.symbol, ctx.tf)
        # 確保有足夠的數據來計算慢線
        if len(df) < self.params["slow"]:
            return

        close_series = df["close"]
        # 注意：原檔案使用 sma，這裡保留原樣，但策略名為 ema_cross。
        # 如需使用 EMA，請將 ta.sma 改為 ta.ema
        fast = ta.sma(close_series, self.params["fast"])
        slow = ta.sma(close_series, self.params["slow"])

        # 使用 .iloc[-1] 和 .iloc[-2] 獲取當前和前一根 K 棒的值，更穩健
        prev_fast = fast.iloc[-2]
        prev_slow = slow.iloc[-2]
        curr_fast = fast.iloc[-1]
        curr_slow = slow.iloc[-1]
        
        pos = ctx.position

        # 訊號 1：黃金交叉 -> 做多
        if prev_fast <= prev_slow and curr_fast > curr_slow:
            if pos.is_short():
                ctx.close_short() # 如果有空單，先平倉
            if pos.is_flat(): # 確保無倉位後再開多單
                ctx.open_long()

        # 訊號 2：死亡交叉 -> 做空
        elif prev_fast >= prev_slow and curr_fast < curr_slow:
            if pos.is_long():
                ctx.close_long() # 如果有多單，先平倉
            if pos.is_flat(): # 確保無倉位後再開空單
                ctx.open_short()