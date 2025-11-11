# 檔案: quantx/strategy/z_score.py
# 版本: v4 (修正：使用標準 ta.zscore)
# 說明:
# - 修正了 Z-Score 趨勢策略，使其使用核心庫提供的 ta.zscore 函數，確保指標計算的準確性。
# - 價格突破 Z-score 上軌 -> 追高做多 (Trend Long)
# - 價格跌破 Z-score 下軌 -> 殺跌做空 (Trend Short)

from __future__ import annotations
from quantx.core.context import ContextBase
from quantx.core.executor.base import BaseExecutor
from quantx.ta import indicators as ta # 引入ta指標

class z_score(BaseExecutor):
    """
    Z-Score 趨勢策略 (多空雙向)
    - Z-score > threshold: 認為趨勢將繼續向上，做多。
    - Z-score < -threshold: 認為趨勢將繼續向下，做空。
    - Z-score 回歸 0 軸: 認為趨勢結束，平倉。
    """

    params = {
        "sma_len": 40,
        "z_th": 0.5,
    }

    def on_bar(self, ctx: ContextBase):
        length = int(self.params["sma_len"])
        z_th = float(self.params["z_th"])

        df = ctx.data(ctx.symbol, ctx.tf)
        # 策略應在數據長度滿足指標計算窗口時再執行
        if len(df) < length:
            return

        close = df["close"]
        
        # 🟢 修正：使用核心庫提供的標準 Z-Score 函式
        z = ta.zscore(close, length)

        curr_z = z.iloc[-1]
        pos = ctx.get_position(ctx.symbol)

        # === 做多邏輯 (Trend Long) ===
        # 開倉條件：無倉位且 Z-score 向上突破閾值
        if pos.is_flat() and curr_z > z_th:
            ctx.open_long()
        # 平倉條件：持有多單，且 Z-score 回落至 0 軸以下
        elif pos.is_long() and curr_z < 0:
            ctx.close_long()

        # === 做空邏輯 (Trend Short) ===
        # 開倉條件：無倉位且 Z-score 向下跌破負閾值
        if pos.is_flat() and curr_z < -z_th:
            ctx.open_short()
        # 平倉條件：持有空單，且 Z-score 回升至 0 軸以上
        elif pos.is_short() and curr_z > 0:
            ctx.close_short()