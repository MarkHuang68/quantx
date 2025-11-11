# 檔案: quantx/strategy/z_score_reverse.py
# 版本: v3 (反轉策略版)
# 說明:
# - 實現了 Z-Score 均值回歸 (反轉) 交易邏輯。
# - 價格跌破 Z-score 下軌 -> 抄底做多 (Reverse Long)
# - 價格突破 Z-score 上軌 -> 摸頂做空 (Reverse Short)

from __future__ import annotations
from quantx.core.context import ContextBase
from quantx.core.executor.base import BaseExecutor
from quantx.ta import indicators as ta

class z_score_reverse(BaseExecutor):
    """
    Z-Score 反轉策略 (多空雙向)
    - Z-score < -threshold: 認為價格超跌將反彈，做多。
    - Z-score > threshold: 認為價格超漲將回調，做空。
    - Z-score 回歸 0 軸: 認為反彈/回調結束，平倉。
    """

    params = {
        "sma_len": 40,
        "z_th": 0.5,
    }

    def on_bar(self, ctx: ContextBase):
        length = int(self.params["sma_len"])
        z_th = float(self.params["z_th"])

        df = ctx.data(ctx.symbol, ctx.tf)
        if len(df) < length:
            return

        close = df["close"]
        sma = ta.sma(close, length)
        rolling_std = close.rolling(window=length).std(ddof=1).replace(0, float("nan")).ffill()
        
        z = (close - sma) / rolling_std
        z = z.fillna(0)

        curr_z = z.iloc[-1]
        pos = ctx.get_position(ctx.symbol)

        # === 做多邏輯 (Reverse Long) ===
        # 開倉條件：無倉位且 Z-score 向下跌破負閾值
        if pos.is_flat() and curr_z < -z_th:
            ctx.open_long()
        # 平倉條件：持有多單，且 Z-score 回升至 0 軸以上
        elif pos.is_long() and curr_z > 0:
            ctx.close_long()

        # === 做空邏輯 (Reverse Short) ===
        # 開倉條件：無倉位且 Z-score 向上突破閾值
        if pos.is_flat() and curr_z > z_th:
            ctx.open_short()
        # 平倉條件：持有空單，且 Z-score 回落至 0 軸以下
        elif pos.is_short() and curr_z < 0:
            ctx.close_short()