# 檔案: quantx/strategy/macd_zero_axis_cross.py
# 版本: v1 (全新策略)
# 說明:
# - 實現 MACD 零軸交叉策略。
# - 做多開倉: MACD 線在 0 軸上方，且黃金交叉信號線。
# - 做多平倉: MACD 線向下跌破 0 軸。
# - 做空開倉: MACD 線在 0 軸下方，且死亡交叉信號線。
# - 做空平倉: MACD 線向上漲破 0 軸。

from __future__ import annotations
import pandas as pd
# 由於您的專案結構，ta.trend.MACD 可能無法直接使用，我們從 quantx.ta 引入
# 如果 quantx.ta 中沒有 MACD，我們需要在此處實現或確保 ta 庫已安裝並可導入
try:
    from ta.trend import MACD
except ImportError:
    # 如果 'ta' 庫不存在，提供一個後備方案或明確的錯誤
    # 根據您的專案結構，您似乎有自己的 ta.indicators 模組，但它不包含 MACD。
    # 因此，我們假設 `ta` 庫 (pip install ta) 是環境的一部分。
    # 如果不是，這段程式碼會引發 ImportError，提示需要安裝。
    raise ImportError("請安裝 'ta' 技術指標庫: pip install ta")

from quantx.core.context import ContextBase
from quantx.core.executor.base import BaseExecutor


class macd_zero_axis_cross(BaseExecutor):
    """
    MACD 零軸交叉策略

    開倉條件:
    - 多單: MACD 線與信號線均在 0 軸上方，且 MACD 線向上穿越信號線 (黃金交叉)。
    - 空單: MACD 線與信號線均在 0 軸下方，且 MACD 線向下穿越信號線 (死亡交叉)。

    平倉條件:
    - 多單: 持有多單時，MACD 線由上往下跌破 0 軸。
    - 空單: 持有空單時，MACD 線由下往上漲破 0 軸。

    持倉期間，忽略所有反向開倉信號。
    """

    # 策略參數，使用業界標準的 MACD 預設值
    params = {
        "fast": 12,    # 快線 EMA 週期
        "slow": 26,    # 慢線 EMA 週期
        "signal": 9,   # 信號線 EMA 週期
    }

    def on_bar(self, ctx: ContextBase):
        # 1. 獲取參數與數據
        fast_len = int(self.params["fast"])
        slow_len = int(self.params["slow"])
        signal_len = int(self.params["signal"])

        df = ctx.data(ctx.symbol, ctx.tf)

        # 確保數據長度足以計算最長的 EMA 週期
        if len(df) < slow_len + signal_len:
            return

        # 2. 計算 MACD 指標
        # 使用 'ta' 庫來計算 MACD，確保準確性
        macd_indicator = MACD(
            close=df["close"],
            window_slow=slow_len,
            window_fast=fast_len,
            window_sign=signal_len
        )
        
        # MACD 線 (快線)
        macd_line = macd_indicator.macd()
        # 信號線 (慢線)
        signal_line = macd_indicator.macd_signal()

        # 3. 獲取判斷所需的指標值
        # 獲取當前 K 棒 (-1) 和前一根 K 棒 (-2) 的值，用於判斷「穿越」事件
        prev_macd = macd_line.iloc[-2]
        curr_macd = macd_line.iloc[-1]
        prev_signal = signal_line.iloc[-2]
        curr_signal = signal_line.iloc[-1]
        
        # 獲取當前倉位狀態
        pos = ctx.get_position(ctx.symbol)

        # 4. 定義交易條件
        
        # --- 開倉條件 ---
        # 黃金交叉: 快線從下方上穿慢線
        is_golden_cross = (prev_macd <= prev_signal) and (curr_macd > curr_signal)
        # 死亡交叉: 快線從上方下穿慢線
        is_death_cross = (prev_macd >= prev_signal) and (curr_macd < curr_signal)

        # 條件過濾: 是否在 0 軸之上或之下
        is_above_zero = curr_signal > 0
        is_below_zero = curr_signal < 0

        # --- 平倉條件 ---
        # 多單平倉: MACD 線由上往下跌破 0 軸
        is_exit_long_signal = (prev_macd >= 0) and (curr_macd < 0)
        # 空單平倉: MACD 線由下往上漲破 0 軸
        is_exit_short_signal = (prev_macd <= 0) and (curr_macd > 0)

        # 5. 執行交易指令
        # 只有在無倉位 (is_flat) 時才考慮開倉
        if pos.is_flat():
            if is_golden_cross and is_above_zero:
                ctx.open_long()
            elif is_death_cross and is_below_zero:
                ctx.open_short()
        
        # 如果持有多單，只監控多單的平倉條件
        elif pos.is_long():
            if is_exit_long_signal:
                ctx.close_long()

        # 如果持有空單，只監控空單的平倉條件
        elif pos.is_short():
            if is_exit_short_signal:
                ctx.close_short()