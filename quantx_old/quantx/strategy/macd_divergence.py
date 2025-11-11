# 檔案: quantx/strategy/macd_divergence.py
# 版本: v2 (改用 ta.ATR)
# 說明: 
# - 實現基於 YouTube 影片的 "半木夏" 高階 MACD 背離策略。
# - 策略使用 MACD(13, 34) 和 ATR(13)。
# - 偵測 MACD Histogram 與價格的背離。
# - 使用基於 ATR 的動態停損和基於風險回報比 (R:R) 的停利。
# - 由於框架限制，SL/TP 邏輯在 on_bar 內部手動管理。

from __future__ import annotations
from quantx.core.context import ContextBase
from quantx.core.executor.base import BaseExecutor
from quantx.ta import indicators as ta # 引入ta指標
import pandas as pd
import numpy as np

class macd_divergence(BaseExecutor):
    """
    半木夏 MACD 背離策略 (多空雙向)
    - 底背離 (Long): MACD 零軸下波谷抬高 + 價格低點降低。
    - 頂背離 (Short): MACD 零軸上波峰降低 + 價格高點抬高。
    - 進場: 背離發生後，MACD Histogram 開始收縮時進場。
    - SL: ATR 
    - TP: R:R
    """

    params = {
        "macd_fast": 13,
        "macd_slow": 34,
        "macd_signal": 9,
        "atr_len": 13,
        "rr_ratio": 2.0,       # 風險回報比 (Risk/Reward Ratio)
        "peak_window": 5,      # 尋找波峰/谷的左右窗口大小
        "min_div_peaks": 2,    # 至少要比較的波峰/谷數量 (固定為2)
        "min_data_len": 60     # 至少需要多少數據才開始
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 用於儲存動態 SL/TP 價格
        self.sl_price = 0.0
        self.tp_price = 0.0
        self.default_stop_loss_pct = 1
        self.default_take_profit_pct = 2

    # 🟢 移除內部的 _calculate_atr 函數 (已移至 indicators.py)

    def _find_recent_peaks_valleys(self, series: pd.Series, window: int, min_peaks: int):
        """
        簡易實現尋找最近的波峰/谷 (使用 rolling，避免 scipy 依賴)
        window: 左右窗口大小
        min_peaks: 至少需要 N 個波峰/谷
        """
        N = window
        # 使用 center=True 確保找到的是局部的真正高/低點
        roll_max = series.rolling(N*2+1, center=True, min_periods=1).max()
        roll_min = series.rolling(N*2+1, center=True, min_periods=1).min()
        
        # 波峰 (必須 > 0)
        is_peak = (series == roll_max) & (series > 0)
        # 篩選掉平台期（只取第一個點）
        peak_indices = series.index[is_peak & (is_peak != is_peak.shift(1))]
        
        # 波谷 (必須 < 0)
        is_valley = (series == roll_min) & (series < 0)
        # 篩選掉平台期
        valley_indices = series.index[is_valley & (is_valley != is_valley.shift(1))]
        
        recent_peaks = peak_indices[-min_peaks:] if len(peak_indices) >= min_peaks else []
        recent_valleys = valley_indices[-min_peaks:] if len(valley_indices) >= min_peaks else []
        
        return recent_peaks, recent_valleys

    def on_bar(self, ctx: ContextBase):
        # 1. 獲取參數
        p = self.params
        macd_fast, macd_slow, macd_signal = int(p["macd_fast"]), int(p["macd_slow"]), int(p["macd_signal"])
        atr_len = int(p["atr_len"])
        rr_ratio = float(p["rr_ratio"])
        peak_window = int(p["peak_window"])
        min_div_peaks = int(p["min_div_peaks"])
        min_data_len = int(p["min_data_len"])

        # 2. 獲取數據
        df = ctx.data(ctx.symbol, ctx.tf)
        if len(df) < min_data_len:
            return

        close, high, low = df["close"], df["high"], df["low"]
        
        # 3. 計算指標
        # 使用 ta.MACD
        macd_df = ta.MACD(close, fast=macd_fast, slow=macd_slow, signal=macd_signal)
        histogram = macd_df['MACD_histogram']
        
        # 🟢 修正：直接呼叫 ta.ATR
        atr = ta.ATR(high, low, close, length=atr_len)
        
        # 獲取最新值
        curr_hist = histogram.iloc[-1]
        prev_hist = histogram.iloc[-2]
        curr_atr = atr.iloc[-1]
        curr_price_close = close.iloc[-1]
        curr_price_high = high.iloc[-1]
        curr_price_low = low.iloc[-1]
        
        # 確保 ATR 已收斂
        if pd.isna(curr_atr) or curr_atr == 0:
            return

        # 4. 獲取倉位
        pos = ctx.get_position(ctx.symbol)

        # --- 狀態重置 (如果已平倉，則清除 SL/TP 價格) ---
        if pos.is_flat() and (self.sl_price != 0 or self.tp_price != 0):
            self.sl_price = 0.0
            self.tp_price = 0.0

        # --- 5. 檢查平倉 (如果持倉) ---
        # 模擬 K 線內的 SL/TP 觸發
        if pos.is_long():
            # 檢查 SL (優先)
            if curr_price_low <= self.sl_price:
                ctx.close_long()
            # 檢查 TP
            elif curr_price_high >= self.tp_price:
                ctx.close_long()
            return # 持倉時不檢查開倉
            
        elif pos.is_short():
            # 檢查 SL (優先)
            if curr_price_high >= self.sl_price:
                ctx.close_short()
            # 檢查 TP
            elif curr_price_low <= self.tp_price:
                ctx.close_short()
            return # 持倉時不檢查開倉

        # --- 6. 檢查開倉 (僅在空倉時執行) ---
        
        # 尋找最近的 N 個波峰/谷 (這是昂貴操作，只在空倉時執行)
        recent_peaks, recent_valleys = self._find_recent_peaks_valleys(
            histogram, peak_window, min_div_peaks
        )

        # 檢查做多 (底背離)
        # 觸發條件: 柱狀圖在零軸下開始縮短 (回升)
        if (curr_hist > prev_hist) and (curr_hist < 0):
            if len(recent_valleys) >= min_div_peaks:
                idx_v1 = recent_valleys[-2] # 前一個波谷
                idx_v2 = recent_valleys[-1] # 最後一個波谷
                
                # 條件1: MACD 谷底抬高
                cond1_macd = histogram[idx_v2] > histogram[idx_v1]
                # 條件2: 價格低點降低
                cond2_price = low[idx_v2] < low[idx_v1]
                
                if cond1_macd and cond2_price:
                    # 計算 SL 和 TP
                    sl = curr_price_low - curr_atr
                    tp = curr_price_close + (curr_price_close - sl) * rr_ratio
                    
                    # 儲存狀態並開倉
                    self.sl_price = sl
                    self.tp_price = tp
                    ctx.open_long()
                    return # 開倉後結束

        # 檢查做空 (頂背離)
        # 觸發條件: 柱狀圖在零軸上開始縮短 (回落)
        if (curr_hist < prev_hist) and (curr_hist > 0):
            if len(recent_peaks) >= min_div_peaks:
                idx_p1 = recent_peaks[-2] # 前一個波峰
                idx_p2 = recent_peaks[-1] # 最後一個波峰
                
                # 條件1: MACD 峰頂降低
                cond1_macd = histogram[idx_p2] < histogram[idx_p1]
                # 條件2: 價格高點抬高
                cond2_price = high[idx_p2] > high[idx_p1]
                
                if cond1_macd and cond2_price:
                    # 計算 SL 和 TP
                    sl = curr_price_high + curr_atr
                    tp = curr_price_close - (sl - curr_price_close) * rr_ratio
                    
                    # 儲存狀態並開倉
                    self.sl_price = sl
                    self.tp_price = tp
                    ctx.open_short()
                    return # 開倉後結束