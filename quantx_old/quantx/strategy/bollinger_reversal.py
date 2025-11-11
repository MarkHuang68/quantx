# 檔案: quantx/strategy/bollinger_reversal.py
# 版本: v5 (最終修正：參數名稱與防禦性編程)
# 說明:
# - 統一策略內部參數名稱為 'stddev'，匹配 train.yaml 和 bt_run 的輸入。
# - 增加了 SD 非零檢查，排除極端平靜或數據錯誤導致的零標準差。

from __future__ import annotations
import pandas as pd
import numpy as np
from quantx.core.context import ContextBase
from quantx.core.executor.base import BaseExecutor

class bollinger_reversal(BaseExecutor):
    """
    布林帶均值回歸策略 (多空雙向)
    - 價格觸碰下軌 -> 做多
    - 價格觸碰上軌 -> 做空
    - 價格回歸中軌 (SMA) -> 平倉
    """
    # 🟢 修正：確保策略的參數定義使用 'stddev'
    params = {"length": 20, "stddev": 2.0} 

    def on_bar(self, ctx: ContextBase) -> None:
        length = int(self.params["length"])
        # 🟢 讀取 'stddev' 參數
        k = float(self.params["stddev"])
        
        df = ctx.data(ctx.symbol, ctx.tf)
        # 確保有足夠的數據來計算 SMA 和 SD
        if len(df) < length:
            return

        # 確保數據是 float
        close = pd.to_numeric(df["close"], errors="coerce")
        
        # 檢查最新的收盤價是否有效
        px_now = close.iloc[-1]
        if pd.isna(px_now):
            return

        # 計算 SMA 和 SD
        sma = close.rolling(window=length).mean()
        # 這裡的 ddof=0 保持不變，與您原本使用的指標計算一致
        sd = close.rolling(window=length).std(ddof=0)

        # 獲取當前 K 棒的指標值
        sma_now = sma.iloc[-1]
        sd_now = sd.iloc[-1]
        
        # 檢查指標是否有效
        if pd.isna(sma_now) or pd.isna(sd_now):
            return
        
        # 🟢 防禦性編程：標準差必須大於零
        if sd_now <= 1e-9: 
            return

        upper_now = sma_now + k * sd_now
        lower_now = sma_now - k * sd_now
        
        pos = ctx.get_position(ctx.symbol)

        # === 做多邏輯 ===
        # 開倉條件：無倉位且價格觸碰或跌破下軌
        if pos.is_flat() and px_now <= lower_now:
            ctx.open_long()
        # 平倉條件：持有多單且價格回歸至中軌上方
        elif pos.is_long() and px_now >= sma_now:
            ctx.close_long()

        # === 做空邏輯 ===
        # 開倉條件：無倉位且價格觸碰或突破上軌
        if pos.is_flat() and px_now >= upper_now:
            ctx.open_short()
        # 平倉條件：持有空單且價格回歸至中軌下方
        elif pos.is_short() and px_now <= sma_now:
            ctx.close_short()