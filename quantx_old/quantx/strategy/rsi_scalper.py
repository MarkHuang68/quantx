# 檔案: quantx/strategy/rsi_scalper.py
# 版本: v3 (Context 接口修復版)
# 說明:
# - 修正策略以適應新的 Context 接口，移除不必要的 order_type 和 price 參數。
# - 交易執行策略 (Maker 優先) 已被底層的 LiveTradeManager/TradeManager 接管。

from __future__ import annotations
import pandas as pd
from quantx.core.context import ContextBase
from quantx.core.executor.base import BaseExecutor
from quantx.ta import indicators as ta

class rsi_scalper(BaseExecutor):
    """
    RSI 剝頭皮策略 (多空雙向)
    1. BBW < 閾值 => 認定為盤整，允許交易
    2. RSI < 超賣區 / RSI > 超買區 => 開倉訊號
    3. 開倉意圖：發出買入/賣出信號，底層 LiveTradeManager 會強制使用 Maker 優先政策執行。
    4. RSI 回到中線 => 平倉 (Market Order)
    """
    params = {
        "rsi_len": 14,
        "bb_len": 20,
        "bb_std": 2.0,
        "bbw_threshold": 0.01,         # 布林帶寬度需小於 1%
        "oversold": 30,
        "overbought": 70,
        "exit_level": 50,
        "entry_spread_pct": 0.0005,    # (保留參數) Maker 掛單價格參考價差
    }

    def on_bar(self, ctx: ContextBase) -> None:
        length = int(self.params["bb_len"])
        df = ctx.data(ctx.symbol, ctx.tf)
        if len(df) < length:
            return

        close = df["close"]
        
        # --- 1. 計算指標 ---
        rsi = ta.rsi(close, int(self.params["rsi_len"]))
        
        sma = close.rolling(window=length).mean()
        sd = close.rolling(window=length).std(ddof=0)
        
        upper = sma + self.params["bb_std"] * sd
        lower = sma - self.params["bb_std"] * sd
        
        # 計算布林帶寬度百分比
        bbw = (upper - lower) / sma
        
        # 獲取當前 K 棒的指標值
        px_now = close.iloc[-1]
        curr_rsi = rsi.iloc[-1]
        curr_bbw = bbw.iloc[-1]
        
        pos = ctx.get_position(ctx.symbol)

        # --- 2. 判斷是否允許交易 ---
        is_ranging = curr_bbw < self.params["bbw_threshold"]

        # --- 3. 執行交易邏輯 ---
        
        # 4. 離場邏輯 (發出平倉意圖，TradeManager 將執行 Taker/Market)
        # 多單離場：RSI 回到中線
        if pos.is_long() and curr_rsi >= self.params["exit_level"]:
            ctx.close_long()
            
        # 空單離場：RSI 回到中線
        elif pos.is_short() and curr_rsi <= self.params["exit_level"]:
            ctx.close_short()
            
        # 3. 開倉邏輯 (發出開倉意圖，TradeManager 將執行 Maker 優先政策)
        if is_ranging and pos.is_flat():
            
            # 開多倉條件：盤整期 + 無倉位 + RSI 進入超賣區
            if curr_rsi < self.params["oversold"]:
                # 策略意圖：開多倉
                # 🟢 修正：移除 order_type/price 參數，交由 TradeManager 執行 Maker 優先
                ctx.open_long()
            
            # 開空倉條件：盤整期 + 無倉位 + RSI 進入超買區
            elif curr_rsi > self.params["overbought"]:
                # 策略意圖：開空倉
                # 🟢 修正：移除 order_type/price 參數，交由 TradeManager 執行 Maker 優先
                ctx.open_short()