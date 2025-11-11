# quantx/core/risk.py
# 檔案: quantx/core/risk.py
# 版本: v4 (最終修復：增強字串匹配健壯性 & 處理全稱)
# 說明:
# - 修復了 size_mode 傳遞 'percent_equity' 時，與 Literal 'pct_equity' 不匹配的錯誤。
# - 確保 mode 字符串在比較前被清理並處理全稱。

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


QtyMode = Literal["fixed_qty", "notional", "pct_equity"]


@dataclass
class RiskConfig:
    """Configuration for calculating order size."""

    size_mode: QtyMode
    leverage: float = 1.0
    risk_pct: float = 0.01
    max_notional: float = 0.0
    min_qty: float = 0.0


def compute_order_size(
    equity: float,
    price: float,
    config: RiskConfig,
) -> float:
    """Compute the order quantity based on the risk configuration.

    Parameters
    ----------
    equity : float
        Current account equity.
    price : float
        Current price of the instrument.
    config : RiskConfig
        Risk configuration.

    Returns
    -------
    float
        Quantity to trade.
    """
    
    # 核心修正 1：確保 mode 字符串是乾淨的，移除空格並轉為小寫
    raw_mode = config.size_mode.strip().lower()
    
    # 🟢 核心修正 2：標準化全稱到縮寫 Literal
    if 'percent_equity' in raw_mode:
        mode = 'pct_equity'
    else:
        mode = raw_mode
        
    value = config.risk_pct
    leverage = config.leverage or 1.0
    
    if mode == "fixed_qty":
        return float(value)
    if mode == "notional":
        # Notional amount in terms of quote currency (e.g. USD)
        return float(value) / price
    if mode == "pct_equity":
        # Percentage of equity per trade, optionally leveraged
        notional = equity * value * leverage
        return float(notional) / price
    
    # 如果 mode 是正確的，執行流程不會到達這裡
    raise ValueError(f"Unsupported quantity mode: {mode}")