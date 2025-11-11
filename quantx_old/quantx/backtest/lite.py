# -*- coding: utf-8 -*-
# quantx/backtest/lite.py
from __future__ import annotations
import numpy as np
import pandas as pd
from quantx.core.timeframe import parse_tf_minutes

class LiteBacktester:
    """
    極簡、向量化：輸入 close、signal ∈ {-1,0,1}，回傳績效指標（含成本）。
    假設單向滿倉（或固定權重），換向/進出場才產生成本。
    """
    def __init__(
        self,
        fee_bps: float = 5.5,
        slip_bps: float = 1.0,
        max_gross_leverage: float = 1.0,
        tf: str | None = None,
        tf_minutes: int | None = None,
        annualize_days: int = 365,
    ):
        self.fee = fee_bps * 1e-4
        self.slip = slip_bps * 1e-4
        self.leverage = max_gross_leverage

        if tf_minutes is None:
            if tf is None:
                raise ValueError("Either tf or tf_minutes must be provided.")
            tf_minutes = parse_tf_minutes(tf)
        self.tf_minutes = int(tf_minutes)
        self.per_year = int(annualize_days * (1440 // self.tf_minutes))


    def run(self, close: pd.Series, signal: pd.Series) -> dict:
        if not close.index.equals(signal.index):
            # 確保對齊（取交集並排序）
            idx = close.index.intersection(signal.index)
            close = close.loc[idx]
            signal = signal.loc[idx]
        px = close.values.astype(float)
        sig = signal.values.astype(float)

        # 前一時點持倉，用於計算換倉
        prev = np.roll(sig, 1)
        prev[0] = 0.0

        # 報酬（簡化：收/收）
        ret = np.zeros_like(px)
        ret[1:] = (px[1:] / px[:-1] - 1.0)

        # 交易成本：持倉變動的絕對值 * (fee+slip)
        turnover = np.abs(sig - prev)
        cost = turnover * (self.fee + self.slip)

        # 策略報酬：持倉 * 基礎報酬 - 成本
        strat_r = sig * ret * self.leverage - cost

        eq = (1.0 + strat_r).cumprod()
        dd = eq / np.maximum.accumulate(eq) - 1.0
        mdd = float(dd.min())

        # 年化 Sharpe（簡化，假設 30m 則日 48 根，年 ~ 252 日）
        mu = strat_r.mean()
        sd = strat_r.std(ddof=1) + 1e-12
        sr = float((mu / sd) * np.sqrt(self.per_year))

        trades = int(np.count_nonzero(turnover > 0))
        return {
            "sr": sr,
            "mdd": mdd,
            "trades": trades,
            "curve": pd.Series(eq, index=close.index),
        }
