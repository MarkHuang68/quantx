# 檔案: quantx/core/labelers/future_return.py
# 版本: v2 (整合 Triple-Barrier Method)
# 說明:
# - 從舊的 quantx/ml/labelers.py 中遷移了 make_labels_triple_barrier 函式。
# - 現在所有基於未來價格的標籤方法都集中在此檔案中，方便統一管理。

import pandas as pd
import numpy as np
from math import ceil
from .base import LabelBase
from quantx.core.timeframe import parse_tf_minutes

class FutureReturnBinary(LabelBase):
    """
    二分類標籤: 未來報酬率是否超過一個固定閾值。
    """
    def __init__(self, horizon=10, threshold=0.01, cfg=None):
        super().__init__(cfg)
        self.horizon = horizon
        self.threshold = threshold

    def transform(self, df: pd.DataFrame) -> pd.Series:
        future = df["close"].shift(-self.horizon)
        ret = (future - df["close"]) / df["close"]
        labels = (ret > self.threshold).astype(int)
        return labels

class FutureReturnTriple(LabelBase):
    """
    三分類標籤: 根據固定的上下閾值，將未來報酬率分為上漲、盤整、下跌。
    """
    def __init__(self, horizon=10, up=0.01, down=-0.01, cfg=None):
        super().__init__(cfg)
        self.horizon = horizon
        self.up = up
        self.down = down

    def transform(self, df: pd.DataFrame) -> pd.Series:
        future = df["close"].shift(-self.horizon)
        ret = (future - df["close"]) / df["close"]

        def classify(x):
            if x > self.up:
                return 2  # 上漲
            elif x < self.down:
                return 0  # 下跌
            else:
                return 1  # 盤整

        labels = ret.apply(classify)
        return labels

# 🟢 === 從 quantx/ml/labelers.py 遷移過來的核心函式 ===
def make_labels_triple_barrier(df: pd.DataFrame,
                               tf: str,
                               max_hours: float = 8.0,
                               atr_n: int = 14,
                               up_k: float = 1.5,
                               dn_k: float = 1.5) -> pd.DataFrame:
    """
    三重關卡標籤法 (Triple-Barrier Method):
      - 上軌 (止盈): 基於 ATR 動態計算。
      - 下軌 (止損): 基於 ATR 動態計算。
      - 時間關卡: 最長持倉時間。
    
    回傳的 DataFrame 會包含一個 'y' 欄位: 0=空, 1=盤整, 2=多。
    """
    out = df.copy()
    if "close" not in out.columns:
        out["close"] = out.iloc[:, 0].astype(float)

    # 計算 ATR (平均真實波幅)
    if all(c in out.columns for c in ("high", "low", "close")):
        tr = np.maximum(out["high"] - out["low"],
                        np.maximum((out["high"] - out["close"].shift(1)).abs(),
                                   (out["low"] - out["close"].shift(1)).abs()))
    else:
        # 如果缺少 H/L，則用收盤價變動來近似
        tr = (out["close"] - out["close"].shift(1)).abs()
        
    atr = tr.ewm(span=atr_n, adjust=False).mean()
    atr_pct = (atr / out["close"]).clip(lower=1e-6)

    # 計算時間關卡 (將小時轉換為 K 棒數量)
    tf_minutes = parse_tf_minutes(tf)
    max_holding_bars = max(int(ceil(max_hours * 60.0 / tf_minutes)), 2)

    # 計算動態的止盈 (上軌) 和止損 (下軌)
    upper_barrier = out["close"] * (1.0 + up_k * atr_pct)
    lower_barrier = out["close"] * (1.0 - dn_k * atr_pct)

    labels = np.full(len(out), 1, dtype="int8")  # 預設所有標籤為 1 (盤整)
    high_prices = out["high"].values
    low_prices = out["low"].values

    for i in range(len(out) - max_holding_bars):
        # 確定未來要檢查的區間
        future_slice = slice(i + 1, i + max_holding_bars + 1)
        
        # 檢查未來價格是否觸碰到上軌或下軌
        hit_upper = (high_prices[future_slice] >= upper_barrier.iloc[i]).any()
        hit_lower = (low_prices[future_slice] <= lower_barrier.iloc[i]).any()
        
        if hit_upper and hit_lower:
            # 如果同時觸碰到，以先到者為準
            first_hit_upper_idx = np.argmax(high_prices[future_slice] >= upper_barrier.iloc[i])
            first_hit_lower_idx = np.argmax(low_prices[future_slice] <= lower_barrier.iloc[i])
            if first_hit_upper_idx < first_hit_lower_idx:
                labels[i] = 2 # 先碰到上軌
            else:
                labels[i] = 0 # 先碰到下軌
        elif hit_upper:
            labels[i] = 2 # 只碰到上軌
        elif hit_lower:
            labels[i] = 0 # 只碰到下軌
        # else: 維持預設的 1 (盤整)
            
    out["y"] = labels
    return out