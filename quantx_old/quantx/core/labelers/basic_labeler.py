# quantx/core/labelers/basic_labeler.py
"""
Labeler 模組 - 基本三分類
0 = 下跌
1 = 盤整
2 = 上漲
"""

import pandas as pd
import numpy as np

class ThreeClassLabeler:
    def __init__(self, horizon: int = 10, threshold: float = 0.01):
        """
        Args:
            horizon (int): 預測未來幾根 K 線
            threshold (float): 漲跌幅門檻，例如 0.01 = 1%
        """
        self.horizon = horizon
        self.threshold = threshold

    def label(self, df: pd.DataFrame) -> pd.Series:
        """
        產生三分類標籤
        0 = 下跌, 1 = 盤整, 2 = 上漲
        """
        df = df.copy()
        future_close = df["close"].shift(-self.horizon)
        future_return = (future_close - df["close"]) / df["close"]

        labels = np.where(
            future_return > self.threshold, 2,  # 上漲
            np.where(future_return < -self.threshold, 0, 1)  # 下跌 / 盤整
        )

        return pd.Series(labels, index=df.index, name="label").dropna()
