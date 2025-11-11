# quantx/core/features/base.py
"""
Feature 模組基底
- 統一輸入: DataFrame (OHLCV)
- 輸出: DataFrame (加上新特徵欄位)
"""

import pandas as pd
from abc import ABC, abstractmethod

class FeatureBase(ABC):
    def __init__(self, cfg=None):
        self.cfg = cfg or {}

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """輸入 df(OHLCV)，輸出包含新特徵的 df"""
        pass
