"""
base.py
=======
Labeler 抽象基底
所有標籤生成模組需繼承此基底，統一 API。
"""

from abc import ABC, abstractmethod
import pandas as pd


class LabelBase(ABC):
    def __init__(self, cfg=None):
        self.cfg = cfg or {}

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.Series:
        """
        根據 df 建立標籤 (Series)
        """
        pass
