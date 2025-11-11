# 檔案: quantx/core/features/basic_features.py
# 版本: v4 (極簡版)
# 說明:
# - 根據新的架構設計，此檔案只負責生成最基礎的特徵。
# - 目前只包含一個 regime 特徵，用於表示與前一根K棒的漲跌關係。

import pandas as pd
import numpy as np
from .base import FeatureBase

class BasicFeatures(FeatureBase):
    """
    基礎特徵產生器
    - ohlcv (由呼叫端傳入)
    - regime: 與前一根K棒的漲跌關係 (-1, 0, 1)
    """

    def __init__(self, cfg=None):
        super().__init__(cfg)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not set(["open", "high", "low", "close", "volume"]).issubset(df.columns):
            raise ValueError("DataFrame 缺少必要的 OHLCV 欄位")

        features = pd.DataFrame(index=df.index)
        
        # 計算 regime: 1 為上漲, -1 為下跌, 0 為平
        features['regime'] = (df['close'] > df['close'].shift(1)).astype(int) - \
                             (df['close'] < df['close'].shift(1)).astype(int)
        
        return features