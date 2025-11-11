# quantx/core/features/main_features.py
# -*- coding: utf-8 -*-
# 版本: v3 (特徵參數化集成)
# 說明:
# - 將傳入的配置 (cfg) 依照子特徵的需要進行切片 (例如：ta_features_params)。
# - 這確保了每個特徵產生器只接收和使用其相關的參數，提高模組的內聚性。

import pandas as pd
from .base import FeatureBase
from .basic_features import BasicFeatures
from .ta_features import TAFeatures
from .smc_features import SMCFeatures

class MainFeatures(FeatureBase):
    """
    特徵總指揮官 (Main Feature Generator)
    
    負責協調所有子特徵生成器 (Basic, TA, SMC)，並處理多時間框 (HTF) 數據的聚合。
    """

    def __init__(self, cfg=None):
        """
        初始化 MainFeatures。
        
        Args:
            cfg (dict, optional): 來自 train.yaml 的 features 區塊設定。
        """
        super().__init__(cfg)
        
        # 🟢 根據設定檔切分並初始化子產生器
        # 確保每個產生器只收到它需要的參數 slice
        ta_cfg = self.cfg.get('ta_features_params', {})
        smc_cfg = self.cfg.get('smc_features_params', {})
        basic_cfg = self.cfg.get('basic_features_params', {})
        
        self.basic_gen = BasicFeatures(basic_cfg)
        self.ta_gen = TAFeatures(ta_cfg)
        self.smc_gen = SMCFeatures(smc_cfg) 

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成所有特徵並合併。

        Args:
            df (pd.DataFrame): 原始 OHLCV 數據 (已是目標時間框)。

        Returns:
            pd.DataFrame: 包含所有特徵的 DataFrame。
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("傳入的 DataFrame 必須使用 DatetimeIndex。")

        # 1. 生成 Basic 和 TA 特徵
        basic_features = self.basic_gen.transform(df)
        ta_features = self.ta_gen.transform(df)
        
        # 2. 處理 SMC 特徵 (需要 HTF 數據)
        
        # 從配置中讀取 HTF 週期
        htf_tf = self.cfg.get('htf_timeframe', '4h')
        
        # 聚合為更高時間週期
        htf_df = df.resample(htf_tf).agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
        
        # 生成 SMC 特徵
        smc_features = self.smc_gen.transform(df, htf_df=htf_df)

        # 3. 合併與清理
        final_df = pd.concat([basic_features, ta_features, smc_features], axis=1)
        final_df = final_df.dropna()

        return final_df