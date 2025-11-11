# quantx/core/features/smc_features.py
# -*- coding: utf-8 -*-
# 版本: v2 (參數配置化)
# 說明:
# - 將 SMC 特徵生成器的所有閾值和窗口長度參數化，從 self.cfg 中讀取配置。

import pandas as pd
import numpy as np
from .base import FeatureBase
from typing import Dict, Any, Optional

class SMCFeatures(FeatureBase):
    """
    Smart Money Concepts (SMC) 特徵生成器 (參數化)
    -----------------------------------
    功能：
      - 結構方向 (Structure Direction)
      - 結構突破 (MSB/BOS)
      - Order Block (OB) 區間距離
      - Fair Value Gap (FVG)
      - Liquidity Sweep (掃流)
      - 多時間框共振 (HTF Alignment)
    """
    
    DEFAULT_CFG = {
        "msb_window": 2,          # MSB 判斷所需的 K 棒回溯數 (例如：2 根 K 棒前的 H/L)
        "msb_confirm_count": 2,   # 連續 MSB 訊號確認數
    }

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        """
        初始化 SMCFeatures。
        
        Args:
            cfg (dict, optional): 來自 train.yaml 的 smc_features_params 區塊設定。
        """
        super().__init__(cfg)
        
        # 🟢 合併配置與預設值
        self.config = self.DEFAULT_CFG.copy()
        if self.cfg:
            self.config.update(self.cfg)
        
        # 確保所有參數都是整數/浮點數
        self.msb_window = int(self.config.get("msb_window"))
        self.msb_confirm_count = int(self.config.get("msb_confirm_count"))

    # === 結構方向 ===
    def _calc_structure_direction(self, df: pd.DataFrame) -> pd.Series:
        """計算與前一根 K 棒相比的結構方向 (-1, 0, 1)"""
        dir_val = np.where(
            (df["high"] > df["high"].shift(1)) & (df["low"] > df["low"].shift(1)), 1,
            np.where(
                (df["high"] < df["high"].shift(1)) & (df["low"] < df["low"].shift(1)), -1, 0
            ),
        )
        return pd.Series(dir_val, index=df.index, name="smc_dir")

    # === 結構突破 MSB/BOS ===
    def _calc_msb(self, df: pd.DataFrame) -> pd.Series:
        """計算市場結構突破 (Market Structure Break) 訊號 (-1, 0, 1)"""
        w = self.msb_window
        # 突破前 w 根 K 棒的高點 (向上突破)
        up = (df["close"] > df["high"].shift(w)).astype(int)
        # 跌破前 w 根 K 棒的低點 (向下突破)
        down = (df["close"] < df["low"].shift(w)).astype(int)
        return pd.Series(up - down, index=df.index, name="msb_dir")

    # === Order Block (OB) 距離 ===
    def _calc_order_block(self, df: pd.DataFrame) -> pd.Series:
        """計算當前收盤價距離前一根 K 棒 (潛在 OB) 頂部的相對距離。"""
        ob_high = df["high"].shift(1)
        ob_low = df["low"].shift(1)
        # 避免除以零
        range_ = ob_high - ob_low
        range_ = range_.replace(0, np.nan).fillna(1e-9) 
        ob_distance = (df["close"] - ob_high) / range_
        return pd.Series(ob_distance, index=df.index, name="ob_distance")

    # === Fair Value Gap (FVG) ===
    def _calc_fvg(self, df: pd.DataFrame) -> pd.Series:
        """計算是否存在 Fair Value Gap (效率缺口) 訊號。"""
        # 上漲 FVG (看多): 當前 K 棒的 low > 前 2 根 K 棒的 high
        fvg_up = ((df["low"] > df["high"].shift(2))).astype(int)
        # 下跌 FVG (看空): 當前 K 棒的 high < 前 2 根 K 棒的 low
        fvg_down = ((df["high"] < df["low"].shift(2))).astype(int)
        return pd.Series(fvg_up - fvg_down, index=df.index, name="fvg_dir")

    # === Liquidity Sweep (掃流) ===
    def _calc_sweep(self, df: pd.DataFrame) -> pd.Series:
        """計算是否存在 Liquidity Sweep (流動性掃蕩) 訊號。"""
        # 向上掃流: 突破前高但收盤價低於前高
        sweep_up = (
            (df["high"] > df["high"].shift(1)) & (df["close"] < df["high"].shift(1))
        ).astype(int)
        # 向下掃流: 跌破前低但收盤價高於前低
        sweep_down = (
            (df["low"] < df["low"].shift(1)) & (df["close"] > df["low"].shift(1))
        ).astype(int)
        return pd.Series(sweep_up - sweep_down, index=df.index, name="sweep_dir")

    # === 多時間框共振 ===
    def _calc_dir_alignment(self, df: pd.DataFrame, htf_df: pd.DataFrame | None) -> pd.Series:
        """計算當前結構方向是否與 HTF 的結構方向一致。"""
        if htf_df is None or "smc_dir" not in htf_df.columns:
            return pd.Series(0, index=df.index, name="dir_align")
        
        # 將 HTF 的 smc_dir 重新索引到當前 TF (ffill 確保數據可用)
        htf_dir = htf_df["smc_dir"].reindex(df.index, method="ffill")
        # 檢查當前 TF 的 smc_dir 是否與 HTF 一致
        return pd.Series((df["smc_dir"] == htf_dir).astype(int), index=df.index, name="dir_align")

    # === 主函式 ===
    def transform(self, df: pd.DataFrame, htf_df: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        輸入 K 線資料 (open, high, low, close)
        回傳包含 SMC 特徵的 DataFrame
        """
        df = df.copy()

        df["smc_dir"] = self._calc_structure_direction(df)
        df["msb_dir"] = self._calc_msb(df)
        df["ob_distance"] = self._calc_order_block(df)
        df["fvg_dir"] = self._calc_fvg(df)
        df["sweep_dir"] = self._calc_sweep(df)
        
        # 計算 HTF 對齊 (需要先在 HTF 上計算 SMC 方向)
        if htf_df is not None:
             # 在 HTF 數據上執行 SMC 結構判斷
            htf_df["smc_dir"] = self._calc_structure_direction(htf_df)
        
        df["dir_align"] = self._calc_dir_alignment(df, htf_df)

        # 🟢 結構確認強度 (使用配置的確認數)
        confirm_count = self.msb_confirm_count
        df["msb_confirmed"] = (df["msb_dir"].rolling(confirm_count).sum().abs() >= confirm_count).astype(int)
        
        df["structure_strength"] = (
            df["fvg_dir"].abs() + df["sweep_dir"].abs() + df["msb_confirmed"]
        )

        return df[
            ["smc_dir", "msb_dir", "ob_distance", "fvg_dir", "sweep_dir", "dir_align", "msb_confirmed", "structure_strength"]
        ]