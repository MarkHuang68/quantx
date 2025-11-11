# quantx/core/features/ta_features.py
# -*- coding: utf-8 -*-
# 版本: v5 (參數配置化)
# 說明:
# - 將技術指標的窗口長度 (window lengths) 參數化，從 self.cfg 中讀取配置。
# - 這使得特徵生成器可以與 train.yaml 中的 feature 配置相容。

import pandas as pd
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from .base import FeatureBase # 引用 FeatureBase，它包含了 __init__ 和 self.cfg

class TAFeatures(FeatureBase):
    """
    強化版基礎特徵 (參數化)
    - 允許透過設定檔配置 EMA/RSI/MACD/ATR/ADX 的窗口長度。
    """
    
    # 預設參數 (如果設定檔中沒有提供)
    DEFAULT_CFG = {
        "ema_fast": 10,
        "ema_slow": 30,
        "rsi_len": 14,
        "adx_len": 14,
        "atr_len": 14,
        "roc_short": 5,
        "roc_long": 10,
    }

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        根據配置檔中的參數生成技術指標特徵。
        
        Args:
            df (pd.DataFrame): 包含 OHLCV 數據的 DataFrame。
            
        Returns:
            pd.DataFrame: 包含參數化技術指標特徵的 DataFrame。
        """
        df = df.copy()
        
        # 🟢 從配置中讀取或使用預設值
        cfg = self.DEFAULT_CFG.copy()
        cfg.update(self.cfg)

        # --- EMA (趨勢) ---
        ema_fast_len = int(cfg.get("ema_fast"))
        ema_slow_len = int(cfg.get("ema_slow"))
        
        ema_fast = EMAIndicator(df["close"], window=ema_fast_len).ema_indicator()
        ema_slow = EMAIndicator(df["close"], window=ema_slow_len).ema_indicator()
        
        df[f"ema_fast_{ema_fast_len}"] = ema_fast
        df[f"ema_slow_{ema_slow_len}"] = ema_slow
        df["ema_diff"] = ema_fast - ema_slow           # 快慢差值
        df["ema_cross"] = (ema_fast > ema_slow).astype(int)

        # --- RSI (動能) ---
        rsi_len = int(cfg.get("rsi_len"))
        rsi = RSIIndicator(df["close"], window=rsi_len).rsi()
        
        df[f"rsi_{rsi_len}"] = rsi
        df["rsi_norm"] = (rsi - 50) / 50               # 相對強弱（-1~+1）

        # --- MACD (動能/趨勢) ---
        # MACD 預設窗口為 12/26/9，這裡使用 ta 庫的預設
        macd = MACD(df["close"])
        df["macd_line"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_diff"] = macd.macd_diff()

        # --- ADX (趨勢強度) ---
        adx_len = int(cfg.get("adx_len"))
        adx = ADXIndicator(df["high"], df["low"], df["close"], window=adx_len)
        
        df[f"adx_{adx_len}"] = adx.adx()
        df[f"pdi_{adx_len}"] = adx.adx_pos()
        df[f"ndi_{adx_len}"] = adx.adx_neg()
        df["trend_strength"] = df[f"pdi_{adx_len}"] - df[f"ndi_{adx_len}"]   # 趨勢方向性

        # --- ATR (波動度) ---
        atr_len = int(cfg.get("atr_len"))
        atr = AverageTrueRange(df["high"], df["low"], df["close"], window=atr_len).average_true_range()
        
        df[f"atr_{atr_len}"] = atr
        df["atr_pct"] = (atr / df["close"]).clip(upper=0.05) # 使用 atr_pct 作為波動度特徵

        # --- ROC (動能變化率) ---
        roc_short = int(cfg.get("roc_short"))
        roc_long = int(cfg.get("roc_long"))
        
        df[f"roc_{roc_short}"] = df["close"].pct_change(roc_short)
        df[f"roc_{roc_long}"] = df["close"].pct_change(roc_long)
        df["roc_diff"] = df[f"roc_{roc_short}"] - df[f"roc_{roc_long}"]

        # 移除 NaN 值
        df = df.dropna()
        
        # 篩選出新生成的特徵欄位（例如：以 rsi_, ema_ 等開頭）
        feature_cols = [c for c in df.columns if any(c.startswith(p) for p in ("ema_", "rsi_", "macd_", "adx_", "pdi_", "ndi_", "trend_", "atr_", "roc_"))]

        return df[feature_cols]