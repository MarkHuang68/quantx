# 檔案: quantx/core/utils.py
# 版本: v3 (新增 sanitize 及 AutoPolicy 相關工具)
# 說明:
# - 新增 sanitize 函式，用於清理數據以便序列化為 JSON。
# - 增加 normalize_candidates 和 score_candidates 等工具，以便在 LiveRunner 中處理策略分數。

from __future__ import annotations
import logging
import math
import numpy as np
import pandas as pd
import time
from typing import Dict, Any, List


def setup_logger(name: str = "quantx", level: int = logging.INFO) -> logging.Logger:
    """
    設定並回傳一個 logger。
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger

def sanitize(obj):
    """
    🟢 處理 NaN / numpy 型別，確保能正確地寫入 JSON。
    這是一個遞迴函式，可以處理巢狀的字典和列表。
    """
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize(v) for v in obj]
    elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None # 將 NaN 和 inf 轉換為 JSON 的 null
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    # 其他類型原樣返回
    return obj


# ------------------------------------------------
# 🟢 AutoPolicy/Scoring Helper Functions (移自 auto_policy.py)
# ------------------------------------------------

def normalize_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """確保每個候選者字典都包含必要的評分/過濾欄位，避免 DataFrame 轉換時出錯。"""
    default_metrics = {
        "sharpe": 0.0,
        "mdd": 1.0,  
        "trades": 0,
        "acc": 0.0,
        "val_acc": 0.0,
        "total_return": 0.0,
    }
    
    normalized = []
    for c in candidates:
        new_c = c.copy()
        for key, default_val in default_metrics.items():
            value = new_c.get(key)
            if value is None:
                 new_c[key] = default_val
            elif isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                 new_c[key] = default_val
            elif isinstance(value, str):
                 try:
                      new_c[key] = float(value)
                 except ValueError:
                      new_c[key] = default_val
        normalized.append(new_c)
    return normalized

def score_candidates(df: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    """
    對通過 Gate 的候選者進行加權計分 (Scoring)。
    """
    if df.empty:
        df["score"] = pd.Series(dtype=float)
        return df
        
    # 計算時間衰減分數 (age_penalty)，單位是分鐘
    now = time.time()
    # 由於 df['time'] 是 isoformat，這裡需進行轉換
    df["age_minutes"] = (now - pd.to_datetime(df["time"]).astype('int64') // 10**9) / 60
    
    # 從設定檔讀取權重，若無則使用預設值
    w_sharpe = weights.get("sharpe", 100.0)
    w_mdd = weights.get("mdd", -10.0) # MDD 是負向指標，權重為負
    w_trades = weights.get("trades", 0.1)
    w_time_decay = weights.get("time_decay", -0.01) # 時間也是負向指標

    # 計算總分
    df["score"] = (
        df["sharpe"].fillna(0) * w_sharpe +
        # 注意：mdd 在 CandidateStore 中儲存的是絕對值 (0.1, 0.2)，但懲罰項權重是負的
        # 由於我們在 normalize_candidates 中已經處理了 NaN，這裡直接使用即可
        df["mdd"].fillna(1) * w_mdd + 
        df["trades"].fillna(0) * w_trades +
        df["age_minutes"].fillna(9999) * w_time_decay
    )
    return df