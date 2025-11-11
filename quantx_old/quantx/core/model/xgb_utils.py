# quantx/core/model/xgb_utils.py
# -*- coding: utf-8 -*-
# 檔案: quantx/core/model/xgb_utils.py
# 版本: v2 (最終確認版)
# 說明:
# - 此檔案集中存放所有與 XGBoost 模型相關的輔助工具，如模型建立、訊號轉換等。

from __future__ import annotations
from xgboost import XGBClassifier
import numpy as np
from typing import Dict, Any

def build_xgb(num_class: int = 3, seed: int = 42, **kwargs) -> XGBClassifier:
    """
    建立一個帶有預設參數的 XGBoost 分類器。
    
    Args:
        num_class (int): 分類的類別數量 (例如 2 或 3)。
        seed (int): 隨機種子，確保可重複性。
        **kwargs: 額外的 XGBoost 參數，可用於覆蓋預設值。

    Returns:
        XGBClassifier: 一個已配置好但尚未訓練的 XGBoost 模型實例。
    """
    if num_class == 2:
        params = dict(
            objective="binary:logistic",
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            reg_lambda=2.0, tree_method="hist", random_state=seed
        )
    else:
        # 多分類模型的預設參數
        params = dict(
            objective="multi:softprob", num_class=num_class,
            n_estimators=800, max_depth=5, learning_rate=0.035,
            subsample=0.8, colsample_bytree=0.8,
            reg_lambda=3.0, reg_alpha=0.5, gamma=1.0,
            tree_method="hist", random_state=seed
        )
    
    # 允許外部傳入的參數覆蓋預設值
    params.update(kwargs)
    return XGBClassifier(**params)

def to_signal_with_gap(proba: np.ndarray, gap: float = 0.10, neutral_idx: int = 1) -> np.ndarray:
    """
    將模型輸出的機率陣列，根據指定的 "gap" 轉換為交易訊號 (-1, 0, 1)。
    
    Args:
        proba (np.ndarray): 模型輸出的機率陣列, shape (n_samples, n_classes)。
        gap (float): 判斷為趨勢的最小機率差。
        neutral_idx (int): 代表「中性/觀望」的類別索引。

    Returns:
        np.ndarray: 代表交易訊號的陣列 (-1: 空, 0: 觀望, 1: 多)。
    """
    proba = np.asarray(proba)
    # 找出機率最高的類別及其機率值
    cls_argmax = proba.argmax(axis=1)
    prob_max = proba.max(axis=1)
    # 取得中性類別的機率值
    prob_neutral = proba[:, neutral_idx]
    
    # 只有當最高機率與中性機率的差距足夠大時，才採取行動
    take_action = (prob_max - prob_neutral) >= gap
    
    # 根據條件決定最終類別
    final_class = np.where(take_action, cls_argmax, neutral_idx)
    
    # 將類別 (0, 1, 2) 映射到訊號 (-1, 0, 1)
    # 這裡假設 0 -> -1 (bear), 1 -> 0 (neutral), 2 -> 1 (bull)
    return final_class - 1