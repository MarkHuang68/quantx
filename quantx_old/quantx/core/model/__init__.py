# 檔案: quantx/core/model/__init__.py
# 版本: v2 (架構重構版)
# 說明:
# - 移除了對已刪除的舊訓練器 (xgb_trainer, lstm_trainer, hybrid_trainer) 的引用。
# - 更新 __all__ 列表，只導出當前架構中真正對外提供服務的核心訓練函式。

"""
核心訓練器模組 (Core Trainer Modules)
- strategy_trainer: 提供策略回測與網格搜尋功能。
- ml_wfo_trainer: 提供機器學習模型的步進優化 (WFO) 訓練功能。
"""

# 從現有的、有效的模組中導出核心函式
from .strategy_trainer import train_strategy_grid
from .ml_wfo_trainer import run_ml_wfo

# 更新 __all__ 以反映當前可用的公開接口
__all__ = [
    "train_strategy_grid",
    "run_ml_wfo",
]