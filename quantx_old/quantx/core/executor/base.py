# quantx/core/executor/base.py
# 檔案: quantx/core/executor/base.py
# 版本: v3 (支援雙向持倉模型)
# 說明:
# - 徹底重構 Position 類別，使其能獨立記錄多頭 (long) 與空頭 (short) 的部位資訊。
# - BaseExecutor 中的倉位屬性從單一的 self.position 升級為 self.positions 字典，
#   結構為 { "symbol": Position }，從而原生支援單策略管理多標的、多空倉位。

from abc import ABC, abstractmethod
from typing import Any, Dict, List
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class Position:
    """
    雙向持倉模型 (Hedge Mode Position)
    - 獨立記錄多頭與空頭的數量與均價。
    """
    long_qty: float = 0.0
    long_entry: float = 0.0
    short_qty: float = 0.0
    short_entry: float = 0.0

    def is_long(self) -> bool:
        """是否持有多頭倉位"""
        return self.long_qty > 1e-12

    def is_short(self) -> bool:
        """是否持有空頭倉位"""
        return self.short_qty > 1e-12

    def is_flat(self) -> bool:
        """是否為空倉 (多空皆無)"""
        return not self.is_long() and not self.is_short()

class BaseExecutor(ABC):
    """抽象化的執行單位 (策略或 ML 模型)，支援多標的雙向持倉。"""

    params: Dict[str, Any] = {}

    # --- [風控開關] ---
    # 策略是否同意接受全局風控的平倉指令
    accepts_global_drawdown_action: bool = False
    accepts_flash_crash_action: bool = False

    # [熱更新] 策略是否處於 "只平倉" 模式
    is_winding_down: bool = False

    # 策略級別的默認止損/止盈百分比 (可選)
    # 例如: default_stop_loss_pct = 2.0 (代表 2%)
    default_stop_loss_pct: float | None = None
    default_take_profit_pct: float | None = None
    # --- [風控開關結束] ---

    def __init__(self, **kwargs):
        """
        初始化執行器，合併傳入的參數並設定預設容器。
        """
        # 合併預設參數與傳入的參數
        merged = dict(self.params)
        merged.update(kwargs)
        self.params = merged

        # 紀錄交易與資金曲線的容器
        self.trades: List[Dict[str, Any]] = []
        self.equity_curve: List[float] = []

        # 🟢 核心升級：倉位屬性改為 positions 字典
        # 使用 defaultdict，當訪問一個不存在的 symbol 時，會自動為其建立一個空的 Position 物件。
        self.positions: Dict[str, Position] = defaultdict(Position)


    # -------------------------------------------------
    # 必要方法 (由子類別實作)
    # -------------------------------------------------
    @abstractmethod
    def on_bar(self, ctx: Any) -> None: # 使用 Any 避免對 ContextBase 的循環依賴
        """每根 K 棒呼叫一次，策略邏輯的主要進入點，必須實作。"""
        raise NotImplementedError

    # -------------------------------------------------
    # 工具方法 (供子類別或框架使用)
    # -------------------------------------------------
    def record_trade(self, trade: Dict[str, Any]) -> None:
        """紀錄一筆交易。"""
        self.trades.append(trade)

    def record_equity(self, equity: float) -> None:
        """紀錄當下的資金淨值。"""
        self.equity_curve.append(equity)