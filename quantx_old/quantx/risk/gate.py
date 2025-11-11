# -*- coding: utf-8 -*-
from __future__ import annotations
import json, os, time
from dataclasses import dataclass

@dataclass
class GateConfig:
    enabled: bool
    cool_down_bars: int
    min_oos_acc: float
    max_oos_mdd_pct: float
    min_oos_sharpe: float

class StrategyGate:
    """
    依 OOS 指標決定策略是否可用。
    你可在回測/優化後產出 summary_xxx.json，內含 oos_acc / oos_mdd / oos_sharpe。
    若不達標 → 進入冷卻期 cool_down_bars；實盤期間每根 bar 檢查一次。
    """
    def __init__(self, conf: GateConfig, state_path: str):
        self.conf = conf
        self.state_path = state_path
        self._load_state()

    def _load_state(self):
        self.state = {"cooldown_left": 0}
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    self.state = json.load(f)
            except Exception:
                pass

    def _save_state(self):
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False)

    def _metrics_ok(self, metrics: dict) -> bool:
        if metrics.get("oos_acc", 0.0) < self.conf.min_oos_acc: return False
        if metrics.get("oos_sharpe", 0.0) < self.conf.min_oos_sharpe: return False
        if metrics.get("oos_mdd_pct", 9e9) > self.conf.max_oos_mdd_pct: return False
        return True

    def check(self, metrics: dict | None) -> bool:
        """
        回傳 True 表示可用；False 表示進冷卻或不合格
        """
        if not self.conf.enabled:
            return True
        # 冷卻中
        if self.state.get("cooldown_left", 0) > 0:
            self.state["cooldown_left"] -= 1
            self._save_state()
            return False
        # 首次或冷卻結束：檢查指標
        if metrics is None:
            return False  # 無指標則保守處理
        ok = self._metrics_ok(metrics)
        if not ok:
            self.state["cooldown_left"] = int(self.conf.cool_down_bars)
            self._save_state()
            return False
        return True
