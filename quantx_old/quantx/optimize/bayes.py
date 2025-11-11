# quantx/quantx/optimize/bayes.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import random
from typing import Dict, Any, List, Optional
import pandas as pd

from .base import BaseOptimizer
from ..core.runtime import Runtime
from ..core.risk import RiskConfig

class BayesLikeOptimizer(BaseOptimizer):
    """
    簡化版貝葉斯啟發式優化器。
    """
    def __init__(
        self,
        runtime: Runtime,
        strategy_cls,
        symbols: List[str],
        tf: str,
        data: Dict[str, pd.DataFrame],
        param_grid: Dict[str, List[Any]],
        n_trials: int = 50,
        equity: float = 10000.0,
        risk_config: Optional[RiskConfig] = None,
    ):
        super().__init__(runtime, strategy_cls, symbols, tf, data, param_grid, equity, risk_config)
        self.n_trials = n_trials

    def _propose(self) -> Dict[str, Any]:
        """提議一組參數 (簡化：先 uniform 採樣)。"""
        return {k: random.choice(vs) for k, vs in zip(self.param_keys, self.param_values)}

    def run(self):
        """執行貝葉斯啟發式優化。"""
        records: List[Dict[str, Any]] = []
        best: Optional[Dict[str, Any]] = None

        symbols_str = "_".join(self.symbols)
        self.logger.info(f"[bayes] 開始對 {self.strategy_cls.__name__} ({symbols_str}) 執行 {self.n_trials} 次嘗試...")

        for i in range(self.n_trials):
            params = self._propose()
            try:
                metrics, perf = self._run_once(params)
            except Exception as e:
                self.logger.warning(f"[bayes] trial {i+1} failed: {e}", exc_info=True)
                continue

            rec = {
                "trial": i + 1,
                "params": params,
                "final_equity": metrics.get("final_equity"),
                "total_return": perf.get("total_return"),
                "sharpe_ratio": perf.get("sharpe_ratio"),
                "max_drawdown": perf.get("max_drawdown"),
                "num_trades": metrics.get("n_trades"),
                "win_rate": perf.get("win_rate"),
            }
            records.append(rec)

            best_sharpe = float("-inf") if not best else (best["perf"].get("sharpe_ratio") or float("-inf"))
            if (rec.get("sharpe_ratio") or float("-inf")) > best_sharpe:
                best = {"params": params, "metrics": metrics, "perf": perf}

            self.logger.info(
                f"[bayes] trial {i+1}/{self.n_trials} sharpe={rec.get('sharpe_ratio', 0):.4f} "
                f"mdd={rec.get('max_drawdown', 0):.4f} ret={rec.get('total_return', 0):.4f} params={params}"
            )

        df = pd.DataFrame.from_records(records) if records else pd.DataFrame()

        return best, df