# 檔案: quantx/optimize/grid.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import itertools
from typing import Dict, Any, List, Optional
import pandas as pd

from .base import BaseOptimizer

class GridOptimizer(BaseOptimizer):
    """
    網格搜尋優化器。
    繼承自 BaseOptimizer，專注於窮舉參數組合的邏輯。
    """
    def run(self):
        """
        執行網格搜尋。
        """
        records: List[Dict[str, Any]] = []
        best: Optional[Dict[str, Any]] = None
        total_trials = len(list(itertools.product(*self.param_values)))

        symbols_str = "_".join(self.symbols)
        self.logger.info(f"[grid] 開始對 {self.strategy_cls.__name__} ({symbols_str}) 執行 {total_trials} 組參數窮舉...")

        for i, values in enumerate(itertools.product(*self.param_values)):
            params = dict(zip(self.param_keys, values))
            try:
                metrics, perf = self._run_once(params)
            except Exception as e:
                self.logger.warning(f"[grid] failed params={params}: {e}", exc_info=True)
                continue

            rec = {
                "params": params,
                "final_equity": metrics.get("final_equity"),
                "total_return": perf.get("total_return"),
                "sharpe_ratio": perf.get("sharpe_ratio"),
                "max_drawdown": perf.get("max_drawdown"),
                "num_trades": metrics.get("n_trades"),
                "win_rate": perf.get("win_rate"),
            }
            records.append(rec)

            current_sharpe = rec.get("sharpe_ratio") or float("-inf")
            best_sharpe = float("-inf") if not best else (best["perf"].get("sharpe_ratio") or float("-inf"))
            
            if current_sharpe > best_sharpe:
                best = {"params": params, "metrics": metrics, "perf": perf}

            self.logger.info(
                f"[grid] Trial {i+1}/{total_trials}: sharpe={rec.get('sharpe_ratio', 0):.4f} mdd={rec.get('max_drawdown', 0):.4f} "
                f"ret={rec.get('total_return', 0):.4f} params={params}"
            )

        df = pd.DataFrame.from_records(records) if records else pd.DataFrame()

        return best, df