# 檔案: quantx/optimize/base.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd

from ..core.risk import RiskConfig
from ..backtest.engine import BacktestEngine
from ..backtest.evaluator import compute_kpis
from ..core.runtime import Runtime


class BaseOptimizer(ABC):
    """
    優化器基類，提供多交易對回測的共享基礎架構。
    """
    def __init__(
        self,
        runtime: Runtime,
        strategy_cls,
        symbols: List[str],
        tf: str,
        data: Dict[str, pd.DataFrame],
        param_grid: Dict[str, List[Any]],
        equity: float = 10000.0,
        risk_config: Optional[RiskConfig] = None,
    ):
        self.runtime = runtime
        self.strategy_cls = strategy_cls
        self.symbols = symbols
        self.tf = tf
        self.data = data
        self.param_grid = param_grid
        self.equity = equity

        cost_model = self.runtime.get_cost_model()
        self.maker_fee_bps = cost_model.get('maker_fee_bps', 2.0)
        self.taker_fee_bps = cost_model.get('taker_fee_bps', 5.5)
        self.slippage_bps = cost_model.get('slip_bps', 1.0)

        self.risk_config = risk_config
        self.logger = self.runtime.log

        self.param_keys = list(param_grid.keys())
        self.param_values = list(param_grid.values())

    def _run_once(self, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        執行單次回測。
        現在這個方法可以處理多交易對的數據。
        """
        engine = BacktestEngine(
            datas=self.data,  # 修正: 參數應為 datas
            strategy_cls=self.strategy_cls,
            runtime=self.runtime,
            # 移除: symbols 參數已不再需要
            tf=self.tf,
            params=params,
            equity=self.equity,
            maker_fee_bps=self.maker_fee_bps,
            taker_fee_bps=self.taker_fee_bps,
            slippage_bps=self.slippage_bps,
            risk_config=self.risk_config,
        )

        metrics = engine.run()

        # 注意：對於多交易對，KPI 計算可能需要更複雜的邏輯，
        # 目前暫時使用第一個 symbol 的數據來計算 pnl，這是一個簡化。
        # 理想情況下，應該基於合併的權益曲線來計算。
        master_timeline = engine.master_timeline
        if len(engine.equity_curve) > 2:
            # 使用 master_timeline 的索引來對齊權益曲線
            pnl = pd.Series(engine.equity_curve[1:-1], index=master_timeline)
        else:
            pnl = pd.Series(dtype=float)

        perf = compute_kpis(pnl, self.tf)

        return metrics, perf

    @abstractmethod
    def run(self):
        """
        子類必須實現此方法來執行其特定的優化策略 (grid, random, etc.)。
        """
        raise NotImplementedError