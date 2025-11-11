# quantx/core/runner/interval_runner.py
# -*- coding: utf-8 -*-
# 版本: v6 (重構完成)
# 說明:
# - run 方法改為 async，以支援並行回測。

from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING
import pandas as pd

from quantx.core.context import IntervalContext, Position
from quantx.core.report.reporter import ReportGenerator
from quantx.core.executor.base import BaseExecutor

if TYPE_CHECKING:
    from quantx.core.runtime import Runtime
    from quantx.core.policy.auto_policy import AutoPolicy
    from quantx.core.config import Config

class IntervalRunner:
    """區間模擬執行器"""
    def __init__(
        self,
        runtime: 'Runtime',
        auto_policy: 'AutoPolicy',
        symbol: str,
        tf: str,
        start_dt: datetime,
        end_dt: datetime,
        cfg: 'Config'
    ):
        self.runtime = runtime
        self.log = runtime.log
        self.symbol = symbol
        self.tf = tf
        self.start_dt = start_dt
        self.end_dt = end_dt
        self.auto_policy = auto_policy
        self.cfg = cfg

    async def run(self):
        self.log.info(f"[IntervalRunner] 啟動 {self.symbol}-{self.tf}, {self.start_dt.strftime('%Y-%m-%d')} → {self.end_dt.strftime('%Y-%m-%d')}")

        executor_tuple = self.auto_policy.select_executor(self.symbol, self.tf)
        executor: BaseExecutor = executor_tuple[0] if isinstance(executor_tuple, tuple) else executor_tuple
        
        if executor is None:
            self.log.warning(f"[IntervalRunner] {self.symbol}-{self.tf} 沒有可用決策者，停止模擬")
            return

        self.log.info(f"[IntervalRunner] 使用執行器: {executor.__class__.__name__} with params {executor.params}")

        main_df = self.runtime.loader.load_ohlcv(self.symbol, self.tf, self.start_dt, self.end_dt)
        if main_df.empty:
            self.log.warning(f"[IntervalRunner] 在指定區間內找不到 {self.symbol}-{self.tf} 的數據，跳過模擬")
            return

        risk_cfg = self.runtime.risk
        initial_equity = risk_cfg.get('capital', {}).get('initial_equity', 10000.0)
        if not initial_equity:
             initial_equity = risk_cfg.get('risk', {}).get('initial_equity', 10000.0)

        if not hasattr(executor, 'equity'): setattr(executor, 'equity', initial_equity)
        if not hasattr(executor, 'equity_curve'): setattr(executor, 'equity_curve', [initial_equity])
        if not hasattr(executor, 'trades'): setattr(executor, 'trades', [])
        if not hasattr(executor, 'position'): setattr(executor, 'position', Position())

        ctx = IntervalContext(
            self.runtime, executor, self.symbol, self.tf, self.start_dt, self.end_dt
        )

        self.log.info(f"[IntervalRunner] 共 {len(main_df)} 根 K 棒，開始逐 bar 模擬...")
        for ts, _ in main_df.iterrows():
            ctx._step(ts)
            try:
                executor.on_bar(ctx)
            except Exception as e:
                self.log.error(f"[IntervalRunner] on_bar 執行失敗 at {ts}: {e}", exc_info=True)
            
            ctx._execute_commands_via_manager()
            ctx._clear_intent()

        self.log.info("[IntervalRunner] 模擬結束，準備生成報表...")
        try:
            reporter = ReportGenerator(
                runtime=self.runtime, symbol=self.symbol, tf=self.tf,
                strategy_name=executor.__class__.__name__, mode="interval_simulation"
            )
            final_equity_curve = executor.equity_curve[1:]
            
            if len(final_equity_curve) > len(main_df.index):
                 eq_curve_series = pd.Series(final_equity_curve[:len(main_df.index)], index=main_df.index)
            else:
                 eq_curve_series = pd.Series(final_equity_curve, index=main_df.index[:len(final_equity_curve)])

            reporter.generate(
                ohlcv=main_df,
                equity_curve=eq_curve_series,
                trades=executor.trades,
                strategy_params=executor.params
            )
            self.log.info(f"[IntervalRunner] 報表已生成至: {reporter.result_dir}")
        except Exception as e:
            self.log.error(f"[IntervalRunner] 生成報表失敗: {e}", exc_info=True)