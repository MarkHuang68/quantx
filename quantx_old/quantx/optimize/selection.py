"""Selection and gating for optimisation results."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from ..backtest.engine import BacktestEngine
from ..backtest.evaluator import compute_kpis


def evaluate_candidates(
    candidates: List[Dict[str, Any]],
    oos_data: pd.DataFrame,
    symbol: str,
    equity: float,
    maker_fee_bps: float,
    taker_fee_bps: float,
    slippage_bps: float,
    risk_config: Any,
    tf: str,
    gate: Dict[str, Any],
    strategy_cls: type,
) -> Optional[Dict[str, Any]]:
    """Evaluate candidates on OOS data and apply gate conditions.

    The gate is a dict with keys such as ``min_oos_fills``,
    ``min_oos_sharpe`` and ``max_oos_mdd``.  A candidate passes if
    it meets all specified thresholds.  Among the passing candidates
    the one with the highest Sharpe ratio is returned; otherwise None.
    """
    best = None
    for cand in candidates:
        params = cand["params"]
        engine = BacktestEngine(
            oos_data,
            cand["params"].get("strategy_cls", cand.get("strategy_cls")) or cand.get("strategy_cls") or strategy_cls,
            symbol=symbol,
            params=params,
            equity=equity,
            maker_fee_bps=maker_fee_bps,
            taker_fee_bps=taker_fee_bps,
            slippage_bps=slippage_bps,
            risk_config=risk_config,
        )
        metrics = engine.run()
        kpis = compute_kpis(engine.pnl_series, tf)
        # Apply gate conditions
        passes = True
        if gate is not None:
            min_fills = gate.get("min_oos_fills")
            if min_fills is not None and len(engine.trades) < min_fills:
                passes = False
            min_sr = gate.get("min_oos_sharpe")
            if min_sr is not None and kpis.get("sharpe_ratio", 0) < min_sr:
                passes = False
            max_mdd = gate.get("max_oos_mdd")
            if max_mdd is not None and abs(kpis.get("max_drawdown", 0)) > max_mdd:
                passes = False
        if not passes:
            continue
        if best is None or kpis["sharpe_ratio"] > best["perf"]["sharpe_ratio"]:
            best = {
                "params": params,
                "metrics": metrics,
                "perf": kpis,
            }
    return best