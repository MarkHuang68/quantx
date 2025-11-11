# quantx/backtest/report.py
# -*- coding: utf-8 -*-
from pathlib import Path
import math
from typing import Dict, Any, List
import pandas as pd
import os

try:
    _tz_out = (os.getenv("REPORT_TZ") or "Asia/Taipei").strip()
except Exception:
    _tz_out = "Asia/Taipei"

def _convert_ts_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: 
        return df
    cand = [c for c in df.columns if c.lower() in ("ts","timestamp","time") or c.lower().endswith("_ts")]
    for c in cand:
        try:
            s = pd.to_datetime(df[c], utc=True, errors="coerce")
            if s.notna().any():
                df[c] = s.dt.tz_convert(_tz_out).dt.strftime("%Y-%m-%d %H:%M:%S%z")
        except Exception:
            pass
    return df

from ..backtest.evaluator import compute_kpis
from ..reports.analyzer import plot_equity

def generate_report(
    trades: List[Dict[str, Any]],
    pnl_series: pd.Series,
    performance: Dict[str, Any],
    output_dir: str | Path,
    *,
    symbol: str,
    strategy_name: str,
    tf: str,
    scope: str,
) -> None:
    out_base = Path(output_dir)
    run_dir = (out_base / scope / strategy_name / f"{symbol}_{tf}") if scope else (out_base / strategy_name / f"{symbol}_{tf}")
    run_dir.mkdir(parents=True, exist_ok=True)

    # trades.csv
    if trades:
        _df_tr = _convert_ts_df(pd.DataFrame(trades))
        _df_tr.to_csv(run_dir / "trades.csv", index=False, encoding="utf-8")

    # equity.csv + equity.png
    if pnl_series is not None and not pnl_series.empty:
        eq = pd.Series(pnl_series, name="equity")
        eq.index = pd.RangeIndex(start=0, stop=len(eq), step=1, name="i")
        eq.to_csv(run_dir / "equity.csv", encoding="utf-8")
        try:
            plot_equity(run_dir / "equity.csv", out_png=run_dir / "equity.png")
        except Exception:
            pass
    else:
        pd.DataFrame(columns=["equity"]).to_csv(run_dir / "equity.csv", index=False, encoding="utf-8")

    # KPI JSON
    (run_dir / "kpis.json").write_text(pd.Series(performance).to_json(orient="index", force_ascii=False, indent=2), encoding="utf-8")
