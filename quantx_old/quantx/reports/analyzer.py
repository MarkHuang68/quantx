# quantx/reports/analyzer.py
# -*- coding: utf-8 -*-
"""
Analyzer helpers (robust equity reader + legacy shim + richer metrics)

修正/新增：
- read_equity_series(): 支援「ts/timestamp+equity」或「i+equity」；若為 bar 索引，不轉 1970。
- plot_equity(): 非時間索引用 Bar # 畫橫軸。
- run_all_bt(): 回傳 initial_equity, final_equity, total_pnl, n_trades, win_rate 等欄位，
  以滿足 summary 產生器不再出現 N/A。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union, Dict, Any

import numpy as np
import pandas as pd


# -----------------------------
# 讀取 equity.csv（健壯）
# -----------------------------

def _detect_dt_format(sample: str) -> Optional[str]:
    s = str(sample).strip()
    if len(s) == 19 and s[4] == "-" and s[7] == "-" and s[10] == " " and s[13] == ":" and s[16] == ":":
        return "%Y-%m-%d %H:%M:%S"
    if len(s) == 16 and s[4] == "-" and s[7] == "-" and s[10] == " " and s[13] == ":":
        return "%Y-%m-%d %H:%M"
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        return "%Y-%m-%d"
    return None


def read_equity_series(equity_csv: Union[str, Path]) -> pd.Series:
    """
    讀取 equity.csv → pd.Series
    - 若檔案含「ts/timestamp + equity」→ 以 ts 為 DatetimeIndex
    - 若檔案含「i + equity」→ 以 i（整數）為索引，不轉時間
    - 否則嘗試用第一欄為時間索引（指定 format 避免 fallback 警告）
    """
    p = Path(equity_csv)

    # 先不設 index 讀一次，根據欄位決策
    df0 = pd.read_csv(p)
    if df0.empty:
        return pd.Series(dtype="float64")

    cols = {c.lower(): c for c in df0.columns}
    # 案例 A：有 ts/timestamp + equity
    for tcol_key in ("ts", "timestamp", "time", "date", "datetime"):
        if tcol_key in cols and "equity" in cols:
            tcol = cols[tcol_key]
            s = pd.Series(df0["equity"].astype(float).values, index=pd.to_datetime(df0[tcol], utc=True, errors="coerce"))
            s = s[~s.index.isna()]
            s.index = pd.DatetimeIndex(s.index)  # 保證型別
            return s

    # 案例 B：i + equity（僅有 bar 編號）
    if ("i" in cols) and ("equity" in cols):
        s = pd.Series(df0["equity"].astype(float).values, index=df0["i"].astype(int).values)
        s.index.name = "i"
        return s

    # 案例 C：第一欄是時間索引 + 第二欄 equity
    df1 = pd.read_csv(p, index_col=0)
    if df1.empty:
        return pd.Series(dtype="float64")

    first_val = str(df1.index[0])
    fmt = _detect_dt_format(first_val)
    if fmt is not None:
        try:
            df2 = pd.read_csv(p, parse_dates=[0], date_format={0: fmt}, index_col=0)
            s = df2.iloc[:, 0].astype(float)
            s.index = pd.DatetimeIndex(s.index)
            return s
        except TypeError:
            pass  # 舊版 pandas 無 date_format，落到手動轉

    idx = pd.to_datetime(df1.index, format=fmt if fmt else None, errors="coerce", utc=True)
    mask = ~idx.isna()
    if not np.all(mask):
        df1 = df1.loc[mask]
        idx = idx[mask]
    df1.index = pd.DatetimeIndex(idx)
    return df1.iloc[:, 0].astype(float)


# -----------------------------
# KPI（本地簡版）
# -----------------------------

def _infer_bar_seconds(index: pd.Index) -> float:
    if isinstance(index, pd.DatetimeIndex) and len(index) >= 2:
        deltas = (index[1:] - index[:-1]).total_seconds()
        return float(np.median(deltas)) if len(deltas) else 60.0
    return 60.0

def _compute_kpis_from_equity(equity: pd.Series) -> Dict[str, float]:
    if equity is None or len(equity) < 2:
        return {"total_return": 0.0, "sharpe_ratio": 0.0, "max_drawdown": 0.0}

    eq = equity.astype(float)
    ret = eq.pct_change().dropna()
    if ret.empty:
        return {"total_return": 0.0, "sharpe_ratio": 0.0, "max_drawdown": 0.0}

    bar_sec = _infer_bar_seconds(eq.index)
    per_year = max(1.0, (365.0 * 24 * 3600) / bar_sec)
    mu, sd = float(ret.mean()), float(ret.std(ddof=0))
    sharpe = (mu / sd) * np.sqrt(per_year) if sd > 0 else 0.0

    roll_max = np.maximum.accumulate(eq.values)
    dd = eq.values / roll_max - 1.0
    mdd = float(dd.min()) if len(dd) else 0.0

    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    return {"total_return": total_return, "sharpe_ratio": sharpe, "max_drawdown": mdd}


# -----------------------------
# 交易統計（從 trades.csv 推導）
# -----------------------------

def _trades_stats(trades_csv: Optional[Union[str, Path]]) -> Dict[str, float]:
    n_trades = 0
    win_rate = 0.0
    if not trades_csv:
        return {"n_trades": n_trades, "win_rate": win_rate}

    p = Path(trades_csv)
    if not p.exists():
        return {"n_trades": n_trades, "win_rate": win_rate}

    try:
        tdf = pd.read_csv(p)
    except Exception:
        return {"n_trades": n_trades, "win_rate": win_rate}

    # 以「close」列且有 pnl 欄位者作為一筆交易結果
    if "side" in tdf.columns:
        closes = tdf[tdf["side"].astype(str).str.lower() == "close"]
    else:
        closes = tdf

    if "pnl" in closes.columns:
        n_trades = int(len(closes))
        if n_trades > 0:
            wins = int((closes["pnl"].astype(float) > 0).sum())
            win_rate = float(wins / n_trades)
    else:
        # 沒有 pnl 欄位時，退回以 open 次數估算
        opens = tdf[getattr(tdf["side"].astype(str).str.lower(), "isin", lambda *_: pd.Series([], dtype=bool))(["buy", "short"])]
        n_trades = int(len(opens)) if len(opens) else int(len(tdf))
        win_rate = 0.0

    return {"n_trades": n_trades, "win_rate": win_rate}


# -----------------------------
# 畫圖（索引自動判斷時間/非時間）
# -----------------------------

def plot_equity(equity_csv: Union[str, Path], out_png: Optional[Union[str, Path]] = None) -> Path:
    import matplotlib.pyplot as plt

    s = read_equity_series(equity_csv)
    out_png = Path(out_png) if out_png else Path(equity_csv).with_suffix(".png")

    fig = plt.figure(figsize=(10, 3.5))
    ax = fig.add_subplot(111)

    if isinstance(s.index, pd.DatetimeIndex):
        ax.plot(s.index, s.values, lw=1.2)
        ax.set_xlabel("Time")
        try:
            fig.autofmt_xdate()
        except Exception:
            pass
    else:
        x = np.arange(len(s), dtype=int)
        ax.plot(x, s.values, lw=1.2)
        ax.set_xlabel("Bar #")

    ax.set_title("Equity Curve")
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return out_png


# -----------------------------
# 舊版相容：run_all_bt（加強回傳欄位）
# -----------------------------

def run_all_bt(
    equity_csv: Union[str, Path],
    trades_csv: Optional[Union[str, Path]] = None,
    out_dir: Optional[Union[str, Path]] = None,
    title: str = "Backtest Report",
    make_plot: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    equity_csv = Path(equity_csv)
    trades_csv = Path(trades_csv) if trades_csv else None
    out_dir = Path(out_dir) if out_dir else equity_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    equity = read_equity_series(equity_csv)
    metrics = _compute_kpis_from_equity(equity)

    # 追加：初末資金與總損益
    initial_equity = float(equity.iloc[0]) if len(equity) else None
    final_equity = float(equity.iloc[-1]) if len(equity) else None
    total_pnl = (final_equity - initial_equity) if (initial_equity is not None and final_equity is not None) else None

    # 交易統計（如有 trades.csv）
    tstats = _trades_stats(trades_csv) if trades_csv else {"n_trades": 0, "win_rate": 0.0}

    equity_png = None
    if make_plot and len(equity):
        equity_png = str(plot_equity(equity_csv, out_dir / "equity.png"))

    result = {
        "title": title,
        "equity_csv": str(equity_csv),
        "trades_csv": str(trades_csv) if trades_csv else None,
        "equity_png": equity_png,
        "metrics": metrics,
        # 舊欄位維持
        "total_return": metrics.get("total_return", 0.0),
        "sharpe_ratio": metrics.get("sharpe_ratio", 0.0),
        "max_drawdown": metrics.get("max_drawdown", 0.0),
        "out_dir": str(out_dir),
        # 新增欄位（供 summary 直接使用）
        "initial_equity": initial_equity,
        "final_equity": final_equity,
        "total_pnl": total_pnl,
        "n_trades": tstats["n_trades"],
        "win_rate": tstats["win_rate"],
    }
    return result


__all__ = ["read_equity_series", "plot_equity", "run_all_bt"]
