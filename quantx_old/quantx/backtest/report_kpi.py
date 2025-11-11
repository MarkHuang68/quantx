from __future__ import annotations
import pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
def plot_kpi(equity_csv: str|Path, decisions_csv: str|Path, outdir: str|Path):
    outdir=Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    eq=pd.read_csv(equity_csv, parse_dates=[0], index_col=0).iloc[:,0].astype(float)
    dec=pd.read_csv(decisions_csv, parse_dates=["ts"]) if Path(decisions_csv).exists() else pd.DataFrame()
    roll=eq.cummax(); plt.figure(); eq.plot(title="Equity with Drawdown Shading")
    plt.fill_between(eq.index, eq.values, roll.values, where=(eq<roll), alpha=0.2)
    plt.tight_layout(); plt.savefig(outdir/"kpi_equity_dd.png"); plt.close()
    ret=eq.pct_change().fillna(0.0)
    if not dec.empty and "regime" in dec.columns:
        reg=dec.set_index("ts")["regime"].reindex(ret.index, method="ffill").fillna("unknown")
        contrib=ret.groupby(reg).sum().sort_values(ascending=False)
        plt.figure(); contrib.plot(kind="bar", title="Regime Contribution (sum of returns)")
        plt.tight_layout(); plt.savefig(outdir/"kpi_regime_contrib.png"); plt.close()
