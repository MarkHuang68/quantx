"""
metrics.py
==========
績效統計工具
取代原本的 backtest/evaluator.py
"""

import numpy as np
import pandas as pd


def compute_kpis(equity_curve: pd.Series, tf: str = "1d") -> dict:
    """
    從 equity curve 計算 Sharpe、MDD、總報酬
    ----------
    equity_curve : pd.Series
        資金曲線，index 為時間，values 為資產價值
    tf : str
        回測時間框 (影響年化 Sharpe)

    回傳:
        dict {sharpe_ratio, max_drawdown, total_return, trades}
    """
    if equity_curve is None or len(equity_curve) < 2:
        return {"sharpe_ratio": 0.0, "max_drawdown": 0.0, "total_return": 0.0, "trades": 0}

    returns = equity_curve.pct_change().dropna()
    mean_ret = returns.mean()
    std_ret = returns.std()

    # 計算年化因子
    if tf.endswith("m"):
        n = int(tf[:-1])
        factor = 525600 / n     # 每年分鐘數 / tf 分鐘
    elif tf.endswith("h"):
        n = int(tf[:-1])
        factor = 8760 / n       # 每年小時數 / tf 小時
    elif tf.endswith("d"):
        n = int(tf[:-1])
        factor = 365 / n
    else:
        factor = 252            # 預設交易日數

    sharpe = (mean_ret / std_ret * np.sqrt(factor)) if std_ret > 1e-8 else 0.0

    # 最大回撤
    roll_max = equity_curve.cummax()
    dd = (equity_curve / roll_max - 1.0)
    mdd = dd.min()

    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0

    return {
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(abs(mdd)),
        "total_return": float(total_return),
        "trades": len(returns)
    }


def evaluate(equity_curve: pd.Series, tf: str = "1d") -> dict:
    """
    KPI 統一入口
    """
    return compute_kpis(equity_curve, tf)
