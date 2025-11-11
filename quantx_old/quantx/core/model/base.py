import os
import pickle
import pandas as pd
from datetime import datetime, timezone

from quantx.core.eval.metrics import compute_kpis
from quantx.backtest.lite import LiteBacktester

def save_model(model, model_name: str, cfg) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("models", model_name)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{model_name}_{ts}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path

def evaluate_predictions(y_true, y_pred, close, tf):
    """
    統一計算 ML/策略預測績效:
      - Accuracy (分類正確率)
      - Backtest 績效 (Sharpe, MDD, Trades)

    Parameters
    ----------
    y_true : array-like
        真實標籤 (0, 1, 2 ...).
    y_pred : array-like
        模型預測標籤 (或概率轉類別).
    close : pd.Series
        收盤價，用於模擬回測.
    tf : str
        時間週期字串 (ex: "15m", "1h").

    Returns
    -------
    dict
        { "accuracy": float, "sharpe": float, "mdd": float, "trades": int }
    """
    # ---- 1) Accuracy ----
    total = len(y_true)
    correct = (y_true == y_pred).sum() if total > 0 else 0
    accuracy = correct / total if total > 0 else 0.0

    # ---- 2) Backtest ----
    # 轉成 Series，並對齊收盤價的 index
    if not isinstance(y_pred, pd.Series):
        y_pred = pd.Series(y_pred, index=close.index[-len(y_pred):])

    bt = LiteBacktester(tf=tf)
    result = bt.run(close, y_pred)
    kpis = compute_kpis(result.get("curve"), tf)

    return {
        "accuracy": float(accuracy),
        "sharpe": float(kpis.get("sharpe_ratio", 0)),
        "mdd": float(kpis.get("max_drawdown", 0)),
        "trades": int(kpis.get("trades", 0)),
    }