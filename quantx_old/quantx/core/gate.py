# 檔案: quantx/core/gate.py
# 版本: v5 (智能檢查版)
# 說明:
# - 修正了 check_gate 函式，使其在檢查準確率時更智能。
# - 只有當候選人的結果中明確包含 accuracy 指標時，才會進行 min_acc 的檢查。
# - 這解決了對傳統策略錯誤地要求準確率的問題。

from typing import Dict, Tuple, List

def check_gate(result: Dict, gate: Dict) -> Tuple[bool, List[str]]:
    """
    檢查結果是否通過 Gate 條件。
    """
    reasons = []
    
    acc = result.get("acc") or result.get("accuracy") or result.get("val_acc")
    sharpe = result.get("sharpe") or result.get("sharpe_ratio")
    mdd = result.get("mdd") or result.get("max_drawdown")
    trades = result.get("trades") or result.get("num_trades")

    # --- 🟢 核心修改：Accuracy Check (智能版) ---
    if "min_acc" in gate:
        # 只在 acc 是一個有效數值時，才進行檢查
        if acc is not None:
            if acc < gate["min_acc"]:
                reasons.append(f"Accuracy {acc:.3f} < {gate['min_acc']}")
        # 如果 acc 是 None (例如對於一個策略)，則不進行檢查，直接跳過

    # --- Sharpe Ratio Check ---
    if "min_sharpe" in gate:
        if sharpe is None or sharpe < gate["min_sharpe"]:
            sharpe_str = f"{sharpe:.3f}" if sharpe is not None else "N/A"
            reasons.append(f"Sharpe {sharpe_str} < {gate['min_sharpe']}")

    # --- Max Drawdown Check ---
    if "max_mdd" in gate:
        if mdd is None or abs(mdd) > gate["max_mdd"]:
            mdd_str = f"|{mdd:.3f}|" if mdd is not None else "N/A"
            reasons.append(f"Max Drawdown {mdd_str} > {gate['max_mdd']}")

    # --- Trades Check ---
    if "min_trades" in gate:
        if trades is None or trades < gate["min_trades"]:
            reasons.append(f"Trades {trades or 0} < {gate['min_trades']}")

    ok = len(reasons) == 0
    return ok, reasons