"""
auto_policy.py - 候選評分與選擇器
-----------------------------------
1. 從候選池讀取紀錄 (CandidatePool.load)
2. 先通過 Gate 過濾 (Sharpe / MDD / Trades / Acc)
3. 再依權重計算分數，挑選最佳候選
"""

import math
from quantx.core.candidate_pool import CandidatePool
from quantx.core.config import Config


def pass_gate(c, gate_cfg):
    """檢查候選是否通過 Gate 條件"""
    if c["sharpe"] is not None and c["sharpe"] < gate_cfg["min_sharpe"]:
        return False
    if c["mdd"] is not None and c["mdd"] > gate_cfg["max_mdd"]:
        return False
    if c["trades"] is not None and c["trades"] < gate_cfg["min_trades"]:
        return False
    if c["type"] == "ml" and c.get("val_acc", 0) < gate_cfg["min_acc"]:
        return False
    return True


def normalize(x, base=1.0):
    """簡單正規化 (避免 None)"""
    if x is None:
        return 0
    return x / base


def score(c, weights):
    """計算候選分數"""
    return (
        weights["sharpe"] * normalize(c.get("sharpe"), 2.0) +
        weights["winrate"] * normalize(c.get("winrate"), 100.0) -
        weights["mdd"] * normalize(c.get("mdd"), 1.0) +
        weights["trades"] * normalize(math.log1p(c.get("trades", 0)), 5.0) +
        weights.get("val_acc", 0) * normalize(c.get("val_acc", 0), 1.0)
    )


def select_best_candidate(symbol, tf, cand_type=None):
    """
    從候選池挑選最佳候選
    Args:
        symbol (str): 幣種，例如 BTCUSDT
        tf (str): 時間框，例如 15m
        cand_type (str): 可選，"ml" 或 "strategy"，預設兩者都查
    Returns:
        dict | None: 最佳候選 (含 metrics / params)，或 None 如果沒找到
    """
    cfg = Config.load_train()
    gate_cfg = cfg.auto_policy["gate"]
    weights = cfg.auto_policy["weights"]

    cands = CandidatePool.load(symbol, tf, cand_type)
    valid = [c for c in cands if pass_gate(c, gate_cfg)]

    if not valid:
        return None

    best = max(valid, key=lambda c: score(c, weights))
    return best


def explain(symbol, tf, cand_type=None, topn=5):
    """
    說明候選清單與分數
    Args:
        symbol (str): 幣種
        tf (str): 時間框
        cand_type (str): 可選，"ml" 或 "strategy"
        topn (int): 取前 N 名 (預設 5)
    Returns:
        list[dict]: 每個候選的詳細資訊
    """
    cfg = Config.load_train()
    gate_cfg = cfg.auto_policy["gate"]
    weights = cfg.auto_policy["weights"]

    cands = CandidatePool.load(symbol, tf, cand_type)

    explained = []
    for c in cands:
        passed = pass_gate(c, gate_cfg)
        sc = score(c, weights) if passed else None
        explained.append({
            "id": c.get("id"),
            "type": c.get("type"),
            "name": c.get("name"),
            "sharpe": c.get("sharpe"),
            "mdd": c.get("mdd"),
            "winrate": c.get("winrate"),
            "trades": c.get("trades"),
            "val_acc": c.get("val_acc"),
            "gate": passed,
            "score": sc,
        })

    # 按 score 排序（未過 gate 的放最後）
    explained.sort(key=lambda x: (x["score"] is not None, x["score"]), reverse=True)
    return explained[:topn]
