# quantx/core/model/strategy_trainer.py
# -*- coding: utf-8 -*-
# 版本: v2 (DI 重構)
# 說明:
# - 移除了對 get_runtime 的依賴。
# - train_strategy_grid 函式簽名被修改，現在接收一個 runtime 參數，由外部調用者 (SchedulerTrain) 注入。

from typing import Dict, Any

from quantx.core.runtime import Runtime # 引入 Runtime 類型
from quantx.optimize.grid import GridOptimizer

def train_strategy_grid(runtime: Runtime, cfg: Any, name: str, param_grid: dict) -> dict:
    """
    在給定的參數網格(param_grid)中執行網格搜尋，找到最佳的策略參數組合。

    Args:
        runtime (Runtime): 核心運行環境實例。
        cfg (Any): 包含執行所需資訊的設定物件 (例如 symbol, tf, data)。
        name (str): 策略的名稱。
        param_grid (dict): 包含參數搜尋範圍的字典。

    Returns:
        dict: 表現最佳的參數組合及其績效。
    """
    StrategyCls = runtime.load_strategy(name)
    
    # 風險配置從 runtime.risk 獲取
    risk_cfg = runtime.risk.get('risk_management') or {"size_mode": "percent_equity", "risk_pct": 0.01}
    
    # 初始化網格搜尋優化器
    optimizer = GridOptimizer(
        runtime=runtime,
        strategy_cls=StrategyCls,
        symbol=cfg.symbol,
        tf=cfg.tf,
        data=cfg.data.df,
        param_grid=param_grid,
        equity=10_000.0,
        risk_config=risk_cfg
    )

    # 執行優化
    runtime.log.info(f"[策略訓練器] 開始對 {name} 進行網格搜尋...")
    best_result, results_df = optimizer.run()

    if best_result is None or not isinstance(best_result, dict):
        runtime.log.warning(f"[策略訓練器] 策略 {name} 的網格搜尋沒有找到任何有效結果。")
        return {}

    best_perf = best_result.get('perf', {})
    best_params = best_result.get('params', {})
    n_trades = best_result.get('metrics', {}).get('n_trades', 0)

    final_result = {
        "sharpe": best_perf.get("sharpe_ratio", 0.0),
        "mdd": best_perf.get("max_drawdown", 0.0),
        "trades": n_trades,
        "ret": best_perf.get("total_return", 0.0),
        "model": name,
        "params": best_params,
        "path": None,
    }

    runtime.log.info(f"[策略訓練器] {name} 網格搜尋完成。最佳 Sharpe: {final_result['sharpe']:.4f}, 參數: {final_result['params']}")

    return final_result