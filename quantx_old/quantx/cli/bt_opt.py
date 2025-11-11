# 檔案: quantx/cli/bt_opt.py
# -*- coding: utf-8 -*-
# 版本: v2 (DI 重構)
# 說明:
# - 使用 AppContainer 來組裝和獲取 runtime，取代 get_runtime()。
# - 將 runtime 實例傳遞給所有優化器 (Optimizer) 類別。

from __future__ import annotations
import argparse
import json
from pathlib import Path

import pandas as pd

from quantx.core.config import Config
from quantx.containers import AppContainer
from quantx.core.risk import RiskConfig
from quantx.optimize.grid import GridOptimizer
from quantx.optimize.random import RandomOptimizer
from quantx.optimize.bayes import BayesLikeOptimizer

def parse_args():
    p = argparse.ArgumentParser(description="Backtest parameter optimisation")
    p.add_argument("--strategy", required=True, help="strategy module name (e.g., z_score)")
    p.add_argument("--symbols", type=str, nargs='+', required=True, help="一個或多個交易對 (以空格分隔)")
    p.add_argument("--tf", required=True, help="timeframe, e.g., 15m")
    p.add_argument("--param-grid", required=True, help='JSON dict of params, e.g. {"sma_len":[10,20], "z_th":[1.5,2.0]}')
    p.add_argument("--method", choices=["grid", "random", "bayes"], default="random")
    p.add_argument("--start", default=None, help="optional ISO date start")
    p.add_argument("--end", default=None, help="optional ISO date end")
    p.add_argument("--n-trials", type=int, default=50, help="random/bayes trials")
    p.add_argument("--output", type=str, default="results", help="Output directory for reports")
    p.add_argument("--equity", type=float, default=10000.0, help="initial equity")
    return p.parse_args()

def main():
    args = parse_args()
    
    # 1. 初始化容器並獲取 runtime
    cfg = Config()
    container = AppContainer(cfg) 
    runtime = container.runtime
    logger = runtime.log
    
    strategy_cls = runtime.load_strategy(args.strategy)

    try:
        grid = json.loads(args.param_grid)
        if not isinstance(grid, dict):
            raise ValueError
    except Exception:
        raise ValueError("--param-grid 必須是 JSON 物件")
    
    risk_cfg = {"size_mode": "percent_equity", "risk_pct": 0.01}

    # --- 多交易對數據加載 ---
    symbols_str = "_".join(args.symbols)
    logger.info(f"正在為 {symbols_str}-{args.tf} 預載數據...")

    all_data = {}
    for symbol in args.symbols:
        data = runtime.loader.load_ohlcv(symbol, args.tf, args.start, args.end)
        if data is None or data.empty:
            logger.warning(f"找不到 {symbol} 的數據，將跳過。")
            continue
        all_data[symbol] = data

    if not all_data:
        logger.error("所有請求的交易對均無數據，無法繼續優化。")
        return
    # --- 數據加載結束 ---

    out_dir = Path(args.output) / runtime.scope / f"optimize/{args.method}" / args.strategy / f"{symbols_str}_{args.tf}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2. 根據方法選擇優化器，並注入多交易對參數
    common_args = {
        "runtime": runtime,
        "strategy_cls": strategy_cls,
        "symbols": list(all_data.keys()), # 只使用成功加載的 symbols
        "tf": args.tf,
        "data": all_data,
        "param_grid": grid,
        "equity": args.equity,
        "risk_config": risk_cfg
    }

    if args.method == "grid":
        opt = GridOptimizer(**common_args)
    elif args.method == "bayes":
        opt = BayesLikeOptimizer(**common_args, n_trials=args.n_trials)
    else: # random
        opt = RandomOptimizer(**common_args, n_trials=args.n_trials)

    best, results_df = opt.run()

    # 3. 處理並儲存結果
    if results_df is not None and len(results_df) > 0:
        sort_cols = [c for c in ["sharpe_ratio", "max_drawdown", "total_return"] if c in results_df.columns]
        if sort_cols:
            ascending = [False, True, False][:len(sort_cols)]
            results_df = results_df.sort_values(by=sort_cols, ascending=ascending).reset_index(drop=True)

        try:
            front = [c for c in ["sharpe_ratio", "max_drawdown", "total_return", "num_trades", "win_rate", "params"] if c in results_df.columns]
            cols = front + [c for c in results_df.columns if c not in front]
            results_df = results_df[cols]
            results_df.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8")
        except Exception as e:
            logger.warning(f"儲存 summary.csv 失敗: {e}")
        logger.info(f"優化結果已儲存至: {out_dir}")

        try:
            if best:
                (out_dir / "best.json").write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
                logger.info(
                    f"BEST sharpe={best['perf'].get('sharpe_ratio', float('nan')):.4f} "
                    f"mdd={best['perf'].get('max_drawdown', float('nan')):.4f} "
                    f"ret={best['perf'].get('total_return', float('nan')):.4f} "
                    f"params={best.get('params')}"
                )
            records = results_df.to_dict(orient="records")
            (out_dir / "results.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"儲存 JSON 報告失敗: {e}")
    else:
        logger.warning("沒有任何優化結果。")

if __name__ == "__main__":
    main()