# quantx/cli/bt_wfo.py
# -*- coding: utf-8 -*-
# 版本: v2 (DI 重構)
# 說明:
# - 使用 AppContainer 來組裝和獲取 runtime，取代 get_runtime()。
# - 將 runtime 實例傳遞給 WalkForwardOptimizer。

from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
import sys
import pandas as pd

from quantx.core.config import Config
from quantx.containers import AppContainer # 引入 DI 容器
from quantx.optimize.wfo import WalkForwardOptimizer
from quantx.core.utils import setup_logger

def _json_default(o):
    if isinstance(o, pd.Timestamp) or hasattr(o, "isoformat"): return o.isoformat()
    return str(o)
def _fmt4(x):
    try: return f"{float(x):.4f}"
    except Exception: return str(x)
def _fmt8(x):
    try: return f"{float(x):.8f}"
    except Exception: return str(x)

def _write_wfo_summary(out_dir: str | Path, results):
    lines = ["# WFO Summary", ""]
    for i, res in enumerate(results):
        is_start, is_end, oos_end, cand = res.get("is_start"), res.get("is_end"), res.get("oos_end"), res.get("candidate")
        lines.append(f"Window {i+1}: IS {is_start} -> {is_end} OOS -> {oos_end}")
        if not cand:
            lines.append("- no candidate passed gate\n")
            continue
        params, perf = cand.get("params", {}), cand.get("perf", {})
        lines.append(f"- Params: {params}")
        lines.append(f"- Sharpe: {_fmt4(perf.get('sharpe_ratio', float('nan')))}")
        lines.append(f"- Max Drawdown: {_fmt8(perf.get('max_drawdown', float('nan')))}\n")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "wfo_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)

def parse_args(argv):
    p = argparse.ArgumentParser(description="Run walk-forward optimisation on a strategy")
    p.add_argument("--strategy", type=str, required=True)
    p.add_argument("--symbol", type=str, required=True)
    p.add_argument("--tf", type=str, required=True)
    p.add_argument("--start", type=str, required=True)
    p.add_argument("--end", type=str, required=True)
    p.add_argument("--param-grid", type=str, required=True, help="JSON dict of parameter lists")
    p.add_argument("--is-days", type=int, default=21)
    p.add_argument("--oos-days", type=int, default=7)
    p.add_argument("--gate", type=str, default="{}", help="JSON dict of gate conditions")
    p.add_argument("--equity", type=float, default=10000.0)
    p.add_argument("--output", type=str, default="results")
    return p.parse_args(argv)

def main(argv=None):
    args = parse_args(argv or sys.argv[1:])
    
    # 1. 初始化容器並獲取 runtime
    cfg = Config()
    container = AppContainer(cfg, mode_override="live")
    runtime = container.runtime
    logger = runtime.log

    # 2. 執行 WFO 邏輯
    data_tf = runtime.loader.load_ohlcv(args.symbol, args.tf, args.start, args.end)
    risk_cfg = {"size_mode": "percent_equity", "risk_pct": 0.01}
    strategy_cls = runtime.load_strategy(args.strategy)
    param_grid = json.loads(args.param_grid)
    gate = json.loads(args.gate)

    wfo = WalkForwardOptimizer(
        strategy_cls,
        param_grid,
        runtime=runtime, # <--- 注入 runtime
        is_days=args.is_days,
        oos_days=args.oos_days,
        gate=gate,
        risk_cfg=risk_cfg,
    )
    results = wfo.run(data_tf, symbol=args.symbol, tf=args.tf, equity=args.equity)

    # 3. 處理輸出
    out_dir = Path(args.output) / runtime.scope / args.strategy / f"{args.symbol}_{args.tf}" / "wfo"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for res in results:
        base_row = {"is_start": res.get("is_start"), "is_end": res.get("is_end"), "oos_end": res.get("oos_end")}
        cand, params, perf = res.get("candidate", {}), {}, {}
        if cand:
            params, perf = cand.get("params", {}), cand.get("perf", {})
        row = {**base_row, "params": json.dumps(params, ensure_ascii=False), **perf}
        rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "windows.csv", index=False)

    (out_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    logger.info(f"Saved WFO reports to: {out_dir}")
    _write_wfo_summary(out_dir, results)

    logger.info(f"Completed WFO across {len(results)} windows")
    for i, res in enumerate(results):
        if cand := res.get("candidate"):
            perf = cand.get("perf", {})
            logger.info(f"Window {i+1}: ... params={cand.get('params', {})} sharpe={_fmt4(perf.get('sharpe_ratio'))} mdd={_fmt8(perf.get('max_drawdown'))}")
        else:
            logger.info(f"Window {i+1}: ... no candidate passed gate")

if __name__ == "__main__":
    main()