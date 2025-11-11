# 檔案: quantx/optimize/wfo.py
# 版本: v4 (DI 重構)
# 說明:
# - 移除了對 get_runtime as _rt 的導入和後備調用。
# - __init__ 建構函數現在強制要求傳入 runtime 實例，確保依賴的明確性。

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional, Union

import pandas as pd

from ..core.runtime import Runtime # 引入 Runtime 類型
from .grid import GridOptimizer
from ..core.risk import RiskConfig

def _to_int_days(v: Union[str, int, float, None], default: int) -> int:
    if v is None: return default
    if isinstance(v, (int, float)): return int(v)
    s = str(v).strip().lower()
    if s.endswith("d"): s = s[:-1]
    try: return int(float(s))
    except Exception: return default

class WalkForwardOptimizer:
    def __init__(
        self,
        strategy_cls: Any,
        param_grid: Dict[str, Any],
        runtime: Runtime, # <--- runtime 現在是必要參數
        *,
        is_days: int = 21,
        oos_days: int = 7,
        risk_cfg: Optional[Dict[str, Any] | RiskConfig] = None,
        gate: Optional[Dict[str, Any]] = None,
        **_ignored: Any,
    ):
        self.strategy_cls = strategy_cls
        self.param_grid = dict(param_grid or {})
        self.is_days = _to_int_days(is_days, is_days)
        self.oos_days = _to_int_days(oos_days, oos_days)
        
        self.runtime = runtime # <--- 直接使用傳入的 runtime
        
        self.risk_cfg = risk_cfg or {
            "size_mode": "percent_equity", "risk_pct": 0.01
        }
        self.gate = dict(gate or {})

    def _window_splits(self, df: pd.DataFrame) -> List[Tuple[datetime, datetime, datetime]]:
        if df.empty: return []
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        else:
            df.index = df.index.tz_convert('UTC')
            
        i0, i1 = df.index[0].to_pydatetime(), df.index[-1].to_pydatetime()
        out: List[Tuple[datetime, datetime, datetime]] = []
        cur = i0
        while True:
            is_start = cur
            is_end = is_start + timedelta(days=self.is_days)
            oos_end = is_end + timedelta(days=self.oos_days)
            if (is_end > i1) or (oos_end > i1): break
            out.append((is_start, is_end, oos_end))
            cur = cur + timedelta(days=self.oos_days)
        return out

    def _risk_cfg_as_dict(self) -> Dict[str, Any]:
        rc = self.risk_cfg
        if isinstance(rc, RiskConfig):
            try: return asdict(rc)
            except Exception: return {"size_mode": rc.size_mode, "risk_pct": rc.risk_pct}
        return dict(rc) if isinstance(rc, dict) else {}

    @staticmethod
    def _first_item(o: Any) -> Any:
        if isinstance(o, (list, tuple)) and o: return o[0]
        return o

    @staticmethod
    def _extract_params(o: Any) -> Dict[str, Any]:
        o = WalkForwardOptimizer._first_item(o)
        if isinstance(o, dict):
            if "params" in o and isinstance(o["params"], dict): return o["params"]
            if "best" in o and isinstance(b := o["best"], dict) and isinstance(b.get("params"), dict): return b["params"]
            if "results" in o and isinstance(r := o["results"], list) and r and isinstance(f := r[0], dict) and isinstance(f.get("params"), dict): return f["params"]
            if any(k in o for k in ("length", "std", "stddev", "window", "period")): return o
        if isinstance(o, list) and o: return WalkForwardOptimizer._extract_params(o[0])
        return {}

    @staticmethod
    def _extract_perf(o: Any) -> Dict[str, Any]:
        src, item = None, WalkForwardOptimizer._first_item(o)
        if isinstance(item, dict):
            for key in ("perf", "metrics", "kpis", "summary"):
                if isinstance(item.get(key), dict):
                    src = item[key]; break
            if src is None: src = item
        elif isinstance(o, list) and o: return WalkForwardOptimizer._extract_perf(o[0])
        if not isinstance(src, dict): return {}

        def _f(name: str, *alts: str):
            for k in (name,) + alts:
                if k in src: return src[k]
            return None

        sharpe, mdd, fills = _f("sharpe_ratio", "oos_sharpe", "sharpe", "sr"), _f("max_drawdown", "oos_mdd", "mdd"), _f("fills", "oos_fills", "num_trades", "n_trades", "trades")
        out = {}
        if sharpe is not None:
            try: out["sharpe_ratio"] = float(sharpe)
            except Exception: pass
        if mdd is not None:
            try: out["max_drawdown"] = float(mdd)
            except Exception: pass
        if fills is not None:
            try: out["fills"] = int(fills)
            except Exception: pass
        return out

    def _check_gate(self, perf: Dict[str, Any]) -> Tuple[bool, List[str]]:
        g, reasons = self.gate or {}, []
        fills, sharpe, mdd = int(perf.get("fills", 0) or 0), perf.get("sharpe_ratio"), perf.get("max_drawdown")
        if "min_oos_fills" in g and fills < int(g["min_oos_fills"]): reasons.append(f"fills<{g['min_oos_fills']}")
        if "min_oos_sharpe" in g and (sharpe is None or float(sharpe) < float(g["min_oos_sharpe"])): reasons.append(f"sharpe<{g['min_oos_sharpe']}")
        if "max_oos_mdd" in g and (mdd is None or abs(float(mdd)) > float(g["max_oos_mdd"])): reasons.append(f"|mdd|>{g['max_oos_mdd']}")
        return len(reasons) == 0, reasons

    def run(self, datas: Dict[str, pd.DataFrame], *, symbols: List[str], tf: str, equity: float = 10_000.0) -> List[Dict[str, Any]]:
        if not symbols or not datas:
            return []

        # 使用第一個 symbol 的數據作為時間劃分的主數據
        main_df = datas[symbols[0]]
        folds, results = self._window_splits(main_df), []
        if not folds: return results

        risk_cfg_dict = self._risk_cfg_as_dict()

        for (is_start, is_end, oos_end) in folds:
            # 為所有 symbol 的數據進行切片
            is_datas = {s: df.loc[is_start:is_end] for s, df in datas.items()}
            oos_datas = {s: df.loc[is_end:oos_end] for s, df in datas.items()}
            
            go_is = GridOptimizer(runtime=self.runtime, strategy_cls=self.strategy_cls, symbols=symbols, tf=tf, data=is_datas, param_grid=self.param_grid, equity=equity, risk_config=risk_cfg_dict)
            is_out = go_is.run()
            best_params = self._extract_params(is_out)

            row: Dict[str, Any] = {"is_start": is_start.isoformat(), "is_end": is_end.isoformat(), "oos_end": oos_end.isoformat(), "best_is": self._first_item(is_out)}
            if not best_params:
                results.append(row)
                continue

            single_grid = {k: [v] for k, v in best_params.items()}
            go_oos = GridOptimizer(runtime=self.runtime, strategy_cls=self.strategy_cls, symbols=symbols, tf=tf, data=oos_datas, param_grid=single_grid, equity=equity, risk_config=risk_cfg_dict)
            oos_out = go_oos.run()
            perf = self._extract_perf(oos_out)

            if perf:
                ok, reasons = self._check_gate(perf)
                if ok: row["candidate"] = {"params": best_params, "perf": perf}
                else: row["reject"] = {"params": best_params, "perf": perf, "reasons": reasons}
            results.append(row)
        return results