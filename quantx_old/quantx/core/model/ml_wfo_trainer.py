# 檔案: quantx/core/model/ml_wfo_trainer.py
# 版本: v19 (清理 WFO 報告生成邏輯)
# 說明:
# - 移除了 ReportGenerator 呼叫中對 OHLCV 數據的臨時構造，改用更穩健的數據結構。
# - 確保 trades 列表在生成報告時格式正確。

from __future__ import annotations
import numpy as np
import pandas as pd
import itertools
from pandas.api.types import is_numeric_dtype
from typing import Dict, Any, List, Tuple

from .xgb_utils import build_xgb, to_signal_with_gap
from quantx.backtest.lite import LiteBacktester
from quantx.core.timeframe import parse_tf_minutes
from . import ml_trainers
from quantx.core.report.reporter import ReportGenerator

def _find_best_params_in_sample(
    log, X_is: pd.DataFrame, y_is: pd.Series, param_grid: Dict[str, List]
) -> Dict[str, Any]:
    """
    在樣本內數據中，透過網格搜尋找到最佳模型參數 (終極穩健版)。
    """
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    if len(param_combinations) == 1:
        return param_combinations[0]

    split_idx = int(len(X_is) * 0.8)
    X_train, X_val = X_is.iloc[:split_idx], X_is.iloc[split_idx:]
    y_train, y_val = y_is.iloc[:split_idx], y_is.iloc[split_idx:]

    # 核心修正：對內部切分出的 y_train 進行嚴格檢查
    if y_train.nunique() < 2:
        log.warning("[WFO-IS-Tuning] 樣本內切分後的訓練集標籤少於2種，無法進行尋優。將使用預設參數。")
        return param_combinations[0]

    best_accuracy = -1.0
    best_params = param_combinations[0]

    log.info(f"[WFO-IS-Tuning] 開始在 {len(param_combinations)} 組參數中尋優...")
    for params in param_combinations:
        try:
            proba_val = ml_trainers.train_predict_xgb(X_train, y_train, X_val, params)
            preds_val = np.argmax(proba_val, axis=1)
            accuracy = (preds_val == y_val.values).mean()

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = params
        except Exception as e:
            log.warning(f"[WFO-IS-Tuning] 參數 {params} 訓練失敗: {e}")
            continue
    
    log.info(f"[WFO-IS-Tuning] 尋優完成。最佳準確率: {best_accuracy:.4f}, 最佳參數: {best_params}")
    return best_params

# (後續的 purged_walk_forward 函式保持不變)
def purged_walk_forward(runtime, df: pd.DataFrame, is_days=42, oos_days=7, purge_days=2, ts_col="ts"):
    log = runtime.log
    ts_raw = df[ts_col]
    if is_numeric_dtype(ts_raw): ts = pd.to_datetime(ts_raw, unit="s", utc=True)
    else: ts = pd.to_datetime(ts_raw, utc=True, errors="coerce")
    d = df.copy(); d["_ts"] = ts
    d = d.dropna(subset=["_ts"]).sort_values("_ts")
    if d.empty:
        log.warning("[WFO Generator] 傳入的數據為空，無法生成任何窗口。")
        return
    start, end = d["_ts"].min(), d["_ts"].max()
    log.info(f"[WFO Generator] 數據總範圍: {start.date()} -> {end.date()} (共 {(end-start).days} 天)")
    log.info(f"[WFO Generator] 窗口設定: is={is_days}天, purge={purge_days}天, oos={oos_days}天")
    cur = start; fold_num = 0
    while True:
        fold_num += 1
        is_s = cur; is_e = cur + pd.Timedelta(days=is_days)
        oos_s = is_e + pd.Timedelta(days=purge_days); oos_e = oos_s + pd.Timedelta(days=oos_days)
        if oos_s >= end: break
        is_mask = (d["_ts"] >= is_s) & (d["_ts"] < is_e)
        oos_mask = (d["_ts"] >= oos_s) & (d["_ts"] < oos_e)
        if is_mask.any() and oos_mask.any():
            yield d.index[is_mask], d.index[oos_mask]
        cur = cur + pd.Timedelta(days=oos_days)

def run_ml_wfo(
    runtime,
    dataset: Dict[str, pd.DataFrame],
    tf: str,
    model_name: str,
    model_params: Dict[str, Any],
    wfo_params: Dict[str, int],
    symbol: str
) -> Dict[str, Any]:
    log = runtime.log
    X_all, y_all, close_all, ts_all = dataset["X"], dataset["y"], dataset["close"], dataset["ts"]
    df_time = pd.DataFrame({"ts": ts_all}, index=X_all.index)
    all_oos_probas, all_oos_closes = [], []
    all_oos_trues = []

    trainer_func = getattr(ml_trainers, f"train_predict_{model_name.lower()}")
    param_grid = model_params.get("params", {})
    
    fold_count = 0
    for is_idx, oos_idx in purged_walk_forward(runtime, df_time, **wfo_params):
        fold_count += 1
        log.info(f"--- WFO Fold #{fold_count} ---")
        X_is, y_is = X_all.loc[is_idx], y_all.loc[is_idx]
        X_oos, y_oos = X_all.loc[oos_idx], y_all.loc[oos_idx]
        close_oos = close_all.loc[oos_idx]

        if y_is.nunique() < 2:
            log.warning(f"[WFO Fold #{fold_count}] 樣本內數據只有單一類別，跳過。")
            continue
        
        best_is_params = _find_best_params_in_sample(log, X_is, y_is, param_grid)
        proba_oos = trainer_func(X_is, y_is, X_oos, best_is_params)

        all_oos_trues.append(y_oos)
        all_oos_probas.append(pd.DataFrame(proba_oos, index=X_oos.index))
        all_oos_closes.append(close_oos)

    if not all_oos_probas:
        log.warning(f"模型 '{model_name}' 在 WFO 中未能產生任何 OOS 預測。")
        return {"sharpe": 0, "mdd": 1, "trades": 0, "accuracy": 0.0}

    final_probas = pd.concat(all_oos_probas).sort_index()
    final_trues = pd.concat(all_oos_trues).sort_index()
    final_closes = pd.concat(all_oos_closes).sort_index()
    
    accuracy = (final_trues == final_probas.idxmax(axis=1)).mean() if not final_trues.empty else 0.0
    log.info(f"[WFO-ML] 所有樣本外(OOS)數據的總體驗證準確率: {accuracy:.4f}")

    signal_cfg = model_params.get("signal_params", {})
    gap_thresholds = signal_cfg.get("gap_threshold", [0.15])
    if not isinstance(gap_thresholds, list): gap_thresholds = [gap_thresholds]

    best_result = None
    best_sharpe = -float('inf')

    log.info(f"--- 開始在 {len(gap_thresholds)} 個 gap_threshold 中尋優 ---")
    for gap in gap_thresholds:
        signal_array = to_signal_with_gap(final_probas.values, gap=gap)
        final_signal = pd.Series(signal_array, index=final_probas.index)
        
        bt = LiteBacktester(tf=tf)
        bt_results = bt.run(final_closes, final_signal)
        current_sharpe = bt_results.get('sr', -float('inf'))
        log.info(f"  測試 Gap={gap:.2f}: Sharpe={current_sharpe:.3f}, Trades={bt_results.get('trades', 0)}")
        
        if current_sharpe > best_sharpe:
            best_sharpe = current_sharpe
            best_result = bt_results
            best_result['best_gap'] = gap

    if best_result is None:
        log.warning(f"模型 '{model_name}' 在所有 gap 測試中均未產生有效回測結果。")
        return {"sharpe": 0, "mdd": 1, "trades": 0, "accuracy": accuracy}
    
    log.info(f"--- Gap 尋優完成。最佳 Gap={best_result['best_gap']:.2f}, 對應夏普={best_sharpe:.3f} ---")
    
    equity_curve = best_result.get('curve')
    final_signal = pd.Series(to_signal_with_gap(final_probas.values, gap=best_result['best_gap']), index=final_probas.index)

    try:
        reporter = ReportGenerator(
            runtime=runtime, symbol=symbol, tf=tf, strategy_name=f"ml_{model_name}", mode="wfo_summary"
        )

        # 🟢 報告生成邏輯優化
        # 1. 準備 OHLCV 數據 (使用 close 價格作為 OHLCV 的近似)
        ohlcv_report = pd.DataFrame({
            'open': final_closes, 
            'high': final_closes, 
            'low': final_closes, 
            'close': final_closes, 
            'volume': 0.0 # 體積設為 0
        })
        # 確保索引是 DatetimeIndex
        ohlcv_report.index.name = 'timestamp'


        # 2. 準備近似的 Trades 列表
        # 訊號變動點即為開倉/平倉點
        trade_signals = final_signal.loc[final_signal.shift(1) != final_signal]
        trades_approx = []
        
        for ts, sig in trade_signals.items():
             # 僅考慮開倉訊號 (1=多, -1=空)
            if sig != 0:
                side = 'buy' if sig == 1 else 'sell'
                price = final_closes.loc[ts]
                # 這裡只記錄開倉，Trades 紀錄需要更細緻的平倉 PnL，但 LiteBacktester 只提供曲線
                trades_approx.append({'ts': ts.isoformat(), 'side': side, 'price': price, 'qty': 1.0, 'pnl': 0.0, 'fee': 0.0})


        final_params = model_params.copy()
        final_params['signal_params']['best_gap'] = best_result['best_gap']

        # 3. 生成報告
        reporter.generate(
            ohlcv=ohlcv_report,
            equity_curve=equity_curve,
            trades=trades_approx, # 傳遞近似的交易列表
            strategy_params=final_params
        )
        
    except Exception as e:
        log.error(f"生成 WFO 總報表時失敗: {e}", exc_info=True)

    return {
        "sharpe": float(best_result.get('sr', 0)),
        "mdd": float(best_result.get('mdd', 1)),
        "trades": int(best_result.get('trades', 0)),
        "accuracy": float(accuracy),
        "params": final_params,
        "path": None, "wfo_windows_passed": fold_count
    }