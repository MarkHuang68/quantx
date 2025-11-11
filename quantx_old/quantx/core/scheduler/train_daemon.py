# 檔案: quantx/core/scheduler/train_daemon.py
# 版本: v31 (DI 最終修正)
# 說明:
# - __init__ 建構函數現在接收顯式傳入的 runtime 和 store 實例。
# - 修正了 SchedulerState 初始化時缺少 runtime 參數的 TypeError。
# - 提供所有方法的完整程式碼，不再使用省略詞。

import time
from datetime import datetime, timedelta, timezone
import pandas as pd
from collections import Counter
import itertools
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING
from pathlib import Path
import logging

# 依賴項
if TYPE_CHECKING:
    from quantx.core.runtime import Runtime
    from quantx.core.data.loader import DataLoader

from quantx.core.policy.candidate_store import CandidateStore
from quantx.core.gate import check_gate
from quantx.core.signal_handler import should_stop
from quantx.core.config import Config, TrainConfig
from quantx.optimize.wfo import WalkForwardOptimizer
from quantx.core.timeframe import parse_tf_minutes
from quantx.core.model.ml_wfo_trainer import run_ml_wfo
from quantx.core.features.main_features import MainFeatures
from quantx.core.labelers.future_return import make_labels_triple_barrier
from quantx.core.scheduler.scheduler_state import SchedulerState

def _generate_param_combinations(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """將參數網格轉換為參數組合的列表。"""
    if not grid:
        return [{}]
    keys, values = grid.keys(), grid.values()
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    for params in combinations:
        for k, v in params.items():
            if isinstance(v, (int, float)): continue
            try: params[k] = float(v)
            except (TypeError, ValueError): pass
    return combinations

class SchedulerTrain:
    def __init__(self, cfg: Config):
        """
        初始化訓練排程器。
        注意：核心依賴將由 DI 容器在外部注入。
        """
        self.cfg = cfg
        self.runtime: Optional[Runtime] = None
        self.log: Optional[logging.Logger] = None
        self.store: Optional[CandidateStore] = None
        self.loader: Optional[DataLoader] = None
        
        self.cfg_train: TrainConfig = self.cfg.load_train()
        self.symbols: List[List[str]] = self.cfg.load_symbol()
        
        self.state_manager: Optional[SchedulerState] = None

    def _ensure_dependencies(self):
        """檢查核心依賴是否已被注入，並初始化依賴於 runtime 的組件。"""
        if not all([self.runtime, self.log, self.store, self.loader]):
            raise RuntimeError("SchedulerTrain 核心依賴 (runtime, log, store, loader) 未被注入！")
        
        if self.state_manager is None:
            self.state_manager = SchedulerState(self.runtime)

    def _get_latest_train_success_time(self, symbols_str: str, tf: str, name: str) -> Optional[datetime]:
        self._ensure_dependencies()
        # list_candidates 仍然使用單一 symbol，但存儲時我們用了組合 str。
        # 這裡我們需要一個策略來處理。最簡單的方式是假設第一個 symbol 代表了整個組。
        # 注意：這是一個簡化，一個更健壯的系統可能需要修改 CandidateStore。
        main_symbol = symbols_str.split('_')[0]
        candidates = self.store.list_candidates(main_symbol, tf)
        latest_time = None
        for c in candidates:
            # 我們需要檢查存儲的 symbol 標識符是否匹配
            if c.get("name") == name and c.get("symbol") == symbols_str:
                try:
                    time_str = c.get("time")
                    if time_str is None: continue
                    dt_obj = datetime.fromisoformat(time_str)
                    current_time = dt_obj.astimezone(timezone.utc) if dt_obj.tzinfo else dt_obj.replace(tzinfo=timezone.utc)
                    if latest_time is None or current_time > latest_time:
                        latest_time = current_time
                except Exception as e:
                    self.log.debug(f"[Freshness] 解析時間字串失敗: {c.get('time')} - {e}")
                    continue
        return latest_time

    def _should_train_now(self, task_key: Tuple[str, str, str], interval_minutes: int) -> Tuple[bool, str]:
        self._ensure_dependencies()
        symbols_str, tf, name = task_key
        now = datetime.now(timezone.utc)
        last_success_time = self._get_latest_train_success_time(symbols_str, tf, name)
        last_attempt_time = self.state_manager.get_last_attempt_time(task_key)
        retry_minutes = self.cfg_train.schedule.train_retry_minutes

        if last_attempt_time is None:
            return True, "從未執行過"
        if last_success_time and last_success_time >= last_attempt_time:
            required_interval = timedelta(minutes=interval_minutes)
            time_elapsed = now - last_success_time
            if time_elapsed >= required_interval:
                return True, f"已超過成功週期 ({interval_minutes} 分鐘)"
            else:
                remaining = (required_interval - time_elapsed).total_seconds() / 60
                return False, f"成功週期內，跳過 (剩餘 {remaining:.1f} 分鐘)"
        else:
            required_interval = timedelta(minutes=retry_minutes)
            time_elapsed = now - last_attempt_time
            if time_elapsed >= required_interval:
                return True, f"已超過失敗重試週期 ({retry_minutes} 分鐘)"
            else:
                remaining = (required_interval - time_elapsed).total_seconds() / 60
                return False, f"失敗冷卻中，等待重試 (剩餘 {remaining:.1f} 分鐘)"
    
    def _run_ml_tasks(self, tasks):
        self._ensure_dependencies()
        self.log.info(f"====== [ML 訓練週期啟動] ======")
        end_dt, start_dt = datetime.now(timezone.utc), datetime.now(timezone.utc) - timedelta(days=self.cfg_train.schedule.train_window_days)
        for task in tasks:
            if should_stop(): break
            if not getattr(task, 'ml_models', []): continue

            # 檢查是否為多交易對任務，如果是，則跳過 ML 訓練
            symbols_to_load = getattr(task, 'symbols', getattr(task, 'symbol', None))
            if isinstance(symbols_to_load, list) and len(symbols_to_load) > 1:
                self.log.warning(f"任務 '{getattr(task, 'name', 'Unnamed')}' 是多交易對任務，目前不支持 ML 訓練，將跳過。")
                continue

            # 確保我們有一個單一的 symbol 來加載數據
            symbol_to_load = symbols_to_load[0] if isinstance(symbols_to_load, list) else symbols_to_load
            if not symbol_to_load:
                continue

            df_full = self.loader.load_ohlcv(symbol_to_load, task.tf, start_dt, end_dt)
            if df_full.empty:
                self.log.warning(f"數據為空，跳過 {symbol_to_load}-{task.tf} 的 ML 訓練")
                continue

            # 將 symbol 存回 task 物件，以確保 _run_ml_training_for_task 能正常工作
            task.symbol = symbol_to_load
            self._run_ml_training_for_task(task, df_full.copy())
            
    def _run_strategy_tasks(self, tasks):
        self._ensure_dependencies()
        self.log.info(f"====== [策略訓練週期啟動] ======")
        end_dt, start_dt = datetime.now(timezone.utc), datetime.now(timezone.utc) - timedelta(days=self.cfg_train.schedule.train_window_days)
        for task in tasks:
            if should_stop(): break
            if not getattr(task, 'strategies', []): continue

            # --- [統一化] 多/單交易對數據加載 ---
            # 優先使用 symbols 列表，如果沒有，則從舊的 symbol 屬性創建列表
            symbols_to_load = getattr(task, 'symbols', None)
            if symbols_to_load is None:
                single_symbol = getattr(task, 'symbol', None)
                if single_symbol:
                    symbols_to_load = [single_symbol]
                else:
                    self.log.warning(f"任務 {getattr(task, 'name', 'Unnamed')} 既沒有 'symbols' 也沒有 'symbol' 屬性，跳過。")
                    continue

            # 將 symbols 列表存回 task 物件，以便後續使用
            task.symbols = symbols_to_load

            all_data = {}
            for symbol in symbols_to_load:
                data = self.loader.load_ohlcv(symbol, task.tf, start_dt, end_dt)
                if data is None or data.empty:
                    self.log.warning(f"為任務 '{getattr(task, 'name', '')}' 加載數據時，找不到 {symbol} 的數據，將跳過整個任務。")
                    all_data = {} # 如果任何一個 symbol 數據缺失，則清空 all_data
                    break
                all_data[symbol] = data

            if not all_data:
                continue
            # --- 數據加載結束 ---

            self._run_strategy_wfo_for_task(task, all_data)

    def _run_ml_training_for_task(self, task, df_full):
        self._ensure_dependencies()
        global_feature_cfg = getattr(self.cfg_train, 'features', {})
        feature_grid = getattr(task, 'ml_models', [{}])[0].get('feature_params_grid', {})
        feature_combos = _generate_param_combinations(feature_grid) if feature_grid else [{}]
        self.log.info(f"--- [ML-TASK: {task.symbol}-{task.tf}] 總共 {len(feature_combos)} 個特徵組合 ---")
        for i_feat, feat_combo in enumerate(feature_combos):
            if should_stop(): break
            for ml_item in getattr(task, 'ml_models', []):
                if should_stop(): break
                name = ml_item.get("name")
                task_key = (task.symbol, task.tf, name)
                can_train, reason = self._should_train_now(task_key, self.cfg_train.schedule.ml_train_minutes)
                if not can_train:
                    self.log.info(f"  [Skip ML] {name} ({task.symbol}-{task.tf}) - {reason}")
                    continue
                self.log.info(f"  [RUN ML] {name} ({task.symbol}-{task.tf}) - {reason}")
                self.state_manager.record_attempt_time(task_key)
                current_feature_cfg = {**global_feature_cfg, **feat_combo}
                try:
                    feature_generator = MainFeatures(cfg=current_feature_cfg)
                    df_features = feature_generator.transform(df_full)
                except Exception as e:
                    self.log.error(f"生成特徵時發生嚴重錯誤: {e}", exc_info=True)
                    continue
                label_cfg_list = getattr(self.cfg_train, 'labeling', [])
                if not isinstance(label_cfg_list, list): label_cfg_list = [label_cfg_list]
                for i_label, label_cfg in enumerate(label_cfg_list):
                    if should_stop(): break
                    self.log.info(f"--- [ML: {name}] [標籤策略 {i_label+1}/{len(label_cfg_list)}, 特徵 {i_feat+1}/{len(feature_combos)}] ---")
                    try:
                        labeled_df = make_labels_triple_barrier(df=df_full, tf=task.tf, max_hours=label_cfg.max_hours, up_k=label_cfg.up_k, dn_k=label_cfg.dn_k)
                        df_labels = labeled_df['y']
                        common_index = df_features.index.intersection(df_labels.index)
                        if common_index.empty:
                            self.log.warning("特徵與標籤對齊後數據為空，跳過此輪訓練。")
                            continue
                        X, y = df_features.loc[common_index], df_labels.loc[common_index]
                        close, ts = df_full.loc[common_index]['close'], pd.Series(common_index.astype('int64') // 10**9, index=common_index)
                        dataset = {"X": X, "y": y, "close": close, "ts": ts}
                        wfo_params = ml_item.get('wfo_params', {'is_days': 42, 'oos_days': 7, 'purge_days': 2})
                        res = run_ml_wfo(self.runtime, dataset, task.tf, name, ml_item, wfo_params, symbol=task.symbol)
                        res['params']['labeling'], res['params']['features'], res['params']['wfo_window'] = label_cfg.__dict__, feat_combo, wfo_params
                        ok, reasons = check_gate(res, self.cfg_train.auto_policy.get("gate", {}))
                        if ok and res.get("wfo_windows_passed", 0) > 0:
                            self.log.info(f"✅ ML 模型 '{name}' WFO 成功並通過 Gate！")
                            self.store.add("ml", name, res, task.symbol, task.tf)
                        else:
                            self.log.warning(f"❌ ML 模型 '{name}' 未通過 Gate 或 WFO 驗證。原因: {reasons}")
                    except Exception as e:
                        self.log.error(f"處理 ML 模型 {name} 時發生嚴重錯誤: {e}", exc_info=True)
                        continue
            
    def _run_strategy_wfo_for_task(self, task, all_data: Dict[str, pd.DataFrame]):
        self._ensure_dependencies()

        symbols_in_task = getattr(task, 'symbols', [])
        if not symbols_in_task:
            self.log.warning(f"任務 {task.name} 在 WFO 執行時缺少 'symbols' 列表。")
            return

        # 使用第一個 symbol 的數據來做為長度檢查的基準
        df_main = all_data[symbols_in_task[0]]

        try: min_bars_needed = (21 + 7) * (1440 / parse_tf_minutes(task.tf))
        except (ValueError, ZeroDivisionError): min_bars_needed = 200
        if len(df_main) < min_bars_needed:
            self.log.warning(f"主數據不足 ({len(df_main)} < {min_bars_needed})，無法執行策略 WFO。")
            return

        for strat_cfg in getattr(task, 'strategies', []):
            if should_stop(): break
            strat_name, param_grid = strat_cfg["name"], strat_cfg.get("params", {})

            # 使用多個 symbol 組成 task_key
            symbols_str = "_".join(symbols_in_task)
            task_key = (symbols_str, task.tf, strat_name)

            can_train, reason = self._should_train_now(task_key, self.cfg_train.schedule.strategy_train_minutes)
            if not can_train:
                self.log.info(f"  [Skip Strat] {strat_name} ({symbols_str}-{task.tf}) - {reason}")
                continue
            self.log.info(f"  [RUN Strat] {strat_name} ({symbols_str}-{task.tf}) - {reason}")
            self.state_manager.record_attempt_time(task_key)
            self.log.info(f"--- 開始對策略 '{strat_name}' ({symbols_str}) 執行 WFO ---")
            try:
                strategy_cls = self.runtime.load_strategy(strat_name)
                wfo_gate = self.cfg_train.auto_policy.get("gate", {})
                wfo = WalkForwardOptimizer(strategy_cls=strategy_cls, param_grid=param_grid, runtime=self.runtime, gate=wfo_gate)
                # 將整個 all_data 字典傳遞給 WFO
                wfo_results = wfo.run(datas=all_data, symbols=symbols_in_task, tf=task.tf, equity=10000.0)
                successful_folds = [f for f in wfo_results if 'candidate' in f]
                if not successful_folds:
                    self.log.warning(f"策略 '{strat_name}' WFO 未產生任何成功的候選。")
                    continue
                avg_sharpe = sum(f['candidate']['perf'].get('sharpe_ratio', 0) for f in successful_folds) / len(successful_folds)
                avg_mdd = sum(f['candidate']['perf'].get('max_drawdown', 1) for f in successful_folds) / len(successful_folds)
                total_trades = sum(f['candidate']['perf'].get('fills', 0) for f in successful_folds)
                param_tuples = [tuple(sorted(f['candidate']['params'].items())) for f in successful_folds]
                best_params = dict(Counter(param_tuples).most_common(1)[0][0])
                res = {"sharpe": avg_sharpe, "mdd": avg_mdd, "trades": total_trades, "params": best_params, "path": None, "wfo_windows_passed": len(successful_folds)}
                self.log.info(f"--- WFO 最終結果總結 (策略) ---\n  - 績效: Sharpe={res['sharpe']:.4f}, MDD={res['mdd']:.4f}, Trades={res['trades']}\n  - 代表性參數: {best_params}\n--------------------------------")
                ok, reasons = check_gate(res, self.cfg_train.auto_policy.get("gate", {}))
                if ok:
                    self.log.info(f"✅ 策略 '{strat_name}' WFO 成功並通過 Gate！")
                    self.store.add("strategy", strat_name, res, symbols_str, task.tf) # 使用組合的 symbols_str
                else:
                    self.log.warning(f"❌ 策略 '{strat_name}' 未通過 Gate 或 WFO 驗證。原因: {reasons}")
            except Exception as e:
                self.log.exception(f"策略 '{strat_name}' WFO 訓練失敗: {e}")

    def run_forever(self):
        """常駐執行主迴圈，實現熱更新與排程。"""
        self._ensure_dependencies()
        last_ml_run_time = 0
        last_strategy_run_time = 0
        master_check_interval = 30

        self.log.info("########### 訓練排程器啟動 (DI 最終修正版) ###########")

        while not should_stop():
            current_time = time.time()
            try:
                self.cfg_train = self.cfg.load_train()
                self.symbols = self.cfg.load_symbol()
                ml_interval_seconds = self.cfg_train.schedule.ml_train_minutes * 60
                strategy_interval_seconds = self.cfg_train.schedule.strategy_train_minutes * 60
            except Exception as e:
                self.log.error(f"[熱更新] 重新載入設定檔失敗: {e}", exc_info=True)
                time.sleep(master_check_interval)
                continue

            tasks = self.cfg_train.expand_tasks(self.symbols)
            if not tasks:
                self.log.warning("熱更新後，沒有可執行的訓練任務，排程器將閒置。")
                time.sleep(master_check_interval)
                continue

            if current_time - last_ml_run_time >= ml_interval_seconds:
                self.log.info(f"########### ML 訓練週期觸發 ###########")
                try:
                    self._run_ml_tasks(tasks)
                except Exception as e:
                    self.log.error(f"ML 訓練週期發生嚴重錯誤: {e}", exc_info=True)
                last_ml_run_time = time.time()
                self.log.info(f"########### ML 訓練週期結束 ###########\n")

            if current_time - last_strategy_run_time >= strategy_interval_seconds:
                self.log.info(f"########### 策略訓練週期觸發 ###########")
                try:
                    self._run_strategy_tasks(tasks)
                except Exception as e:
                    self.log.error(f"策略訓練週期發生嚴重錯誤: {e}", exc_info=True)
                last_strategy_run_time = time.time()
                self.log.info(f"########### 策略訓練週期結束 ###########\n")

            time.sleep(master_check_interval)

        self.log.info("########### 訓練排程器已安全停止 ###########")