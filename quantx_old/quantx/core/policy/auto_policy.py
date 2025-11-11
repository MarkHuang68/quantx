# 檔案: quantx/core/policy/auto_policy.py
# 版本: v11 (最終修正：日誌降級)
# 說明:
# - 修正 _get_best_candidate_for_symbol 中的全局緩存 (global _SYMBOL_BEST_CACHE)，
#   新增時間戳來判斷緩存是否過期，確保 LiveRunner 能讀取到訓練守護進程新增的新策略。
# - 將例行性檢查的 INFO/WARNING 日誌降級到 DEBUG，以減少 Console 輸出噪音。

import time
import pandas as pd
import logging # 確保 logging 被正確導入
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import json

from .candidate_store import CandidateStore
from quantx.core.executor.base import BaseExecutor
from quantx.core.runtime import Runtime
from quantx.core.utils import normalize_candidates, score_candidates

# 🟢 核心修改 1: 緩存結構變更
# 儲存 (Executor, Score, selected_tf, last_read_timestamp)
_SYMBOL_BEST_CACHE: Dict[str, Tuple[Optional[BaseExecutor], Optional[float], Optional[str], float]] = {}
# 🟢 緩存生命週期（例如 5 分鐘，確保不會頻繁讀取磁碟，但也不會永久過期）
CACHE_LIFETIME_SECONDS = 300.0 

class AutoPolicy:
    """
    自動策略/模型決策器 (Auto-Policy)
    """

    def __init__(self, runtime: Runtime, risk_cfg: dict, store: CandidateStore):
        """
        初始化 AutoPolicy。

        Args:
            runtime (Runtime): 核心運行環境實例。
            risk_cfg (dict): 從 live.yaml 載入的風險與 auto_policy 設定。
            store (CandidateStore): 候選池的儲存服務實例。
        """
        self.runtime = runtime
        self.log = runtime.log
        self.risk_cfg = risk_cfg.get('auto_policy', {}).get('gate', {})
        self.weights = risk_cfg.get('auto_policy', {}).get('weights', {})
        self.store = store 
        
        self.log.info(f"[AutoPolicy] 初始化完成。Gate 條件: {self.risk_cfg}")
        global _SYMBOL_BEST_CACHE
        # 在初始化時清除所有舊的緩存，確保啟動時是全新狀態
        _SYMBOL_BEST_CACHE = {} 

    def _load_executor(self, candidate: pd.Series) -> Optional[BaseExecutor]:
        """根據候選者資訊，動態載入並實例化對應的策略或 ML 執行器。"""
        kind, name = candidate['kind'], candidate['name']
        params = candidate.get('params', {})
        # 處理 params 可能是 JSON 字串的情況
        if isinstance(params, str):
            try: params = json.loads(params)
            except: params = {}
        try:
            if kind == 'strategy':
                StrategyCls = self.runtime.load_strategy(name)
                if not StrategyCls: raise ImportError(f"無法從 runtime 載入策略 '{name}'")
                return StrategyCls(**params)
            elif kind == 'ml':
                self.log.warning(f"[AutoPolicy] ML 模型的載入邏輯尚未實現: {name}")
                return None
            else:
                self.log.error(f"[AutoPolicy] 未知的執行器類型: {kind}")
                return None
        except Exception as e:
            self.log.error(f"[AutoPolicy] 載入執行器 {kind}/{name} 失敗: {e}", exc_info=True)
            return None

    def _apply_gate_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Gate 過濾：篩掉不滿足基本門檻的候選者。
        """
        if df.empty: return df
        min_sharpe = self.risk_cfg.get("min_sharpe", -999)
        max_mdd = self.risk_cfg.get("max_mdd", 999)
        min_trades = self.risk_cfg.get("min_trades", 0)
        
        df_filtered = df[(df["sharpe"].fillna(-999) >= min_sharpe) & 
                         (df["mdd"].fillna(999) <= max_mdd) & 
                         (df["trades"].fillna(0) >= min_trades)].copy()
        
        # 處理 ML 模型的 acc 檢查 (這裡使用 df.query 進行過濾)
        if 'min_acc' in self.risk_cfg:
             min_acc = self.risk_cfg['min_acc']
             # 僅針對 type='ml' 的紀錄，檢查 val_acc
             ml_filter = (df_filtered['kind'] != 'ml') | (df_filtered.get('val_acc', 0.0) >= min_acc)
             df_filtered = df_filtered[ml_filter].copy()

        return df_filtered

    def _score_candidates(self, df: pd.DataFrame) -> pd.DataFrame:
        """對通過 Gate 的候選者進行加權計分 (Scoring)。"""
        return score_candidates(df, self.weights)

    def _get_best_candidate_for_symbol(self, symbol: str) -> Tuple[Optional[BaseExecutor], Optional[float], Optional[str]]:
        """
        核心邏輯：從所有 TF 中找出指定 Symbol 的最佳策略，並使用具有過期機制的緩存。
        """
        global _SYMBOL_BEST_CACHE
        now = time.time()
        
        # 1. 檢查緩存的新鮮度
        if symbol in _SYMBOL_BEST_CACHE:
            executor, score, best_tf, timestamp = _SYMBOL_BEST_CACHE[symbol]
            time_since_read = now - timestamp

            # 🟢 判斷緩存是否過期
            is_stale = time_since_read >= CACHE_LIFETIME_SECONDS
            if not is_stale:
                return executor, score, best_tf # 返回緩存結果
            
            # 🟢 核心修改：將緩存過期日誌降級到 DEBUG
            self.log.debug(f"[AutoPolicy] {symbol} 緩存已過期 ({time_since_read:.1f}s)，強制重新載入。")
        else:
            # 首次載入，保持 INFO 級別
            self.log.info(f"[AutoPolicy] 首次執行 Symbol 級別評分: {symbol} (從 {self.store.base_dir} 載入新數據)")


        # 2. 從所有 TF 中載入所有候選者 (從磁碟讀取最新數據)
        candidates = self.store.list_candidates_for_symbol(symbol)
        
        if not candidates:
            # 🟢 核心修改：將無候選日誌降級到 DEBUG
            self.log.debug(f"[AutoPolicy] {symbol} 在所有時間框中無任何候選策略/模型。")
            _SYMBOL_BEST_CACHE[symbol] = (None, None, None, now) 
            return None, None, None

        df = pd.DataFrame(normalize_candidates(candidates))
        df_filtered = self._apply_gate_filter(df)
        
        if df_filtered.empty:
            # 🟢 核心修改：將未通過 Gate 日誌降級到 DEBUG
            self.log.debug(f"[AutoPolicy] {symbol} 沒有任何候選者通過 Gate 風險門檻。")
            _SYMBOL_BEST_CACHE[symbol] = (None, None, None, now) 
            return None, None, None

        df_scored = self._score_candidates(df_filtered)
        if df_scored.empty:
            # 🟢 核心修改：將評分後無合格者日誌降級到 DEBUG
            self.log.debug(f"[AutoPolicy] {symbol} 評分後沒有合格的候選者。")
            _SYMBOL_BEST_CACHE[symbol] = (None, None, None, now) 
            return None, None, None

        best_candidate = df_scored.sort_values("score", ascending=False).iloc[0]
        best_score, best_tf = best_candidate['score'], best_candidate['tf']

        # 3. 載入並緩存結果 (包含時間戳)
        executor = self._load_executor(best_candidate)
        result = (executor, best_score, best_tf)
        _SYMBOL_BEST_CACHE[symbol] = (executor, best_score, best_tf, now)
        
        # 🟢 保持 INFO 級別的成功日誌，以便用戶知道策略已被選中
        self.log.info(f"--- {symbol} 最佳策略 (選定) ---\n  TF: {best_tf}, 策略: {best_candidate['kind']}/{best_candidate['name']}\n  Sharpe: {best_candidate['sharpe']:.3f}, MDD: {best_candidate['mdd']:.3f}\n  最終分數: {best_score:.2f}\n--------------------------")
        
        return result


    def select_executor(self, symbol: str, tf: str) -> Tuple[Optional[BaseExecutor], Optional[float]]:
        """
        為指定的 symbol-tf 挑選最佳的執行單位 (Executor) (用於 LiveRunner 啟動時)。
        """
        executor, best_score, best_tf = self._get_best_candidate_for_symbol(symbol)
        
        if executor is None: return None, None
        
        # 篩選：只有當前 LiveRunner 任務的 TF 與最佳 TF 相符時，才返回策略
        if tf == best_tf:
             # 保持 INFO 級別的匹配成功日誌
             self.log.info(f"[AutoPolicy] ✅ 匹配成功：選定 {symbol}-{tf} (Score: {best_score:.2f})")
             return executor, best_score
        else:
             # 🟢 核心修改：將不匹配日誌降級到 DEBUG
             self.log.debug(f"[AutoPolicy] ❌ 匹配失敗：{symbol}-{tf} 跳過。最佳 TF 是 {best_tf}。")
             return None, None


    def check_for_better_executor(self, symbol: str, tf: str, current_executor_score: float) -> Tuple[Optional[BaseExecutor], Optional[float]]:
        """
        從候選池中尋找分數高於目前執行器的新最佳執行器 (用於空倉切換)。
        """
        
        candidates = self.store.list_candidates(symbol, tf)
        if not candidates: return None, None
        df = pd.DataFrame(normalize_candidates(candidates))
        df_filtered = self._apply_gate_filter(df)
        df_scored = self._score_candidates(df_filtered)
        if df_scored.empty: return None, None
        best_candidate = df_scored.sort_values("score", ascending=False).iloc[0]
        best_score = best_candidate['score']
        
        # 確保新策略的分數顯著優於舊策略 (1.01 = 1% 優勢)
        if best_score > 0 and best_score > current_executor_score * 1.01: 
            # 保持 INFO 級別的策略切換日誌
            self.log.info(f"[AutoPolicy] 發現更優策略！舊分數: {current_executor_score:.2f} < 新分數: {best_score:.2f} 於 {symbol}-{tf}")
            if new_executor := self._load_executor(best_candidate):
                return new_executor, best_score
        return None, None

    def select_best_for_legacy_takeover(self, symbol: str) -> Tuple[Optional[BaseExecutor], Optional[float], Optional[str]]:
        """
        為遺留倉位選擇策略時，忽略 Gate 門檻，只使用評分權重。
        """
        self.log.info(f"[AutoPolicy-Legacy] 執行 Legacy Takeover 評分 (忽略 Gate): {symbol}")
        candidates = self.store.list_candidates_for_symbol(symbol)
        if not candidates:
            self.log.warning(f"[AutoPolicy-Legacy] {symbol} 在所有時間框中無任何候選策略/模型。")
            return None, None, None
        
        df = pd.DataFrame(normalize_candidates(candidates))
        # 不執行 _apply_gate_filter
        df_scored = self._score_candidates(df)
        
        if df_scored.empty:
            return None, None, None
            
        best_candidate = df_scored.sort_values("score", ascending=False).iloc[0]
        best_score, best_tf = best_candidate['score'], best_candidate['tf']
        
        self.log.info(f"--- {symbol} 遺留倉位接管策略 ---\n  TF: {best_tf}, 策略: {best_candidate['kind']}/{best_candidate['name']}\n  分數: {best_score:.2f}\n----------------------------------")
        
        return self._load_executor(best_candidate), best_score, best_tf