# quantx/core/policy/candidate_store.py
# -*- coding: utf-8 -*-
# 版本: v7 (架構重構：徹底解耦)
# 說明:
# - 遵從模組化設計原則，徹底移除對 'quantx.core.runtime' 的依賴，從根本上解決循環引用問題。
# - __init__ 建構函數不再接收 runtime 物件，而是直接接收必要的依賴：'base_dir' (工作目錄) 和 'log' (日誌記錄器)。
# - 這使得 CandidateStore 成為一個職責單一、可獨立測試的純粹儲存模組。

import os
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Any, List, Dict

from quantx.core.utils import sanitize

class CandidateStore:
    def __init__(self, base_dir: Path, log: logging.Logger):
        """
        初始化候選池管理器。

        Args:
            base_dir (Path): 候選策略檔案的基底儲存目錄。
            log (logging.Logger): 用於記錄日誌的記錄器實例。
        """
        self.log = log
        self.base_dir = base_dir
        
        # 確保基底目錄存在
        os.makedirs(self.base_dir, exist_ok=True)
        
        self.log.info(f"[CandidateStore] 初始化完成，工作目錄: {self.base_dir}")

    # (其餘所有方法 _score_key, _select_top_and_latest, _load_json, _save_json, 
    #  _clean_old_models, add, list_candidates, list_candidates_for_symbol 保持不變)

    def _score_key(self, e: dict):
        """多條件排序: Sharpe 高 > MDD 低 > Trades 多"""
        sharpe = e.get("sharpe", -9999) or -9999
        mdd = e.get("mdd", 9999) or 9999
        trades = e.get("trades", 0) or 0
        return (sharpe, -mdd, trades)

    def _select_top_and_latest(self, entries: list):
        """篩選規則：保留 Sharpe 最佳的 3 筆 + 時間最新的 50 筆。"""
        if not entries: return []
        entries_sorted_time = sorted(entries, key=lambda x: x.get("time", ""), reverse=True)
        latest_50 = entries_sorted_time[:50]
        best3_sharpe = sorted(entries, key=self._score_key, reverse=True)[:3]
        combined = {json.dumps(e, sort_keys=True): e for e in (latest_50 + best3_sharpe)}
        selected = list(combined.values())
        return sorted(selected, key=self._score_key, reverse=True)

    def _load_json(self, path: Path) -> List[Dict]:
        """安全地載入 JSON 檔案。"""
        if not path.exists(): return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            self.log.warning(f"[Candidate] JSON 檔案 '{path}' 損毀或格式錯誤，將其視為空檔案。")
            return []

    def _save_json(self, path: Path, data: list):
        """安全地儲存 JSON 檔案。"""
        sanitized_data = sanitize(data)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(sanitized_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.log.error(f"[Candidate] 儲存 JSON 檔案 '{path}' 失敗: {e}", exc_info=True)

    def _clean_old_models(self, before: list, after: list):
        """比對更新前後的列表，刪除被淘汰的 ML 模型檔案。"""
        after_paths = {e.get("path") for e in after if e.get("kind") == "ml" and e.get("path")}
        for e in before:
            if e.get("kind") == "ml" and (path := e.get("path")) and path not in after_paths and os.path.exists(path):
                try:
                    os.remove(path)
                    self.log.info(f"[Candidate] 移除過舊模型檔: {path}")
                except Exception as ex:
                    self.log.warning(f"[Candidate] 刪除模型檔 '{path}' 失敗: {ex}")

    def add(self, kind: str, name: str, res: dict, symbol: str, tf: str):
        """將一筆新的訓練結果加入候選池。"""
        entry = {"kind": kind, "name": name, "symbol": symbol, "tf": tf, "time": datetime.now(timezone.utc).isoformat(), **res}
        
        subfile = self.base_dir / f"{symbol}_{tf}.json"
        entries = self._load_json(subfile)
        entries.append(entry)
        selected = self._select_top_and_latest(entries)
        
        self._clean_old_models(entries, selected)
        self._save_json(subfile, selected)

        sharpe = entry.get("sharpe")
        sharpe_str = f"{float(sharpe):.3f}" if sharpe is not None else "N/A"
        self.log.info(f"[Candidate] 已加入 {kind}/{name} ({symbol}-{tf})，Sharpe={sharpe_str}")
        return entry
    
    def list_candidates(self, symbol: str, tf: str) -> List[Dict]:
        """從單一檔案中讀取特定 symbol/tf 的候選者列表。"""
        subfile = self.base_dir / f"{symbol}_{tf}.json"
        return self._load_json(subfile)
    
    def list_candidates_for_symbol(self, symbol: str) -> List[Dict]:
        """列出特定 Symbol 在所有時間框 (tf) 下的候選者。"""
        all_candidates: List[Dict] = []
        pattern = f"{symbol}_*.json"
        
        for p in self.base_dir.glob(pattern):
            candidates = self._load_json(p)
            all_candidates.extend(candidates)
            
        return all_candidates