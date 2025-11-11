# quantx/core/scheduler/scheduler_state.py
# -*- coding: utf-8 -*-
# 版本: v2 (DI 重構)
# 說明:
# - 移除了對 get_runtime 的依賴。
# - __init__ 建構函數現在接收一個 runtime 參數，由外部注入。

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Tuple, Optional

from quantx.core.runtime import Runtime # 引入 Runtime 類型

class SchedulerState:
    """
    排程器狀態管理器
    - 負責讀取和寫入 `scheduler_state.json`。
    - 記錄每個訓練任務 (task_key) 的最後嘗試時間 (last_attempt_time)。
    """
    def __init__(self, runtime: Runtime, state_file: Optional[Path] = None):
        """
        初始化狀態管理器。

        Args:
            runtime (Runtime): 核心運行環境實例。
            state_file (Path, optional): 狀態檔案的路徑。若為 None，則使用預設路徑。
        """
        self.runtime = runtime
        if state_file is None:
            self.state_path = Path("results") / self.runtime.scope / "scheduler_state.json"
        else:
            self.state_path = state_file
        
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load()

    def _load(self) -> Dict[str, str]:
        """從 JSON 檔案安全地載入狀態。"""
        if not self.state_path.exists():
            return {}
        try:
            with self.state_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            self.runtime.log.warning(f"[SchedulerState] 狀態檔案 '{self.state_path}' 損毀，將建立新檔案。")
            return {}

    def _save(self):
        """將當前狀態寫入 JSON 檔案。"""
        try:
            with self.state_path.open("w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.runtime.log.error(f"[SchedulerState] 寫入狀態檔案失敗: {e}", exc_info=True)

    def get_last_attempt_time(self, task_key: Tuple[str, str, str]) -> Optional[datetime]:
        """
        獲取指定任務的最後嘗試時間。
        """
        key_str = "|".join(task_key)
        time_iso = self.state.get(key_str)
        if time_iso:
            try:
                return datetime.fromisoformat(time_iso)
            except ValueError:
                return None
        return None

    def record_attempt_time(self, task_key: Tuple[str, str, str]):
        """
        記錄指定任務的當前嘗試時間。
        """
        key_str = "|".join(task_key)
        self.state[key_str] = datetime.now(timezone.utc).isoformat()
        self._save()