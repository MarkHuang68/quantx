# 檔案: quantx/containers.py
# 版本: v3 (最終修正版)
# 說明:
# - 修正了服務實例化的順序，確保 AutoPolicy 和 SchedulerTrain 能獲得正確注入的依賴。
# - 確保 CandidateStore 只被實例化一次，並共享給所有需要的服務。

from __future__ import annotations
import os
from pathlib import Path
from typing import Optional

from quantx.core.config import Config
from quantx.core.runtime import Runtime
from quantx.market.provider.bybit import BybitProvider
from quantx.market.provider.simulated import SimulatedProvider
from quantx.core.data.loader import DataLoader
from quantx.core.policy.candidate_store import CandidateStore
from quantx.core.policy.auto_policy import AutoPolicy
from quantx.core.scheduler.train_daemon import SchedulerTrain

def _load_env():
    """從 .env 檔案載入環境變數。"""
    p = Path(".env")
    if not p.exists(): return
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s: continue
        k, v = s.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

class AppContainer:
    """
    負責實例化並組織所有核心服務的依賴注入容器。
    """
    def __init__(self, cfg: Config, mode_override: Optional[str] = None):
        """
        初始化容器並組裝所有服務。
        """
        _load_env()
        self._cfg = cfg

        # 1. 確定運行模式和交易所
        self.exchange = (os.getenv("exchange") or "bybit").strip().lower()
        self.mode = mode_override.strip().lower() if mode_override else (os.getenv("mode") or "testnet").strip().lower()

        # 2. 載入設定檔
        exchange_config = self._cfg.load_exchange()
        live_config = self._cfg.load_risk()
        test_run_flag = live_config.get('test_run', True)

        # 3. 實例化服務 (遵循依賴順序)
        if self.exchange == 'bybit':
            self._provider = BybitProvider(mode=self.mode, test_run=test_run_flag)
        else:
            self._provider = SimulatedProvider()

        self._data_loader = DataLoader(
            cache_dir="cache",
            scope=f"{self.exchange}_{self.mode}",
            provider=self._provider
        )

        self._runtime = Runtime(
            exchange=self.exchange,
            mode=self.mode,
            loader=self._data_loader,
            exchange_config=exchange_config,
            risk=live_config,
            provider=self._provider
        )

        # 核心修正 1：創建單一的 CandidateStore 實例
        candidate_base_dir = Path("results") / self._runtime.scope / "candidates"
        self._candidate_store = CandidateStore(base_dir=candidate_base_dir, log=self._runtime.log)

        # 核心修正 2：將 store 實例注入到 AutoPolicy
        self._auto_policy = AutoPolicy(
            runtime=self._runtime,
            risk_cfg=self._runtime.risk,
            store=self._candidate_store
        )
        
        # 將 AutoPolicy 實例賦值給 LiveRuntime
        self._runtime.live.auto_policy = self._auto_policy

        # 核心修正 3：創建 SchedulerTrain 並注入所有依賴
        self._scheduler = SchedulerTrain(cfg=self._cfg)
        self._scheduler.runtime = self._runtime
        self._scheduler.log = self._runtime.log
        self._scheduler.store = self._candidate_store
        self._scheduler.loader = self._runtime.loader


    # --- 提供 Property 讓外部可以取得服務實例 ---
    @property
    def config(self) -> Config:
        return self._cfg

    @property
    def runtime(self) -> Runtime:
        return self._runtime

    @property
    def auto_policy(self) -> AutoPolicy:
        return self._auto_policy

    @property
    def scheduler(self) -> SchedulerTrain:
        return self._scheduler