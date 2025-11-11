# 檔案: quantx/core/config.py
# 版本: v8 (Live Config 讀取修正)
# 說明:
# - 修正了 load_risk/load_live 函數，使其能夠讀取 live.yaml 中的新 reporting 區塊。

import os
import yaml
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import copy # 🟢 引入 copy 模組

def _load_yaml(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config 檔不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        content = yaml.safe_load(f)
        return content if content is not None else {}

@dataclass
class ScheduleConfig:
    ml_train_minutes: int = 1440
    strategy_train_minutes: int = 1440
    train_retry_minutes: int = 180
    train_window_days: int = 90

@dataclass
class LabelingConfig:
    max_hours: int = 8
    up_k: float = 1.5
    dn_k: float = 1.5

@dataclass
class OverrideConfig:
    symbol: str
    tf: Optional[str] = None
    ml_models: List[dict] = field(default_factory=list)
    strategies: List[dict] = field(default_factory=list)

@dataclass
class TrainTask:
    symbol: str
    tf: str
    ml_models: List[dict]
    strategies: List[dict]

@dataclass
class TrainConfig:
    schedule: ScheduleConfig
    labeling: List[LabelingConfig]
    features: dict = field(default_factory=dict)
    ml_models: List[dict] = field(default_factory=list)
    strategies: List[dict] = field(default_factory=list)
    overrides: List[OverrideConfig] = None
    auto_policy: dict = field(default_factory=dict)
    debug: bool = False

    @staticmethod
    def load_from_yaml(path: str) -> "TrainConfig":
        raw = _load_yaml(path)
        schedule_raw = raw.get("schedule", {})
        sched = ScheduleConfig(
            ml_train_minutes=schedule_raw.get("ml_train_minutes", 1440),
            strategy_train_minutes=schedule_raw.get("strategy_train_minutes", 1440),
            train_retry_minutes=schedule_raw.get("train_retry_minutes", 180), 
            train_window_days=schedule_raw.get("train_window_days", 90),
        )
        
        labeling_raw = raw.get("labeling", [])
        if not isinstance(labeling_raw, list):
            labeling_raw = [labeling_raw]
        labels = [LabelingConfig(**lbl) for lbl in labeling_raw if isinstance(lbl, dict)]

        overrides = [OverrideConfig(**ov) for ov in raw.get("overrides", [])]

        return TrainConfig(
            schedule=sched,
            labeling=labels,
            features=raw.get("features", {}),
            ml_models=raw.get("ml_models") or [],
            strategies=raw.get("strategies") or [],
            overrides=overrides,
            auto_policy=raw.get("auto_policy", {}),
            debug=raw.get("debug", False),
        )

    def expand_tasks(self, symbols: List[List[str]]) -> List[TrainTask]:
        tasks: List[TrainTask] = []
        for sym, tf in symbols:
            # 🟢 核心修改：使用 deepcopy 確保不影響原始設定
            ml_base = copy.deepcopy(self.ml_models or [])
            strategies_base = copy.deepcopy(self.strategies or [])

            for ov in self.overrides:
                if ov.symbol == sym and (ov.tf is None or ov.tf == tf):
                    # --- 覆寫 ML 模型參數 ---
                    if ov.ml_models:
                        for override_model in ov.ml_models:
                            for base_model in ml_base:
                                if base_model.get("name") == override_model.get("name"):
                                    # 找到同名模型，只更新 params
                                    if "params" in override_model:
                                        base_model["params"] = override_model["params"]
                                    break # 繼續處理下一個 override_model

                    # --- 覆寫策略參數 ---
                    if ov.strategies:
                        for override_strategy in ov.strategies:
                            for base_strategy in strategies_base:
                                if base_strategy.get("name") == override_strategy.get("name"):
                                    # 找到同名策略，只更新 params
                                    if "params" in override_strategy:
                                        base_strategy["params"] = override_strategy["params"]
                                    break # 繼續處理下一個 override_strategy
            
            tasks.append(TrainTask(
                symbol=sym,
                tf=tf,
                ml_models=ml_base,
                strategies=strategies_base,
            ))
        return tasks


class Config:
    def __init__(self, path="conf/config.yaml"):
        self.path = path
        self.master = _load_yaml(self.path)
        self.train: Optional[TrainConfig] = None
        self.symbols: List[List[str]] = []
        self.exchange: dict = {}
        self.risk: dict = {}

    def reload(self):
        """重新載入主設定檔和符號列表，並清除舊的 risk/live 快取。"""
        self.master = _load_yaml(self.path)
        # 清除舊的快取，以便下次呼叫 load_risk/load_symbol 時能重新載入
        self.risk = {}
        self.symbols = []
        # 注意: train 和 exchange 在 live runner 中通常不會被熱載入，所以暫時不清

    def load_train(self):
        if "train" not in self.master:
            raise KeyError("config.yaml 缺少 train 路徑設定")
        path = self.master["train"]
        self.train = TrainConfig.load_from_yaml(path)
        return self.train

    def load_symbol(self):
        if "symbol" not in self.master:
            raise KeyError("config.yaml 缺少 symbol 路徑設定")
        path = self.master["symbol"]
        self.symbols = _load_yaml(path).get("symbols", [])
        return self.symbols

    def load_exchange(self):
        if "exchange" not in self.master:
            raise KeyError("config.yaml 缺少 exchange 路徑設定")
        path = self.master["exchange"]
        self.exchange = _load_yaml(path)
        return self.exchange

    def load_risk(self):
        path = self.master.get("risk") or self.master.get("live")
        if not path:
            raise KeyError("config.yaml 缺少 risk 或 live 路徑設定")
        # 🟢 核心修正: 確保 load_risk 能夠載入 live.yaml 的所有頂層區塊 (包括 reporting)
        self.risk = _load_yaml(path)
        return self.risk