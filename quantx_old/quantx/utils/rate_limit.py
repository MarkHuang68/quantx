# -*- coding: utf-8 -*-
# quantx/utils/rate_limit.py
from __future__ import annotations
import time, math, random
from typing import Optional

class AdaptivePacer:
    """
    自適應節流器：
    - 以 min_interval 控制節奏（秒/請求）
    - 每次撞到限流就「暫時」提高 interval，冷卻期過後緩慢恢復
    """
    def __init__(self,
                 base_interval: float = 0.05,   # 正常情況每 50ms 可打一發（依你環境調）
                 max_interval: float = 2.0,     # 撞線後暫時把頻率降到這麼慢
                 cooldown_sec: float = 20.0,    # 限流後的冷卻期
                 decay: float = 0.5):           # 冷卻期後恢復比例（越小恢復越快）
        self.base = float(base_interval)
        self.maxi = float(max_interval)
        self.cooldown = float(cooldown_sec)
        self.decay = float(decay)
        self._cur_interval = float(base_interval)
        self._next_allowed_ts = 0.0
        self._last_penalty_ts = 0.0

    def pace(self):
        now = time.time()
        if now < self._next_allowed_ts:
            time.sleep(self._next_allowed_ts - now)
        self._next_allowed_ts = max(self._next_allowed_ts, time.time()) + self._cur_interval

        # 冷卻後逐步恢復到 base
        if (time.time() - self._last_penalty_ts) > self.cooldown and self._cur_interval > self.base:
            self._cur_interval = max(self.base, self._cur_interval * self.decay)

    def penalize(self):
        self._last_penalty_ts = time.time()
        self._cur_interval = min(self.maxi, max(self._cur_interval * 2.0, self._cur_interval + self.base))

class RetryPolicy:
    """
    指數退避 + 抖動（Jitter）的重試策略
    """
    def __init__(self,
                 max_retries: int = 6,
                 base_delay: float = 0.25,      # 起始延遲
                 max_delay: float = 8.0,
                 jitter: float = 0.25):         # 0~1 的比例抖動
        self.max_retries = int(max_retries)
        self.base_delay = float(base_delay)
        self.max_delay = float(max_delay)
        self.jitter = float(jitter)

    def sleep(self, attempt: int):
        # 指數退避 + 抖動
        delay = min(self.max_delay, self.base_delay * (2 ** attempt))
        if self.jitter > 0:
            delta = delay * self.jitter
            delay = delay - delta + random.random() * (2 * delta)
        time.sleep(delay)
