# 檔案: strategies/base_strategy.py

from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    def __init__(self, context):
        self.context = context

    @abstractmethod
    def on_bar(self, dt):
        """
        每個 K 線關閉時執行的主要邏輯。
        """
        pass
