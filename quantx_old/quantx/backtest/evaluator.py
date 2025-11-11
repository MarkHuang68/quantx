"""
evaluator.py
============
向後相容封裝
保留舊路徑引用，轉發到 core.eval.metrics
"""

from quantx.core.eval.metrics import compute_kpis, evaluate
