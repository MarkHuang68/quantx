# quantx/core/log_formatter.py
# -*- coding: utf-8 -*-
# 版本: v1 (全新模組)
# 說明:
# - 引入 rich library 來實現多彩且結構化的日誌輸出。
# - 定義 get_rich_handler 函式，提供一個配置好的 RichHandler，
#   可用於替換標準的 StreamHandler，美化 console 輸出。

import logging
from rich.logging import RichHandler
from rich.console import Console

def get_rich_handler() -> RichHandler:
    """
    獲取一個配置好的 RichHandler 用於美化日誌輸出。

    - INFO: 預設顏色 (通常是白色或灰色)
    - WARNING: 黃色
    - ERROR: 紅色
    - CRITICAL: 加粗紅底白字
    - DEBUG: 藍色

    Returns:
        RichHandler: 配置好的日誌處理器。
    """
    return RichHandler(
        level=logging.INFO,
        show_path=False,         # 不顯示檔案路徑，保持日誌簡潔
        show_level=True,         # 顯示日誌級別 (INFO, WARNING)
        show_time=True,          # 顯示時間
        rich_tracebacks=True,    # 發生錯誤時顯示漂亮的 traceback
        markup=True,             # 允許在日誌訊息中使用 markup 語法
        log_time_format="[%m-%d %H:%M:%S]", # 統一時間格式
        console=Console(color_system="auto"), # 自動偵測終端是否支援顏色
    )