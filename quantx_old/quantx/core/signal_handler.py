# 檔案: quantx/core/signal_handler.py
# 版本: v3 (DI 重構)
# 說明:
# - 不再依賴全局的 runtime_factory，改為提供一個 setup_signal_handler 函式。
# - 應用程式主入口 (如 launch.py) 負責調用此函式，並將 runtime.log 實例傳入。

import signal
import logging
from typing import Optional

# 全局旗標
_stop_flag = False
_logger: Optional[logging.Logger] = None

def _handler(sig, frame):
    """信號處理函式，設置停止旗標並記錄日誌。"""
    global _stop_flag
    if _logger:
        _logger.info("收到中斷信號，準備安全停止 ...")
    else:
        # 在 logger 未設置時，回退到標準輸出
        print("收到中斷信號，準備安全停止 ...")
    _stop_flag = True

def setup_signal_handler(logger: Optional[logging.Logger] = None):
    """
    註冊 SIGINT 和 SIGTERM 的信號處理器。

    Args:
        logger (Optional[logging.Logger]): 用於記錄日誌的記錄器實例。
    """
    global _logger
    _logger = logger
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

def should_stop() -> bool:
    """回傳是否收到停止信號。"""
    return _stop_flag