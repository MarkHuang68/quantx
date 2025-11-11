"""
logging_utils.py - Logging 初始化工具
-----------------------------------
根據 config 設定 debug 模式，決定 log level。
"""

import logging


def setup_logging(debug: bool = False):
    """
    初始化 logging
    Args:
        debug (bool): 是否啟用 DEBUG 模式
    """
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
