# 檔案: quantx/cli/train_daemon.py
# 版本: v9 (DI 重構)
# 說明:
# - 使用 AppContainer 來組裝和獲取所有核心服務，取代 get_runtime()。
# - 從容器中解析出 SchedulerTrain 實例並執行。
# - 注入 logger 到 signal_handler。

import argparse
import logging

from quantx.core.config import Config
from quantx.containers import AppContainer # 引入 DI 容器
from quantx.core.signal_handler import setup_signal_handler

def main():
    """
    訓練排程器主程式 (DI 版本)。
    """
    parser = argparse.ArgumentParser(description="QuantX 訓練排程")
    parser.add_argument("--config", type=str, default="conf/config.yaml", help="設定檔路徑")
    args = parser.parse_args()

    # 1. 初始化 Config 和 AppContainer，強制使用 'live' 模式獲取數據
    cfg = Config(args.config)
    container = AppContainer(cfg, mode_override="live")

    # 2. 從容器獲取需要的服務
    runtime = container.runtime
    sched = container.scheduler

    # 3. 設置信號處理器
    setup_signal_handler(runtime.log)

    # 4. 啟動排程器
    runtime.log.info(f"=== 訓練排程啟動 (數據環境: {runtime.mode}) ===")
    sched.run_forever()

if __name__ == "__main__":
    main()