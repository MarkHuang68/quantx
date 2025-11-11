# 檔案: quantx/cli/launch.py
# -*- coding: utf-8 -*-
# 版本: v17 (重構完成)
# 說明:
# - 回測模式 (Interval) 現在使用 asyncio.gather 並行執行。

import asyncio
import argparse
from datetime import datetime
import pandas as pd
from typing import TYPE_CHECKING

from quantx.core.config import Config
from quantx.containers import AppContainer
# import signal # 不再需要 signal
from quantx.core.runner.live_runner import LiveRunner
from quantx.core.runner.interval_runner import IntervalRunner

if TYPE_CHECKING:
    from quantx.core.policy.auto_policy import AutoPolicy
    from quantx.core.runtime import Runtime


async def main():
    """
    QuantX 啟動器主函數 (DI 版本)。
    """
    parser = argparse.ArgumentParser(description="QuantX 實盤/模擬啟動器")
    parser.add_argument("--start", type=str, default=None, help="區間測試起始日期 (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="區間測試結束日期 (YYYY-MM-DD)")
    args = parser.parse_args()

    cfg = Config()
    container = AppContainer(cfg)

    runtime = container.runtime
    auto_policy = container.auto_policy
    symbols = container.config.load_symbol()

    runtime.log.info(f"[Launch] 系統啟動。模式: {runtime.mode}，監控 {len(symbols)} 組標的。")

    is_live_mode = not (args.start and args.end)

    runner = None
    try:
        if is_live_mode:
            # 實例化 Runner
            runner = LiveRunner(runtime, auto_policy, symbols, cfg)

            # 1. 執行非阻塞的啟動任務
            await runner.startup()

            # 2. 在背景啟動主監控循環
            asyncio.create_task(runner._master_loop())

            # 3. 阻塞主線程，直到有關閉信號
            runtime.log.info("[Launch] 啟動完成，主線程進入永久等待狀態...")
            await runner.run_forever()

        else:
            runtime.log.info("[Launch] 進入 Interval 模擬模式...")
            start_dt = pd.to_datetime(args.start, utc=True)
            end_dt = pd.to_datetime(args.end, utc=True)

            tasks = []
            for sym, tf in symbols:
                interval_runner = IntervalRunner(
                    runtime=runtime,
                    auto_policy=auto_policy,
                    symbol=sym,
                    tf=tf,
                    start_dt=start_dt,
                    end_dt=end_dt,
                    cfg=cfg
                )
                tasks.append(interval_runner.run())

            if tasks:
                runtime.log.info(f"準備並行執行 {len(tasks)} 個回測任務...")
                await asyncio.gather(*tasks)
                runtime.log.info("所有回測任務已完成。")

    finally:
        # 無論如何，在程式結束前嘗試優雅關閉
        if runner and isinstance(runner, LiveRunner):
            runtime.log.info("程式即將退出，執行最後的清理工作...")
            await runner.stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # 這個塊現在可以正常工作，因為我們沒有攔截 SIGINT
        print("\n程式被手動中斷，正在安全退出...")