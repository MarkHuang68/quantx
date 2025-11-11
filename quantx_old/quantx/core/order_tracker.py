# quantx/core/order_tracker.py
import asyncio
import logging
from typing import TYPE_CHECKING, Dict, Any

if TYPE_CHECKING:
    from .trade_manager import LiveTradeManager

log = logging.getLogger("quantx.core")

class OrderTracker:
    """
    負責追蹤一筆在途限價單的生命週期，並在逾時後觸發重試邏輯。
    """
    def __init__(
        self,
        trade_manager: "LiveTradeManager",
        client_order_id: str,
        original_cmd: Dict[str, Any],
        attempt: int,
        timeout: int = 10,
    ):
        self.trade_manager = trade_manager
        self.client_order_id = client_order_id
        self.original_cmd = original_cmd
        self.attempt = attempt
        self.timeout = timeout
        self._task: asyncio.Task | None = None

    def start(self):
        """啟動背景計時器任務。"""
        if not self._task or self._task.done():
            self._task = asyncio.create_task(self._run_timer())

    def stop(self):
        """停止背景計時器任務。"""
        if self._task and not self._task.done():
            self._task.cancel()
            log.info(f"[OrderTracker] 訂單 {self.client_order_id} 已成交或被外部處理，計時器已停止。")

    async def _run_timer(self):
        """計時器主體，等待指定時間後觸發逾時處理。"""
        try:
            log.info(f"[OrderTracker] 開始追蹤訂單 {self.client_order_id} (第 {self.attempt} 次嘗試)，逾時: {self.timeout} 秒。")
            await asyncio.sleep(self.timeout)

            # 如果能執行到這裡，代表計時器沒有被 stop()，訂單已逾時。
            log.warning(f"[OrderTracker] 訂單 {self.client_order_id} 已逾時！觸發逾時處理...")
            # 注意：這裡使用 create_task 是為了避免 handle_timeout 中的 await 阻塞計時器迴圈本身
            # 雖然我們目前沒有迴圈，但這是一個好的異步程式設計習慣
            asyncio.create_task(self.trade_manager.handle_timeout(self.client_order_id, self.attempt))

        except asyncio.CancelledError:
            # 這是預期中的行為，當訂單成交時，stop() 會觸發此異常。
            pass
        except Exception as e:
            log.error(f"[OrderTracker] 追蹤訂單 {self.client_order_id} 時發生未知錯誤: {e}", exc_info=True)
