# 檔案: quantx/core/data/datafeed.py
# 版本: v19 (重構完成)
# 說明:
# - 移除了 lazy_init 方法。
# - 新增了 subscribe_bulk 方法，用於集中式、並行化的數據訂閱和歷史回補。

import asyncio
import pandas as pd
import numpy as np
from collections import deque, defaultdict
from datetime import datetime, timezone
from typing import Dict, Set, TYPE_CHECKING, List, Any, Optional, Callable
from ..timeframe import parse_tf_minutes
import ccxt.pro as ccxtpro

try:
    from ccxt.base.errors import ExchangeError, NetworkError, RequestTimeout 
except ImportError:
    class ExchangeError(Exception): pass
    class NetworkError(Exception): pass
    class RequestTimeout(Exception): pass

if TYPE_CHECKING:
    from ..runtime import Runtime

class DataFeed:
    """即時數據中心"""
    def __init__(self, runtime: "Runtime", buffer_size: int = 1000):
        self.runtime = runtime
        self.log = runtime.log
        self.buffer_size = buffer_size
        self.ohlcv_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.buffer_size))
        self.last_processed_ts: Dict[str, int] = defaultdict(int)
        self.main_symbols: Set[str] = set()
        self.subscribed_ohlcv_topics: Set[tuple[str, str]] = set()
        self.subscribed_orderbook_symbols: Set[str] = set()
        self._ohlcv_task: asyncio.Task | None = None
        self._orderbook_task: asyncio.Task | None = None
        self._order_task: asyncio.Task | None = None
        self.orderbook_buffers: Dict[str, Dict[str, List[List[float]]]] = {}
        # [優雅重連] 新增狀態和延遲變量
        self._is_ohlcv_connected = False
        self._is_orderbook_connected = False
        self._reconnect_delay = 5  # 初始延遲5秒

    def start(self):
        self.log.info("[DataFeed] 正在啟動 WebSocket 背景服務...")
        if not self._ohlcv_task or self._ohlcv_task.done():
            self._ohlcv_task = asyncio.create_task(self._ohlcv_handler())
        if not self._orderbook_task or self._orderbook_task.done():
            self._orderbook_task = asyncio.create_task(self._orderbook_handler())
        if not self._order_task or self._order_task.done():
            self._order_task = asyncio.create_task(self._order_handler())

    async def stop(self):
        self.log.info("[DataFeed] 正在停止 WebSocket 背景服務...")
        tasks = [self._ohlcv_task, self._orderbook_task, self._order_task]
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try: await task
                except asyncio.CancelledError: pass
        
        provider = self.runtime.loader.provider
        if hasattr(provider, "close_ws"):
            await provider.close_ws()
        self.log.info("[DataFeed] 服務已停止。")

    async def subscribe_bulk(self, targets: List[tuple[str, str]], main_symbols_tfs: List[tuple[str, str]]):
        self.log.info(f"[DataFeed] 開始批次訂閱 {len(targets)} 個目標...")

        for symbol, tf in main_symbols_tfs:
            key = f"{symbol}-{tf}"
            self.main_symbols.add(key)
            self.log.info(f"[DataFeed] {key} 已被標記為主要觸發標的。")

        for symbol, tf in targets:
            self.subscribed_ohlcv_topics.add((symbol, tf))
            base_symbol = symbol.split(':')[0]
            self.subscribed_orderbook_symbols.add(base_symbol)

        backfill_tasks = []
        for symbol, tf in targets:
            key = f"{symbol}-{tf}"
            if key not in self.ohlcv_buffers:
                self.ohlcv_buffers[key] = deque(maxlen=self.buffer_size)
                backfill_tasks.append(self._backfill_ohlcv(symbol, tf))

        if backfill_tasks:
            self.log.info(f"正在為 {len(backfill_tasks)} 個新的數據流並行回補歷史數據...")
            await asyncio.gather(*backfill_tasks)
            self.log.info("所有歷史數據回補完成。")

    async def _backfill_ohlcv(self, symbol: str, tf: str):
        key = f"{symbol}-{tf}"
        try:
            minutes_to_fetch = parse_tf_minutes(tf) * (self.buffer_size + 5)
            start_dt = datetime.now(timezone.utc) - pd.Timedelta(minutes=minutes_to_fetch)
            end_dt = datetime.now(timezone.utc)

            self.log.info(f"[{key}] 開始回補歷史數據...")
            hist_df = await asyncio.to_thread(
                self.runtime.loader.load_ohlcv, symbol, tf, start_dt, end_dt
            )
            
            if not hist_df.empty:
                hist_df['timestamp'] = (hist_df.index.astype(np.int64) // 10**9).astype(int)
                records = hist_df.to_dict('records')
                self.ohlcv_buffers[key].extend(records)
                self.log.info(f"[{key}] 成功填入 {len(records)} 筆歷史數據。")
                if records:
                    self.last_processed_ts[key] = records[-1]['timestamp']
        except Exception as e:
            self.log.error(f"為 {key} 回補歷史數據時失敗: {e}", exc_info=True)

    def get_orderbook(self, symbol: str) -> Optional[Dict[str, List[List[float]]]]:
        return self.orderbook_buffers.get(symbol)

    def _on_raw_bar_received(self, symbol: str, timeframe: str, ohlcv: list):
        key = f"{symbol}-{timeframe}"
        if key not in self.ohlcv_buffers: return

        for bar in ohlcv:
            try:
                if not isinstance(bar, list) or len(bar) < 6: continue
                bar_ts = bar[0] // 1000
                bar_dict = {'timestamp': bar_ts, 'open': float(bar[1]), 'high': float(bar[2]), 'low': float(bar[3]), 'close': float(bar[4]), 'volume': float(bar[5])}
            except (ValueError, TypeError, IndexError): continue
            
            self.ohlcv_buffers[key].append(bar_dict)
            
            if self.runtime.live.on_tick_callbacks.get(key):
                self.runtime.live.on_tick_callbacks[key](key, bar_dict) 
            
            if bar_dict['timestamp'] > self.last_processed_ts.get(key, 0):
                self.log.info(f"[DataFeed] 收到新 K 棒 -> {key} @ {datetime.fromtimestamp(bar_dict['timestamp'], tz=timezone.utc)}")
                if key in self.main_symbols and self.runtime.live.on_closed_bar_callbacks.get(key):
                    self.log.info(f"====== {key} 觸發策略 on_bar (新 K 棒完成) ======")
                    try: self.runtime.live.on_closed_bar_callbacks[key](key)
                    except Exception as e: self.log.error(f"執行 on_bar 回呼時發生錯誤: {e}", exc_info=True)
                self.last_processed_ts[key] = bar_dict['timestamp']

    def _on_orderbook_received(self, symbol: str, orderbook: Any):
        if not isinstance(orderbook, dict): return
        # [核心修正] 直接使用 ccxt.pro 提供的原始 symbol (例如 'CRV/USDT') 作為鍵
        self.orderbook_buffers[symbol] = {
            'bids': orderbook.get('bids', []),
            'asks': orderbook.get('asks', []),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

    async def _ohlcv_handler(self):
        provider = self.runtime.loader.provider
        while True:
            try:
                if not self.subscribed_ohlcv_topics:
                    await asyncio.sleep(1); continue

                self.log.info(f"[DataFeed-OHLCV] 正在連接 WS，訂閱 {len(self.subscribed_ohlcv_topics)} 個 K 線主題...")
                self._is_ohlcv_connected = True
                self._reconnect_delay = 5 # 連接成功後重置延遲

                subscriptions = [[symbol, tf] for symbol, tf in self.subscribed_ohlcv_topics]
                await provider.watch_ohlcv_stream(subscriptions, self._on_raw_bar_received)

            except asyncio.CancelledError:
                self.log.info("[DataFeed-OHLCV] WS 處理器被外部取消。")
                break
            except Exception as e:
                self.log.error(f"[DataFeed-OHLCV] WS 處理器發生錯誤: {e}", exc_info=False)
                self._is_ohlcv_connected = False
                self.log.info(f"[DataFeed-OHLCV] 將在 {self._reconnect_delay} 秒後重試...")
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, 60) # 指數退避，最長60秒

    async def _orderbook_handler(self):
        provider = self.runtime.loader.provider
        reconnect_delay_ob = 5
        while True:
            try:
                if not self.subscribed_orderbook_symbols:
                    await asyncio.sleep(1); continue

                self.log.info(f"[DataFeed-OB] 正在連接 WS，訂閱 {len(self.subscribed_orderbook_symbols)} 個訂單簿 Symbol...")
                self._is_orderbook_connected = True
                reconnect_delay_ob = 5

                subscriptions = list(self.subscribed_orderbook_symbols)
                await provider.watch_orderbook_stream(subscriptions, self._on_orderbook_received)

            except asyncio.CancelledError:
                self.log.info("[DataFeed-OB] WS 處理器被外部取消。")
                break
            except Exception as e:
                self.log.error(f"[DataFeed-OB] WS 處理器發生錯誤: {e}", exc_info=False)
                self._is_orderbook_connected = False
                self.log.info(f"[DataFeed-OB] 將在 {reconnect_delay_ob} 秒後重試...")
                await asyncio.sleep(reconnect_delay_ob)
                reconnect_delay_ob = min(reconnect_delay_ob * 2, 60)

    def _on_order_update(self, orders: List[Dict[str, Any]]):
        if not orders: return
        self.log.info(f"[_on_order_update] 收到 {len(orders)} 筆訂單更新。")

        for order in orders:
            self.log.info(f"--- RAW ORDER UPDATE --- \n{order}\n--------------------")
            try:
                status = order.get('status')
                symbol_raw = order.get('symbol')
                if not symbol_raw: continue
                # [核心修正] 更穩健地解析 symbol，例如 'ETH/USDT:USDT' -> 'ETHUSDT'
                symbol = symbol_raw.split(':')[0].replace('/', '')
                self.log.info(f"  - Symbol 解析: '{symbol_raw}' -> '{symbol}'")
                order_id = order.get('id')
                filled = order.get('filled', 0.0)

                self.log.info(f"  - WS 訂單更新: {symbol} (ID: {order_id}), 狀態: {status}, 已成交: {filled}")

                if not (status == 'closed' and filled > 0): continue

                price = order.get('average', 0.0)
                if price <= 0: price = order.get('price', 0.0)

                # [穩健性修正] 嘗試從多個位置獲取 client order id
                client_order_id = order.get('clientOrderId')
                if not client_order_id:
                    client_order_id = order.get('info', {}).get('orderLinkId')

                if not client_order_id:
                    self.log.warning(f"訂單 {order_id} 缺少 clientOrderId/orderLinkId，無法判斷意圖，跳過。")
                    self.log.info(f"--- FAILED RAW ORDER --- \n{order}\n--------------------")
                    continue

                intent = client_order_id.split('-')[0]
                self.log.info(f"  - 訂單 {order_id} 已成交。意圖: {intent.upper()}, 價格: {price}, 數量: {filled}")

                target_ctx = next((ctx for ctx in self.runtime.live.contexts if ctx.symbol == symbol), None)
                if not target_ctx:
                    self.log.warning(f"找不到 {symbol} 對應的策略 Context，無法更新倉位。")
                    continue

                # --- [訂單追蹤] 通知 TradeManager 停止追蹤此訂單 ---
                if hasattr(target_ctx, 'trade_manager') and hasattr(target_ctx.trade_manager, 'stop_tracker'):
                    target_ctx.trade_manager.stop_tracker(client_order_id)
                # --- [訂單追蹤結束] ---

                pos = target_ctx.position
                equity_manager = self.runtime.live.equity_manager
                pnl = 0.0

                self.log.info(f"  - {symbol} 更新前: L={pos.long_qty:.4f} @ {pos.long_entry:.4f}, S={pos.short_qty:.4f} @ {pos.short_entry:.4f}")

                if intent == "open_long":
                    new_total_cost = pos.long_entry * pos.long_qty + price * filled
                    pos.long_qty += filled
                    pos.long_entry = new_total_cost / pos.long_qty if pos.long_qty > 0 else 0
                elif intent == "open_short":
                    new_total_cost = pos.short_entry * pos.short_qty + price * filled
                    pos.short_qty += filled
                    pos.short_entry = new_total_cost / pos.short_qty if pos.short_qty > 0 else 0
                elif intent == "close_long":
                    if pos.is_long():
                        pnl = (price - pos.long_entry) * filled
                        equity_manager.apply_pnl(pnl)
                        pos.long_qty -= filled
                        if pos.long_qty <= 0.00001: pos.long_qty = pos.long_entry = 0.0
                elif intent == "close_short":
                    if pos.is_short():
                        pnl = (pos.short_entry - price) * filled
                        equity_manager.apply_pnl(pnl)
                        pos.short_qty -= filled
                        if pos.short_qty <= 0.00001: pos.short_qty = pos.short_entry = 0.0

                self.log.info(f"  - {symbol} 更新後: L={pos.long_qty:.4f} @ {pos.long_entry:.4f}, S={pos.short_qty:.4f} @ {pos.short_entry:.4f}")
                if pnl != 0.0: self.log.info(f"  - 已實現 PNL: {pnl:.4f}。全局資金更新為: {equity_manager.equity:.4f}")

                # 觸發 Live Status 檔案即時更新
                self.runtime.live.update_status_file()

            except Exception as e:
                self.log.error(f"處理訂單更新時發生嚴重錯誤: {e}", exc_info=True)


    async def _order_handler(self):
        provider = self.runtime.loader.provider
        reconnect_delay_order = 5
        while True:
            try:
                self.log.info("[DataFeed-Order] 正在連接 WS，監聽訂單更新...")
                await provider.watch_orders_stream(self._on_order_update)
            except asyncio.CancelledError:
                self.log.info("[DataFeed-Order] WS 處理器被外部取消。")
                break
            except Exception as e:
                self.log.error(f"[DataFeed-Order] WS 處理器發生錯誤: {e}", exc_info=False)
                self.log.info(f"[DataFeed-Order] 將在 {reconnect_delay_order} 秒後重試...")
                await asyncio.sleep(reconnect_delay_order)
                reconnect_delay_order = min(reconnect_delay_order * 2, 60)