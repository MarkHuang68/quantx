# quantx/core/trade_manager.py
# -*- coding: utf-8 -*-
# 版本: v25 (重構完成)
# 說明:
# - LiveTradeManager 在執行平倉指令前，會主動獲取真實倉位資訊。
# - 通用 close 指令能正確處理雙向持倉下的單邊平倉。
# - 整個下單鏈路的參數傳遞更清晰。

from __future__ import annotations
from abc import ABC, abstractmethod
import os
import json
from pathlib import Path
from typing import Dict, Any, TYPE_CHECKING, Tuple, Optional, List
from datetime import datetime, timezone
import numpy as np
import uuid

from quantx.core.executor.base import Position
from quantx.core.risk import RiskConfig, compute_order_size
from .orderbook_utils import calculate_optimal_maker_price
from .order_tracker import OrderTracker
try:
    from ccxt.base.errors import BadRequest, ExchangeError
except ImportError:
    class BadRequest(Exception): pass
    class ExchangeError(Exception): pass

if TYPE_CHECKING:
    from quantx.core.runtime import Runtime
    from quantx.core.executor.base import BaseExecutor


class BaseTradeManager(ABC):
    """交易管理抽象基類"""
    def __init__(self, runtime: 'Runtime', symbol: str, tf: str, risk_cfg: RiskConfig):
        self.runtime, self.log, self.symbol, self.tf, self.risk_cfg = runtime, runtime.log, symbol, tf, risk_cfg
    
    def get_position(self, symbol: str) -> Position:
        # 預設實現，返回一個空的 Position 物件。
        # LiveTradeManager 應從 executor.positions 獲取，那裡儲存了接管的或本地模擬的狀態。
        # 注意：這個狀態可能與真實交易所狀態有延遲。
        if hasattr(self.executor, 'positions'):
            return self.executor.positions[symbol]
        return Position()

    @abstractmethod
    def get_equity(self) -> float: pass
    
    @abstractmethod
    def execute_commands(self, commands: list[Dict[str, Any]], current_price: float, current_ts: Any) -> None: pass


class SimulatedTradeManager(BaseTradeManager):
    """模擬交易管理器"""
    def __init__(self, runtime: 'Runtime', tf: str, risk_cfg: RiskConfig, executor: BaseExecutor):
        # 移除 symbol，因為現在處理多個
        super().__init__(runtime, symbol=None, tf=tf, risk_cfg=risk_cfg)
        self.executor = executor
        cost_model = self.runtime.get_cost_model()
        self.taker_fee_bps, self.slippage_bps = cost_model.get('taker_fee_bps', 5.5), cost_model.get('slip_bps', 1.0)
        
        if not hasattr(self.executor, 'equity'): setattr(self.executor, 'equity', 10000.0)
        if not hasattr(self.executor, 'trades'): setattr(self.executor, 'trades', [])

    def get_equity(self) -> float: return self.executor.equity

    def _apply_slippage_and_fee(self, side: str, price: float, qty: float) -> Tuple[float, float]:
        slip = self.slippage_bps * 1e-4
        exec_price = price * (1 + slip if side in ("buy", "open_long", "close_short") else 1 - slip)
        fee = abs(exec_price * qty) * (self.taker_fee_bps * 1e-4)
        return exec_price, fee

    def _get_order_qty(self, price_now: float, commanded_qty: Optional[float]) -> float:
        if commanded_qty is None or commanded_qty <= 0.0:
            return compute_order_size(self.get_equity(), price_now, self.risk_cfg)
        return commanded_qty

    def execute_commands(self, commands: list[Dict[str, Any]], price_now: float, current_ts: Any) -> None:
        ts_str = current_ts.isoformat() if hasattr(current_ts, 'isoformat') else str(current_ts)
        
        for cmd in commands:
            action = cmd.get("action")
            symbol = cmd.get("symbol")
            if not symbol: continue

            pos = self.get_position(symbol)
            commanded_qty = cmd.get("qty")
            
            if action in ["open_long", "open_short"]:
                qty = self._get_order_qty(price_now, commanded_qty)
                if qty <= 0: continue

                exec_price, fee = self._apply_slippage_and_fee(action, price_now, qty)

                # [核心修正] 模擬策略級別的 SL/TP
                trade_info = {"ts": ts_str, "symbol": symbol, "price": exec_price, "pnl": 0, "fee": fee, "maker": False}
                sl_pct = getattr(self.executor, 'default_stop_loss_pct', None)
                tp_pct = getattr(self.executor, 'default_take_profit_pct', None)

                if action == "open_long":
                    if pos.is_long():
                        new_total_cost = pos.long_entry * pos.long_qty + exec_price * qty
                        pos.long_qty += qty
                        pos.long_entry = new_total_cost / pos.long_qty
                    else:
                        pos.long_qty, pos.long_entry = qty, exec_price

                    trade_info.update({"side": "buy", "qty": qty})
                    if sl_pct: trade_info['sl'] = exec_price * (1 - sl_pct / 100)
                    if tp_pct: trade_info['tp'] = exec_price * (1 + tp_pct / 100)
                    self.executor.trades.append(trade_info)

                else: # open_short
                    if pos.is_short():
                        new_total_cost = pos.short_entry * pos.short_qty + exec_price * qty
                        pos.short_qty += qty
                        pos.short_entry = new_total_cost / pos.short_qty
                    else:
                        pos.short_qty, pos.short_entry = qty, exec_price

                    trade_info.update({"side": "sell", "qty": -qty})
                    if sl_pct: trade_info['sl'] = exec_price * (1 + sl_pct / 100)
                    if tp_pct: trade_info['tp'] = exec_price * (1 - tp_pct / 100)
                    self.executor.trades.append(trade_info)

            elif action in ["close_long", "close_short", "close"]:
                if action == "close_long" or (action == "close" and pos.is_long()):
                    if not pos.is_long(): continue
                    qty_to_close = min(commanded_qty or pos.long_qty, pos.long_qty)
                    if qty_to_close <=0: continue
                    exec_price, fee = self._apply_slippage_and_fee(action, price_now, qty_to_close)
                    pnl = (exec_price - pos.long_entry) * qty_to_close - fee
                    self.executor.equity += pnl
                    pos.long_qty -= qty_to_close
                    if not pos.is_long(): pos.long_entry = 0.0
                    self.executor.trades.append({"ts": ts_str, "symbol": symbol, "side": "close_long", "price": exec_price, "qty": -qty_to_close, "pnl": pnl, "fee": fee, "maker": False})
                
                elif action == "close_short" or (action == "close" and pos.is_short()):
                    if not pos.is_short(): continue
                    qty_to_close = min(commanded_qty or pos.short_qty, pos.short_qty)
                    if qty_to_close <= 0: continue
                    exec_price, fee = self._apply_slippage_and_fee(action, price_now, qty_to_close)
                    pnl = (pos.short_entry - exec_price) * qty_to_close - fee
                    self.executor.equity += pnl
                    pos.short_qty -= qty_to_close
                    if not pos.is_short(): pos.short_entry = 0.0
                    self.executor.trades.append({"ts": ts_str, "symbol": symbol, "side": "close_short", "price": exec_price, "qty": qty_to_close, "pnl": pnl, "fee": fee, "maker": False})


class LiveTradeManager(BaseTradeManager):
    """實時交易管理器"""
    def __init__(self, runtime: 'Runtime', symbol: str, tf: str, risk_cfg: RiskConfig, executor: BaseExecutor):
        super().__init__(runtime, symbol, tf, risk_cfg)
        self.executor = executor
        self.provider = runtime.loader.provider
        if not hasattr(self.executor, 'equity_manager'): self.executor.equity_manager = self.runtime.live.equity_manager
        if not hasattr(self.executor, 'positions'):
             from collections import defaultdict
             self.executor.positions = defaultdict(Position)
        self.active_trackers: Dict[str, OrderTracker] = {}

    def get_equity(self) -> float: return self.executor.equity_manager.equity

    def _get_order_qty(self, price_now: float, commanded_qty: Optional[float]) -> float:
        if commanded_qty is None or commanded_qty <= 0.0:
            return compute_order_size(self.get_equity(), price_now, self.risk_cfg)
        return commanded_qty

    def execute_commands(self, commands: list[Dict[str, Any]], price_now: float, current_ts: datetime) -> None:
        try:
            online_positions_raw = self.provider.get_positions()
        except Exception as e:
            self.log.error(f"[Live] 無法獲取線上倉位資訊，將跳過所有平倉指令。錯誤: {e}")
            online_positions_raw = []

        for cmd in commands:
            action = cmd.get("action")
            symbol = cmd.get("symbol")
            if not symbol: continue

            # 為通用平倉指令產生具體的平倉指令
            if action == "close":
                symbol_positions = [p for p in online_positions_raw if p['symbol'].replace('/', '').split(':')[0] == symbol]
                if not symbol_positions: self.log.debug(f"[Live] 通用平倉 {symbol}：無線上倉位可平，跳過。")
                for pos in symbol_positions:
                    pos_side = pos.get('side', '').lower()
                    close_action = "close_long" if pos_side == 'long' else "close_short"
                    self._execute_single_command({**cmd, "action": close_action}, price_now, online_positions_raw)
                continue

            # 執行單一指令
            self._execute_single_command(cmd, price_now, online_positions_raw)

    def _execute_single_command(self, cmd: Dict[str, Any], price_now: float, online_positions: List[Dict], retry_attempt: Optional[int] = None):
        action = cmd.get("action")
        symbol = cmd.get("symbol")
        if not (action and symbol): return

        # [熱更新] 檢查策略是否處於 "只平倉" 模式
        if action.startswith("open") and self.executor.is_winding_down:
            self.log.warning(f"[Live] {symbol} 策略處於『只平倉』模式，已阻止新的開倉指令: {action}")
            return

        side: str = ""
        is_reduce_only = False
        qty_to_trade: float = 0.0

        if action in ["open_long", "close_short"]: side = "buy"
        elif action in ["open_short", "close_long"]: side = "sell"
        else: self.log.warning(f"[Live] 未知的交易指令 action: '{action}'，已忽略。"); return

        if action in ["close_long", "close_short"]: is_reduce_only = True

        commanded_qty = cmd.get("qty")
        if action.startswith("open"):
            qty_to_trade = self._get_order_qty(price_now, commanded_qty)
        else: # close_long or close_short
            pos_side = 'long' if action == 'close_long' else 'short'
            real_qty = 0
            for pos in online_positions:
                if pos['symbol'].replace('/', '').split(':')[0] == symbol and pos.get('side', '').lower() == pos_side:
                    real_qty = float(pos.get('contracts', 0))
                    break
            qty_to_trade = commanded_qty if commanded_qty is not None and commanded_qty > 0 else real_qty

        if qty_to_trade <= 0:
            self.log.warning(f"[Live] {action} {symbol} 計算後的可交易數量為 0，跳過。")
            return

        # --- [自適應限價單邏輯] ---
        order_type = cmd.get("order_type", "market")
        price = cmd.get("price")

        if order_type == "market" and price is None:
            self.log.info(f"[Live-Adaptive] 市價單 '{action} {symbol}' 觸發自適應限價單邏輯。")
            # [核心修正] 將策略 symbol (例如 'CRVUSDT') 轉換為 ccxt symbol (例如 'CRV/USDT')
            ccxt_symbol = f"{symbol.replace('USDT', '')}/USDT"
            orderbook = self.runtime.live.datafeed.get_orderbook(ccxt_symbol)

            if orderbook and orderbook.get('bids') and orderbook.get('asks'):
                if side == "buy":
                    # 買單：使用賣一價 (best ask)
                    price = orderbook['asks'][0][0]
                    order_type = "limit"
                    self.log.info(f"[Live-Adaptive] 買入：使用賣一價 {price} 作為限價。")
                elif side == "sell":
                    # 賣單：使用買一價 (best bid)
                    price = orderbook['bids'][0][0]
                    order_type = "limit"
                    self.log.info(f"[Live-Adaptive] 賣出：使用買一價 {price} 作為限價。")
            else:
                self.log.warning(f"[Live-Adaptive] 無法獲取 {symbol} 的訂單簿，將按原市價單執行。")
        # --- [自適應限價單邏輯結束] ---

        try:
            submit_params = cmd.get("params", {})

            # [核心改造] 新增 clientOrderId 以追蹤訂單意圖
            # 格式: intent-symbol-timestamp-uuid
            ts_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            unique_id = str(uuid.uuid4()).split('-')[0]
            client_order_id = f"{action}-{symbol}-{ts_ms}-{unique_id}"
            submit_params['clientOrderId'] = client_order_id

            submit_params['intent'] = action
            if is_reduce_only: submit_params['reduce_only'] = True

            # --- [止損/止盈邏輯] ---
            # [核心修正] 自動應用策略級別的 SL/TP
            if action.startswith("open") and price:
                sl_pct = getattr(self.executor, 'default_stop_loss_pct', None)
                tp_pct = getattr(self.executor, 'default_take_profit_pct', None)

                final_sl, final_tp = None, None

                # 計算 SL 價格
                if sl_pct and sl_pct > 0:
                    final_sl = price * (1 - sl_pct / 100) if side == "buy" else price * (1 + sl_pct / 100)

                # 計算 TP 價格
                if tp_pct and tp_pct > 0:
                    final_tp = price * (1 + tp_pct / 100) if side == "buy" else price * (1 - tp_pct / 100)

                # 處理精度並更新提交參數
                if final_sl or final_tp:
                    try:
                        price_precision = self.provider.get_market_precision(symbol, 'price')
                        if final_sl:
                            submit_params['stopLoss'] = self.provider.price_to_precision(final_sl, price_precision)
                            self.log.info(f"  - 自動應用止損: {sl_pct}%, 計算價格: {submit_params['stopLoss']}")
                        if final_tp:
                            submit_params['takeProfit'] = self.provider.price_to_precision(final_tp, price_precision)
                            self.log.info(f"  - 自動應用止盈: {tp_pct}%, 計算價格: {submit_params['takeProfit']}")
                    except Exception as e:
                        self.log.error(f"  - 處理 SL/TP 精度時出錯: {e}")
            # --- [止損/止盈邏輯結束] ---

            self.log.info(f"[Live] 準備提交訂單: {action.upper()} {qty_to_trade:.5f} {symbol} "
                          f"({order_type}) at price {price or 'N/A'}")

            order_result = self.provider.submit_order(
                symbol=symbol,
                order_type=order_type,
                side=side,
                qty=qty_to_trade,
                price=price,
                params=submit_params
            )
            self.log.info(f"[Live] 訂單已成功發送到 Provider: {action.upper()} {symbol}")

            # [訂單追蹤] 如果是限價單，則啟動追蹤器
            if order_type == 'limit' and not order_result.get("dry_run"):
                attempt_count = retry_attempt or 1
                tracker = OrderTracker(
                    trade_manager=self,
                    client_order_id=client_order_id,
                    original_cmd=cmd,
                    attempt=attempt_count
                )
                self.active_trackers[client_order_id] = tracker
                tracker.start()

        except (BadRequest, ExchangeError) as e:
            self.log.error(f"[Live] ❌ {action.upper()} {symbol} 提交失敗: {e}")
        except Exception as e:
            self.log.error(f"[Live] ❌ {action.upper()} {symbol} 提交時發生未知錯誤: {e}", exc_info=True)

    async def handle_timeout(self, client_order_id: str, attempt: int):
        """處理來自 OrderTracker 的逾時事件。"""
        tracker = self.active_trackers.get(client_order_id)
        if not tracker:
            self.log.warning(f"[TimeoutHandler] 收到逾時信號，但找不到對應的 Tracker: {client_order_id}")
            return

        self.log.info(f"[TimeoutHandler] 開始處理訂單 {client_order_id} 的第 {attempt} 次逾時...")

        original_cmd = tracker.original_cmd
        symbol = original_cmd.get("symbol")
        if not symbol:
            self.log.error(f"[TimeoutHandler] 原始指令中缺少 symbol，無法處理逾時。")
            return

        # 1. 取消舊訂單
        try:
            self.log.info(f"[TimeoutHandler] 正在取消舊訂單...")
            self.provider.exchange.cancel_all_orders(self.provider.exchange.symbol(symbol))
            self.log.info(f"[TimeoutHandler] {symbol} 的所有待處理訂單已取消。")
        except Exception as e:
            self.log.error(f"[TimeoutHandler] 取消訂單 {client_order_id} 失敗: {e}", exc_info=True)
            self.active_trackers.pop(client_order_id, None)
            return

        self.active_trackers.pop(client_order_id, None)

        # [核心修正] 在重試前，必須獲取最新的價格來計算數量
        price_now = 0
        try:
            ccxt_symbol = f"{symbol.replace('USDT', '')}/USDT"
            orderbook = self.runtime.live.datafeed.get_orderbook(ccxt_symbol)
            if orderbook and orderbook.get('bids') and len(orderbook['bids']) > 0 and orderbook.get('asks') and len(orderbook['asks']) > 0:
                price_now = (orderbook['bids'][0][0] + orderbook['asks'][0][0]) / 2
            else:
                ticker = self.provider.get_ticker(symbol)
                price_now = ticker['last']

            if price_now <= 0:
                raise ValueError("獲取的價格無效")
            self.log.info(f"[TimeoutHandler] 成功獲取 {symbol} 的當前價格: {price_now}")
        except Exception as e:
            self.log.error(f"[TimeoutHandler] 無法獲取 {symbol} 的最新有效價格，重試失敗: {e}")
            return

        # 2. 根據嘗試次數執行重試邏輯
        try:
            online_positions_raw = self.provider.get_positions()
            if attempt == 1:
                self.log.info(f"[TimeoutHandler] 第 1 次重試：執行『追價』限價單...")
                new_cmd = original_cmd.copy()
                self._execute_single_command(new_cmd, price_now, online_positions_raw, retry_attempt=attempt + 1)
            elif attempt == 2:
                self.log.info(f"[TimeoutHandler] 第 2 次重試：執行『市價單』...")
                market_order_cmd = original_cmd.copy()
                market_order_cmd['order_type'] = 'market'
                market_order_cmd['price'] = None
                self._execute_single_command(market_order_cmd, price_now, online_positions_raw)
        except Exception as e:
            self.log.error(f"[TimeoutHandler] 執行重試邏輯時發生未知錯誤: {e}", exc_info=True)

    def stop_tracker(self, client_order_id: str):
        """外部呼叫：停止一個指定的訂單追蹤器。"""
        if client_order_id in self.active_trackers:
            tracker = self.active_trackers.pop(client_order_id)
            tracker.stop()