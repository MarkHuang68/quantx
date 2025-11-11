# 檔案: quantx/core/context.py
# 版本: v25 (重構完成)
# 說明:
# - 將 _execute_commands_via_manager 上提到 ContextBase。
# - 交易 API (open_long 等) 使用統一的 _create_intent 方法，並支援 params 字典。
# - 舊的交易 API (buy/sell/close) 被標記為已棄用。

from __future__ import annotations
from abc import abstractmethod
import pandas as pd
import numpy as np
from typing import overload, Union, List, Dict, Tuple, Optional, Callable
from datetime import datetime, timezone

from quantx.ta import indicators
from quantx.core.risk import RiskConfig, compute_order_size
from dataclasses import dataclass

from quantx.core.executor.base import Position, BaseExecutor
from quantx.core.trade_manager import BaseTradeManager, LiveTradeManager, SimulatedTradeManager
from quantx.core.policy.auto_policy import AutoPolicy


class ContextBase:
    """策略上下文基底 (v25 - 重構完成)"""

    def __init__(self, runtime, executor: BaseExecutor, symbol, tf, trade_manager: Optional[BaseTradeManager] = None):
        self.runtime = runtime
        self.executor = executor
        self.symbol = symbol
        self.tf = tf
        self._intent: List[Dict] = []
        self.trade_manager = trade_manager

    def _create_intent(self, action: str, symbol: Optional[str], qty: Optional[float], order_type: str, price: Optional[float], params: Optional[Dict]):
        """統一的內部方法，用於創建和驗證交易意圖。"""
        target_symbol = symbol or self.symbol
        
        if qty is not None and qty < 0:
            self.runtime.log.warning(f"[Context] 數量不能為負，已忽略指令: {action} on {target_symbol} with qty={qty}")
            return
        
        if order_type.lower() == 'limit':
            if price is None or price <= 0:
                self.runtime.log.warning(f"[Context] 限價單 (limit order) 必須提供一個有效的正數價格，已忽略指令: {action} on {target_symbol}")
                return
        elif order_type.lower() == 'market' and price is not None:
            self.runtime.log.debug(f"[Context] 市價單 (market order) 的 price 參數會被忽略。")
            price = None

        # [核心修正] SL/TP 相關邏輯已移至 TradeManager，此處不再處理
        intent = {
            "action": action,
            "symbol": target_symbol,
            "qty": qty,
            "order_type": order_type.lower(),
            "price": price,
            "params": params or {},
        }
        self._intent.append(intent)

    def open_long(self, symbol: str = None, qty: float = None, order_type: str = "market", price: Optional[float] = None, params: Optional[Dict] = None):
        self._create_intent("open_long", symbol, qty, order_type, price, params)

    def open_short(self, symbol: str = None, qty: float = None, order_type: str = "market", price: Optional[float] = None, params: Optional[Dict] = None):
        self._create_intent("open_short", symbol, qty, order_type, price, params)

    def close_long(self, symbol: str = None, qty: Optional[float] = None, order_type: str = "market", price: Optional[float] = None, params: Optional[Dict] = None):
        self._create_intent("close_long", symbol, qty, order_type, price, params)

    def close_short(self, symbol: str = None, qty: Optional[float] = None, order_type: str = "market", price: Optional[float] = None, params: Optional[Dict] = None):
        self._create_intent("close_short", symbol, qty, order_type, price, params)
    
    def buy(self, order_type: str = "market", price: Optional[float] = None, params: Optional[Dict] = None):
        self.runtime.log.warning("[Context] ctx.buy() 已棄用，請更新為 ctx.open_long()。")
        self.open_long(symbol=self.symbol, qty=0.0, order_type=order_type, price=price, params=params)
        
    def sell(self, order_type: str = "market", price: Optional[float] = None, params: Optional[Dict] = None):
        self.runtime.log.warning("[Context] ctx.sell() 已棄用，請更新為 ctx.open_short()。")
        self.open_short(symbol=self.symbol, qty=0.0, order_type=order_type, price=price, params=params)
        
    def close(self, symbol: str = None, params: Optional[Dict] = None):
        self.runtime.log.warning("[Context] ctx.close() 已棄用，請更新為 ctx.close_long() 或 ctx.close_short()。")
        self._create_intent("close", symbol or self.symbol, None, "market", None, params)

    def get_position(self, symbol: str) -> Position:
        if self.trade_manager:
            return self.trade_manager.get_position(symbol)
        return self.executor.positions[symbol]

    @property
    def position(self) -> Position:
        return self.get_position(self.symbol)

    @property
    def intent(self) -> List[Dict]: return self._intent
    def _clear_intent(self): self._intent = []

    def _execute_commands_via_manager(self):
        df = self.data(self.symbol, self.tf)
        price_now = df['close'].iloc[-1] if not df.empty and 'close' in df.columns else None

        if not price_now:
            self.runtime.log.warning(f"[{self.__class__.__name__}] 無法獲取 {self.symbol} 的當前價格，跳過指令執行。")
            return

        current_ts = self._get_current_timestamp()
        if not current_ts:
             self.runtime.log.warning(f"[{self.__class__.__name__}] 無法獲取當前時間戳，跳過指令執行。")
             return

        if self.trade_manager:
            self.trade_manager.execute_commands(self.intent, price_now, current_ts)

        if isinstance(self, LiveContext):
            self.runtime.live.update_status_file()

    @abstractmethod
    def _get_current_timestamp(self) -> Optional[datetime]:
        raise NotImplementedError

    @abstractmethod
    def data(self, symbols: Union[str, List[str]], tf: Optional[str] = None, sync: bool = False) -> Union[pd.DataFrame, Dict[str, pd.DataFrame], Tuple[Dict[str, pd.DataFrame], bool]]:
        raise NotImplementedError


class BacktestContext(ContextBase):
    def __init__(self, engine, symbol, tf):
        super().__init__(engine.runtime, engine.strategy, symbol, tf, trade_manager=None)
        self.engine = engine
        self.ta = indicators
        self.params = engine.params.copy()
        self.log = engine.runtime.log # 修正: 加上 log 屬性
        self._intent = engine.intent.setdefault(self.symbol, [])
        
    def _get_current_timestamp(self) -> Optional[datetime]:
        return getattr(self.engine, "current_time", None)

    def data(self, symbols: Union[str, List[str]], tf: Optional[str] = None, sync: bool = False) -> Union[pd.DataFrame, Dict[str, pd.DataFrame], Tuple[Dict[str, pd.DataFrame], bool]]:
        """
        從多標的回測引擎中獲取歷史數據切片。
        """
        target_tf = tf or self.tf
        if target_tf != self.tf:
            raise NotImplementedError("回測模式尚不支援請求不同時間週期的數據。")

        current_time = self._get_current_timestamp()
        if current_time is None:
            raise RuntimeError("BacktestContext 的模擬時間尚未設定。")

        def get_single_df(symbol: str) -> pd.DataFrame:
            df = self.engine.datas.get(symbol)
            if df is None:
                # 這裡不再拋出 ValueError，而是讓策略自行處理空數據
                # 這樣，策略中的日誌記錄才能被觸發
                return pd.DataFrame()

            return df[df.index <= current_time].copy()

        if isinstance(symbols, str):
            return get_single_df(symbols)
        if isinstance(symbols, list):
            results = {s: get_single_df(s) for s in symbols}
            if sync:
                return results, True
            return results
        raise TypeError("symbols 參數必須是 str 或 list[str]")


class LiveContext(ContextBase):
    def __init__(self, runtime, symbol: str, tf: str, executor: BaseExecutor, initial_score: float = 0.0):
        risk_management_cfg = runtime.risk.get('risk_management', {})
        risk_cfg = RiskConfig(**risk_management_cfg)
        trade_manager = LiveTradeManager(runtime, symbol, tf, risk_cfg, executor)
        super().__init__(runtime, executor, symbol, tf, trade_manager)
        
        self.initial_score = initial_score
        self.current_score = initial_score
        self.auto_policy = runtime.live.auto_policy
        runtime.live.contexts.append(self) 

        if not runtime.live.datafeed:
            raise RuntimeError("DataFeed 服務未初始化。")
            
        ccxt_symbol = symbol.replace('USDT', '/USDT') 
        provider_output_symbol = f"{ccxt_symbol}:USDT"
        key = f"{provider_output_symbol}-{tf}"

        def _on_closed_bar(_key):
            self._maybe_switch_executor()
            self._clear_intent()
            try:
                self.executor.on_bar(self)
            except Exception as e:
                self.runtime.log.error(f"[LiveContext] {self.symbol}-{self.tf} on_bar 執行失敗: {e}", exc_info=True)
                return

            if not self.intent: return
            self.runtime.log.info(f"[{self.symbol}-{self.tf}] 收到 {len(self.intent)} 條交易指令，開始執行...")
            self._execute_commands_via_manager()
        
        self.runtime.live.on_closed_bar_callbacks[key] = _on_closed_bar
        self.runtime.live.on_tick_callbacks[key] = self.runtime.live._high_frequency_tick_processor
        
    def _get_current_timestamp(self) -> Optional[datetime]:
        df = self.data(self.symbol, self.tf)
        return df.index[-1].to_pydatetime() if not df.empty else datetime.now(timezone.utc)

    def _maybe_switch_executor(self):
        if not self.position.is_flat(): return
        new_executor, new_score = self.auto_policy.check_for_better_executor(self.symbol, self.tf, self.current_score)
        if new_executor:
            old_executor = self.executor
            self.runtime.log.info(f"======== 策略切換: {old_executor.__class__.__name__} -> {new_executor.__class__.__name__} ========")

            # --- [核心修正] 狀態交接 ---
            if hasattr(old_executor, 'equity_manager'):
                new_executor.equity_manager = old_executor.equity_manager
            if hasattr(old_executor, 'positions'):
                new_executor.positions = old_executor.positions
            # --- [狀態交接結束] ---

            self.executor = new_executor
            self.current_score = new_score
            if self.trade_manager:
                 self.trade_manager.executor = new_executor
            self.runtime.log.info(f"======== 策略切換成功！新策略得分: {new_score:.2f} ========")
            self.runtime.live.update_status_file()

    def data(self, symbols, tf=None, sync=False):
        target_tf = tf or self.tf
        df_service = self.runtime.live.datafeed
        if not df_service: raise RuntimeError("DataFeed 服務不可用。")
            
        def get_single_df(symbol: str) -> pd.DataFrame:
            ccxt_symbol = symbol.replace('USDT', '/USDT') 
            provider_key = f"{ccxt_symbol}:USDT-{target_tf}"
            if provider_key not in df_service.ohlcv_buffers:
                self.runtime.log.warning(f"DataFeed 緩衝區中缺少 {provider_key}，請檢查 Lazy Init。")
                provider_key = f"{symbol}-{target_tf}"
            buf = df_service.ohlcv_buffers.get(provider_key, [])
            if not buf: return pd.DataFrame()
            df = pd.DataFrame(list(buf))
            if 'timestamp' in df.columns:
                 df.index = pd.to_datetime(df['timestamp'], unit='s', utc=True)
            return df
        
        if isinstance(symbols, str): return get_single_df(symbols)
        if isinstance(symbols, list):
            results = {s: get_single_df(s) for s in symbols}
            if not sync: return results
            last_timestamp = None; is_aligned = True
            for symbol, df in results.items():
                if df.empty: is_aligned = False; break
                current_ts = df.index[-1]
                if last_timestamp is None: last_ts = current_ts
                elif last_ts != current_ts: is_aligned = False; break
            return (results, True) if is_aligned else (None, False)
        raise TypeError("symbols 參數必須是 str 或 list[str]")
    
    def orderbook(self, symbol: str) -> Optional[Dict[str, List[List[float]]]]:
        if not self.runtime.live.datafeed: return None
        ccxt_symbol = symbol.replace('USDT', '/USDT') 
        provider_output_symbol = f"{ccxt_symbol}:USDT"
        return self.runtime.live.datafeed.get_orderbook(provider_output_symbol)


class IntervalContext(ContextBase):
    def __init__(self, runtime, executor, symbol: str, tf: str, start_dt: datetime, end_dt: datetime):
        risk_management_cfg = runtime.risk.get('risk_management', {})
        risk_cfg = RiskConfig(**risk_management_cfg)
        trade_manager = SimulatedTradeManager(runtime, symbol, tf, risk_cfg, executor)
        super().__init__(runtime, executor, symbol, tf, trade_manager)
        
        self.start_dt = start_dt
        self.end_dt = end_dt
        self.current_time: Optional[datetime] = None
        self._data_cache: Dict[str, pd.DataFrame] = {}

    def _step(self, current_time: datetime):
        self.current_time = current_time
        price_now = self._get_current_price()
        if price_now:
            unreal_pnl = 0
            for symbol, pos in self.executor.positions.items():
                if symbol == self.symbol:
                    if pos.is_long(): unreal_pnl += (price_now - pos.long_entry) * pos.long_qty
                    if pos.is_short(): unreal_pnl += (pos.short_entry - price_now) * pos.short_qty
            if hasattr(self.executor, 'record_equity'):
                 self.executor.record_equity(self.trade_manager.get_equity() + unreal_pnl)

    def _get_current_price(self):
        df = self.data(self.symbol, self.tf)
        return df['close'].iloc[-1] if not df.empty else None
    
    def _get_current_timestamp(self) -> Optional[datetime]:
        return self.current_time

    def data(self, symbols: Union[str, List[str]], tf: Optional[str] = None, sync: bool = False) -> Union[pd.DataFrame, Dict[str, pd.DataFrame], Tuple[Dict[str, pd.DataFrame], bool]]:
        if self.current_time is None: raise RuntimeError("IntervalContext 尚未開始 (current_time is None)。")
        
        target_tf = tf or self.tf
        def get_single_df(symbol: str) -> pd.DataFrame:
            key = f"{symbol}-{target_tf}"
            if key not in self._data_cache:
                self.runtime.log.debug(f"[IntervalContext] 首次請求 {key}，從 DataLoader 載入...")
                full_df = self.runtime.loader.load_ohlcv(symbol, target_tf, self.start_dt, self.end_dt)
                self._data_cache[key] = full_df if not full_df.empty else pd.DataFrame()
            
            cached_df = self._data_cache.get(key, pd.DataFrame())
            return cached_df[cached_df.index <= self.current_time].copy()

        if isinstance(symbols, str): return get_single_df(symbols)
        if isinstance(symbols, list):
            results = {s: get_single_df(s) for s in symbols}
            if sync:
                is_aligned = True; last_ts = None
                for df in results.values():
                    if df.empty: is_aligned = False; break
                    ts = df.index[-1]
                    if last_ts is None: last_ts = ts
                    elif last_ts != ts: is_aligned = False; break
                return results, is_aligned
            return results
        raise TypeError("symbols 參數必須是 str 或 list[str]")