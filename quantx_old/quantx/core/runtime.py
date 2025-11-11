# 檔案: quantx/core/runtime.py
# 版本: v33 (修正：LiveStatus 熱更新)
# 說明:
# - 新增 LiveRuntime.update_config 方法，用於在運行時熱更新配置。
# - _high_frequency_tick_processor 和 update_status_file 的邏輯保持不變。

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Callable, Any, Dict, List
from pathlib import Path
import os
import inspect
import logging
import importlib
import asyncio
import json
from datetime import datetime, timezone
import time
from collections import defaultdict

# 依賴項
from quantx.core.data.loader import DataLoader
from quantx.core.data.datafeed import DataFeed
from quantx.core.executor.base import BaseExecutor
from quantx.core.log_formatter import get_rich_handler
from quantx.core.signal_handler import should_stop

# 延遲導入
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from quantx.core.policy.auto_policy import AutoPolicy

def _is_concrete_strategy(cls, base_cls):
    if not inspect.isclass(cls) or not issubclass(cls, base_cls) or cls is base_cls: return False
    return ("on_bar" in cls.__dict__) and callable(cls.__dict__["on_bar"])

@dataclass
class Runtime:
    exchange: str
    mode: str
    loader: DataLoader
    exchange_config: dict
    risk: Dict = field(default_factory=dict)
    provider: Any = None

    def get_cost_model(self) -> dict:
        exchange_name = self.exchange
        if exchange_name in self.exchange_config.get('exchanges', {}):
            return self.exchange_config['exchanges'][exchange_name]
        else:
            self.log.warning(f"在 exchange.yaml 中找不到 '{exchange_name}' 的設定，將使用預設成本。")
            return {'maker_fee_bps': 2.0, 'taker_fee_bps': 5.5, 'slip_bps': 1.0}

    @property
    def scope(self) -> str: return f"{self.exchange}_{self.mode}"

    @property
    def log(self) -> logging.Logger:
        logger = logging.getLogger(f"quantx.{self.mode}")
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            rich_handler = get_rich_handler()
            logger.addHandler(rich_handler)
            file_formatter = logging.Formatter(
                fmt="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            log_dir = Path("results"); log_dir.mkdir(exist_ok=True)
            fh = logging.FileHandler(log_dir / f"{self.mode}.log", encoding="utf-8")
            fh.setFormatter(file_formatter)
            logger.addHandler(fh)
            logger.propagate = False
        return logger

    def load_strategy(self, name: str) -> Any:
        mod = importlib.import_module(f"quantx.strategy.{name}")
        for _, cls in inspect.getmembers(mod, inspect.isclass):
            if _is_concrete_strategy(cls, BaseExecutor): return cls
        raise ImportError(f"Strategy {name} not found")

    def load_ml(self, name: str) -> Any:
        mod = importlib.import_module(f"quantx.core.model.{name}_trainer")
        if hasattr(mod, f"train_{name}"): return getattr(mod, f"train_{name}")
        raise ImportError(f"ML model {name} not found")

    @property
    def live(self) -> "LiveRuntime":
        if not hasattr(self, "_live_runtime"): self._live_runtime = LiveRuntime(self)
        return self._live_runtime

class LiveRuntime:
    class SharedEquityManager:
        def __init__(self, initial_equity):
            self.equity = initial_equity
            self.realized_pnl = 0.0
        
        def apply_pnl(self, pnl):
            self.equity += pnl
            self.realized_pnl += pnl

    def __init__(self, runtime: "Runtime"):
        self.runtime = runtime
        self._running = True
        self.on_closed_bar_callbacks: Dict[str, Callable] = {}
        self.on_tick_callbacks: Dict[str, Callable] = {}
        self.datafeed: Optional[DataFeed] = None
        self.contexts: List[Any] = []
        self.status_file = Path("results") / "live_status.json"
        
        # 🟢 核心修改 1: 讀取 Live Status 報告開關
        reporting_cfg = self.runtime.risk.get('reporting', {})
        self._report_status_file = reporting_cfg.get('report_status_file', True)

        self.update_config(self.runtime.risk) # 首次初始化時，使用 runtime.risk 的內容

        self._last_status_report_ms = 0
        self._crash_risk_state: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"breach_time": None})
        self.crash_risk_confirm_delay = 5.0
        
        self.is_trading_active: bool = False
        self.session_initial_equity: float = 0.0

        # AutoPolicy 將由 DI 容器創建後注入
        self.auto_policy: Optional[AutoPolicy] = None

        if self.runtime.mode in ["live", "testnet"]:
            self.runtime.log.info("[LiveRuntime] Live 模式，正在初始化 DataFeed 服務...")
            self.datafeed = DataFeed(self.runtime)
            
            initial_equity = self.runtime.risk.get('initial_equity', 10000.0)
            self.equity_manager = self.SharedEquityManager(initial_equity)
            self.runtime.log.info(f"[LiveRuntime] SharedEquityManager 初始化資金: {self.equity_manager.equity}")
            self.runtime.log.warning("[LiveRuntime] 全局風控處於「待命」狀態。")
        
        # 🟢 核心修改 2: 狀態檔案如果被禁用，立即清除，以防止讀取舊檔案
        if not self._report_status_file and self.status_file.exists():
             try:
                 self.status_file.unlink()
                 self.runtime.log.info("[LiveRuntime] Live Status 檔案輸出已禁用，已移除舊檔案。")
             except Exception as e:
                 self.runtime.log.warning(f"[LiveRuntime] 無法移除舊的 Live Status 檔案: {e}")

    # 🟢 核心修改 3: 新增熱更新配置方法
    def update_config(self, risk_config: Dict):
        """
        在運行時更新 LiveRuntime 的內部配置。
        
        Args:
            risk_config (Dict): 包含 live.yaml 所有內容的字典。
        """
        reporting_cfg = risk_config.get('reporting', {})
        new_report_status_file = reporting_cfg.get('report_status_file', True)
        
        # 檢查是否有狀態變更
        if hasattr(self, '_report_status_file') and self._report_status_file != new_report_status_file:
            self.runtime.log.info(f"[LiveRuntime Hot Reload] Live Status 寫入已從 {self._report_status_file} 變更為 {new_report_status_file}。")
            if not new_report_status_file and self.status_file.exists():
                try:
                    self.status_file.unlink()
                    self.runtime.log.info("[LiveRuntime Hot Reload] Status 檔案已移除。")
                except Exception as e:
                    self.runtime.log.warning(f"[LiveRuntime Hot Reload] 無法移除 Status 檔案: {e}")
        
        self._report_status_file = new_report_status_file
        self.runtime.risk.update(risk_config) # 確保 runtime.risk 也被更新

    @property
    def is_running(self) -> bool:
        return self._running

    def stop(self):
        """停止 LiveRuntime 的主循環。"""
        self.runtime.log.info("[LiveRuntime] 收到停止指令...")
        self._running = False

    def activate_trading_session(self, online_positions: Dict[str, Any], managed_symbols: set):
        if self.is_trading_active:
            return
            
        # 初始資金基準 = 當前可用餘額
        self.session_initial_equity = self.equity_manager.equity
        self.is_trading_active = True
        self.runtime.log.info("==========================================================")
        self.runtime.log.info(f"🚀 全局風控已啟動！初始資金基準已鎖定為: {self.session_initial_equity:.4f} USDT")
        self.runtime.log.info("==========================================================")
        
        self.update_status_file()


    def reset_crash_risk_state(self, symbol: str):
        clean_symbol = symbol.split(':')[0].replace('/', '')
        if clean_symbol in self._crash_risk_state:
            if self._crash_risk_state[clean_symbol]['breach_time'] is not None:
                self._crash_risk_state[clean_symbol]['breach_time'] = None
                self.runtime.log.info(f"✅ [CRASH RISK RESET] {clean_symbol} 的全局風控計時狀態已被外部重置。")

    def _high_frequency_tick_processor(self, key: str, bar_data: Dict[str, Any]):
        # 如果風控未激活，直接返回
        if not self.is_trading_active: return

        # [核心修正] 統一計算全局資產
        # 1. 計算全局的總未實現損益
        total_unrealized_pnl = 0.0
        for context in self.contexts:
            pos = context.position
            latest_price = 0.0

            # 從 datafeed buffer 中獲取最新價格
            price_key = f"{context.symbol.replace('USDT', '/USDT')}:USDT-{context.tf}"
            if self.datafeed and (buf := self.datafeed.ohlcv_buffers.get(price_key)):
                if buf: latest_price = buf[-1]['close']

            if latest_price <= 0:
                 latest_price = pos.long_entry or pos.short_entry

            if not pos.is_flat() and latest_price > 0:
                if pos.is_long(): total_unrealized_pnl += (latest_price - pos.long_entry) * pos.long_qty
                if pos.is_short(): total_unrealized_pnl += (pos.short_entry - latest_price) * pos.short_qty

        # 2. 計算統一的當前總資產
        current_total_equity = self.equity_manager.equity + total_unrealized_pnl

        # 3. 將統一的總資產傳遞給風控檢查
        symbol_base = key.split(':')[0].replace('/', '')
        contexts_for_symbol = [c for c in self.contexts if c.symbol == symbol_base]
        trigger_price = bar_data.get('close', 0.0)

        for ctx in contexts_for_symbol:
            self._check_and_execute_crash_risk(ctx, trigger_price, current_total_equity)
            
        current_time_ms = int(time.time() * 1000)
        if self._report_status_file and (current_time_ms - self._last_status_report_ms > 500):
             self.update_status_file()
             self._last_status_report_ms = current_time_ms


    def _check_and_execute_crash_risk(self, ctx: Any, trigger_price: float, current_total_equity: float):
        if not self.is_trading_active: return

        initial_equity = self.session_initial_equity
        if not (trigger_price > 0 and initial_equity > 0): return

        global_risk_cfg = self.runtime.risk.get('risk', {})
        position = ctx.position

        # --- 1. 全局總資金回撤檢查 ---
        max_total_dd_pct = global_risk_cfg.get('max_total_drawdown_pct', 100.0) / 100.0
        if max_total_dd_pct < 1.0:
            total_drawdown = (initial_equity - current_total_equity) / initial_equity
            if total_drawdown > max_total_dd_pct:
                # 只在倉位非空時執行，避免對已平倉的策略重複操作
                if not position.is_flat() and getattr(ctx.executor, 'accepts_global_drawdown_action', False):
                    self.runtime.log.critical(f"🔥🔥 [GLOBAL DRAWDOWN EXEC] 總資金回撤 ({total_drawdown:.2%}) 超過閾值 ({max_total_dd_pct:.2%})！"
                                              f"策略 '{ctx.executor.__class__.__name__}' 已同意，對 {ctx.symbol} 執行強制平倉。")
                    ctx.trade_manager.execute_commands(
                        [{"action": "close", "symbol": ctx.symbol, "reason": "GLOBAL_TOTAL_DRAWDOWN"}],
                        trigger_price, datetime.now(timezone.utc))

                    self.runtime.log.info(f"  - [SYNC] 強制平倉後，立即將 {ctx.symbol} 的本地倉位狀態清空。")
                    position.long_qty = position.long_entry = position.short_qty = position.short_entry = 0.0
                else:
                    # 增加日誌清晰度
                    reason = "策略未啟用" if not position.is_flat() else "倉位已空"
                    self.runtime.log.warning(f"⚠️ [GLOBAL DRAWDOWN SKIP] 總資金回撤觸發，但因 ({reason}) 跳過對 {ctx.symbol} 的操作。")

        # --- 2. 瞬間波動風控檢查 (Flash Crash) ---
        max_flash_dd_pct = global_risk_cfg.get('max_flash_crash_pct', 10.0) / 100.0
        if max_flash_dd_pct >= 1.0: return

        current_drawdown = (initial_equity - current_total_equity) / initial_equity
        state = self._crash_risk_state[ctx.symbol]
        current_time = time.time()

        if current_drawdown > max_flash_dd_pct:
            if state['breach_time'] is None:
                state['breach_time'] = current_time
                log_msg = (
                    f"⚠️ [FLASH CRASH] {ctx.symbol} 首次觀測到全局資產跌破瞬間風控閾值 ({current_drawdown:.2%} > {max_flash_dd_pct:.2%})。\n"
                    f"  - Initial Equity: {initial_equity:.4f}, Current Total Equity: {current_total_equity:.4f}"
                )
                self.runtime.log.warning(log_msg)
            elif current_time - state['breach_time'] >= self.crash_risk_confirm_delay:
                if not position.is_flat() and getattr(ctx.executor, 'accepts_flash_crash_action', False):
                    self.runtime.log.critical(f"🔥🔥 [FLASH CRASH EXEC] {ctx.symbol} 連續跌破閾值！其策略已同意，立即強制平倉。")
                    ctx.trade_manager.execute_commands(
                        [{"action": "close", "symbol": ctx.symbol, "reason": "GLOBAL_FLASH_CRASH"}],
                        trigger_price, datetime.now(timezone.utc))

                    self.runtime.log.info(f"  - [SYNC] 強制平倉後，立即將 {ctx.symbol} 的本地倉位狀態清空。")
                    position.long_qty = position.long_entry = position.short_qty = position.short_entry = 0.0

                    self.reset_crash_risk_state(ctx.symbol)
                else:
                    reason = "策略未啟用" if not position.is_flat() else "倉位已空"
                    self.runtime.log.warning(f"⚠️ [FLASH CRASH SKIP] {ctx.symbol} 觸發瞬間風控，但因 ({reason}) 跳過平倉。")
                    # 即使跳過，也要重置計時器，避免不斷打印日誌
                    self.reset_crash_risk_state(ctx.symbol)

        elif state['breach_time'] is not None:
            self.runtime.log.info(f"✅ [FLASH CRASH CLEAR] {ctx.symbol} 全局資產淨值已回升。")
            state['breach_time'] = None
    
    def update_status_file(self):
        """
        同步方法：立即計算並寫入 Live Status 檔案。
        用於策略切換或交易執行後的即時更新。
        """
        if not self._report_status_file:
            return

        strategy_statuses, total_unrealized_pnl = [], 0.0
        for ctx in self.contexts:
            executor = ctx.executor
            position = ctx.position 
            current_price = 0.0
            
            try:
                # 嘗試從數據緩衝區獲取最新價格 (使用 datafeed 的 ohlcv_buffers)
                key = f"{ctx.symbol.replace('USDT', '/USDT')}:USDT-{ctx.tf}"
                if (df_service := self.datafeed) and (buf := df_service.ohlcv_buffers.get(key)):
                    if buf:
                        current_price = buf[-1]['close']
                
                if current_price == 0.0:
                     current_price = position.long_entry or position.short_entry
            except Exception: 
                 current_price = position.long_entry or position.short_entry

            unrealized_pnl = 0.0
            
            if not position.is_flat() and current_price > 0:
                if position.is_long(): 
                    unrealized_pnl += (current_price - position.long_entry) * position.long_qty
                if position.is_short(): 
                    unrealized_pnl += (position.short_entry - current_price) * position.short_qty
            
            total_unrealized_pnl += unrealized_pnl
            
            total_size = (position.long_qty or 0.0) - (position.short_qty or 0.0)
            
            strategy_statuses.append({
                "symbol": f"{ctx.symbol}-{ctx.tf}", 
                "strategy": executor.__class__.__name__, 
                "position_size": round(total_size, 5), 
                "entry_price": round(position.long_entry or position.short_entry or 0.0, 5), 
                "current_price": round(current_price, 5), 
                "unrealized_pnl": round(unrealized_pnl, 5),
            })
            
        total_equity = self.equity_manager.equity + total_unrealized_pnl
        final_status = {
            "portfolio_status": {
                "total_equity": round(total_equity, 5), 
                "available_equity": round(self.equity_manager.equity, 5), 
                "total_unrealized_pnl": round(total_unrealized_pnl, 5), 
                "total_realized_pnl": round(self.equity_manager.realized_pnl, 5), 
                "last_update": datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
            }, 
            "strategy_status": strategy_statuses
        }
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.status_file, "w", encoding="utf-8") as f: 
                json.dump(final_status, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.runtime.log.error(f"[LiveStatus] 寫入狀態檔案失敗: {e}")

    def register_context(self, context: Any):
        self.contexts.append(context)
        self.update_status_file()


    async def start_services(self):
        self.runtime.log.info("[LiveRuntime] 正在啟動背景服務...")
        if self.datafeed: self.datafeed.start()
        await asyncio.sleep(1)
        self.runtime.log.info("[LiveRuntime] 背景服務已啟動。")

    async def stop_services(self):
        self.runtime.log.info("[LiveRuntime] 正在停止背景服務...")
        if self.datafeed: await self.datafeed.stop()
        self.runtime.log.info("[LiveRuntime] 背景服務已停止。")
        
    async def run_forever(self):
        await self.start_services()
        try:
            while self.is_running and not should_stop():
                await asyncio.sleep(1)
        except (KeyboardInterrupt, asyncio.CancelledError):
            self.runtime.log.info("[LiveRuntime] 收到中斷信號...")
            self._running = False
        finally:
            await self.stop_services()
            if self._report_status_file:
                 self.update_status_file() 
            self.runtime.log.info("[LiveRuntime] 已安全停止。")