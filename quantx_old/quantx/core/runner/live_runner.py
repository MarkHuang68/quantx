# 檔案: quantx/core/runner/live_runner.py
# 版本: v26 (重構完成)
# 說明:
# - run 方法現在採用集中式數據訂閱。
# - 在啟動時就呼叫 datafeed.subscribe_bulk，並行回補歷史數據。

from __future__ import annotations
import asyncio
import math
from typing import TYPE_CHECKING, List, Dict, Any, Optional, Set
from pathlib import Path
import logging
from collections import defaultdict

from quantx.core.context import LiveContext
from quantx.core.executor.base import Position, BaseExecutor
from quantx.core.policy.auto_policy import AutoPolicy

if TYPE_CHECKING:
    from quantx.core.runtime import Runtime
    from quantx.core.config import Config

class LiveRunner:
    """實盤/紙上交易模式執行器"""
    def __init__(self, runtime: 'Runtime', auto_policy: 'AutoPolicy', symbols_cfg: list, cfg: 'Config'):
        self.runtime = runtime
        self.log = runtime.log
        self.auto_policy = auto_policy
        self.symbols_cfg = symbols_cfg
        self.cfg = cfg
        self.active_contexts: Dict[str, LiveContext] = {}
        self.managed_symbols: Set[str] = set()

    async def _launch_context(self, symbol: str, tf: str, executor: BaseExecutor, score: float, position: Optional[Position] = None):
        self.log.info(f"正在為 {symbol} 啟動新策略: {executor.__class__.__name__} on {tf} (分數: {score:.2f})")
        if position:
            executor.positions[symbol] = position
            self.log.info(f"  - 接管倉位: Long={position.long_qty}, Short={position.short_qty}")
        new_ctx = LiveContext(self.runtime, symbol, tf, executor, initial_score=score)
        self.active_contexts[symbol] = new_ctx
        self.managed_symbols.add(symbol) # [核心修正] 追蹤可管理的標的
        self.log.info(f"✅ 策略 {symbol}-{tf} 已成功啟動並監控。")
        self.runtime.live.update_status_file()

    def _get_online_positions(self) -> Dict[str, Dict[str, Any]]:
        try:
            online_positions_raw = self.runtime.loader.provider.get_positions()
            return {p['symbol'].replace('/', '').split(':')[0]: p for p in online_positions_raw}
        except Exception as e:
            self.log.error(f"查詢線上倉位失敗: {e}", exc_info=True)
            return {}

    async def _handle_existing_position(self, symbol: str, position_data: Dict[str, Any]) -> bool:
        if symbol in self.active_contexts: return True
        self.log.info(f"發現 {symbol} 的現有倉位，嘗試指派策略進行管理...")
        executor, score, tf = self.auto_policy.select_best_for_legacy_takeover(symbol)
        
        if executor and tf:
            amount = float(position_data.get('contracts', 0) or 0)
            entry_price = float(position_data.get('entryPrice', 0) or 0.0)
            side = position_data.get('side')
            position = Position(long_qty=amount, long_entry=entry_price) if side == 'long' else Position(short_qty=amount, short_entry=entry_price)
            await self._launch_context(symbol, tf, executor, score, position)
            return True
        else:
            self.log.warning(f"無法為 {symbol} 的現有倉位找到任何可用的接管策略！")
            return False

    async def _handle_flat_symbol(self, symbol: str):
        if symbol in self.active_contexts: return
        executor, score, tf = self.auto_policy._get_best_candidate_for_symbol(symbol)
        if executor and tf:
            await self._launch_context(symbol, tf, executor, score)
        else:
            self.log.debug(f"為 {symbol} 執行 AutoPolicy 後，未找到任何合格的策略可供開倉。")

    async def _check_orphan_positions_and_reconcile(self):
        """檢查孤兒倉位並對齊本地與遠程的倉位狀態。"""
        self.log.debug("[Reconcile] 開始執行倉位對齊檢查...")
        try:
            online_positions = self._get_online_positions()
        except Exception as e:
            self.log.error(f"[Reconcile] 無法獲取線上倉位進行對齊: {e}")
            return

        online_symbols = set(online_positions.keys())
        local_symbols = set(self.active_contexts.keys())

        # 1. 檢查孤兒倉位 (線上存在，本地不存在)
        orphan_symbols = online_symbols - local_symbols
        if orphan_symbols:
            self.log.warning(f"[Reconcile] 發現 {len(orphan_symbols)} 個孤兒倉位: {list(orphan_symbols)}，正在嘗試接管...")
            for symbol in orphan_symbols:
                await self._handle_existing_position(symbol, online_positions[symbol])

        # 2. 檢查本地與線上的倉位狀態是否一致
        for symbol in local_symbols.intersection(online_symbols):
            ctx = self.active_contexts[symbol]
            local_pos = ctx.position
            remote_pos_data = online_positions[symbol]

            remote_qty = float(remote_pos_data.get('contracts', 0) or 0)
            remote_side = remote_pos_data.get('side', '').lower()

            is_mismatched = False
            if remote_side == 'long' and not (abs(local_pos.long_qty - remote_qty) < 1e-9 and local_pos.is_long()):
                is_mismatched = True
            elif remote_side == 'short' and not (abs(local_pos.short_qty - remote_qty) < 1e-9 and local_pos.is_short()):
                is_mismatched = True

            if is_mismatched:
                self.log.critical(f"🚨 [Reconcile] 倉位狀態不一致！ Symbol: {symbol}\n"
                                  f"  - 本地狀態: Long={local_pos.long_qty}, Short={local_pos.short_qty}\n"
                                  f"  - 線上狀態: Side={remote_side}, Qty={remote_qty}\n"
                                  f"  - 正在以線上狀態為準，強制同步本地倉位...")
                # 以線上狀態為準，更新本地倉位
                if remote_side == 'long':
                    local_pos.long_qty = remote_qty
                    local_pos.long_entry = float(remote_pos_data.get('entryPrice', 0) or 0.0)
                    local_pos.short_qty = 0.0
                    local_pos.short_entry = 0.0
                else: # short
                    local_pos.short_qty = remote_qty
                    local_pos.short_entry = float(remote_pos_data.get('entryPrice', 0) or 0.0)
                    local_pos.long_qty = 0.0
                    local_pos.long_entry = 0.0

        # 3. 檢查已平倉的倉位 (本地不存在，線上也不存在，但之前存在過)
        # 這個邏輯可以在未來的版本中添加，用於更複雜的狀態管理

    def _init_config_timestamps(self):
        """初始化設定檔的時間戳紀錄。"""
        self._config_timestamps = {}
        config_files = ["config.yaml", "live.yaml", "symbol.yaml"]
        for f_name in config_files:
            p = Path(f"conf/{f_name}")
            if p.exists():
                self._config_timestamps[f_name] = p.stat().st_mtime

    async def _check_config_reload(self):
        """檢查設定檔是否有變動，並觸發熱更新。"""
        if not hasattr(self, '_config_timestamps'):
            self._init_config_timestamps()

        has_changed = False
        changed_files = []
        config_files = ["config.yaml", "live.yaml", "symbol.yaml"]

        for f_name in config_files:
            p = Path(f"conf/{f_name}")
            if not p.exists(): continue

            last_mtime = self._config_timestamps.get(f_name)
            current_mtime = p.stat().st_mtime

            if last_mtime is None or current_mtime > last_mtime:
                has_changed = True
                changed_files.append(f_name)
                self._config_timestamps[f_name] = current_mtime

        if has_changed:
            self.log.warning(f"🚨 [Hot-Reload] 偵測到設定檔變更: {', '.join(changed_files)}，正在執行完整的熱載入...")
            try:
                # [核心修正] 執行完整的重載流程，確保狀態一致
                self.cfg.reload()

                # 1. 總是重載 risk/live 設定
                self.runtime.live.update_config(self.cfg.load_risk())
                self.log.info("[Hot-Reload] Live/Risk 設定已更新。")

                # 2. 總是重載 symbols 並處理變更
                new_symbols_cfg = self.cfg.load_symbol()
                await self._handle_symbol_changes(new_symbols_cfg)
                self.log.info("[Hot-Reload] Symbol 列表已更新。")

                self.log.info(f"[Hot-Reload] 所有設定檔已成功重新載入。")
            except Exception as e:
                self.log.error(f"[Hot-Reload] 執行完整熱載入時發生錯誤: {e}", exc_info=True)

    async def _handle_symbol_changes(self, new_symbols_cfg: list):
        """處理 symbol.yaml 變更後的邏輯，實現新增、移除和參數更新。"""
        new_symbols_map = {s[0]: s[1] for s in new_symbols_cfg}
        current_symbols_set = set(self.active_contexts.keys())

        # 1. 處理新增和參數變更
        for symbol, params in new_symbols_map.items():
            if symbol not in self.active_contexts:
                # 這是新增的 symbol
                self.log.info(f"[Hot-Reload] 新增標的: {symbol}")
                await self._handle_flat_symbol(symbol)
            else:
                # 這是可能需要更新參數的 symbol
                ctx = self.active_contexts[symbol]
                # 假設 symbol.yaml 中的參數直接對應到 executor 的屬性
                # 注意：這裡的 params 是 symbol.yaml 中 symbol 底下的整個字典
                strategy_params = params.get('strategy', {}).get('params', {})
                for key, value in strategy_params.items():
                    if hasattr(ctx.executor, key) and getattr(ctx.executor, key) != value:
                        self.log.info(f"[Hot-Reload] 更新 {symbol} 的參數: {key} 從 {getattr(ctx.executor, key)} -> {value}")
                        setattr(ctx.executor, key, value)

        # 2. 處理被移除的 symbol
        removed_symbols = current_symbols_set - set(new_symbols_map.keys())
        for symbol in removed_symbols:
            if symbol in self.active_contexts:
                ctx = self.active_contexts[symbol]
                if not ctx.position.is_flat():
                    self.log.warning(f"[Hot-Reload] {symbol} 已從設定檔移除，但仍有倉位。將其標記為『只平倉』模式。")
                    ctx.executor.is_winding_down = True
                else:
                    self.log.info(f"[Hot-Reload] {symbol} 已從設定檔移除且無倉位，將直接卸載。")
                    # 安全地移除 context (未來的步驟會實作)
                    self.active_contexts.pop(symbol, None)

        # 更新 runner 內部的設定檔鏡像
        self.symbols_cfg = new_symbols_cfg

    async def _master_loop(self):
        """背景主迴圈，用於熱更新、孤兒倉位監管、倉位一致性檢查等。"""
        self._init_config_timestamps() # 首次運行時初始化
        await asyncio.sleep(30) # 首次啟動後延遲30秒，等待所有服務穩定
        while self.runtime.live.is_running:
            try:
                self.log.debug("[MasterLoop] 正在執行背景檢查...")
                await self._check_orphan_positions_and_reconcile()
                await self._check_config_reload()
                await self._cleanup_wound_down_contexts()
            except Exception as e:
                self.log.error(f"[MasterLoop] 背景主迴圈發生錯誤: {e}", exc_info=True)

            await asyncio.sleep(10) # 頻率改為每 10 秒檢查一次

    async def _cleanup_wound_down_contexts(self):
        """清理那些已標記為 'winding down' 且倉位已平的 context。"""
        contexts_to_remove = []
        for symbol, ctx in self.active_contexts.items():
            if ctx.executor.is_winding_down and ctx.position.is_flat():
                contexts_to_remove.append(symbol)

        if contexts_to_remove:
            self.log.info(f"[Auto-Cleanup] 發現 {len(contexts_to_remove)} 個已完成平倉的策略，將進行卸載: {contexts_to_remove}")
            for symbol in contexts_to_remove:
                self.active_contexts.pop(symbol, None)
                # TODO: 未來可以增加取消特定 symbol 數據訂閱的邏輯
            self.runtime.live.update_status_file()

    async def stop(self):
        """
        優雅地停止 LiveRunner。
        """
        self.log.info("接收到停止信號，正在優雅地關閉 LiveRunner...")
        if self.runtime.live.datafeed:
            # [核心修正] 異步方法需要 await
            await self.runtime.live.datafeed.stop()

        # [核心修正] 現在 LiveRuntime 有了 stop 方法
        self.runtime.live.stop()
        self.log.info("LiveRunner 已成功停止。")

    async def startup(self):
        """
        執行所有非阻塞的啟動任務。
        """
        self.log.info("====== [智能調度中心] 系統啟動 ======")

        try:
            self.log.info("正在從交易所查詢帳戶餘額...")
            real_balance = self.runtime.provider.fetch_balance(currency='USDT')
            self.log.info(f"查詢成功！將使用真實帳戶餘額: {real_balance:.2f} USDT 作為初始資金。")
            self.runtime.live.equity_manager.equity = real_balance
        except Exception as e:
            self.log.error(f"無法從交易所獲取真實餘額: {e}。將使用 live.yaml 中的預設資金。")

        online_positions = self._get_online_positions()
        config_symbols = {s[0] for s in self.symbols_cfg}
        all_symbols = config_symbols.union(set(online_positions.keys()))

        if not all_symbols:
            self.log.warning("沒有找到任何需要監控的標的，系統將閒置。")
            return

        self.log.info(f"共發現 {len(all_symbols)} 個相關標的，準備數據訂閱...")

        targets_to_subscribe = []
        main_symbols_tfs = []
        for symbol in all_symbols:
            _, _, tf = self.auto_policy._get_best_candidate_for_symbol(symbol)
            if tf:
                ccxt_symbol = f"{symbol.replace('USDT', '/USDT')}:USDT"
                targets_to_subscribe.append((ccxt_symbol, tf))
                if symbol in config_symbols:
                    main_symbols_tfs.append((ccxt_symbol, tf))
            else:
                self.log.warning(f"無法為 {symbol} 確定 TF，將跳過其數據訂閱。")

        if self.runtime.live.datafeed and targets_to_subscribe:
            self.log.info("正在執行數據批次訂閱和歷史回補...")
            await self.runtime.live.datafeed.subscribe_bulk(targets_to_subscribe, main_symbols_tfs)
            self.log.info("數據訂閱和回補完成。")

        if self.runtime.live.datafeed:
            self.runtime.live.datafeed.start()

        orphan_positions = []
        if online_positions:
            self.log.info(f"發現 {len(online_positions)} 個線上持倉，優先處理...")
            for symbol, position_data in online_positions.items():
                is_managed = await self._handle_existing_position(symbol, position_data)
                if not is_managed:
                    orphan_positions.append(symbol)
        
        # [核心修正] 對孤兒倉位發出一次性警告
        if orphan_positions:
            self.log.critical("="*60)
            self.log.critical(f"⚠️ 偵測到 {len(orphan_positions)} 個孤兒倉位，它們將被排除在自動化風控之外！")
            self.log.critical(f"   請手動處理以下倉位: {', '.join(orphan_positions)}")
            self.log.critical("="*60)

        symbols_without_positions = config_symbols - set(online_positions.keys())
        if symbols_without_positions:
            self.log.info(f"處理 {len(symbols_without_positions)} 個無倉位的標的...")
            for symbol in symbols_without_positions:
                await self._handle_flat_symbol(symbol)

        # [核心修正] 在所有策略都啟動後，再統一啟動全局風控
        self.log.info("所有策略已初始化，正在啟動全局交易風控...")
        self.runtime.live.activate_trading_session(
            online_positions=online_positions,
            managed_symbols=self.managed_symbols
        )

        # [核心修正] 啟動背景監控任務
        self.log.info("正在啟動背景監控迴圈 (孤兒倉位/熱更新)...")
        asyncio.create_task(self._master_loop())

        self.log.info("✅ 啟動階段完成，進入持續監控模式...")

    async def run_forever(self):
        """
        永久運行的阻塞方法，用於保持程式存活。
        """
        await self.runtime.live.run_forever()