# quantx/backtest/engine.py
# 版本: v7 (DI 重構)
# 說明:
# - 移除了對 get_runtime 的依賴。
# - __init__ 建構函數新增 runtime 參數，由外部注入。
# - SimulatedTradeManager 的初始化現在會接收這個傳入的 runtime。

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..core.risk import RiskConfig, compute_order_size
from ..core.timeframe import bars_per_year
from ..core.context import BacktestContext
from ..core.executor.base import Position
from ..core.runtime import Runtime # 引入 Runtime 類型
from ..core.trade_manager import SimulatedTradeManager

class SeriesView:
    def __init__(self, df: pd.DataFrame, current_idx: int):
        self._df = df
        self._idx = int(current_idx)

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @property
    def idx(self) -> int:
        return self._idx


class BacktestEngine:
    def __init__(
        self,
        datas: Dict[str, pd.DataFrame], # 修改: 接收一個包含多個 DataFrame 的字典
        strategy_cls: type,
        *,
        runtime: Runtime,
        tf: str = "1m", # 修改: symbol 參數被移除，但保留 tf
        params: Optional[Dict[str, Any]] = None,
        equity: float = 10_000.0,
        maker_fee_bps: float = 5.5,
        taker_fee_bps: float = 2.0,
        slippage_bps: float = 1.0,
        risk_config: Optional[RiskConfig | Dict[str, Any]] = None,
        max_mdd_threshold: float = 1.0
    ) -> None:
        self.runtime = runtime
        # 修改: 標準化所有傳入的 DataFrame
        self.datas = {symbol: self._normalize_df(df) for symbol, df in datas.items()}
        self.tf = tf
        self.strategy = strategy_cls(**(params or {}))
        self.params = params or {}

        self.initial_equity = float(equity)
        self.strategy.equity = self.initial_equity
        # 修改: position 現在是一個字典，以支持多標的
        self.strategy.position = {} # 將在 get_position 中用 defaultdict 自動處理
        self.strategy.trades = []
        
        self.max_mdd_threshold = float(max_mdd_threshold)

        if isinstance(risk_config, dict):
            self.risk_config = RiskConfig(**risk_config)
        else:
            self.risk_config = risk_config or RiskConfig(size_mode="pct_equity", risk_pct=0.01)

        # 修改: Trade Manager 不再綁定單一 symbol
        self.trade_manager = SimulatedTradeManager(
            runtime,
            tf=tf, # 傳遞 tf
            risk_cfg=self.risk_config,
            executor=self.strategy
        )
        
        self.current_time: Optional[datetime] = None
        self.equity_curve: List[float] = [self.strategy.equity]
        self.intent: Dict[str, List[Dict]] = {}
        self.master_timeline: pd.DatetimeIndex = pd.DatetimeIndex([]) # 新增: 初始化主時間軸

    @staticmethod
    def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if not isinstance(out.index, pd.DatetimeIndex):
            ts_col = next((c for c in ("time", "timestamp", "ts") if c in out.columns), None)
            if ts_col is not None:
                ts = out[ts_col]
                unit = "s" if pd.to_numeric(ts, errors="coerce").max() < 10**11 else "ms"
                out.index = pd.to_datetime(ts, unit=unit, utc=True, errors="coerce")
                out = out.drop(columns=[ts_col])
        if isinstance(out.index, pd.DatetimeIndex):
            out.index = out.index.tz_localize("UTC") if out.index.tz is None else out.index.tz_convert("UTC")
            out.index.name = "time"

        for c in ("open", "high", "low", "close", "volume"):
            if c not in out.columns:
                out[c] = np.nan
            out[c] = pd.to_numeric(out[c], errors="coerce")
        out = out.sort_index()
        return out[["open", "high", "low", "close", "volume"]]

    def get_series_view(self, symbol: str) -> SeriesView:
        # 修改: 從 self.datas 字典中獲取數據
        df = self.datas.get(symbol)
        if df is None:
            # 這裡不應該發生，因為 BacktestContext 會先檢查
            raise ValueError(f"請求了未在 BacktestEngine 中載入的 symbol: {symbol}")

        # 在新的時間驅動模型中，current_idx 不再有意義，但為保持 SeriesView 結構而保留
        return SeriesView(df, 0)

    def run(self) -> Dict[str, Any]:
        # 1. 建立主時間軸
        all_indices = [df.index for df in self.datas.values() if not df.empty]
        if not all_indices:
            self.runtime.log.warning("所有提供的數據均為空，無法執行回測。")
            return {"initial_equity": self.initial_equity, "final_equity": self.initial_equity, "total_return": 0.0, "sharpe_ratio": 0.0, "max_drawdown": 0.0, "n_trades": 0}

        master_timeline = pd.concat([series.to_series() for series in all_indices]).index.unique().sort_values()

        # 修正: 確保主時間軸有 UTC 時區資訊
        if master_timeline.tz is None:
            master_timeline = master_timeline.tz_localize("UTC")

        self.master_timeline = master_timeline
        self.runtime.log.info(f"主時間軸已建立，共 {len(self.master_timeline)} 個唯一時間點。")

        # 2. 初始化狀態
        peak_equity = self.initial_equity
        early_stopped = False
        
        # 3. 遍歷主時間軸
        for ts in master_timeline:
            self.current_time = ts
            
            # 計算當前總 PnL (需要遍歷所有持倉)
            unreal = 0.0
            for symbol, pos in self.strategy.positions.items():
                # 獲取該 symbol 在當前時間點的最新價格
                if symbol in self.datas and ts in self.datas[symbol].index:
                    price_now = float(self.datas[symbol].loc[ts, "close"])
                    if pos.is_long():
                        unreal += (price_now - pos.long_entry) * pos.long_qty
                    if pos.is_short():
                        unreal += (pos.short_entry - price_now) * pos.short_qty

            current_total_equity = self.strategy.equity + unreal
            self.equity_curve.append(current_total_equity)
            
            # MDD 早停邏輯 (保持不變)
            if self.max_mdd_threshold < 1.0:
                peak_equity = max(peak_equity, current_total_equity)
                current_drawdown = (peak_equity - current_total_equity) / peak_equity
                if current_drawdown > self.max_mdd_threshold:
                    self.runtime.log.warning(f"[BacktestEngine] 觸發早停 (MDD: {current_drawdown:.2f} > {self.max_mdd_threshold:.2f}) at {ts}")
                    early_stopped = True
                    break
            
            # 4. 執行策略邏輯
            # 策略現在需要能夠處理多個標的，它內部會決定使用哪個標的作為觸發信號
            # 我們為每個 symbol 都創建一個 context
            # 注意：這裡的設計是，on_bar 會在每個時間點被呼叫一次，策略內部需要自己處理數據

            # 簡化：我們假設策略只需要一個 context，它內部可以請求多個 symbol 的數據
            # 我們需要一個 "主" symbol 來創建 context，這裡我們選擇第一個
            main_symbol = next(iter(self.datas.keys()))
            ctx = BacktestContext(self, main_symbol, self.tf)
            
            try:
                self.strategy.on_bar(ctx)
            except Exception as e:
                self.runtime.log.error(f"策略 {self.strategy.__class__.__name__} 在 {ts} on_bar 執行失敗: {e}", exc_info=False)
                pass
            
            # 5. 執行交易指令 (現在意圖是個字典)
            if self.intent:
                # 這裡需要一個價格來模擬成交，我們可以取主 symbol 的價格
                price_now = float(self.datas[main_symbol].loc[ts, "close"])
                # 遍歷所有 symbol 的意圖
                for symbol, intents in self.intent.items():
                    if intents:
                        # 如果是其他 symbol，也用主 symbol 價格模擬 (簡化處理)
                        # 一個更精確的引擎會用各個 symbol 自己的價格
                        self.trade_manager.execute_commands(intents, price_now, ts.to_pydatetime())
                self.intent = {} # 清空所有意圖

        # 6. 回測結束後的處理
        final_equity = self.strategy.equity
        if len(master_timeline) > 0 and not early_stopped:
            # 平掉所有剩餘倉位
            last_ts = master_timeline[-1]
            for symbol, pos in self.strategy.positions.items():
                if not pos.is_flat() and symbol in self.datas and last_ts in self.datas[symbol].index:
                    price_now = float(self.datas[symbol].loc[last_ts, "close"])
                    self.trade_manager.execute_commands([{"action": "close"}], price_now, last_ts.to_pydatetime())

            final_equity = self.strategy.equity
            self.equity_curve.append(final_equity)
        elif early_stopped:
             final_equity = self.strategy.equity
             
        if len(self.equity_curve) < 2:
            sharpe, mdd = 0.0, 0.0
        else:
            eq = np.array(self.equity_curve[1:])
            if early_stopped and len(eq) > idx + 1:
                 eq = eq[:idx + 1]
            if eq.size <= 1:
                 sharpe, mdd = 0.0, 0.0
            else:
                 rets = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
                 mu = float(np.nanmean(rets))
                 sd = float(np.nanstd(rets, ddof=0))
                 ann = float(bars_per_year(self.tf))
                 sharpe = (mu / sd * math.sqrt(ann)) if sd > 0 else 0.0
                 cummax = np.maximum.accumulate(eq)
                 dd = (eq - cummax) / np.maximum(cummax, 1e-12)
                 mdd = float(np.min(dd)) if dd.size else 0.0
        
        metrics = {
            "initial_equity": float(self.initial_equity),
            "final_equity": final_equity,
            "total_return": (final_equity / self.initial_equity - 1.0) if self.initial_equity > 0 else 0.0,
            "sharpe_ratio": sharpe,
            "max_drawdown": mdd,
            "n_trades": sum(1 for t in self.strategy.trades if t.get("side") in ("buy", "sell")),
        }
        
        self.trades = self.strategy.trades
        return metrics