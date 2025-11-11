# 檔案: quantx/cli/bt_run.py
# -*- coding: utf-8 -*-
# 版本: v9 (DI 重構與 Equity Curve 修正)
# 說明:
# - 使用 AppContainer 來組裝和獲取 runtime。
# - 修正了在生成報告時，因 equity_curve 切片錯誤導致的 "Length of values does not match" ValueError。

from __future__ import annotations
import argparse
import json
import sys
import pandas as pd
from datetime import datetime

from quantx.core.config import Config
from quantx.containers import AppContainer
from quantx.backtest.engine import BacktestEngine
from quantx.core.report.reporter import ReportGenerator

def parse_args(argv):
    parser = argparse.ArgumentParser(description="執行單一或多標的策略回測")
    parser.add_argument("--strategy", type=str, required=True, help="策略模組名稱")
    # 修改: --symbol 改為 --symbols，並允許接收多個值
    parser.add_argument("--symbols", type=str, nargs='+', required=True, help="一個或多個交易對 (以空格分隔)")
    parser.add_argument("--tf", type=str, required=True, help="時間週期")
    parser.add_argument("--start", type=str, required=True, help="回測起始日期 (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="回測結束日期 (YYYY-MM-DD)")
    parser.add_argument("--params", type=str, default="{}", help="策略參數 (JSON 字串)")
    parser.add_argument("--equity", type=float, default=10000.0, help="初始本金")
    return parser.parse_args(argv)

def main(argv=None):
    """
    回測主程式 (支援多標的)。
    """
    args = parse_args(argv or sys.argv[1:])

    cfg = Config()
    container = AppContainer(cfg, mode_override="live")
    runtime = container.runtime
    logger = runtime.log

    start_dt = datetime.fromisoformat(args.start)
    end_dt = datetime.fromisoformat(args.end)

    # 修改: 為所有 symbols 載入數據
    datas = {}
    for symbol in args.symbols:
        logger.info(f"正在為 {symbol} 載入數據...")
        df = runtime.loader.load_ohlcv(symbol, args.tf, start_dt, end_dt)
        if df.empty:
            logger.warning(f"在指定時間範圍內找不到 {symbol}-{args.tf} 的數據，將在回測中忽略此標的。")
        datas[symbol] = df

    if not any(not df.empty for df in datas.values()):
        logger.error("所有請求的標的都沒有數據，無法執行回測。")
        return

    strategy_cls = runtime.load_strategy(args.strategy)
    params = json.loads(args.params)

    cost_model = runtime.get_cost_model()
    taker_bps, maker_bps, slip_bps = cost_model.get('taker_fee_bps', 5.5), cost_model.get('maker_fee_bps', 2.0), cost_model.get('slip_bps', 1.0)
    logger.info(f"使用成本模型: Taker={taker_bps}bps, Maker={maker_bps}bps, Slippage={slip_bps}bps")

    risk_cfg = {"size_mode": "pct_equity", "risk_pct": 0.01}

    # 修改: 傳遞 datas 字典給引擎
    engine = BacktestEngine(
        datas=datas,
        strategy_cls=strategy_cls,
        runtime=runtime,
        tf=args.tf,
        params=params,
        equity=args.equity,
        maker_fee_bps=maker_bps,
        taker_fee_bps=taker_bps,
        slippage_bps=slip_bps,
        risk_config=risk_cfg
    )

    metrics = engine.run()
    trades = engine.trades

    # 修改: 報表生成現在需要一個主 symbol 來對齊時間
    # 我們選擇第一個有數據的 symbol 作為主報表標的
    main_report_symbol = next((s for s, df in datas.items() if not df.empty), None)
    if not main_report_symbol:
        logger.error("沒有可用於生成報表的主數據。")
        return

    main_data_tf = datas[main_report_symbol]

    if len(engine.equity_curve) > 1:
        # The equity curve has N+1 or N+2 points, where N is the length of the timeline.
        # We take the curve corresponding to each bar.
        equity_points = engine.equity_curve[1:len(engine.master_timeline)+1]

        # Ensure we don't have a mismatch if the strategy stopped early
        timeline_points = engine.master_timeline[:len(equity_points)]

        eq_series = pd.Series(equity_points, index=timeline_points)
        # Reindex to the main symbol's timeline for consistent reporting
        eq_series = eq_series.reindex(main_data_tf.index, method='ffill')
    else:
        eq_series = pd.Series(dtype=float, index=main_data_tf.index)

    logger.info(f"回測完成，最終資產：{metrics['final_equity']:.2f}")

    reporter = ReportGenerator(
        runtime=runtime,
        symbol=main_report_symbol,
        tf=args.tf,
        strategy_name=args.strategy,
        mode="backtest"
    )
    
    reporter.generate(
        ohlcv=main_data_tf,
        equity_curve=eq_series,
        trades=trades,
        strategy_params=params
    )

if __name__ == "__main__":
    main()