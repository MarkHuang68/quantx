# 檔案: backtest.py
# 目的：在歷史數據上回測「多時間框架 XGBoost 趨勢模型」的組合策略（1m/5m/15m）
# 修改說明：
# - 全部模型換成 train_trend_model.py 中的 XGBoost 分類模型（漲跌預測）。
# - 使用 3 個時間框架：1m (進場)、5m (中間)、15m (趨勢)。
# - 載入模型：分別從 config 獲取 1m/5m/15m 模型路徑。
# - 數據準備：載入 3 個 TF 數據，計算各自訊號，合併到 1m 為主（ffill 上層訊號）。
# - 進場邏輯：所有 3 個訊號一致（全 1 做多、全 0 做空）；預測機率 > 門檻。
# - 持倉邏輯：固定止盈/止損；Kelly 計算倉位（基於 1m 信心）。
# - 其他：移除 LSTM 相關；調整參數；計算績效及曲線。

import pandas as pd
import numpy as np
import argparse
import xgboost as xgb
import warnings
import os
import json
import matplotlib.pyplot as plt

# --- 1. 引用「設定檔」和「共用工具箱」 ---
import config
from common_utils import fetch_data, create_features_trend

warnings.simplefilter(action='ignore', category=FutureWarning)
np.random.seed(42)

def load_models(symbol, version):
    """ 載入 3 個 XGBoost 模型 (1m/5m/15m)
    - 從 config 獲取路徑。
    - 若檔案不存在，返回 None。
    """
    print(f"--- 正在為 {symbol} 載入 XGBoost 模型 (Version: {version}) ---")
    
    models = {}
    timeframes = ['1m', '5m', '15m']
    
    for tf in timeframes:
        model_path = config.get_trend_model_path(symbol, tf, version)
        if not os.path.exists(model_path):
            print(f"🛑 錯誤：找不到 {tf} 模型 {model_path}。")
            return None
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        models[tf] = model
        print(f"✅ {tf} 模型載入成功。")
    
    return models

def prepare_backtest_data(symbol, models):
    """
    準備回測數據 (計算 3 個 TF 訊號)。
    - 載入 1m/5m/15m 數據。
    - 計算各 TF 漲跌訊號 (1:漲, 0:跌)。
    - 合併到 1m 為主 (ffill 5m/15m 訊號)。
    - dropna 確保完整。
    """
    print("\n--- 正在準備回測數據 ---")
    
    df_15m = fetch_data(symbol, '15m', args.start, args.end)
    df_5m = fetch_data(symbol, '5m', args.start, args.end)
    df_1m = fetch_data(symbol, '1m', args.start, args.end)
    
    if any(df is None for df in [df_15m, df_5m, df_1m]):
        print("🛑 數據獲取失敗。")
        return None
    
    # 計算訊號
    dfs = {'15m': df_15m, '5m': df_5m, '1m': df_1m}
    signals = {}
    
    for tf, df in dfs.items():
        print(f"計算 {tf} 訊號...")
        features, features_list = create_features_trend(df.copy())
        X = features[features_list]
        model = models[tf]
        proba = model.predict_proba(X)[:, 1]  # 漲機率
        signals[tf] = pd.DataFrame({'signal': (proba > 0.5).astype(int), 'proba': proba}, index=features.index).add_suffix(f'_{tf}')
    
    # 合併到 1m
    df_backtest = df_1m.copy()
    df_backtest = df_backtest.join(signals['1m'], rsuffix='_1m')
    df_backtest = df_backtest.join(signals['5m'].reindex(df_backtest.index, method='ffill'), rsuffix='_5m')
    df_backtest = df_backtest.join(signals['15m'].reindex(df_backtest.index, method='ffill'), rsuffix='_15m')
    df_backtest = df_backtest.dropna()
    
    print(f"--- 數據準備完畢，總 {len(df_backtest)} 根 1m K 棒 ---")
    return df_backtest

def run_strategy_backtest(df_backtest, symbol, stop_loss_pct, take_profit_pct, entry_threshold):
    """
    執行回測 (逐根 1m K 棒)。
    - 進場：3 TF 訊號一致，且 1m 機率 > 門檻 (1:多, 0:空)。
    - 倉位：Kelly (基於 1m 信心，限 0.05-0.3)。
    - 平倉：止盈/止損觸發扣費。
    - 計算 BH 曲線、績效；寫 pnl.json。
    """
    if df_backtest.empty:
        return

    print("\n--- 執行策略回測 ---")

    initial_balance = 10000.0
    cash = initial_balance
    position_size = 0.0
    entry_price = 0.0
    in_position = False
    
    trades = []
    equity_curve = [initial_balance]
    
    COMMISSION_FEE = 0.00055
    
    bh_position = initial_balance / df_backtest['Close'].iloc[0]
    bh_curve = [initial_balance]
    
    for i in range(1, len(df_backtest)):
        row = df_backtest.iloc[i]
        current_price = row['Close']
        sig_1m = row['signal_1m']
        sig_5m = row['signal_5m']
        sig_15m = row['signal_15m']
        proba_1m = row['proba_15m']
        
        current_net = cash + (position_size * current_price)
        equity_curve.append(current_net)
        
        if in_position:
            pnl_pct = (current_price - entry_price) / entry_price if position_size > 0 else (entry_price - current_price) / entry_price
            if pnl_pct <= -stop_loss_pct or pnl_pct >= take_profit_pct:
                cash += position_size * current_price
                cash -= abs(position_size * current_price) * COMMISSION_FEE
                trade_pnl = position_size * (current_price - entry_price)
                trades.append(trade_pnl)
                in_position = False
                position_size = 0.0
                continue
        
        if not in_position:
            if sig_1m == sig_5m == sig_15m and abs(proba_1m - 0.5) > entry_threshold:
                direction = 1 if sig_1m == 1 else -1
                p = max(min(proba_1m if direction == 1 else 1 - proba_1m, 0.9), 0.55)
                q = 1 - p
                b = take_profit_pct / stop_loss_pct
                kelly = (p - q) / b if b != 0 else 0.05
                kelly = max(min(kelly, 0.3), 0.05)
                
                size = (cash * kelly) / current_price * direction
                position_size = size
                cash -= abs(size) * current_price * COMMISSION_FEE
                if direction == 1:
                    cash -= size * current_price
                else:
                    cash += abs(size) * current_price
                entry_price = current_price
                in_position = True
        
        bh_net = bh_position * current_price
        bh_curve.append(bh_net)
    
    if in_position:
        final_price = df_backtest['Close'].iloc[-1]
        cash += position_size * final_price
        cash -= abs(position_size * final_price) * COMMISSION_FEE
        trade_pnl = position_size * (final_price - entry_price)
        trades.append(trade_pnl)
    
    final_net = cash
    equity_curve.append(final_net)
    bh_curve.append(bh_position * df_backtest['Close'].iloc[-1])
    
    if not trades:
        print("無交易。")
        return
    
    total_trades = len(trades)
    wins = [t for t in trades if t > 0]
    win_rate = (len(wins) / total_trades) * 100
    total_pnl = final_net - initial_balance
    total_return = (total_pnl / initial_balance) * 100
    
    equity_returns = pd.Series(equity_curve).pct_change().dropna()
    sr = equity_returns.mean() / equity_returns.std() if equity_returns.std() != 0 else 0
    sr_annual = sr * np.sqrt(365 * 24 * 60 / len(equity_curve))  # 1m 年化
    
    peak = np.maximum.accumulate(equity_curve)
    dd = (np.array(equity_curve) - peak) / peak
    mdd = dd.min() * 100
    
    print(f"\n--- 回測報告 ({symbol}) ---")
    print(f"初始: ${initial_balance:.2f}")
    print(f"最終: ${final_net:.2f}")
    print(f"PnL: ${total_pnl:.2f}")
    print(f"報酬: {total_return:.2f}%")
    print(f"交易數: {total_trades}")
    print(f"勝率: {win_rate:.2f}%")
    print(f"Sharpe: {sr_annual:.2f}")
    print(f"MDD: {mdd:.2f}%")
    
    result = {
        "total_pnl": total_pnl,
        "total_return": total_return,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "sr_annual": sr_annual,
        "mdd": mdd
    }
    with open('pnl.json', 'w') as f:
        json.dump(result, f)
    print("✅ pnl.json 已寫。")

    if not args.no_plot:
        plt.rc('font', family='MingLiu')
        plt.figure(figsize=(12, 6))
        plt.plot(equity_curve, label='策略', color='red')
        plt.plot(bh_curve, label='Buy & Hold', color='gray', linestyle='--')
        plt.title(f'權益曲線 - {symbol}')
        plt.xlabel('1m K 棒')
        plt.ylabel('淨值 (USD)')
        plt.grid(True)
        plt.legend()
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='執行 MTF XGBoost 回測')
    parser.add_argument('-s', '--symbol', type=str, required=True, help='交易對 (e.g., ETH/USDT)')
    parser.add_argument('-sd', '--start', type=str, help='起始 (YYYY-MM-DD)')
    parser.add_argument('-ed', '--end', type=str, help='結束 (YYYY-MM-DD)')
    parser.add_argument('-sl', '--stop_loss', type=float, default=0.015, help='止損% (0.015)')
    parser.add_argument('-tp', '--take_profit', type=float, default=0.06, help='止盈% (0.05)')
    parser.add_argument('-et', '--entry_threshold', type=float, default=0.1, help='機率門檻 (0.1)')
    parser.add_argument('--no_plot', action='store_true', help='不顯示圖')
    args = parser.parse_args()
    
    models = load_models(args.symbol, config.TREND_MODEL_VERSION)
    
    if models:
        backtest_df = prepare_backtest_data(args.symbol, models)
        if backtest_df is not None:
            run_strategy_backtest(backtest_df, args.symbol, args.stop_loss, args.take_profit, args.entry_threshold)