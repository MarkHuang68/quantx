# 檔案: backtest.py
# 目的：在歷史數據上回測「趨勢模型 + 進場模型」的組合策略
# 修改說明：
# - 使用 common_utils.py 的 fetch_data（含快取邏輯：當 --start 及 --end 設值時，先查 data/ CSV，若無則抓取並存）。
# - 新增 argparse --stop_loss (預設 0.01)、--take_profit (預設 0.02)、--entry_threshold (預設 0.0001)，run_strategy_backtest 使用之。
# - 移除 RSI 濾波邏輯（依用戶偏好）。
# - Kelly 計算穩定版：p = 0.55 + abs(predicted_return) * 0.5；kelly_fraction 限 0.05-0.3，減少波動。
# - 回測結束寫 pnl.json 含 total_pnl、total_return、total_trades、win_rate、sr_annual、mdd（供 hyperparameter_search.py 使用）。
# - 回測邏輯：逐根 5m K 棒模擬持倉；趨勢 + 預測進場 (動態 Kelly 倉位)；固定止盈止損觸發平倉；計算 Buy&Hold 曲線、績效指標；結束強制平倉。
# - 注意：需確保 config.py 及 common_utils.py 存在；模型路徑等依 config 設定。

import pandas as pd
import numpy as np
import argparse
import tensorflow as tf
import xgboost as xgb
import warnings
import os
import json
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

# --- 1. 引用「設定檔」和「共用工具箱」 ---
import config
from common_utils import fetch_data, create_features_trend, create_features_trend, create_sequences

warnings.simplefilter(action='ignore', category=FutureWarning)
np.random.seed(42)
tf.random.set_seed(42)

def load_models_and_configs(symbol, trend_version, entry_version):
    """ 載入所有需要的模型和配置檔案
    - 趨勢模型：LSTM，從 trend_model_path 載入。
    - 進場模型：XGBoost Booster，從 entry_model_path 載入。
    - 若檔案不存在，輸出錯誤並返回 None。
    """
    print(f"--- 正在為 {symbol} 載入模型 (Trend: {trend_version}, Entry: {entry_version}) ---")
    
    models_data = {}

    # --- 載入趨勢模型 (LSTM) ---
    trend_model_path = config.get_trend_model_path(symbol, trend_version)
    if not os.path.exists(trend_model_path):
        print(f"🛑 錯誤：找不到趨勢模型 {trend_model_path}。")
        return None
        
    models_data['trend_model'] = load_model(trend_model_path)
    print("✅ 趨勢模型載入成功。")

    # --- 載入進場模型 (XGBoost) ---
    entry_model_path = config.get_trend_model_path(symbol, 1, entry_version)
    if not os.path.exists(entry_model_path):
        print(f"🛑 錯誤：找不到進場模型 {entry_model_path}。")
        return None
        
    entry_model = xgb.Booster()
    entry_model.load_model(entry_model_path)
    models_data['entry_model'] = entry_model
    print("✅ 進場模型載入成功。")
    
    return models_data

def prepare_backtest_data(symbol, models_data):
    """
    準備回測所需的多時間框架 (MTF) 數據。
    - 載入 1h (趨勢) 及 5m (進場) 數據（使用 fetch_data，支持快取）。
    - 計算趨勢訊號 (LSTM 預測趨勢方向 1:漲/0:跌)。
    - 計算進場預測 (XGBoost 預測報酬率)。
    - 合併 5m 數據為 df_backtest，ffill 趨勢訊號。
    - dropna 確保無缺失值。
    """
    print("\n--- 正在準備回測數據 (預先計算所有訊號) ---")
    
    # --- 1. 載入數據 ---
    df_1h = fetch_data(symbol, config.TREND_MODEL_TIMEFRAME, args.start, args.end, config.TREND_MODEL_BACKTEST_LIMIT)
    df_5m = fetch_data(symbol, config.TREND_MODEL_TIMEFRAME, args.start, args.end, config.TREND_MODEL_BACKTEST_LIMIT * 12)
    
    if df_1h is None or df_5m is None:
        print("🛑 數據獲取失敗。")
        return None
        
    # --- 2. 預計算「趨勢模型 (LSTM)」訊號 (在 1h 數據上) ---
    print("正在計算 1h 趨勢模型訊號...")
    trend_model = models_data['trend_model']
    
    df_1h_features, features_list_1h = create_features_trend(df_1h.copy())
    
    scaler_1h = MinMaxScaler(feature_range=(0, 1))
    scaled_features_1h = scaler_1h.fit_transform(df_1h_features[features_list_1h])
    
    lookback = config.TREND_MODEL_PARAMS['LOOKBACK_WINDOW'] 
    X_1h, _ = create_sequences(scaled_features_1h, np.zeros(len(scaled_features_1h)), lookback_window=lookback)
    
    trend_predictions_proba = trend_model.predict(X_1h, verbose=0)
    trend_predictions = (trend_predictions_proba > 0.5).astype(int).flatten()
    
    df_1h_features = df_1h_features.iloc[lookback:].copy()
    df_1h_features['trend_signal'] = trend_predictions
    
    # --- 3. 預計算「進場模型 (XGB)」訊號 (在 5m 數據上) ---
    print(f"正在計算 {config.TREND_MODEL_TIMEFRAME} 進場模型訊號...")
    entry_model = models_data['entry_model']
    
    df_5m_features, features_list_5m = create_features_trend(df_5m.copy())
    
    X_5m = xgb.DMatrix(df_5m_features[features_list_5m])
    df_5m_features['entry_prediction'] = entry_model.predict(X_5m)  # 預測報酬率
    
    # --- 4. 合併 MTF 數據 ---
    print(f"正在合併 {config.TREND_MODEL_TIMEFRAME} 和 {config.TREND_MODEL_TIMEFRAME} 數據...")
    
    df_1h_signal_resampled = df_1h_features[['trend_signal']].reindex(df_5m_features.index, method='ffill')
    df_backtest = df_5m_features.join(df_1h_signal_resampled)
    df_backtest = df_backtest.dropna()
    
    print(f"--- 數據準備完畢，總共 {len(df_backtest)} 根 5m K 棒可供回測 ---")
    return df_backtest

def run_strategy_backtest(df_backtest, symbol, stop_loss_pct, take_profit_pct, entry_threshold):
    """
    執行倉位持有回測 (逐根 5m K 棒模擬)。
    - 參數：從 args 傳入 stop_loss_pct, take_profit_pct, entry_threshold。
    - 進場邏輯：趨勢 + 預測報酬 >/< 門檻；使用 Kelly 計算動態倉位 (穩定版)。
    - 持倉邏輯：監控固定止損/止盈觸發平倉扣費。
    - 計算 Buy & Hold 曲線 (初始買入持有)。
    - 計算績效：總 PnL/報酬率/交易數/勝率/年化 Sharpe/MDD。
    - 結束強制平倉；輸出報告及圖；寫 pnl.json。
    """
    if df_backtest is None or df_backtest.empty:
        return

    print("\n--- 執行策略回測 ---")

    initial_balance = 10000.0  # 初始資金
    cash = initial_balance     # 現金餘額
    position_size = 0.0        # 倉位大小 (正:多倉, 負:空倉)
    entry_price = 0.0          # 進場價格
    in_position = False        # 是否持倉旗標
    
    trades = []                # 交易 PnL 列表
    equity_curve = [initial_balance]  # 策略淨值曲線
    
    COMMISSION_FEE = 0.00055   # 手續費率
    
    # Buy & Hold 基準曲線
    bh_position = initial_balance / df_backtest['Close'].iloc[0]  # 初始購買數量
    bh_curve = [initial_balance]  # Buy & Hold 淨值曲線
    
    for i in range(1, len(df_backtest)):
        row = df_backtest.iloc[i]
        current_price = row['Close']  # 當前收盤價
        trend_signal = row.get('trend_signal', None)  # 趨勢訊號 (1:漲, 0:跌)
        predicted_return = row.get('entry_prediction', None)  # 預測報酬
        
        # 更新當前策略淨值 (現金 + 倉位價值)
        current_net_worth = cash + (position_size * current_price)
        equity_curve.append(current_net_worth)
        
        # 若持倉，檢查止盈止損 (pnl_pct <= -stop_loss_pct 或 >= take_profit_pct 即平倉扣費)
        if in_position:
            pnl_pct = (current_price - entry_price) / entry_price if position_size > 0 else (entry_price - current_price) / entry_price
            if pnl_pct <= -stop_loss_pct or pnl_pct >= take_profit_pct:
                exit_price = current_price
                cash += position_size * exit_price
                cash -= abs(position_size * exit_price) * COMMISSION_FEE  # 扣平倉手續費
                trade_pnl = position_size * (exit_price - entry_price)
                trades.append(trade_pnl)
                in_position = False
                position_size = 0.0
                entry_price = 0.0
                continue
        
        # 若未持倉，檢查進場條件 (趨勢 + 預測報酬門檻)
        if not in_position:
            if predicted_return is None or trend_signal is None:
                continue
            
            if (predicted_return > entry_threshold) or \
               (predicted_return < -entry_threshold):
                
                # Kelly 計算倉位比例 (穩定版：p 保守估計；限制 5%-30%)
                p = 0.55 + abs(predicted_return) * 0.5  # 穩定勝率估計
                q = 1 - p
                b = abs(predicted_return) / stop_loss_pct  # 風險報酬比
                kelly_fraction = (p - q) / b if b != 0 else 0.05  # 避免除零，最小 5%
                kelly_fraction = max(min(kelly_fraction, 0.3), 0.05)  # 限制 5%-30%
                
                size = (cash * kelly_fraction) / current_price  # 動態計算數量
                position_size = size if trend_signal == 1 else -size
                if trend_signal == 1:
                    cash -= size * current_price
                else:
                    cash += size * current_price
                cash -= abs(size * current_price) * COMMISSION_FEE  # 扣進場手續費
                entry_price = current_price
                in_position = True
        
        # 更新 Buy & Hold 淨值
        bh_net_worth = bh_position * current_price
        bh_curve.append(bh_net_worth)
    
    # 回測結束，若持倉則強制平倉
    if in_position:
        final_price = df_backtest['Close'].iloc[-1]  # 最後收盤價
        cash += position_size * final_price
        cash -= abs(position_size * final_price) * COMMISSION_FEE  # 扣平倉手續費
        trade_pnl = position_size * (final_price - entry_price)
        trades.append(trade_pnl)
    
    # 更新最終策略淨值
    final_net = cash
    equity_curve.append(final_net)
    
    # 更新最終 Buy & Hold 淨值
    bh_curve.append(bh_position * df_backtest['Close'].iloc[-1])
    
    if not trades:
        print("回測期間沒有發生任何交易。")
        return
    
    # 計算績效指標
    total_trades = len(trades)  # 總交易次數
    wins = [t for t in trades if t > 0]  # 盈利交易
    win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0  # 勝率
    total_pnl = final_net - initial_balance  # 總 PnL
    total_return = (total_pnl / initial_balance) * 100  # 總報酬率
    
    # 年化 Sharpe Ratio (假設 5m 框架)
    equity_returns = pd.Series(equity_curve).pct_change().dropna()  # 日報酬率
    sr = equity_returns.mean() / equity_returns.std() if equity_returns.std() != 0 else 0  # Sharpe Ratio
    sr_annual = sr * np.sqrt(365 * 24 * 12 / len(equity_curve))  # 年化 (365天 * 24小時 * 12根/小時)
    
    # Max Drawdown
    peak = np.maximum.accumulate(equity_curve)  # 累計峰值
    dd = (np.array(equity_curve) - peak) / peak  # 回檔率
    mdd = dd.min() * 100 if len(dd) > 0 else 0   # 最大回檔 (%)
    
    # 輸出報告
    print(f"\n--- 策略回測績效報告 (Symbol: {symbol}) ---")
    print(f"初始資金: ${initial_balance:.2f}")
    print(f"最終淨值: ${final_net:.2f}")
    print(f"總盈虧 (PnL): ${total_pnl:.2f}")
    print(f"總報酬率: {total_return:.2f}%")
    print(f"總交易次數: {total_trades}")
    print(f"勝率 (Win Rate): {win_rate:.2f}%")
    print(f"Sharpe Ratio: {sr_annual:.2f}")
    print(f"Max Drawdown: {mdd:.2f}%")
    
    # 寫 pnl.json (供尋參腳本使用)
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
    print("✅ 已寫 pnl.json 檔案。")

    if not args.no_plot:
    # 繪製權益曲線
        plt.rc('font', family='MingLiu')
        plt.figure(figsize=(12, 6))
        plt.plot(equity_curve, label='策略', color='red')  # 策略曲線 (紅色)
        plt.plot(bh_curve, label='Buy & Hold', color='gray', linestyle='--')  # Buy & Hold (灰色虛線)
        plt.title(f'策略權益曲線 (Equity Curve) - {symbol}')  # 標題
        plt.xlabel(f'{config.TREND_MODEL_TIMEFRAME} K 棒 (時間步)')  # X 軸標籤
        plt.ylabel('淨值 (USD)')  # Y 軸標籤
        plt.grid(True)  # 顯示格線
        plt.legend()    # 顯示圖例
        print("正在顯示權益曲線圖...")
        plt.show()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='執行雙模型策略回測')
    parser.add_argument('-s', '--symbol', type=str, required=True, help='要回測的交易對 (e.g., ETH/USDT)')
    parser.add_argument('-sd', '--start', type=str, help='回測起始日期 (YYYY-MM-DD)')
    parser.add_argument('-ed', '--end', type=str, help='回測結束日期 (YYYY-MM-DD)')
    parser.add_argument('-sl', '--stop_loss', type=float, default=0.015, help='止損百分比 (預設 0.01)')
    parser.add_argument('-tp', '--take_profit', type=float, default=0.05, help='止盈百分比 (預設 0.02)')
    parser.add_argument('-et', '--entry_threshold', type=float, default=0.0005, help='進場門檻 (預設 0.0001)')
    parser.add_argument('--no_plot', action='store_true', help='不顯示權益曲線圖 (用於尋參)')
    args = parser.parse_args()
    
    models_data = load_models_and_configs(
        args.symbol, 
        config.TREND_MODEL_VERSION, 
        config.TREND_MODEL_VERSION
    )
    
    if models_data:
        backtest_df = prepare_backtest_data(args.symbol, models_data)
        if backtest_df is not None:
            run_strategy_backtest(backtest_df, args.symbol, args.stop_loss, args.take_profit, args.entry_threshold)