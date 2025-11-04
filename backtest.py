# 檔案: backtest.py
# 目的：在歷史數據上回測「趨勢模型 + 價格模型」的組合策略 (修正版)

import pandas as pd
import numpy as np
import argparse
import tensorflow as tf
import xgboost as xgb
import warnings
import os
import json 
import math
import ccxt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model # <--- 修正 Keras 導入

# --- 1. 引用「設定檔」和「共用工具箱」 ---
import config
from common_utils import fetch_data, create_features_trend, create_features_price, create_sequences

warnings.simplefilter(action='ignore', category=FutureWarning)
np.random.seed(42)
tf.random.set_seed(42)

# --- (*** 策略回測參數 (可在此調整) ***) ---
STOP_LOSS_PCT = -0.015  # 1.5% 止損
TAKE_PROFIT_PCT = 0.03   # 3.0% 止盈
COMMISSION_FEE = 0.0004  # 0.04% (幣安手續費)

def load_models_and_configs(symbol, trend_version, price_version):
    """ 載入所有需要的模型和配置檔案 """
    print(f"--- 正在為 {symbol} 載入模型 (Trend: {trend_version}, Price: {price_version}) ---")
    
    models_data = {}

    # --- 載入趨勢模型 (LSTM) ---
    trend_model_path = config.get_trend_model_path(symbol, trend_version)
    trend_config_path = trend_model_path.replace('.keras', '_feature_config.json')
    
    if not os.path.exists(trend_model_path) or not os.path.exists(trend_config_path):
        print(f"🛑 錯誤：找不到趨勢模型 {trend_model_path} 或其配置檔案。")
        print(f"請先執行: python train_trend_model.py --symbol {symbol} --version {trend_version}")
        return None
        
    models_data['trend_model'] = load_model(trend_model_path)
    with open(trend_config_path, 'r') as f:
        models_data['trend_config'] = json.load(f)
    print("✅ 趨勢模型載入成功。")

    # --- 載入價格模型 (XGBoost) ---
    price_model_path = config.get_price_model_path(symbol, price_version)
    price_config_path = price_model_path.replace('.json', '_feature_config.json')
    
    if not os.path.exists(price_model_path) or not os.path.exists(price_config_path):
        print(f"🛑 錯誤：找不到價格模型 {price_model_path} 或其配置檔案。")
        print(f"請先執行: python train_price_model.py --symbol {symbol} --version {price_version}")
        return None
        
    price_model = xgb.Booster()
    price_model.load_model(price_model_path)
    models_data['price_model'] = price_model
    
    with open(price_config_path, 'r') as f:
        models_data['price_config'] = json.load(f)
    print("✅ 價格模型配置載入成功。")
    
    return models_data

def prepare_backtest_data(symbol, models_data):
    """
    準備回測所需的多時間框架 (MTF) 數據。
    """
    print("\n--- 正在準備回測數據 (預先計算所有訊號) ---")
    
    # --- 1. 載入數據 ---
    df_1h = fetch_data(symbol, config.TREND_MODEL_TIMEFRAME, config.TREND_MODEL_TRAIN_LIMIT)
    # (抓取 5m 數據，使其時間範圍大致與 1h 匹配)
    df_5m = fetch_data(symbol, config.PRICE_MODEL_TIMEFRAME, config.TREND_MODEL_TRAIN_LIMIT * 12) 
    
    if df_1h is None or df_5m is None:
        print("🛑 數據獲取失敗。")
        return None
        
    # --- 2. 預計算「趨勢模型 (LSTM)」訊號 (在 1h 數據上) ---
    print("正在計算 1h 趨勢模型訊號...")
    
    trend_config = models_data['trend_config']
    trend_model = models_data['trend_model']
    
    # 2a. 計算 1h 特徵 (*** 關鍵：使用載入的 trend_config ***)
    df_1h_features, features_list_1h = create_features_trend(df_1h.copy(), **trend_config)
    
    # 2b. 準備 LSTM 輸入 (Scaler, 序列化)
    df_1h_model = df_1h_features.copy()
    scaler_1h = MinMaxScaler(feature_range=(0, 1))
    scaled_features_1h = scaler_1h.fit_transform(df_1h_model[features_list_1h])
    
    lookback = config.TREND_MODEL_PARAMS['LOOKBACK_WINDOW'] 
    
    X_1h, _ = create_sequences(scaled_features_1h, np.zeros(len(scaled_features_1h)), lookback_window=lookback)
    
    # 2c. 預測「所有」1h 訊號
    trend_predictions_proba = trend_model.predict(X_1h, verbose=0)
    trend_predictions = (trend_predictions_proba > 0.5).astype(int).flatten()
    
    # 2d. 將訊號放回 1h DataFrame (對齊索引)
    df_1h_features = df_1h_features.iloc[lookback:].copy()
    df_1h_features['trend_signal'] = trend_predictions # 1 (漲), 0 (跌)
    
    # --- 3. 預計算「價格模型 (XGB)」訊號 (在 5m 數據上) ---
    print("正在計算 5m 價格模型訊號...")
    
    price_config = models_data['price_config']
    price_model = models_data['price_model']
    
    # 3a. 計算 5m 特徵 (*** 關鍵：使用載入的 price_config ***)
    df_5m_features, features_list_5m = create_features_price(df_5m.copy(), **price_config)
    
    # 3b. 預測「所有」5m 訊號
    X_5m = xgb.DMatrix(df_5m_features[features_list_5m])
    df_5m_features['price_prediction'] = price_model.predict(X_5m)
    
    # --- 4. 合併 MTF 數據 ---
    print("正在合併 1h 和 5m 數據...")
    
    # 4a. 將 1h 訊號 (每小時一個) 擴展到 5m (每 5 分鐘一個)
    df_1h_signal_resampled = df_1h_features[['trend_signal']].reindex(df_5m_features.index, method='ffill')
    
    # 4b. 合併
    df_backtest = df_5m_features.join(df_1h_signal_resampled)
    
    # 4c. 清理 (移除 NaN)
    df_backtest = df_backtest.dropna()
    
    print(f"--- 數據準備完畢，總共 {len(df_backtest)} 根 5m K 棒可供回測 ---")
    return df_backtest

def run_strategy_backtest(df_backtest, symbol, models_data):
    """
    執行「事件驅動」回測 (修正版)：使用歷史 RMSE+相對閾值、比例倉位、冷卻期、最小持倉時間。
    """
    if df_backtest is None or df_backtest.empty:
        return

    print("\n--- 步驟 3: 執行策略回測 (IF/THEN 修正版) ---")

    initial_balance = 10000.0  # 初始資金
    cash = initial_balance     # 現金（未投入的）
    position_size = 0.0        # 持有張數（幣數，正=多，負=空）
    entry_price = 0.0
    in_position = False
    entry_idx = None

    trades = []
    equity_curve = []

    # 參數：可微調
    risk_per_trade_pct = 0.10   # 每次投入本金的比例 (10%)
    STOP_LOSS_PCT = 0.03        # 3% 止損
    TAKE_PROFIT_PCT = 0.06      # 6% 止盈
    COMMISSION_FEE = 0.00055
    cooldown_bars = 0           # 平倉後冷卻多少根 5m K 棒才允許再進場
    min_hold_bars = 1           # 最少持倉時間（避免立即反向平倉）
    rmse_multiplier = 2.0       # 歷史 rmse 乘數
    threshold_pct = 0.003       # 或用相對價格的百分比 (0.3%)

    # 計算歷史 RMSE：用 model 預測（已存在的 price_prediction）比對下一根實際價
    # （注意最後一根沒有下一根，會產生 NaN，忽略）
    if 'price_prediction' in df_backtest.columns:
        rmse_hist = np.sqrt(np.nanmean((df_backtest['price_prediction'] - df_backtest['Close'].shift(-1)) ** 2))
        if np.isnan(rmse_hist) or rmse_hist <= 0:
            rmse_hist = 0.0
    else:
        rmse_hist = 0.0

    print(f"歷史 RMSE (on backtest data): {rmse_hist:.4f}")

    last_trade_idx = -9999

    for i in range(1, len(df_backtest) - 1):  # 到倒數第二根，因為我們會參照 shift(-1)
        row = df_backtest.iloc[i]
        current_price = row['Close']
        trend_signal = row.get('trend_signal', None)
        predicted_price = row.get('price_prediction', None)

        # 計算淨值（當前）
        current_net_worth = cash + (position_size * current_price)
        equity_curve.append(current_net_worth)

        # 如果持倉，檢查止盈止損與最小持倉時間
        if in_position:
            pnl_pct = (current_price - entry_price) / entry_price if position_size > 0 else (entry_price - current_price) / entry_price
            # 檢查最小持倉時間
            held_bars = i - entry_idx if entry_idx is not None else 9999

            if held_bars >= min_hold_bars:
                if pnl_pct <= -STOP_LOSS_PCT or pnl_pct >= TAKE_PROFIT_PCT:
                    # 平倉
                    exit_price = current_price
                    # 平倉時加入手續費 (假設開倉時已扣現金)
                    cash += position_size * exit_price * (1 - COMMISSION_FEE)
                    trade_pnl = position_size * (exit_price - entry_price)
                    trades.append(trade_pnl)
                    # reset
                    in_position = False
                    position_size = 0.0
                    entry_price = 0.0
                    entry_idx = None
                    last_trade_idx = i
                    # print(f"平倉 @ {exit_price:.2f}, PnL: {trade_pnl:.2f}, cash: {cash:.2f}")

        # 若不在倉，且通過冷卻期，檢查開倉條件
        if (not in_position) and (i - last_trade_idx > cooldown_bars):
            # 需要有預測值與趨勢濾網
            if predicted_price is None or trend_signal is None:
                continue

            # 閾值：混合 rmse 與相對百分比
            threshold_amount = max(rmse_hist * rmse_multiplier, current_price * threshold_pct)

            # Long 条件：趨勢多頭，且預測下一根價格足夠高於 current_price
            if trend_signal == 1:
                if predicted_price > (current_price + threshold_amount):
                    # 按風險比例下單
                    allocation = initial_balance * risk_per_trade_pct
                    # 幣數
                    size = (allocation / current_price) * (1 - COMMISSION_FEE)
                    position_size = size
                    cash -= allocation  # 扣除現金
                    entry_price = current_price
                    in_position = True
                    entry_idx = i
                    # print(f"開多 @ {current_price:.2f}, size: {size:.6f}, cash left: {cash:.2f}")

            # Short 条件：趨勢空頭，且預測下一根價格足夠低於 current_price
            elif trend_signal == 0:
                if predicted_price < (current_price - threshold_amount):
                    allocation = initial_balance * risk_per_trade_pct
                    size = (allocation / current_price) * (1 - COMMISSION_FEE)
                    # 以負數表示空單(注意：簡化，計算時仍用 size*price)
                    position_size = -size
                    cash -= allocation
                    entry_price = current_price
                    in_position = True
                    entry_idx = i
                    # print(f"開空 @ {current_price:.2f}, size: {-size:.6f}, cash left: {cash:.2f}")

    # 結束回測：若仍在倉，強制以最後價格平倉
    final_price = df_backtest['Close'].iloc[-1]
    if in_position:
        cash += position_size * final_price * (1 - COMMISSION_FEE)
        trade_pnl = position_size * (final_price - entry_price)
        trades.append(trade_pnl)
        in_position = False
        position_size = 0.0

    final_net = cash
    if equity_curve:
        # 最後一根沒 append final net，補上
        equity_curve.append(final_net)

    # 計算績效
    if not trades:
        print("回測期間沒有發生任何交易。")
        return

    total_trades = len(trades)
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t < 0]
    win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
    total_pnl = final_net - initial_balance
    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = abs(sum(losses) / len(losses)) if losses else 0
    profit_factor = sum(wins) / abs(sum(losses)) if sum(losses) != 0 else 999

    print(f"\n--- 策略回測績效報告 (Symbol: {symbol}) ---")
    print(f"回測週期: {df_backtest.index[0]} to {df_backtest.index[-1]}")
    print(f"初始資金: ${initial_balance:.2f}")
    print(f"最終淨值: ${final_net:.2f}")
    print(f"總盈虧 (PnL): ${total_pnl:.2f}")
    print(f"總報酬率: {(total_pnl / initial_balance) * 100:.2f}%")
    print(f"-----------------------------------")
    print(f"總交易次數: {total_trades}")
    print(f"勝率 (Win Rate): {win_rate:.2f}%")
    print(f"平均獲利: ${avg_win:.2f}")
    print(f"平均虧損: ${avg_loss:.2f}")
    print(f"盈虧比 (Profit Factor): {profit_factor:.2f}")

    # 繪製權益曲線
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve)
    plt.title(f'策略權益曲線 (Equity Curve) - {symbol}')
    plt.xlabel('5m K 棒 (時間步)')
    plt.ylabel('淨值 (USD)')
    plt.grid(True)
    print("正在顯示權益曲線圖...")
    plt.show()

if __name__ == "__main__":
    
    # 1. 建立「參數解析器」
    parser = argparse.ArgumentParser(description='執行「階段 1.5」：雙模型策略回測 (IF/THEN 邏輯)')
    
    parser.add_argument(
        '-s', '--symbol', 
        type=str, 
        required=True, 
        help='要回測的交易對 (例如: ETH/USDT)'
    )
    
    args = parser.parse_args()
    
    # 2. 載入模型
    models_data = load_models_and_configs(
        args.symbol, 
        config.TREND_MODEL_VERSION, 
        config.PRICE_MODEL_VERSION
    )
    
    if models_data:
        # 3. 準備數據
        backtest_df = prepare_backtest_data(args.symbol, models_data)
        
        # 4. 執行回測
        if backtest_df is not None:
            # (*** 修正：從 models_data 傳入 price_config ***)
            run_strategy_backtest(backtest_df, args.symbol, models_data)