# 檔案: backtest.py
# 目的：在歷史數據上回測「趨勢模型 + 進場模型」的組合策略

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
from common_utils import fetch_data, create_features_trend, create_features_entry, create_sequences

warnings.simplefilter(action='ignore', category=FutureWarning)
np.random.seed(42)
tf.random.set_seed(42)

def load_models_and_configs(symbol, trend_version, entry_version):
    """ 載入所有需要的模型和配置檔案 """
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
    entry_model_path = config.get_entry_model_path(symbol, entry_version)
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
    """
    print("\n--- 正在準備回測數據 (預先計算所有訊號) ---")
    
    # --- 1. 載入數據 ---
    df_1h = fetch_data(symbol, config.TREND_MODEL_TIMEFRAME, config.TREND_MODEL_TRAIN_LIMIT)
    df_5m = fetch_data(symbol, config.ENTRY_MODEL_TIMEFRAME, config.TREND_MODEL_TRAIN_LIMIT * 12)
    
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
    print("正在計算 5m 進場模型訊號...")
    entry_model = models_data['entry_model']
    
    df_5m_features, features_list_5m = create_features_entry(df_5m.copy())
    
    X_5m = xgb.DMatrix(df_5m_features[features_list_5m])
    df_5m_features['entry_prediction'] = entry_model.predict(X_5m) # <-- 預測的是報酬率
    
    # --- 4. 合併 MTF 數據 ---
    print("正在合併 1h 和 5m 數據...")
    
    df_1h_signal_resampled = df_1h_features[['trend_signal']].reindex(df_5m_features.index, method='ffill')
    df_backtest = df_5m_features.join(df_1h_signal_resampled)
    df_backtest = df_backtest.dropna()
    
    print(f"--- 數據準備完畢，總共 {len(df_backtest)} 根 5m K 棒可供回測 ---")
    return df_backtest

def run_strategy_backtest(df_backtest, symbol):
    """
    執行「事件驅動」回測。
    """
    if df_backtest is None or df_backtest.empty:
        return

    print("\n--- 步驟 3: 執行策略回測 ---")

    initial_balance = 10000.0
    cash = initial_balance
    position_size = 0.0
    entry_price = 0.0
    in_position = False

    trades = []
    equity_curve = []

    ENTRY_THRESHOLD = 0.0001
    STOP_LOSS_PCT = 0.015
    TAKE_PROFIT_PCT = 0.03
    COMMISSION_FEE = 0.0004

    for i in range(1, len(df_backtest)):
        row = df_backtest.iloc[i]
        current_price = row['Close']
        trend_signal = row.get('trend_signal', None)
        predicted_return = row.get('entry_prediction', None)

        current_net_worth = cash + (position_size * current_price)
        equity_curve.append(current_net_worth)

        if in_position:
            pnl_pct = (current_price - entry_price) / entry_price if position_size > 0 else (entry_price - current_price) / entry_price
            if pnl_pct <= -STOP_LOSS_PCT or pnl_pct >= TAKE_PROFIT_PCT:
                exit_price = current_price
                cash += position_size * exit_price * (1 - COMMISSION_FEE)
                trade_pnl = position_size * (exit_price - entry_price)
                trades.append(trade_pnl)
                in_position = False
                position_size = 0.0
                entry_price = 0.0

        if not in_position:
            if predicted_return is None or trend_signal is None:
                continue

            if trend_signal == 1 and predicted_return > ENTRY_THRESHOLD:
                size = (cash * 0.5) / current_price
                position_size = size * (1 - COMMISSION_FEE)
                cash -= size * current_price
                entry_price = current_price
                in_position = True
            elif trend_signal == 0 and predicted_return < -ENTRY_THRESHOLD:
                size = (cash * 0.5) / current_price
                position_size = -size * (1 - COMMISSION_FEE)
                cash -= size * current_price
                entry_price = current_price
                in_position = True

    if in_position:
        final_price = df_backtest['Close'].iloc[-1]
        cash += position_size * final_price * (1 - COMMISSION_FEE)
        trade_pnl = position_size * (final_price - entry_price)
        trades.append(trade_pnl)

    final_net = cash
    equity_curve.append(final_net)

    if not trades:
        print("回測期間沒有發生任何交易。")
        return

    total_trades = len(trades)
    wins = [t for t in trades if t > 0]
    win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
    total_pnl = final_net - initial_balance

    print(f"\n--- 策略回測績效報告 (Symbol: {symbol}) ---")
    print(f"初始資金: ${initial_balance:.2f}")
    print(f"最終淨值: ${final_net:.2f}")
    print(f"總盈虧 (PnL): ${total_pnl:.2f}")
    print(f"總報酬率: {(total_pnl / initial_balance) * 100:.2f}%")
    print(f"總交易次數: {total_trades}")
    print(f"勝率 (Win Rate): {win_rate:.2f}%")

    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve)
    plt.title(f'策略權益曲線 (Equity Curve) - {symbol}')
    plt.xlabel('5m K 棒 (時間步)')
    plt.ylabel('淨值 (USD)')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='執行雙模型策略回測')
    parser.add_argument('-s', '--symbol', type=str, required=True, help='要回測的交易對')
    args = parser.parse_args()
    
    models_data = load_models_and_configs(
        args.symbol, 
        config.TREND_MODEL_VERSION, 
        config.ENTRY_MODEL_VERSION
    )
    
    if models_data:
        backtest_df = prepare_backtest_data(args.symbol, models_data)
        if backtest_df is not None:
            run_strategy_backtest(backtest_df, args.symbol)
