# 檔案: train_entry_model.py

import pandas as pd
import numpy as np
import argparse
import xgboost as xgb
import warnings
import os
import json
import math
from sklearn.metrics import mean_squared_error

# --- 1. 引用「設定檔」和「共用工具箱」 ---
import config
from common_utils import fetch_data, create_features_entry # <-- 改為引用 create_features_entry

warnings.simplefilter(action='ignore', category=FutureWarning)

# --- XGBoost 訓練基礎參數 ---
XGB_BASE_PARAMS = {
    'n_estimators': 1000,
    'learning_rate': 0.01,
    'objective': 'binary:logistic',
    'n_jobs': -1,
    'random_state': 42,
    'early_stopping_rounds': 10,
    'max_depth': 3,
    'reg_lambda': 5,
}

def train_xgb_regressor(df_features, features_list):
    """
    訓練 XGBoost 回歸模型
    """
    if df_features is None: return None, np.inf

    # --- 數據準備 (Target) ---
    df_model = df_features.copy()
    # *** 核心修改: Target 改為報酬率 ***
    df_model['target'] = df_model['Close'].shift(-1) > df_model['Close']
    df_model = df_model.dropna()

    # 2. 獲取 X 和 Y
    X = df_model[features_list]
    y = df_model['target']

    # 3. 分割訓練/測試集
    split_index = int(len(X) * 0.9) # 90% 訓練, 10% 驗證 (用於 early stopping)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    # 4. 訓練邏輯
    xgb_reg = xgb.XGBRegressor(**XGB_BASE_PARAMS)

    try:
        xgb_reg.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    except Exception as e:
        print(f"訓練時出錯: {e}") # 顯示錯誤
        return None, np.inf

    y_pred = xgb_reg.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return xgb_reg, rmse

if __name__ == "__main__":

    # --- 3. 建立「參數解析器」 ---
    parser = argparse.ArgumentParser(description=f'訓練 {config.ENTRY_MODEL_TIMEFRAME} XGBoost 進場模型')

    parser.add_argument('-s', '--symbol', type=str, required=True, help='要訓練的交易對 (例如: ETH/USDT 或 BTC/USDT)')
    parser.add_argument('-l', '--limit', type=int, default=config.ENTRY_MODEL_TRAIN_LIMIT, help=f'K 線筆數 (預設: {config.ENTRY_MODEL_TRAIN_LIMIT})')
    parser.add_argument('-v', '--version', type=str, default=config.ENTRY_MODEL_VERSION, help=f'要訓練的模型版本 (預設: {config.ENTRY_MODEL_VERSION})')

    args = parser.parse_args()

    # --- 4. 執行訓練 ---
    print(f"--- 開始執行: {args.symbol} ({config.ENTRY_MODEL_TIMEFRAME}), 資料量={args.limit} ---")

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    raw_df = fetch_data(symbol=args.symbol, timeframe=config.ENTRY_MODEL_TIMEFRAME, total_limit=args.limit)

    current_price = raw_df['Close'].iloc[-1]

    # --- 計算特徵 ---
    df_features, features_list = create_features_entry(raw_df.copy())
    if df_features is None or features_list is None:
        print(f"特徵計算失敗，結束訓練。")
        exit()

    # --- 訓練和評估 ---
    model, rmse = train_xgb_regressor(df_features, features_list)

    if math.isinf(rmse) or model is None:
        print(f"訓練失敗 (RMSE=inf)。")
        exit()

    print(f"訓練完成: RMSE={rmse:.6f}")

    # --- 最終模型儲存 ---
    abs_max_rmse = config.ABS_MAX_RMSE_PCT # <-- 直接使用百分比作為 RMSE 閾值

    if rmse > abs_max_rmse:
        print(f"\n❌ 訓練失敗！最佳 RMSE ({rmse:.6f}) 超過絕對極限 ({abs_max_rmse:.6f})。不儲存模型。")
        exit()
    else:
        print(f"\n✅ 質量門通過！最佳 RMSE ({rmse:.6f}) 優於絕對極限 ({abs_max_rmse:.6f})。")

    model_filename = config.get_entry_model_path(args.symbol, args.version)
    config_filename = model_filename.replace('.json', '_feature_config.json')

    # 儲存 XGBoost 模型
    print(f"\n--- 正在儲存「進場模型」... ---")
    model.save_model(model_filename)
    print(f"模型儲存完畢！({model_filename})")

    # 儲存空的 config (因為特徵工程是固定的)
    with open(config_filename, 'w') as f:
        json.dump({}, f, indent=4)
    print(f"✅ 特徵配置儲存完畢：{config_filename}")
