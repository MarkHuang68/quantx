# 檔案: train_price_model.py (最終專業版)

import pandas as pd
import numpy as np
import argparse
import xgboost as xgb
import warnings
import os
import itertools
import json 
import math
from sklearn.metrics import mean_squared_error

# --- 1. 引用「設定檔」和「共用工具箱」 ---
import config
from common_utils import fetch_data, create_features_price
from hyperparameter_search import SearchIterator

warnings.simplefilter(action='ignore', category=FutureWarning)

# --- 您的尋參空間 (保持不變) ---
PRICE_SEARCH_SPACE = {
    'macd_fast': [6, 12],
    'macd_slow': [13, 26],
    'bbands': [2, 20, 6],
    'learning_rate': [0.01, 0.03, 0.05], 
    'max_depth': [2, 4, 6],
}
# --- 您的 XGBoost 訓練基礎參數 (保持不變) ---
XGB_BASE_PARAMS = {
    'n_estimators': 1000, 'objective': 'reg:squarederror',
    'n_jobs': -1, 'random_state': 42, 'early_stopping_rounds': 50
}


def train_xgb_regressor(df_features, features_list, params):
    """ (您的訓練函數，保持不變) """
    if df_features is None: return None, np.inf

    # --- 數據準備 (Target) ---
    df_model = df_features.copy()
    df_model['target'] = df_model['Close'].shift(-1)
    df_model = df_model.dropna()
    
    # 2. 獲取 X 和 Y
    X = df_model[features_list]
    y = df_model['target']
    
    # 3. 分割訓練/測試集
    split_index = int(len(X) * 0.9) # 90% 訓練, 10% 驗證 (用於 early stopping)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    # 4. 訓練邏輯
    xgb_reg = xgb.XGBRegressor(**params, **XGB_BASE_PARAMS)
    
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
    parser = argparse.ArgumentParser(description='訓練 5m XGBoost 價格模型')
    
    parser.add_argument('-s', '--symbol', type=str, required=True, help='要訓練的交易對 (例如: ETH/USDT 或 BTC/USDT)')
    parser.add_argument('-l', '--limit', type=int, default=config.PRICE_MODEL_TRAIN_LIMIT, help=f'K 線筆數 (預設: {config.PRICE_MODEL_TRAIN_LIMIT})')
    parser.add_argument('-v', '--version', type=str, default=config.PRICE_MODEL_VERSION, help=f'要訓練的模型版本 (預設: {config.PRICE_MODEL_VERSION})')
    
    args = parser.parse_args()
    
    # --- 4. 執行訓練 (網格搜索) ---
    print(f"--- 開始執行: {args.symbol} ({config.PRICE_MODEL_TIMEFRAME}), 資料量={args.limit} ---")
    
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    raw_df = fetch_data(symbol=args.symbol, timeframe=config.PRICE_MODEL_TIMEFRAME, total_limit=args.limit)
    
    # (*** 關鍵修正：使用 raw_df 的最新價格，而不是重新 fetch ***)
    current_price = raw_df['Close'].iloc[-1]
    
    # 設定參數格式, 生成參數組合
    f_type = {
        'learning_rate': 'discrete', 
        'max_depth': 'discrete',
    }
    iterator = SearchIterator(PRICE_SEARCH_SPACE, search_type='grid', format_types=f_type)

    print(f"--- 總共需要執行 {iterator.get_total_runs()} 次訓練 ---")
    
    best_rmse = np.inf
    best_model = None
    best_feature_params = None # <--- *** 只儲存特徵配置 ***
    
    # 定義哪些鍵是特徵參數 (必須匹配 common_utils.py)
    FEATURE_KEYS = ['macd_fast', 'macd_slow', 'bbands']
    
    for i, params in enumerate(iterator):
        
        # 1a. 分離特徵參數和模型參數
        feature_params = {k: params[k] for k in FEATURE_KEYS if k in params}
        xgb_params = {k: params[k] for k in params.keys() if k not in FEATURE_KEYS}

        # 1b. 計算特徵 (傳入特徵參數)
        # (*** 警告：您的 common_utils.py 必須更新以接受 **feature_params ***)
        df_features, features_list = create_features_price(raw_df.copy(), **feature_params)
        if df_features is None or features_list is None: 
            print(f"Iter {i+1:02d}/{iterator.get_total_runs()}: 特徵計算失敗，跳過。")
            continue
        
        # 1c. 訓練和評估
        current_model, rmse = train_xgb_regressor(df_features, features_list, xgb_params)
        
        if math.isinf(rmse):
             print(f"Iter {i+1:02d}/{iterator.get_total_runs()}: 訓練失敗 (RMSE=inf)。 (Params: {feature_params}, LR={xgb_params.get('learning_rate')})")
             continue

        print(f"Iter {i+1:02d}/{iterator.get_total_runs()}: RMSE={rmse:.2f} (Params: {feature_params}, LR={xgb_params.get('learning_rate')})")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = current_model
            best_feature_params = feature_params 
    
    # --- 2. 最終模型儲存和質量控制 ---
    
    if not best_model:
        print("🛑 訓練失敗：未能找到最佳模型。")
        exit()
        
    # 2a. (*** 補上的「絕對 RMSE 檢查」***)
    abs_max_rmse = current_price * config.ABS_MAX_RMSE_PCT
    
    if best_rmse > abs_max_rmse:
        print(f"\n❌ 訓練失敗！最佳 RMSE ({best_rmse:.2f}) 超過絕對極限 (${abs_max_rmse:.2f})。不儲存模型。")
        print("請調整 PRICE_SEARCH_SPACE 參數以獲得更精確的模型。")
        exit()
    else:
        print(f"\n✅ 質量門通過！最佳 RMSE ({best_rmse:.2f}) 優於絕對極限 (${abs_max_rmse:.2f})。")

    # 2b. (*** 補上的「競爭標準檢查」 ***)
    # 載入現行模型和配置
    current_model_path = config.get_price_model_path(args.symbol, args.version)
    current_config_path = current_model_path.replace('.json', '_feature_config.json')
    
    current_model = None
    historical_rmse = np.inf
    
    if os.path.exists(current_model_path) and os.path.exists(current_config_path):
        try:
            print(f"--- 正在載入現行模型 ({args.version}) 進行競爭比較... ---")
            current_model = xgb.Booster()
            current_model.load_model(current_model_path)
            
            with open(current_config_path, 'r') as f:
                current_feature_config = json.load(f)
            
            # (*** 關鍵：在「相同」的 raw_df 上，用「舊」的特徵參數回測 ***)
            print(f"正在使用現行模型的特徵參數 {current_feature_config} 進行回測...")
            df_features_old, features_list_old = create_features_price(raw_df.copy(), **current_feature_config)
            
            # (我們需要一個獨立的評估函數，因為 train_xgb_regressor 包含了訓練)
            df_model_old = df_features_old.copy()
            df_model_old['target'] = df_model_old['Close'].shift(-1)
            df_model_old = df_model_old.dropna()
            X_old = df_model_old[features_list_old]
            y_old = df_model_old['target']
            
            split_index_old = int(len(X_old) * 0.9)
            X_test_old = X_old.iloc[split_index_old:]
            y_test_old = y_old.iloc[split_index_old:]
            
            y_pred_old = current_model.predict(xgb.DMatrix(X_test_old))
            historical_rmse = np.sqrt(mean_squared_error(y_test_old, y_pred_old))
            
        except Exception as e:
            print(f"警告：載入或評估現行模型失敗：{e}")
            historical_rmse = np.inf
    else:
        print("--- 找不到現行模型，將直接儲存新模型。 ---")

    # (*** 質量門 2: 競爭標準檢查 ***)
    if best_rmse >= historical_rmse:
        print(f"\n❌ 訓練失敗！新模型 RMSE ({best_rmse:.2f}) 並未優于現行模型 ({historical_rmse:.2f})。不儲存模型。")
        exit()
    else:
        print(f"\n✅ 質量門 2 (競爭標準) 通過！新模型 ({best_rmse:.2f}) 擊敗 現行模型 ({historical_rmse:.2f})。")


    # 2c. 儲存模型和最佳參數 (*** 核心步驟 ***)
    model_filename = config.get_price_model_path(args.symbol, args.version)
    config_filename = config.get_price_model_path(args.symbol, args.version).replace('.json', '_feature_config.json')

    # 儲存 XGBoost 模型
    if best_model:
        print(f"\n--- 正在儲存「價格模型」... ---")
        best_model.save_model(model_filename)
        print(f"模型儲存完畢！({model_filename})")
    
    # (*** 關鍵修正：只儲存特徵參數 (best_feature_params) ***)
    if best_feature_params:
        with open(config_filename, 'w') as f:
            json.dump(best_feature_params, f, indent=4)
        print(f"✅ 最佳特徵配置儲存完畢：{config_filename}")
