# -*- coding: utf-8 -*-
"""
SMC-ML 訓練腳本
把 feature_engine + smc_features 產出的特徵 → XGBoost 二元分類
目標：預測「下一個 K 棒是否會觸發 SMC 進場」(label_entry == 1)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
import shap
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

# -------------------------------------------------
# 1. 載入你的特徵工程模組
# -------------------------------------------------
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
from feature_engine import add_all_features

# -------------------------------------------------
# 2. 讀取原始 OHLCV (CSV 範例)
# -------------------------------------------------
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    df = df.set_index('timestamp')
    # 必須要有 open, high, low, close, volume
    return df

# -------------------------------------------------
# 3. 產生 ML 標籤 (SMC SOP)
# -------------------------------------------------
# def create_labels(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
#     """
#     修正版標籤產生：
#     1. 進場 = Extreme POI 被觸發（SMC 進場點）
#     2. 出場 = Target 被觸發
#     3. 用 shift(-1) 預測「下一根 K 棒」是否進場
#     """
#     df = df.copy()

#     # --- 1. 進場訊號：Extreme POI 被觸發的那一根 ---
#     entry_bull = (df['feat_smc_state_EXTREME'] == 1) & (df['feat_smc_state_EXTREME'].shift(1) == 0)
#     entry_bear = (df['feat_smc_state_EXTREME'] == -1) & (df['feat_smc_state_EXTREME'].shift(1) == 0)
#     entry_signal = entry_bull | entry_bear

#     # --- 2. 出場訊號：Target 被觸發 ---
#     exit_bull = (df['feat_smc_state_TARGET'] == 1) & (df['feat_smc_state_TARGET'].shift(1) == 0)
#     exit_bear = (df['feat_smc_state_TARGET'] == -1) & (df['feat_smc_state_TARGET'].shift(1) == 0)
#     exit_signal = exit_bull | exit_bear

#     # --- 3. 標籤：預測「未來 horizon 根 K 棒內」是否會進場 ---
#     future_entry = entry_signal.shift(-horizon).fillna(False)
#     df['label_entry'] = future_entry.astype(int)

#     df['label_exit'] = exit_signal.shift(-1).fillna(False).astype(int)  # 出場用下一根

#     # --- 4. 回歸標的：目標價差 ---
#     df['label_target'] = 0.0
#     mask_bull = (df['feat_smc_state_TARGET'] == 1)
#     mask_bear = (df['feat_smc_state_TARGET'] == -1)
#     df.loc[mask_bull, 'label_target'] = df['high'].shift(-horizon) - df['close']
#     df.loc[mask_bear, 'label_target'] = df['close'] - df['low'].shift(-horizon)

#     # --- 5. 除錯：印出正例數量 ---
#     print(f"[LABEL DEBUG] 正例數量 (label_entry==1): {df['label_entry'].sum()} / {len(df)} ({df['label_entry'].mean():.6f})")
#     print(f"Extreme POI 觸發次數: {entry_signal.sum()}")
#     print(f"Target 觸發次數: {exit_signal.sum()}")

#     return df

def create_labels(df: pd.DataFrame, horizon: int = 3) -> pd.DataFrame:
    df = df.copy()

    # 進場 = Extreme POI 被觸發
    entry = (df['feat_smc_state_EXTREME'] != 0) & (df['feat_smc_state_EXTREME'].shift(1) == 0)
    
    # 讓標籤提前 horizon 根（模型能學）
    df['label_entry'] = entry.shift(-horizon).fillna(0).astype(int)
    
    print(f"[LABEL] Extreme 觸發次數: {entry.sum()}")
    print(f"[LABEL] label_entry 正例: {df['label_entry'].sum()}")

    return df

# -------------------------------------------------
# 4. 主流程
# -------------------------------------------------
def main(csv_path: str, model_path: str = "smc_xgb_v1.pkl"):
    raw = load_data(csv_path)
    print(f"原始資料筆數: {len(raw)}")

    df = add_all_features(raw)
    print(f"特徵工程後欄位數: {len(df.columns)}")
    print(f"特徵工程後筆數: {len(df)}")

    # === 除錯：檢查 SMC 關鍵欄位 ===
    smc_cols = ['bos_bullish', 'choch_bullish', 'hq_bullish_ob_top', 
                'feat_smc_state_EXTREME', 'feat_smc_state_TARGET']
    for col in smc_cols:
        if col in df.columns:
            print(f"{col:25}: {df[col].notna().sum():4} 個非空值, 正例 { (df[col]!=0).sum() if df[col].dtype != bool else df[col].sum() }")
        else:
            print(f"{col:25}: 欄位不存在！")

    df = create_labels(df, horizon=5)
    print(f"label_entry 正例數: {df['label_entry'].sum()}")

    # --- 特徵清單 (去除價格、時間、標籤) ---
    drop_cols = ['open','high','low','close','volume',
                'poi_bullish','poi_bearish',  # 避免洩漏
                'label_entry','label_exit','label_target']
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].copy()
    y_entry = df['label_entry'].copy()

    # --- 正規化 (只 fit train，test 用相同 scaler) ---
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)

    # --- TimeSeriesSplit (5 folds) ---
    tscv = TimeSeriesSplit(n_splits=5)

    # --- 超參數搜尋空間 ---
    search_spaces = {
        'max_depth': (3, 12),
        'learning_rate': (0.01, 0.3, 'log-uniform'),
        'n_estimators': (100, 1000),
        'subsample': (0.6, 1.0),
        'colsample_bytree': (0.6, 1.0),
        'gamma': (0, 5),
        'min_child_weight': (1, 10)
    }

    # 計算 class_weight (SMC 訊號非常稀疏)
    pos_weight = (y_entry == 0).sum() / (y_entry == 1).sum()

    xgb_base = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        scale_pos_weight=pos_weight,
        n_jobs=-1,
        random_state=42
    )

    opt = BayesSearchCV(
        estimator=xgb_base,
        search_spaces=search_spaces,
        n_iter=50,
        cv=tscv,
        scoring='f1',               # 對稀疏正例更敏感
        verbose=1,
        n_jobs=-1,
        random_state=42
    )

    #========================================
    # train_smc_ml.py → main() 內
    from imblearn.over_sampling import SMOTE

    # 在 fit 之前
    smote = SMOTE(random_state=42, k_neighbors=1)
    X_res, y_res = smote.fit_resample(X_scaled, y_entry)

    # 更新 pos_weight
    pos_weight = (y_res == 0).sum() / (y_res == 1).sum()

    xgb_base = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        scale_pos_weight=pos_weight,
        n_jobs=-1,
        random_state=42
    )
    #=========================================

    print("開始超參數搜尋…")
    opt.fit(X_scaled, y_entry)

    best_model = opt.best_estimator_
    print(f"Best F1: {opt.best_score_:.4f}")
    print("Best params:", opt.best_params_)

    # --- 儲存模型 + scaler ---
    joblib.dump({
        'model': best_model,
        'scaler': scaler,
        'feature_cols': feature_cols
    }, model_path)
    print(f"模型已儲存至 {model_path}")

    # -------------------------------------------------
    # 5. SHAP 解釋 (看模型到底學到什麼 SMC 特徵)
    # -------------------------------------------------
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_scaled)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_scaled, max_display=20, show=False)
    plt.title("Top 20 SMC Feature Importance (SHAP)")
    plt.tight_layout()
    plt.savefig("shap_summary.png")
    print("SHAP 圖已儲存為 shap_summary.png")

    # -------------------------------------------------
    # 6. 簡易回測 (只看 entry 預測)
    # -------------------------------------------------
    df['pred_entry'] = best_model.predict(X_scaled)
    df['pred_proba'] = best_model.predict_proba(X_scaled)[:, 1]

    # 只在預測為 1 時進場，持倉到 label_exit == 1 或 20 根 K 棒
    position = 0
    entry_price = 0.0
    pnl = []

    for i in range(len(df)-20):
        row = df.iloc[i]

        if position == 0 and row['pred_entry'] == 1:
            position = 1
            entry_price = row['close']

        elif position == 1:
            # 強制出場條件
            if row['label_exit'] == 1 or (i+20 < len(df) and df.iloc[i+20]['label_exit'] == 1):
                exit_price = row['close']
                pnl.append(exit_price - entry_price)
                position = 0
            # 時間到也出場
            elif i + 20 >= len(df):
                exit_price = row['close']
                pnl.append(exit_price - entry_price)
                position = 0

    total_pnl = sum(pnl)
    win_rate = sum(1 for p in pnl if p > 0) / len(pnl) if pnl else 0
    print(f"\n簡易回測結果：")
    print(f"交易次數 = {len(pnl)}  勝率 = {win_rate:.2%}  總盈虧 = {total_pnl:.2f}")

if __name__ == "__main__":
    # 請自行替換成你的 CSV 路徑
    CSV_PATH = "data/btc_1h.csv"
    main(CSV_PATH, model_path="smc_xgb_v1.pkl")