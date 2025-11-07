# 檔案: train_trend_model.py

import pandas as pd
import numpy as np
import argparse
import xgboost as xgb
import warnings
import os
import json
import math
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from scipy.stats import uniform, randint
import matplotlib.pyplot as plt  # 用於繪圖
from sklearn.metrics import ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE  # 新增: 用於過採樣平衡類別

# --- 1. 引用「設定檔」和「共用工具箱」 ---
import config
from common_utils import fetch_data, create_features_trend

warnings.simplefilter(action='ignore', category=FutureWarning)

# --- XGBoost 訓練基礎參數 (分類模型, 添加 class_weight='balanced' 處理不平衡) ---
XGB_BASE_PARAMS = {
    'n_estimators': 1000,
    'learning_rate': 0.013142918568673426,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'n_jobs': -1,
    'random_state': 42,
    'early_stopping_rounds': 10,
    'max_depth': 3,
    'reg_alpha': 3.143559810763267,
    'reg_lambda': 5.085706911647028,
    'colsample_bytree': 0.6431565707973218,
    'subsample': 0.9630265895704372,
}

# --- 超參數調校範圍 ---
PARAM_DIST = {
    'max_depth': randint(3, 8),
    'learning_rate': uniform(0.01, 0.1),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'reg_lambda': uniform(0, 10),
    'reg_alpha': uniform(0, 10)
}

def train_xgb_classifier(df_features, features_list):
    """
    訓練 XGBoost 分類模型 (預測漲跌: 1=漲, 0=跌)
    - 計算 target = (Close.shift(-1) > Close).astype(int)  # 漲跌二元標籤
    - 分割訓練/測試集 (80/20)
    - 使用 SMOTE 過採樣訓練集平衡類別 (減緩漲偏多問題)
    - 若不關閉調校: 使用 RandomizedSearchCV 調校超參數 (TimeSeriesSplit 交叉驗證)
    - 否則: 直接 fit (傳 eval_set 啟用 early stopping)
    - 評估測試集準確率 (accuracy_score)
    - 打印特徵重要性排行 (feature_importances_)
    - 高信心回測: 篩選信心 > threshold 的預測，計算準確率並繪製混淆矩陣 (threshold=0.55, 0.60, 0.65)
    - 向量化回測: 
      - 產生 signal (y_pred_proba > 0.5 = 1 (買), else 0)
      - 計算 strategy_return = signal.shift(1) * actual_return  # 毛收益
      - 計算 transaction_costs = trades * FEE_RATE  # 手續費
      - 計算 strategy_net_return = strategy_return - transaction_costs  # 淨收益
      - 計算 strategy_gross_equity = (1 + strategy_return).cumprod()  # 未扣費淨值曲線
      - 計算 strategy_net_equity = (1 + strategy_net_return).cumprod()  # 扣費後淨值曲線
      - 計算 bh_equity = (1 + actual_return).cumprod()  # Buy&Hold 淨值曲線
      - 繪製三條曲線: 策略 (未扣費, 扣費後), Buy&Hold
    - 返回模型及準確率
    """
    if df_features is None: return None, 0.0

    # --- 數據準備 (Target) ---
    df_model = df_features.copy()
    # *** 核心修改: Target 改為漲跌分類 (1=漲, 0=跌) ***
    df_model['target'] = (df_model['Close'].shift(-1) > df_model['Close'].shift(2)).astype(int)
    df_model = df_model.dropna()

    # 2. 獲取 X 和 Y
    X = df_model[features_list]
    y = df_model['target']

    # 3. 分割訓練/測試集
    split_index = int(len(X) * 0.8)  # 80% 訓練, 20% 測試
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    # 新增: 使用 SMOTE 過採樣訓練集 (平衡漲/跌樣本)
    print("\n--- 使用 SMOTE 平衡訓練集類別 ---")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"原訓練漲/跌: {sum(y_train == 1)}/{sum(y_train == 0)}")
    print(f"平衡後漲/跌: {sum(y_train_resampled == 1)}/{sum(y_train_resampled == 0)}")

    # 4. 超參數調校或直接訓練
    xgb_clf = xgb.XGBClassifier(**XGB_BASE_PARAMS)

    if not args.no_search_params:
        print("\n--- 開始超參數調校 (分類模型，這會花費 5-15+ 分鐘...) ---")
        xgb_clf.early_stopping_rounds = None
        xgb_clf.feature_weights
        
        tscv = TimeSeriesSplit(n_splits=3)

        random_search = RandomizedSearchCV(
            estimator=xgb_clf,
            param_distributions=PARAM_DIST,
            n_iter=25,
            cv=tscv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=2,
            random_state=42
        )
        
        random_search.fit(X_train_resampled, y_train_resampled)
        xgb_clf = random_search.best_estimator_
        print("\n--- 調校完成! ---")
        print(f"最佳交叉驗證 (CV) 準確率: {random_search.best_score_:.2%}")
        print("找到的最佳參數組合:")
        print(random_search.best_params_)
    else:
        # 直接訓練 (傳 eval_set 啟用 early stopping)
        print("\n--- 無調校，直接訓練模型 ---")
        xgb_clf.fit(X_train_resampled, y_train_resampled, eval_set=[(X_test, y_test)], verbose=False)

    # 5. 使用最佳模型評估測試集
    y_pred_proba = xgb_clf.predict_proba(X_test)[:, 1]  # 漲機率 (用於 signal)
    y_pred = (y_pred_proba > 0.5).astype(int)  # 二元預測

    acc = accuracy_score(y_test, y_pred)
    print(f"\n測試集準確率: {acc * 100:.2f}%")

    # 6. 打印特徵重要性排行
    features = X.columns
    importances = xgb_clf.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    print("\n--- 特徵重要性排行 ---")
    print(feature_importance_df)

    # 7. 高信心回測 (threshold=0.55, 0.60, 0.65)
    evaluate_with_confidence(xgb_clf, X_test, y_test, threshold=0.55)
    evaluate_with_confidence(xgb_clf, X_test, y_test, threshold=0.60)
    evaluate_with_confidence(xgb_clf, X_test, y_test, threshold=0.65)

    # --- 向量化回測 (計算收益曲線 vs Buy&Hold) ---
    # 1. 計算實際報酬率 (用於回測收益)
    df_test = df_model.iloc[split_index:].copy()  # 測試集資料
    df_test['actual_return'] = df_test['Close'].pct_change().shift(-1)  # 下一根 K 棒報酬 (實際漲跌幅)

    # 2. 產生訊號 (基於 y_pred_proba > 0.5 = 1 (買), else 0)
    THRESHOLD = 0.65  # 分類機率門檻 (可調整)
    # df_test['signal'] = np.where(y_pred_proba > THRESHOLD, 1, 0)  # 僅多頭 (漲預測 > 門檻 進場)
    df_test['signal'] = 0
    df_test.loc[y_pred_proba > THRESHOLD, 'signal'] = 1  # (信心做多)
    df_test.loc[y_pred_proba < (1 - THRESHOLD), 'signal'] = -1 # (信心做空)

    # 3. 計算策略毛收益 (signal.shift(1) * actual_return)
    df_test['strategy_return'] = df_test['signal'].shift(1) * df_test['actual_return']

    # 4. 計算手續費 (每次訊號變化扣費)
    FEE_RATE = 0.00055  # 手續費率
    df_test['trades'] = df_test['signal'].diff().abs().fillna(0)
    df_test['transaction_costs'] = df_test['trades'] * FEE_RATE

    # 5. 計算淨收益
    df_test['strategy_net_return'] = df_test['strategy_return'] - df_test['transaction_costs']

    # 6. 計算策略累計淨值 (從 1 開始累乘)
    df_test['strategy_gross_equity'] = (1 + df_test['strategy_return']).cumprod()  # 未扣費淨值曲線
    df_test['strategy_net_equity'] = (1 + df_test['strategy_net_return']).cumprod()  # 扣費後淨值曲線

    # 7. 計算 Buy&Hold 累計淨值
    df_test['bh_return'] = df_test['actual_return']
    df_test['bh_equity'] = (1 + df_test['bh_return']).cumprod()

    # 8. 繪製收益曲線 (三條: 未扣費策略、扣費策略、Buy&Hold)
    plt.rc('font', family='MingLiu')  # 支援中文字型
    plt.figure(figsize=(12, 6))
    df_test['strategy_gross_equity'].plot(label='策略淨值 (未扣費)', color='green')  # 未扣費 (綠色)
    df_test['strategy_net_equity'].plot(label='策略淨值 (扣費後)', color='red')  # 扣費後 (紅色)
    df_test['bh_equity'].plot(label='Buy & Hold', color='gray', linestyle='--')  # Buy&Hold (灰色虛線)
    plt.title('測試集回測收益曲線')
    plt.xlabel('時間步')
    plt.ylabel('累計淨值 (從 1 開始)')
    plt.grid(True)
    plt.legend()
    print("正在顯示回測收益曲線圖...")
    plt.show()

    return xgb_clf, acc  # 返回模型及準確率

def evaluate_with_confidence(classifier, X_test, y_test, threshold=0.60):
    """
    只評估「信心」高於 'threshold' 的交易。
    - 計算信心分數 (max predict_proba)
    - 篩選高信心預測及實際
    - 計算準確率、交易頻率
    - 繪製混淆矩陣
    """
    print(f"\n--- 高信心策略回測 (信心門檻 > {threshold:.0%}) ---")
    
    try:
        probabilities = classifier.predict_proba(X_test)
        confidence_scores = np.max(probabilities, axis=1)
        predictions = classifier.predict(X_test)
        
        high_confidence_mask = confidence_scores > threshold
        
        if np.sum(high_confidence_mask) == 0:
            print(f"警告：在 {len(y_test)} 筆資料中，沒有任何預測的信心 > {threshold:.0%}")
            print("請嘗試降低門檻 (例如 0.55)。")
            return

        high_confidence_predictions = predictions[high_confidence_mask]
        high_confidence_actuals = y_test[high_confidence_mask]
        
        new_accuracy = accuracy_score(high_confidence_actuals, high_confidence_predictions)
        
        total_trades = len(y_test)
        trades_taken = len(high_confidence_actuals)
        
        print("\n--- 高信心策略回測結果 ---")
        print(f"總 K 棒 (測試集): {total_trades}")
        print(f"觸發交易 (信心 > {threshold:.0%}): {trades_taken} 次")
        print(f"交易頻率: {trades_taken / total_trades:.2%}")
        print("---------------------------------")
        print(f"高信心交易的準確率 (Accuracy): {new_accuracy:.2%}")
        print("---------------------------------")
        
        # 繪製混淆矩陣
        plt.rc('font', family='MingLiu')  # 支援中文字型
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.set_title(f'混淆矩陣 (Confidence > {threshold:.0%})')
        
        ConfusionMatrixDisplay.from_predictions(
            high_confidence_actuals,
            high_confidence_predictions,
            ax=ax,
            cmap=plt.cm.Blues,
            display_labels=['實際 跌 (0)', '實際 漲 (1)']
        )
        ax.xaxis.set_ticklabels(['預測 跌 (0)', '預測 漲 (1)'])
        ax.yaxis.set_ticklabels(['實際 跌 (0)', '實際 漲 (1)'])
        
        print(f"正在顯示 信心 > {threshold:.0%} 的混淆矩陣...")
        plt.show()

    except Exception as e:
        print(f"執行高信心評估時出錯: {e}")

if __name__ == "__main__":

    # --- 3. 建立「參數解析器」 ---
    parser = argparse.ArgumentParser(description=f'訓練 {config.TREND_MODEL_TIMEFRAME} XGBoost 趨勢模型')

    parser.add_argument('-s', '--symbol', type=str, required=True, help='要訓練的交易對 (例如: ETH/USDT 或 BTC/USDT)')
    parser.add_argument('-tf', '--timeframe', type=str, required=True, help='要訓練的TimeFrame 例如:5m, 15m, 1h')
    parser.add_argument('-sd', '--start', type=str, help='回測起始日期 (YYYY-MM-DD)')
    parser.add_argument('-ed', '--end', type=str, help='回測結束日期 (YYYY-MM-DD)')
    parser.add_argument('-ns', '--no_search_params', action='store_true', help='關閉尋找模型最佳參數')
    parser.add_argument('-l', '--limit', type=int, help=f'K 線筆數限制')
    parser.add_argument('-v', '--version', type=str, default=config.TREND_MODEL_VERSION, help=f'要訓練的模型版本 (預設: {config.TREND_MODEL_VERSION})')

    args = parser.parse_args()

    # --- 4. 執行訓練 ---
    print(f"--- 開始執行: {args.symbol} ({args.timeframe}), 資料量={args.limit} ---")

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    raw_df = fetch_data(symbol=args.symbol, start_date=args.start, end_date=args.end, timeframe=args.timeframe, total_limit=args.limit)

    # --- 計算特徵 ---
    df_features, features_list = create_features_trend(raw_df.copy())
    if df_features is None or features_list is None:
        print(f"特徵計算失敗，結束訓練。")
        exit()

    # --- 訓練和評估 (改為分類模型) ---
    model, acc = train_xgb_classifier(df_features, features_list)

    if acc == 0.0 or model is None:
        print(f"訓練失敗 (準確率=0.0)。")
        exit()

    print(f"訓練完成: 準確率={acc * 100:.2f}%")

    # --- 最終模型儲存 (改用準確率閾值) ---
    abs_min_acc = 0.55  # *** 修改: 準確率最低閾值 (可調整) ***

    if acc < abs_min_acc:
        print(f"\n❌ 訓練失敗！最佳準確率 ({acc * 100:.2f}%) 低於絕對極限 ({abs_min_acc * 100:.2f}%)。不儲存模型。")
        exit()
    else:
        print(f"\n✅ 質量門通過！最佳準確率 ({acc * 100:.2f}%) 優於絕對極限 ({abs_min_acc * 100:.2f}%)。")

    model_filename = config.get_trend_model_path(args.symbol, args.timeframe, args.version)
    config_filename = model_filename.replace('.json', '_feature_config.json')

    # 儲存 XGBoost 模型
    print(f"\n--- 正在儲存「趨勢模型」... ---")
    model.save_model(model_filename)
    print(f"模型儲存完畢！({model_filename})")

    # 儲存空的 config (因為特徵工程是固定的)
    with open(config_filename, 'w') as f:
        json.dump({}, f, indent=4)
    print(f"✅ 特徵配置儲存完畢：{config_filename}")