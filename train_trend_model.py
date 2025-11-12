# 檔案: train_trend_model.py

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import argparse
import xgboost as xgb
import warnings
import os
import json
import math
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, learning_curve
from scipy.stats import uniform, randint
import matplotlib.pyplot as plt  # 用於繪圖
from sklearn.metrics import ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE  # 新增: 用於過採樣平衡類別

# --- 1. 引用「設定檔」和「共用工具箱」 ---
import settings
from utils.common import fetch_data, create_features_trend

warnings.simplefilter(action='ignore', category=FutureWarning)

# --- 新增: 定義「中立區」門檻 ---
# 漲跌幅在 +/- 0.2% 以內，都視為「持有 (0)」
# (您可以調整這個值，越大，交易越少)
HOLD_THRESHOLD = 0.008

train = False

# --- XGBoost 訓練基礎參數 (分類模型, 添加 class_weight='balanced' 處理不平衡) ---
XGB_BASE_PARAMS = {
    'n_estimators': 300,
    'learning_rate': 0.013142918568673426,
    'objective': 'multi:softmax',     # 從 'binary:logistic' 改為 'multi:softmax'
    'num_class': 3,                   # 新增: 告訴 XGBoost 我們有 3 個類別 (0=持有, 1=買入, 2=賣出)
    'eval_metric': 'mlogloss',        # 從 'logloss' 改為 'mlogloss'
    'n_jobs': -1,
    'random_state': 42,
    'early_stopping_rounds': 50,
    'max_depth': 5,
    'reg_alpha': 5,
    'reg_lambda': 5,
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
    訓練 XGBoost 分類模型 (預測趨勢: 1=漲, 0=持有, 2=跌)
    - 計算 target (三分類標籤)
    - 分割訓練/測試集 (80/20)
    - 使用 SMOTE 過採樣訓練集平衡類別 (減緩漲偏多問題)
    - 若不關閉調校: 使用 RandomizedSearchCV 調校超參數 (TimeSeriesSplit 交叉驗證)
    - 否則: 直接 fit (傳 eval_set 啟用 early stopping)
    - 評估測試集準確率 (accuracy_score)
    - 打印特徵重要性排行 (feature_importances_)
    - 高信心回測: 篩選信心 > threshold 的預測，計算準確率並繪製混淆矩陣 (threshold=0.55, 0.60, 0.65)  # 預設不顯示，需 --show_confidence
    - 向量化回測: 
      - 產生 signal (y_pred 0=0, 1=1, 2=-1)
      - 計算 strategy_return = signal.shift(1) * actual_return  # 毛收益
      - 計算 transaction_costs = trades * FEE_RATE  # 手續費
      - 計算 strategy_net_return = strategy_return - transaction_costs  # 淨收益
      - 計算 strategy_gross_equity = (1 + strategy_return).cumprod()  # 未扣費淨值曲線
      - 計算 strategy_net_equity = (1 + strategy_net_return).cumprod()  # 扣費後淨值曲線
      - 計算 bh_equity = (1 + actual_return).cumprod()  # Buy&Hold 淨值曲線
      - 繪製三條曲線: 策略 (未扣費, 扣費後), Buy&Hold  # 預設不顯示，需 --show_equity
    - 過擬合檢測: 使用 learning_curve 繪製學習曲線 (訓練 vs 驗證準確率)  # 預設不顯示，需 --show_overfit
    - 返回模型及準確率
    """
    if df_features is None: return None, 0.0

    # --- 數據準備 (Target) ---
    df_model = df_features.copy()
    
    # --- 修改: 建立三分類 target ---
    # 1. 計算未來 1 根 K 棒的報酬率
    future_return = (df_model['Close'].shift(-1) - df_model['Close']) / df_model['Close']
    
    # 2. 預設標籤為 0 (持有)
    df_model['target'] = 0
    
    # 3. 如果未來報酬 > 門檻，標記為 1 (買入)
    df_model.loc[future_return > HOLD_THRESHOLD, 'target'] = 1
    
    # 4. 如果未來報酬 < -門檻，標記為 2 (賣出)
    df_model.loc[future_return < -HOLD_THRESHOLD, 'target'] = 2
    
    df_model = df_model.dropna()
    # --- 修改結束 ---

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
    
    # --- 修改: 更新 SMOTE 打印訊息 ---
    print(f"原訓練 (持有/買入/賣出): {sum(y_train == 0)}/{sum(y_train == 1)}/{sum(y_train == 2)}")
    print(f"平衡後 (持有/買入/賣出): {sum(y_train_resampled == 0)}/{sum(y_train_resampled == 1)}/{sum(y_train_resampled == 2)}")
    # --- 修改結束 ---

    # 4. 超參數調校或直接訓練
    xgb_clf = xgb.XGBClassifier(**XGB_BASE_PARAMS)

    if not args.no_search_params:
        print("\n--- 開始超參數調校 (分類模型，這會花費 5-15+ 分鐘...) ---")
        xgb_clf.early_stopping_rounds = None  # RandomizedSearchCV 不支援 early stopping

        tscv = TimeSeriesSplit(n_splits=3)

        random_search = RandomizedSearchCV(
            estimator=xgb_clf,
            param_distributions=PARAM_DIST,
            n_iter=25,
            cv=tscv,
            scoring='accuracy', # 保持 accuracy，或可改用 'balanced_accuracy'
            n_jobs=-1,
            verbose=2,
            random_state=42
        )
        
        random_search.fit(X_train_resampled, y_train_resampled)

        print("\n--- 最佳參數 (分類模型) ---")
        print(random_search.best_params_)

        model = random_search.best_estimator_
    else:
        print("\n--- 開始直接訓練 (分類模型)... ---")
        eval_set = [(X_test, y_test)]  # 用測試集作為驗證集 (early stopping)
        model = xgb_clf.fit(X_train_resampled, y_train_resampled, eval_set=eval_set, verbose=True)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return model, acc

def backtest(model, df_features, features_list):
    # --- 數據準備 (Target) ---
    df_model = df_features.copy()
    
    # --- 修改: 建立三分類 target ---
    # 1. 計算未來 1 根 K 棒的報酬率
    future_return = (df_model['Close'].shift(-1) - df_model['Close']) / df_model['Close']
    
    # 2. 預設標籤為 0 (持有)
    df_model['target'] = 0
    
    # 3. 如果未來報酬 > 門檻，標記為 1 (買入)
    df_model.loc[future_return > HOLD_THRESHOLD, 'target'] = 1
    
    # 4. 如果未來報酬 < -門檻，標記為 2 (賣出)
    df_model.loc[future_return < -HOLD_THRESHOLD, 'target'] = 2
    
    df_model = df_model.dropna()
    # --- 修改結束 ---

    # 2. 獲取 X 和 Y
    X = df_model[features_list]
    y = df_model['target']

    # 3. 分割訓練/測試集
    if train:
        split_index = int(len(X) * 0.8)  # 80% 訓練, 20% 測試
    else:
        split_index = 0

    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]

    # 5. 測試集評估
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n測試集準確率 (Accuracy): {acc:.2%}")

    # --- 修改: 多分類需要 predict_proba (用於 show_confidence) ---
    # predict_proba 現在會返回 (n_samples, 3) 的矩陣
    y_pred_proba_all = model.predict_proba(X_test)
    # --- 修改結束 ---


    # 6. 特徵重要性
    print("\n--- 特徵重要性排行 ---")
    feature_importances = pd.DataFrame({
        'feature': features_list,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)
    print(feature_importances.to_string(index=False))

    # --- 高信心回測 (預設不顯示，需 --show_confidence) ---
    if args.show_confidence:
        print("\n--- 高信心策略回測 ---")
        
        # --- 修改: 高信心邏輯 (多分類) ---
        # 獲取模型對其預測的信心 (取最大機率)
        y_pred_confidence = y_pred_proba_all.max(axis=1)

        for threshold in [0.55, 0.60, 0.65]:
            # 篩選信心 > 門檻的預測
            high_confidence_mask = (y_pred_confidence >= threshold)
            
            high_confidence_predictions = y_pred[high_confidence_mask]
            high_confidence_actuals = y_test[high_confidence_mask]

            if len(high_confidence_actuals) == 0:
                print(f"警告：在 {len(y_test)} 筆資料中，沒有任何預測的信心 > {threshold:.0%}")
                print("請嘗試降低門檻 (例如 0.55)。")
                continue

            new_accuracy = accuracy_score(high_confidence_actuals, high_confidence_predictions)
            
            total_trades = len(y_test)
            trades_taken = len(high_confidence_actuals)
            
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
            
            # 定義 3 個類別的標籤
            display_labels = ['實際 持有(0)', '實際 買入(1)', '實際 賣出(2)']
            tick_labels = ['預測 持有(0)', '預測 買入(1)', '預測 賣出(2)']

            ConfusionMatrixDisplay.from_predictions(
                high_confidence_actuals,
                high_confidence_predictions,
                ax=ax,
                cmap=plt.cm.Blues,
                labels=[0, 1, 2],
                # tick_labels=tick_labels,
                # display_labels=display_labels
            )
            ax.xaxis.set_ticklabels(tick_labels)
            ax.yaxis.set_ticklabels(display_labels)
            
            print(f"正在顯示 信心 > {threshold:.0%} 的混淆矩陣...")
            plt.show()
        # --- 修改結束 ---


    # --- 過擬合檢測 (學習曲線，預設不顯示，需 --show_overfit) ---
    if args.show_overfit:
        print("\n--- 過擬合檢測: 學習曲線 ---")
        
        # 定義新模型 (避免使用已訓練 model)
        xgb_clf = xgb.XGBClassifier(**XGB_BASE_PARAMS)
        xgb_clf.early_stopping_rounds = None  # learning_curve 不支援 early stopping，直接關閉
        
        train_sizes, train_scores, val_scores = learning_curve(
            xgb_clf, X, y, cv=TimeSeriesSplit(n_splits=3), n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5),
            scoring='accuracy'
        )
        
        train_mean = np.mean(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)

        train_val_gap = train_mean[-1] - val_mean[-1]
        print(f"最終訓練-驗證差距: {train_val_gap:.2%}")  # 若 >5%，過擬合嚴重
        
        plt.rc('font', family='MingLiu')  # 支援中文字型
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, label='訓練準確率', color='blue')
        plt.plot(train_sizes, val_mean, label='驗證準確率', color='red')
        plt.title('學習曲線 (檢測過擬合)')
        plt.xlabel('訓練樣本數')
        plt.ylabel('準確率')
        plt.legend()
        plt.grid(True)
        print("正在顯示學習曲線... 若訓練高於驗證且差距大，即可能過擬合。")
        plt.show()

    # --- 修改: 更新最終打印訊息 ---
    hold_ratio = sum(y_pred == 0) / len(y_pred)
    buy_ratio = sum(y_pred == 1) / len(y_pred)
    sell_ratio = sum(y_pred == 2) / len(y_pred)
    print(f"預測分佈 (持有/買入/賣出): {hold_ratio:.2%} / {buy_ratio:.2%} / {sell_ratio:.2%}")
    # --- 修改結束 ---

def iterative_backtest(df_test, y_pred):
    print("\n--- [二次修正版] 事件驅動回測 (已修正會計邏輯) ---")
    initial_capital = 10000.0
    cash = initial_capital
    position_contracts = 0.0  # 持倉合約數量 (正為多, 負為空)
    equity_curve = [initial_capital]

    signal_map = {0: 0, 1: 1, 2: -1}
    signals = pd.Series(y_pred).map(signal_map).values

    df_test = df_test.copy()
    df_test['signal'] = np.nan
    df_test.iloc[0:len(signals), df_test.columns.get_loc('signal')] = signals

    for i in range(1, len(df_test)):
        # --- 時間點與價格 ---
        # 訊號在 t-1 的收盤後產生
        # 交易在 t 的開盤時執行
        current_open = df_test['Open'].iloc[i]
        current_close = df_test['Close'].iloc[i]
        signal = df_test['signal'].iloc[i-1]

        # --- 權益計算: 每個 bar 開始時，先更新持倉市值 ---
        # 權益 = 現金 + 持倉市值
        current_equity = cash + (position_contracts * current_close)

        # --- 交易執行 (在 t 的開盤價) ---
        # --- 交易執行 (在 t 的開盤價) ---
        # 核心邏輯：只有在訊號與現有倉位反向時，才需要動作 (先平後開)
        # 訊號為 0 或與現有倉位同向，則持倉不動

        # 1. 平倉邏輯：訊號與倉位反向
        if (signal == -1 and position_contracts > 0) or \
           (signal == 1 and position_contracts < 0):
            revenue = abs(position_contracts) * current_open
            fee = revenue * settings.FEE_RATE
            cash += (revenue - fee)
            print(f"[{df_test.index[i]}] 反向平倉 {'多' if position_contracts > 0 else '空'} @ {current_open:.2f}, 數量: {abs(position_contracts):.4f}, 收入: {revenue-fee:.2f}, 現金: {cash:.2f}")
            position_contracts = 0

        # 2. 開倉邏輯：收到開倉訊號且當前為空手
        if signal != 0 and position_contracts == 0:
            trade_value = cash * 0.95  # 使用95%現金開倉
            fee = trade_value * settings.FEE_RATE

            if cash > (trade_value + fee):
                cash -= (trade_value + fee)
                contracts_to_buy = trade_value / current_open
                position_contracts = contracts_to_buy if signal == 1 else -contracts_to_buy
                print(f"[{df_test.index[i]}] 開倉 {'多' if signal == 1 else '空'} @ {current_open:.2f}, 數量: {abs(position_contracts):.4f}, 成本: {trade_value+fee:.2f}, 現金: {cash:.2f}")

        # --- 權益更新 ---
        # 重新計算迴圈結束時的權益
        current_equity = cash + (position_contracts * current_close)
        equity_curve.append(current_equity)

    # --- 繪圖 ---
    df_test['equity'] = pd.Series(equity_curve, index=df_test.index[:len(equity_curve)])
    df_test['bh_return'] = df_test['Close'].pct_change().fillna(0)
    df_test['bh_equity'] = (1 + df_test['bh_return']).cumprod() * initial_capital

    plt.rc('font', family='MingLiu')
    plt.figure(figsize=(12, 6))
    plt.plot(df_test['bh_equity'], label='Buy & Hold', color='gray')
    plt.plot(df_test['equity'], label='策略 (扣費後, 修正版)', color='red')
    plt.title(f'回測淨值曲線 (二次修正版) ({args.timeframe})')
    plt.xlabel('時間')
    plt.ylabel('淨值 (USDT)')
    plt.legend()
    plt.grid(True)
    print("正在顯示二次修正後的資金曲線...")
    plt.show()


if __name__ == "__main__":

    # --- 3. 建立「參數解析器」 ---
    parser = argparse.ArgumentParser(description=f'訓練 XGBoost 趨勢模型')

    parser.add_argument('-s', '--symbol', type=str, required=True, help='要訓練的交易對 (例如: ETH/USDT 或 BTC/USDT)')
    parser.add_argument('-tf', '--timeframe', type=str, required=True, help='要訓練的TimeFrame 例如:5m, 15m, 1h')
    parser.add_argument('-sd', '--start', type=str, help='回測起始日期 (YYYY-MM-DD)')
    parser.add_argument('-ed', '--end', type=str, help='回測結束日期 (YYYY-MM-DD)')
    parser.add_argument('-ns', '--no_search_params', action='store_true', help='關閉尋找模型最佳參數')
    parser.add_argument('-l', '--limit', type=int, help=f'K 線筆數限制')
    parser.add_argument('-v', '--version', type=str, default=settings.TREND_MODEL_VERSION, help=f'要訓練的模型版本 (預設: {settings.TREND_MODEL_VERSION})')
    parser.add_argument('--show_confidence', action='store_true', help='顯示高信心混淆矩陣 (預設不顯示)')
    parser.add_argument('--show_equity', action='store_true', help='顯示資金曲線 (預設不顯示)')
    parser.add_argument('--show_overfit', action='store_true', help='顯示過擬合檢測學習曲線 (預設不顯示)')
    parser.add_argument('--force_train', action='store_true', help='強制訓練模型，即使模型已存在')

    args = parser.parse_args()

    # --- 4. 執行訓練 ---
    print(f"--- 開始執行: {args.symbol} ({args.timeframe}), 資料量={args.limit} ---")

    os.makedirs(settings.MODEL_DIR, exist_ok=True)
    raw_df = fetch_data(symbol=args.symbol, start_date=args.start, end_date=args.end, timeframe=args.timeframe, total_limit=args.limit)

    # --- 計算特徵 ---
    df_features, features_list = create_features_trend(raw_df.copy())
    if df_features is None or features_list is None:
        print(f"特徵計算失敗，結束訓練。")
        exit()

    model_filename = settings.get_trend_model_path(args.symbol, args.timeframe, args.version)
    config_filename = model_filename.replace('.json', '_feature_config.json')

    if os.path.exists(model_filename) and not args.force_train:
        print(f"模型已存在 ({model_filename})，載入並直接回測...")
        model = xgb.XGBClassifier()
        model.load_model(model_filename)
        acc = None  # 不重新計算準確率
    else:
        train = True
        print("模型不存在或強制訓練，開始訓練...")
        model, acc = train_xgb_classifier(df_features, features_list)

        if acc == 0.0 or model is None:
            print(f"訓練失敗 (準確率=0.0)。")
            exit()

        print(f"訓練完成: 準確率={acc * 100:.2f}%")

        # --- 最終模型儲存 (改用準確率閾值) ---
        abs_min_acc = 0.30  # *** 修改: 準確率最低閾值 (可調整)。三分類的 55% 太高，先降到 40% ***

        if acc < abs_min_acc:
            print(f"\n❌ 訓練失敗！最佳準確率 ({acc * 100:.2f}%) 低於絕對極限 ({abs_min_acc * 100:.2f}%)。不儲存模型。")
            exit()
        else:
            print(f"\n✅ 質量門通過！最佳準確率 ({acc * 100:.2f}%) 優於絕對極限 ({abs_min_acc * 100:.2f}%)。")

        model_filename = settings.get_trend_model_path(args.symbol, args.timeframe, args.version)
        config_filename = model_filename.replace('.json', '_feature_config.json')

        # 儲存 XGBoost 模型
        print(f"\n--- 正在儲存「趨勢模型」... ---")
        model.save_model(model_filename)
        print(f"模型儲存完畢！({model_filename})")

        # 儲存空的 config (因為特徵工程是固定的)
        with open(config_filename, 'w') as f:
            json.dump({}, f, indent=4)
        print(f"✅ 特徵配置儲存完畢：{config_filename}")

    # 無論訓練或載入，都執行回測 (基於 args)
    backtest(model, df_features, features_list)

    # --- 新增：如果使用者要求，執行更真實的疊代回測 ---
    if args.show_equity:
        # 重新分割一次測試集以傳遞給新函式
        split_index = 0
        if train:
            split_index = int(len(df_features) * 0.8)

        df_test_data = df_features.iloc[split_index:]

        # 重新獲取預測結果 (因為 backtest 內部作用域限制)
        X_test_for_pred = df_test_data[features_list]
        y_pred_for_iter = model.predict(X_test_for_pred)

        iterative_backtest(df_test_data, y_pred_for_iter)