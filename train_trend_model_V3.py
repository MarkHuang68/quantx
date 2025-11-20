# 檔案: train_trend_model_V3_Integrated.py (基於 train_trend_model copy.py 流程)
# 【核心重構：流程維持不變，但模型邏輯已替換為 V3 的三分類 MTF DMI】

import os
import sys
# 假設外部庫皆已安裝，且路徑正確
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 

import pandas as pd
import numpy as np
import argparse
import xgboost as xgb
import warnings
import os
import json
import math
import uuid
from sklearn.metrics import accuracy_score, f1_score 
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, learning_curve
from sklearn.base import clone
from scipy.stats import uniform, randint
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
# from imblearn.over_sampling import SMOTE # 【已移除】V3 使用類別權重取代 SMOTE

# --- 1. 引用「設定檔」和「共用工具箱」 ---
import settings 
from settings import get_best_model_config, load_registry, save_registry
# 【修正】: 僅保留 fetch_data，特徵計算改為使用 V3 邏輯
from utils.common import fetch_data 
from hyperparameter_search import SearchIterator

# 【新增】: 外部庫檢查 (V3 核心依賴)
try:
    import talib
except ImportError:
    print("Warning: talib not installed. Some features will not be calculated.")
    class MockTalib:
        def __getattr__(self, name):
            return lambda *args, **kwargs: np.nan
    talib = MockTalib()

warnings.simplefilter(action='ignore', category=FutureWarning)

INITIAL_CAPITAL = 10000

# ⭐【保持 V1 參數】
PREDICTION_HORIZON = 2  
HOLD_THRESHOLD = 0.004  

train = False # (全域變數，用於控制載入模式)

# ⭐【V3 模型參數】: 三分類模型 (Multi-class)
XGB_BASE_PARAMS = {
    'objective': 'multi:softprob',     
    'num_class': 3,                    # 【V3 核心】: 三分類 (0=盤整, 1=大漲, 2=大跌)
    'eval_metric': 'mlogloss',         # 【V3 核心】: 評估指標改為多分類對數損失
    'n_estimators': 500, 
    'learning_rate': 0.014424625102595975, 
    'max_depth': 3,               
    'min_child_weight': 1,             # 【V3 核心】: 降至 1 (極低)，強迫模型對稀有事件敏感
    # 'gamma': 1.021874235023117,                 
    # 'subsample': 0.8,             
    # 'colsample_bytree': 0.8,      
    'n_jobs': -1,
    'random_state': 42,
    'early_stopping_rounds': 20,
    # 'reg_alpha': 5.805860121746745,
    # 'reg_lambda': 5.222669243390757,
}

# 【保持 V1 尋參範圍，但需要注意 num_class 的影響】
PARAM_DIST = {
    # 'max_depth': randint(1, 5),
    # 'learning_rate': uniform(0.01, 0.05),
    # 'min_child_weight': uniform(1, 10),
    # 'gamma': uniform(0.1, 1),
    # 'reg_lambda': uniform(1, 5),
    # 'reg_alpha': uniform(1, 5)
}

# ⭐【V3 特徵尋參空間】: 專注於 MTF DMI 參數
FEATURE_SEARCH_SPACE = {
    'dmi_len_ma_4h': [10, 30, 5], 
    'dmi_len_ma_1h': [8, 20, 4], 
    'dmi_len_ma_15m': [5, 15, 2],
    'rsi_period': [7, 21, 7],
}
FEATURE_FORMAT_TYPES = {k: 'range' for k in FEATURE_SEARCH_SPACE.keys()}


# --------------------- 【V3 核心函數：MTF DMI 特徵計算】 ---------------------

def calculate_dmi_hist(df, len_di, len_ma, len_adx_smooth=14):
    """計算 DMI 柱體、MA 線和 ADX 數值，並加入長度檢查。"""
    df_temp = df.copy()
    if len(df_temp) < len_di + 30: 
        for col in ['histValue', 'histMa', 'histMa_Up', 'histMa_BelowZero', 'ADX']: df_temp[col] = np.nan
        return df_temp
    
    plus_di  = talib.PLUS_DI(df_temp['High'], df_temp['Low'], df_temp['Close'], timeperiod=len_di)
    minus_di = talib.MINUS_DI(df_temp['High'], df_temp['Low'], df_temp['Close'], timeperiod=len_di)
    adx_val  = talib.ADX(df_temp['High'], df_temp['Low'], df_temp['Close'], timeperiod=len_di)
    
    df_temp['histValue'] = plus_di - minus_di
    alpha = 1 / len_ma
    df_temp['histMa'] = df_temp['histValue'].ewm(alpha=alpha, adjust=False).mean()
    df_temp['histMa_Up'] = (df_temp['histMa'] >= df_temp['histMa'].shift(1)).astype(float) # 轉為 float 避免 merge 問題
    df_temp['histMa_BelowZero'] = (df_temp['histMa'] < 0).astype(float)
    df_temp['ADX'] = adx_val
    return df_temp

def create_mtf_dmi_features(df_15m_raw, DMI_DI_LEN=14, **feature_params):
    """產生跨時間框架的 DMI 趨勢特徵並合併到 15M K 線上 (含 ADX)。"""
    df_15m = df_15m_raw.copy()
    
    df_4h = df_15m_raw.resample('4H').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
    df_4h = calculate_dmi_hist(df_4h, DMI_DI_LEN, feature_params.get('dmi_len_ma_4h', 10))
    
    df_1h = df_15m_raw.resample('1H').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
    df_1h = calculate_dmi_hist(df_1h, DMI_DI_LEN, feature_params.get('dmi_len_ma_1h', 10))
    
    df_15m = calculate_dmi_hist(df_15m, DMI_DI_LEN, feature_params.get('dmi_len_ma_15m', 10))

    # 合併 4H 特徵
    df_15m = df_15m.merge(
        df_4h[['histMa', 'histMa_Up', 'histMa_BelowZero', 'ADX']].shift(1).rename(
            columns={'histMa': '4H_Ma', 'histMa_Up': '4H_Ma_Up', 'histMa_BelowZero': '4H_Ma_DownTrend', 'ADX': '4H_ADX'}
        ), left_index=True, right_index=True, how='left'
    ).ffill()
    # 確保 bool 轉 float
    df_15m['4H_Ma_Up'] = df_15m['4H_Ma_Up'].astype(float)
    df_15m['4H_Ma_DownTrend'] = df_15m['4H_Ma_DownTrend'].astype(float)
    
    # 合併 1H 特徵
    df_15m = df_15m.merge(
        df_1h[['histMa', 'histMa_Up', 'histMa_BelowZero', 'ADX']].shift(1).rename(
            columns={'histMa': '1H_Ma', 'histMa_Up': '1H_Ma_Up', 'histMa_BelowZero': '1H_Ma_DownTrend', 'ADX': '1H_ADX'}
        ), left_index=True, right_index=True, how='left'
    ).ffill()
    df_15m['1H_Ma_Up'] = df_15m['1H_Ma_Up'].astype(float)
    df_15m['1H_Ma_DownTrend'] = df_15m['1H_Ma_DownTrend'].astype(float)
    
    df_15m['15M_Ma_Delta'] = df_15m['histMa'] - df_15m['histMa'].shift(1)

    mtf_features = [
        '4H_Ma', '4H_Ma_Up', '4H_Ma_DownTrend', '4H_ADX',
        '1H_Ma', '1H_Ma_Up', '1H_Ma_DownTrend', '1H_ADX',
        'histValue', 'histMa', 'histMa_Up', '15M_Ma_Delta', 'ADX',
    ]
    return df_15m, mtf_features

def create_features_trend(df_raw, **feature_params):
    """產生所有特徵並合併 MTF 邏輯 (V3 邏輯)。"""
    if df_raw is None: return None, None
    df = df_raw.copy()
    
    # 1. 執行 MTF DMI 特徵計算
    try:
        df, mtf_features = create_mtf_dmi_features(df, **feature_params)
    except Exception as e:
        print(f"❌ MTF DMI 特徵計算失敗: {e}")
        mtf_features = []

    # 2. 傳統/衍生品特徵計算
    close_prices = df['Close']
    
    # EMA 結構特徵 (用於捕捉反轉風險)
    ema_s, ema_m, ema_l = talib.EMA(close_prices, 10), talib.EMA(close_prices, 30), talib.EMA(close_prices, 60)
    df['CLOSE_EMA_L'] = (close_prices - ema_l) / ema_l
    df['EMA_M_EMA_L'] = (ema_m - ema_l) / ema_l
    
    # RSI
    df['RSI'] = talib.RSI(close_prices, timeperiod=feature_params.get('rsi_period', 14))
    
    # 衍生品特徵 (FundingRate)
    if 'FundingRate' in df.columns:
        df['FR_ROC'] = df['FundingRate'].pct_change().replace([np.inf, -np.inf], np.nan)
        df['FR_ABS'] = df['FundingRate'] * 1000 
        df['FR_ROC'] = df['FR_ROC'].fillna(0.0) 
        df['FR_ABS'] = df['FR_ABS'].fillna(0.0)
    else:
        df['FR_ROC'], df['FR_ABS'] = 0.0, 0.0

    # 3. 組合最終特徵列表
    features_list = [
        *mtf_features, 
        'CLOSE_EMA_L', 'EMA_M_EMA_L', 'RSI', 
        'FR_ROC', 'FR_ABS'
    ]
    
    features_list = list(set(features_list))
    df_features = df.dropna(subset=features_list)
    return df_features, features_list

# --------------------- 【V3 核心函數：三分類訓練 (含類別權重)】 ---------------------

def train_xgb_classifier(df_train, df_val, features_list, 
                         base_model_params, model_param_dist=None, no_search_model=False, show_overfit=False):
    """
    訓練 XGBoost 三分類模型 (1=大漲, 2=大跌, 0=盤整)，並應用類別權重。
    """
    if df_train is None or df_val is None: return None, 0.0, 0.0

    # --- 1. 準備數據 (Target) ---
    def prepare_xy(df):
        df_model = df.copy()
        
        df_model['future_close'] = df_model['Close'].shift(-PREDICTION_HORIZON)
        df_model['future_return'] = (df_model['future_close'] - df_model['Close']) / df_model['Close']
        
        df_model['target'] = 0 # 預設為盤整 (Hold)
        df_model.loc[df_model['future_return'] > HOLD_THRESHOLD, 'target'] = 1 # 大漲
        df_model.loc[df_model['future_return'] < -HOLD_THRESHOLD, 'target'] = 2 # 大跌
        
        df_model = df_model.dropna(subset=features_list + ['target', 'future_return'])
        
        X = df_model[features_list]
        y = df_model['target']
        return X, y

    X_train_all, y_train_all = prepare_xy(df_train)
    X_val_all, y_val_all = prepare_xy(df_val)

    # 【V3 修正】：不再使用欠採樣 (SMOTE)，直接使用全量數據 (0, 1, 2)
    X_train_resampled = X_train_all
    y_train_resampled = y_train_all
    
    if X_train_resampled.empty or y_train_resampled.empty:
        print("❌ 錯誤：訓練集在特徵處理後為空，無法訓練模型。")
        return None, 0.0, 0.0
    
    X_val_filtered, y_val_filtered = X_val_all, y_val_all
    eval_set = [(X_val_filtered, y_val_filtered)]

    # --- 2. 關鍵修正：計算類別權重 (用於解決不平衡問題) ---
    class_counts = y_train_resampled.value_counts()
    for cls in [0, 1, 2]:
        if cls not in class_counts:
            class_counts[cls] = 1 # 避免除以零
            
    max_samples = class_counts.max()
    # 權重計算公式：MaxSamples / ClassCount
    class_weights = {cls: max_samples / count for cls, count in class_counts.items()}
    sample_weights = y_train_resampled.map(class_weights)
    
    # 3. 超參數調校或直接訓練
    xgb_clf = xgb.XGBClassifier(**base_model_params)
    model = None # 確保 model 有定義

    if len(y_train_resampled) == 0:
        print("錯誤: 欠採樣後沒有 (1/2) 樣本，無法訓練！")
        return None, 0.0, 0.0

    if not no_search_model:
        # (此為「新 階段一」: 模型尋參)
        print("\n--- 開始超參數調校 (三分類模型...) ---")
        xgb_clf.early_stopping_rounds = None # 尋參時關閉 early_stopping
        tscv = TimeSeriesSplit(n_splits=2)
        random_search = RandomizedSearchCV(
            estimator=xgb_clf,
            param_distributions=model_param_dist,
            n_iter=25,
            cv=tscv,
            # 【修正】: 使用 'f1_weighted' 進行三分類評估
            scoring='f1_weighted', 
            n_jobs=4,
            verbose=2,
            random_state=42
        )
        # 【修正】：訓練時帶入 sample_weight
        random_search.fit(X_train_resampled, y_train_resampled, sample_weight=sample_weights) 
        
        print("\n--- 最佳參數 (三分類模型) ---")
        print(random_search.best_params_)
        print(f"平均交叉驗證 Weighted F1-Score (調參模式): {random_search.best_score_:.4f}")
        model = random_search.best_estimator_
    else:
        # (此為「新 階段二」: 特徵尋參/直接訓練)
        print("\n--- 開始直接訓練 (使用最佳模型參數)... ---")
        # 【修正】：訓練時帶入 sample_weight
        model = xgb_clf.fit(X_train_resampled, y_train_resampled, sample_weight=sample_weights, eval_set=eval_set, verbose=False)

    # --- 4. 計算 Train F1 vs Val F1 ---
    y_pred_train = model.predict(X_train_resampled)
    # 【修正】：F1-Score 計算使用 average='weighted'
    train_f1 = f1_score(y_train_resampled, y_pred_train, average='weighted') if len(y_train_resampled) > 0 else 0.0
        
    y_pred_val_all = model.predict(X_val_all)
    # 【修正】：F1-Score 計算使用 average='weighted'
    val_f1 = f1_score(y_val_filtered, y_pred_val_all, average='weighted') if len(y_val_filtered) > 0 else 0.0

    # ( ... 繪圖/打印邏輯保持不變 ... )
    if not no_search_model and model is not None:
        print("\n--- 特徵重要性排行 (預設特徵) ---")
        feature_importance = sorted(zip(features_list, model.feature_importances_), key=lambda x: x[1], reverse=True)
        for feat, imp in feature_importance:
            print(f"{feat}: {imp:.4f}")

    if show_overfit and not no_search_model and model is not None: 
        print("\n--- 過擬合檢測: 學習曲線 ---")
        # 由於 RandomizedSearchCV 可能會產生新參數，為確保能使用新模型，這裡使用 clone
        temp_model = clone(model) 
        train_sizes, train_scores, val_scores = learning_curve(
            temp_model, X_train_resampled, y_train_resampled, cv=TimeSeriesSplit(n_splits=3), 
            scoring='f1_weighted', n_jobs=-1,
            # 【重要】: 傳遞 sample_weight 給 learning_curve
            fit_params={'sample_weight': sample_weights} 
        )
        train_acc = np.mean(train_scores, axis=1)
        val_acc = np.mean(val_scores, axis=1)
        plt.rc('font', family='MingLiu')
        plt.figure(figsize=(8, 6))
        plt.plot(train_sizes, train_acc, label='訓練 Weighted F1')
        plt.plot(train_sizes, val_acc, label='驗證 Weighted F1')
        plt.title('學習曲線 (Weighted F1-Score)')
        plt.xlabel('訓練樣本數')
        plt.ylabel('Weighted F1-Score')
        plt.legend()
        plt.grid(True)
        plt.show()

    return model, train_f1, val_f1


# (calculate_sharpe_ratio 函數保持不變)
def calculate_sharpe_ratio(strategy_net_equity, timeframe):
    """
    計算「年化夏普比率」
    """
    daily_equity = strategy_net_equity.resample('D').last()
    daily_returns = daily_equity.pct_change().dropna()
    
    if daily_returns.empty or daily_returns.std() == 0 or np.isnan(daily_returns.std()):
        return 0.0 
        
    sharpe_ratio = daily_returns.mean() / daily_returns.std()
    annualized_sharpe_ratio = sharpe_ratio * np.sqrt(365)
    
    return annualized_sharpe_ratio


# --------------------- 【V3 核心函數：三分類向量回測】 ---------------------

def vector_backtest(model, df_features, features_list, conf_buy, conf_sell, args):
    """
    向量化回測 (三分類模型)。
    """
    df_model = df_features.copy()
    
    # 1. 數據和目標準備 (三分類)
    df_model['future_close'] = df_model['Close'].shift(-PREDICTION_HORIZON)
    df_model['future_return'] = (df_model['future_close'] - df_model['Close']) / df_model['Close']
    
    df_model['target'] = 0 # 預設為盤整
    df_model.loc[df_model['future_return'] > HOLD_THRESHOLD, 'target'] = 1 # 大漲
    df_model.loc[df_model['future_return'] < -HOLD_THRESHOLD, 'target'] = 2 # 大跌
    df_model = df_model.dropna(subset=features_list + ['target', 'future_return']) 

    # --- 特徵對齊：確保 X 的欄位順序與模型預期一致 ---
    # 取得模型訓練時的 feature names（可能存在於 model.feature_names 或 model.get_booster().feature_names）
    model_feature_names = None
    if hasattr(model, 'feature_names') and model.feature_names:
        model_feature_names = list(model.feature_names)
    else:
        try:
            booster = model.get_booster()
            if hasattr(booster, 'feature_names') and booster.feature_names:
                model_feature_names = list(booster.feature_names)
        except Exception:
            model_feature_names = None

    # 如果無法取得 model 的 feature_names，就使用當前傳入的 features_list（最保守）
    if not model_feature_names:
        model_feature_names = list(features_list)

    # 在 df_model 中補上 model 需要但不存在的欄位（以 0 填滿）
    for col in model_feature_names:
        if col not in df_model.columns:
            df_model[col] = 0.0

    # 丟棄 df_model 中模型不需要的多餘欄位，並把欄位排序成 model_feature_names
    X = df_model.reindex(columns=model_feature_names, copy=False)

    # 若仍有 NaN（理論上不該有），以 0 填補以免 XGBoost 拋錯
    X = X.fillna(0.0)

    y = df_model['target']

    if model is None or not hasattr(model, 'predict_proba'):
        print("錯誤: 模型未訓練或無法執行 predict_proba。")
        return INITIAL_CAPITAL, 0.0

    # 現在安全地呼叫 predict_proba（欄位順序已對齊）
    y_prob = model.predict_proba(X)
    
    # 【V3 修正】: Class 1 (漲) = prob_buy, Class 2 (跌) = prob_sell
    prob_buy = y_prob[:, 1] 
    prob_sell = y_prob[:, 2] 
    
    CONF_THRESH_BUY = conf_buy
    CONF_THRESH_SELL = conf_sell
    
    # 2. 產生原始訊號
    df_model['signal'] = 0 
    df_model.loc[prob_buy > CONF_THRESH_BUY, 'signal'] = 1  # Buy
    df_model.loc[prob_sell > CONF_THRESH_SELL, 'signal'] = -1 # Sell (Short)

    # 3. 計算回報 (時序修正：訊號 shift(1) 應用於下一根 K 線)
    actual_return = df_model['Close'].pct_change()
    
    df_model['strategy_return'] = df_model['signal'].shift(1).fillna(0) * actual_return
    df_model['signal_diff'] = df_model['signal'].diff().abs().fillna(0)
    df_model['transaction_costs'] = df_model['signal_diff'] * settings.FEE_RATE
    df_model['strategy_net_return'] = df_model['strategy_return'] - df_model['transaction_costs']
    strategy_net_equity = (1 + df_model['strategy_net_return']).cumprod() * INITIAL_CAPITAL
    bh_equity = (1 + actual_return).cumprod() * INITIAL_CAPITAL

    # ( ... 繪圖邏輯 ... )
    plt.rc('font', family='MingLiu') 
    if args.show_equity:
        plt.figure(figsize=(12, 6))
        plt.plot(df_model.index, strategy_net_equity, label='策略 (扣費後)', color='red')
        plt.plot(df_model.index, bh_equity, label='Buy & Hold', color='gray')
        plt.title(f'向量回測淨值曲線 ({args.timeframe}) - CONF={conf_buy:.2f}/{conf_sell:.2f}')
        plt.xlabel('時間')
        plt.ylabel('淨值 (USDT)')
        plt.legend()
        plt.grid(True)
        plt.show()

    if args.show_confidence:
        # 【V3 修正】: 繪製三分類混淆矩陣
        y_true = df_model['target'] 
        y_pred_class = np.argmax(y_prob, axis=1) # 模型預測的類別 (0, 1, 2)
        
        labels_map = {0: '盤整 (0)', 1: '大漲 (1)', 2: '大跌 (2)'}
        y_true_mapped = y_true.map(labels_map).fillna('N/A')
        y_pred_mapped = pd.Series(y_pred_class).map(labels_map).values
        
        valid_labels = ['盤整 (0)', '大漲 (1)', '大跌 (2)']
        
        print(f"\n--- 回測訊號 混淆矩陣 (CONF={conf_buy:.2f}/{conf_sell:.2f}) ---")
        if len(y_true_mapped) > 0:
            ConfusionMatrixDisplay.from_predictions(y_true_mapped, y_pred_mapped, 
                                                labels=valid_labels, 
                                                normalize='true',
                                                values_format='.2%')
            plt.title(f'三分類混淆矩陣 (預測 vs 真實) - CONF={conf_buy:.2f}/{conf_sell:.2f}')
            plt.show()
        else:
            print("無有效的樣本可供繪製混淆矩陣。")
    
    # ( ... 回傳 ... )
    final_equity = INITIAL_CAPITAL
    sharpe_ratio = 0.0
    if not strategy_net_equity.empty:
        final_equity = strategy_net_equity.iloc[-1]
        sharpe_ratio = calculate_sharpe_ratio(strategy_net_equity, args.timeframe) 
    return final_equity, sharpe_ratio


# --- 【主執行區塊：if __name__ 流程 (保持 V1 結構)】 ---
# --- 【F. 主執行區塊：最終完整版 (補齊所有尋參邏輯與流程)】 ---

if __name__ == "__main__":
    
    # --- 0. 參數解析與路徑設定 ---
    parser = argparse.ArgumentParser(description=f'訓練 XGBoost 趨勢模型')
    parser.add_argument('-s', '--symbol', type=str, required=True)
    parser.add_argument('-tf', '--timeframe', type=str, required=True)
    parser.add_argument('-sd', '--start', type=str)
    parser.add_argument('-ed', '--end', type=str)
    parser.add_argument('-l', '--limit', type=int)
    parser.add_argument('-nsm', '--no_search_model', action='store_true')
    parser.add_argument('-nsp', '--no_search_params', action='store_true')
    parser.add_argument('-nsc', '--no_search_conf', action='store_true')
    parser.add_argument('--show_equity', action='store_true')
    parser.add_argument('--show_confidence', action='store_true')
    parser.add_argument('--show_overfit', action='store_true')
    parser.add_argument('--force_train', action='store_true')
    parser.add_argument('--search_type', type=str, default='random')
    parser.add_argument('--n_iter', type=int, default=25)

    parser.add_argument('--force_save', action='store_true')
    args = parser.parse_args()

    symbol_safe = args.symbol.replace('/', '')
    symbol_tf_key = f"{symbol_safe}_{args.timeframe}" 
    model_dir = os.path.join(settings.MODEL_DIR, symbol_safe)
    os.makedirs(model_dir, exist_ok=True) 

    # --- 1. 載入資料 ---
    raw_df = fetch_data(symbol=args.symbol, start_date=args.start, end_date=args.end, timeframe=args.timeframe, total_limit=args.limit)
    if raw_df is None or raw_df.empty:
        print("❌ 錯誤: 無法獲取資料。")
        sys.exit(1)
    start_date_used = raw_df.index.min().strftime('%Y-%m-%d')
    end_date_used = raw_df.index.max().strftime('%Y-%m-%d')
    
    # --- 2. 初始化/載入判斷 ---
    registry = load_registry()
    current_best_config = get_best_model_config(args.symbol, args.timeframe) 
    
    train = False
    model = None
    df_features = None
    features_list = None
    best_feature_params = {}
    best_conf_buy = 0.51
    best_conf_sell = 0.51
    objective_sharpe = 0.0
    best_acc = 0.0 
    best_train_f1 = 0.0 
    best_sharpe = -float('inf')

    # 暫存原始繪圖設定
    original_show_equity = args.show_equity
    original_show_confidence = args.show_confidence

    if current_best_config and not args.force_train:
        # --- PATH A: 載入模式 (Load Mode) ---
        print(f"--- 模式: 載入 (Load Mode) ---")
        
        model_filename = current_best_config['model_file']
        config_filename = current_best_config['config_file']
        
        try:
            model = xgb.XGBClassifier()
            model.load_model(model_filename)
            model.n_classes_ = XGB_BASE_PARAMS['num_class'] 
            with open(config_filename, 'r') as f:
                best_feature_params = json.load(f)
            
            df_features, features_list = create_features_trend(raw_df.copy(), **best_feature_params)
            
            best_conf_buy = current_best_config.get('reference_conf_buy', 0.51)
            best_conf_sell = current_best_config.get('reference_conf_sell', 0.51)
            
            print(f"✅ 成功載入模型。參考門檻: {best_conf_buy:.2f}/{best_conf_sell:.2f}")

        except Exception as e:
            print(f"❌ 載入模型失敗: {e}，強制進入訓練模式。")
            args.force_train = True
    
    if args.force_train or (current_best_config is None and not train):
        # --- PATH B: 訓練模式 (Train Mode) ---
        train = True
        print(f"--- 模式: 訓練 (Train Mode) ---")
        
        # 尋參過程關閉繪圖
        args.show_equity = False
        args.show_confidence = False
        
        # --- 階段一: 模型尋參 (使用預設特徵) ---
        print("\n--- 階段一: 正在尋找最佳「模型參數」 (使用預設特徵) ---")
        
        default_feature_params = {k: v[0] for k, v in FEATURE_SEARCH_SPACE.items()}
        df_features_default, features_list_default = create_features_trend(raw_df.copy(), **default_feature_params)
        
        if df_features_default is None or features_list_default is None: 
            print("❌ 錯誤: 無法產生預設特徵。")
            sys.exit(1)
            
        train_split_idx = int(len(df_features_default) * 0.6)
        val_split_idx = int(len(df_features_default) * 0.8)
        df_train = df_features_default.iloc[:train_split_idx]
        df_val = df_features_default.iloc[train_split_idx:val_split_idx]

        best_model_step_1, train_f1_step_1, acc_step_1 = train_xgb_classifier(
            df_train, df_val, features_list_default,
            model_param_dist=PARAM_DIST,      
            base_model_params=XGB_BASE_PARAMS, 
            no_search_model=args.no_search_model, 
            show_overfit=args.show_overfit
        )
        if best_model_step_1 is None: sys.exit(1)
            
        best_model_params = best_model_step_1.get_params()
        print(f"\n--- 階段一 完成: 找到最佳模型參數 (Val Weighted F1 {acc_step_1*100:.2f}%) ---")

        # --- 階段二: 特徵尋參 (使用最佳模型) ---
        print(f"\n--- 階段二: 正在使用「最佳模型」尋找最佳「特徵參數」 ---")
        
        best_acc_step_2 = acc_step_1
        best_train_f1_step_2 = train_f1_step_1
        best_feature_params = default_feature_params 
        best_model_step_2 = best_model_step_1
        final_features_list = features_list_default 
        df_val_for_phase3 = df_val.copy()
        df_test = df_features_default.iloc[val_split_idx:].copy()

        iterator = SearchIterator(FEATURE_SEARCH_SPACE, search_type=args.search_type, n_iter=args.n_iter, format_types=FEATURE_FORMAT_TYPES)
        
        # 只有在模型尋參被開啟時，才執行特徵尋參迴圈
        if not args.no_search_params:
            for config in iterator:
                
                df_features_full, features_list_2 = create_features_trend(raw_df.copy(), **config)
                if df_features_full is None or features_list_2 is None:
                    continue
                    
                train_split_idx = int(len(df_features_full) * 0.6)
                val_split_idx = int(len(df_features_full) * 0.8)
                df_train_current = df_features_full.iloc[:train_split_idx]
                df_val_current = df_features_full.iloc[train_split_idx:val_split_idx]

                # 使用階段一的最佳模型參數，跳過尋參
                model_step_2_current, train_f1_2, acc_2 = train_xgb_classifier(
                    df_train_current, df_val_current, features_list_2,
                    model_param_dist=None,
                    base_model_params=best_model_params, 
                    no_search_model=args.no_search_model,             
                    show_overfit=False
                )

                if acc_2 == 0.0 or model_step_2_current is None:
                    continue

                print(f"配置 F1 (Train/Val): {train_f1_2 * 100:.2f}% / {acc_2 * 100:.2f}%")

                if acc_2 > best_acc_step_2:
                    best_acc_step_2 = acc_2
                    best_train_f1_step_2 = train_f1_2
                    best_feature_params = config 
                    best_model_step_2 = model_step_2_current
                    final_features_list = features_list_2
                    
                    df_val_for_phase3 = df_val_current.copy()
                    df_test = df_features_full.iloc[val_split_idx:].copy()

            if not best_feature_params:
                print("錯誤: 階段二 未找到任何有效的特徵組合。")
                sys.exit(1)
                    
            print(f"--- 階段二 完成: 最佳特徵 {best_feature_params} (Val Weighted F1-Score {best_acc_step_2*100:.2f}%) ---")
        else:
            print(f"--- 階段二 完成: 已跳過特徵尋參 (--no_search_params)。使用預設特徵。")
        
        # 賦值給最終變數
        best_model = best_model_step_2
        best_train_f1 = best_train_f1_step_2
        best_acc = best_acc_step_2 
        features_list = final_features_list

        # 重新計算 100% 數據的特徵 (以確保 df_test, df_val_for_phase3 匹配最佳特徵)
        df_features_full, features_list = create_features_trend(raw_df.copy(), **best_feature_params)
        val_split_idx = int(len(df_features_full) * 0.8)
        # 確保 df_val_for_phase3 和 df_test 匹配最終的最佳特徵
        df_val_for_phase3 = df_features_full.iloc[train_split_idx:val_split_idx].copy()
        df_test = df_features_full.iloc[val_split_idx:].copy()
        
        
        # --- 階段三: 信心門檻尋參 ---
        if args.no_search_conf:
             print("\n--- 階段三: (已跳過) 使用預設信心門檻 0.51 / 0.51 ---")
             best_equity_at_best_sharpe, best_sharpe = vector_backtest(best_model, df_val_for_phase3.copy(), features_list, best_conf_buy, best_conf_sell, args)
        else:
            print("\n--- 階段三: 正在尋找最佳「非對稱」信心門檻 (在「驗證集」上評估「夏普比率」) ---")
            
            best_sharpe = -float('inf') 
            best_equity_at_best_sharpe = 0 
            best_conf_buy = 0.51
            best_conf_sell = 0.51
            
            thresholds_buy = np.arange(0.50, 0.76, 0.02) 
            thresholds_sell = np.arange(0.50, 0.76, 0.02) 
            
            for conf_b in thresholds_buy:
                for conf_s in thresholds_sell:
                    conf_buy = round(conf_b, 2)
                    conf_sell = round(conf_s, 2)
                    
                    final_equity, sharpe_ratio = vector_backtest(best_model, df_val_for_phase3.copy(), features_list, conf_buy, conf_sell, args)
                    
                    if sharpe_ratio > best_sharpe:
                        best_sharpe = sharpe_ratio
                        best_equity_at_best_sharpe = final_equity
                        best_conf_buy = conf_buy
                        best_conf_sell = conf_sell

            print(f"--- 階段三 完成: 找到最佳門檻 {best_conf_buy:.2f}/{best_conf_sell:.2f} (驗證集最高夏普: {best_sharpe:.2f}) ---")
        
        # --- 【階段四: 客觀測試集評估 (強制繪圖)】 ---
        print(f"\n--- 階段四: 正在使用 (最佳模型/門檻) 在「客觀測試集」上執行最終評估 ---")
        
        # 暫時強制開啟繪圖
        args.show_confidence = True
        args.show_equity = True
        
        objective_equity, objective_sharpe = vector_backtest(
            best_model, df_test.copy(), features_list, best_conf_buy, best_conf_sell, args
        )
        
        # 恢復原始繪圖設定
        args.show_confidence = original_show_confidence
        args.show_equity = original_show_equity
        
        print(f"--- 客觀分數 (測試集): 淨值: {objective_equity:.0f} USDT, 夏普: {objective_sharpe:.2f} ---")
        
        # --- 【核心：模型儲存與註冊邏輯】 ---
        
        current_best_f1 = registry.get(symbol_tf_key, {}).get('best_f1_score', 0.0)
        
        if (best_acc > current_best_f1) or args.force_save:
            
            if args.force_save and not (best_acc > current_best_f1):
                print(f"⚠️ 警告: 強制儲存 (--force_save) 啟動。")
            else:
                print(f"✅ 新紀錄！Val Weighted F1 ({best_acc:.4f}) > 目前最佳 ({current_best_f1:.4f})。正在儲存...")
            
            # ( 垃圾桶邏輯 )
            if symbol_tf_key in registry and 'model_file' in registry[symbol_tf_key]:
                old_model_config = registry[symbol_tf_key]
                old_model_file = old_model_config['model_file']
                old_config_file = old_model_config['config_file']
                trash_dir = os.path.join(model_dir, 'trash')
                os.makedirs(trash_dir, exist_ok=True)
                try:
                    if os.path.exists(old_model_file):
                        trash_model_path = os.path.join(trash_dir, os.path.basename(old_model_file))
                        os.rename(old_model_file, trash_model_path)
                        print(f"♻️ 舊模型已移至: {trash_model_path}")
                    if os.path.exists(old_config_file):
                        trash_config_path = os.path.join(trash_dir, os.path.basename(old_config_file))
                        os.rename(old_config_file, trash_config_path)
                        print(f"♻️ 舊配置已移至: {trash_config_path}")
                except Exception as e:
                    print(f"⚠️ 警告: 移動舊模型至垃圾桶時發生錯誤: {e}")
            
            # 儲存新模型
            unique_id = uuid.uuid4().hex[:6]
            model_filename = os.path.join(model_dir, f"model_{unique_id}.json")
            config_filename = os.path.join(model_dir, f"model_{unique_id}_feature_config.json")
            
            best_model.save_model(model_filename)
            print(f"模型儲存完畢！({model_filename})")
            
            with open(config_filename, 'w') as f:
                json.dump(best_feature_params, f, indent=4)
            print(f"✅ 特徵配置儲存完畢：{config_filename}")
            
            # 更新 Registry
            registry[symbol_tf_key] = {
                'best_f1_score': best_acc, 
                'train_f1_score': best_train_f1,
                'reference_sharpe_ratio': best_sharpe, 
                'objective_sharpe_ratio': objective_sharpe,
                'reference_equity': best_equity_at_best_sharpe if 'best_equity_at_best_sharpe' in locals() else 0.0,
                'reference_conf_buy': best_conf_buy, 
                'reference_conf_sell': best_conf_sell,
                'model_file': model_filename,
                'config_file': config_filename,
                'feature_params': best_feature_params,
                'start_date': start_date_used, 
                'end_date': end_date_used,   
                'last_updated': pd.Timestamp.now().isoformat()
            }
            save_registry(registry)
            print(f"✅ 最佳模型紀錄已更新至: {settings.REGISTRY_FILE}")
            
        else:
             print(f"✅ 訓練完成。但 Val Weighted F1 ({best_acc:.4f}) 未超過目前最佳紀錄 ({current_best_f1:.4f})。不儲存。")
             print(f"  (客觀測試集 SR 為: {objective_sharpe:.2f})")
        
        # 確保 model 變數指向最新的最佳模型，並更新 feature_names (無論是否儲存)
        model = best_model
        df_features = df_features_full 
        try: model.feature_names = features_list 
        except AttributeError: pass
    
    # --- 3. 最終回測報告 ---
    if model is None or df_features is None or features_list is None:
        print("❌ 錯誤：模型或資料未能成功載入/訓練，無法執行回測。")
        sys.exit(1)

    if train:
        # 【訓練模式結束】: 已經在 Test Set 跑過一次，只印出最終報告
        print(f"\n✅ 訓練模式完成。最終報告已基於客觀測試集 (Test Set) 的夏普 {objective_sharpe:.2f}。")
    else:
        # 【載入模式結束】: 執行 100% 數據回測
        print(f"\n--- 執行最終回測 (在 100% 資料集上，使用參考門檻 {best_conf_buy:.2f}/{best_conf_sell:.2f}) ---")
        
        # 使用使用者原本指定的繪圖旗標
        args.show_equity = original_show_equity
        args.show_confidence = original_show_confidence

        final_equity, final_sharpe = vector_backtest(model, df_features.copy(), features_list, best_conf_buy, best_conf_sell, args)
        print(f"最終回測 (100% 資料) 淨值: {final_equity:.0f} USDT, 最終夏普: {final_sharpe:.2f}")