# 檔案: validate_models.py

import os
import json
import argparse
from datetime import datetime, timedelta

import pandas as pd

from core.backtest_engine import BacktestEngine
from core.context import Context
from core.portfolio import Portfolio
from core.exchange import PaperExchange
from strategies.xgboost_trend_strategy import XGBoostTrendStrategy
from settings import SYMBOLS_TO_TRADE, TREND_MODEL_VERSION

DATA_DIR = "data"

# --- 模型上線的品質標準 (Production Gate Criteria) ---
VALIDATION_CRITERIA = {
    "sharpe_ratio": 1.0,
    "max_drawdown": 0.20, # 最大允許回撤 20%
    "profit_factor": 1.5,
    "total_pnl_pct": 0.05 # 樣本外測試中至少要有 5% 的獲利
}

PRODUCTION_MODELS_FILE = "production_models.json"

def get_latest_model_path(model_dir, model_prefix):
    """找到一個目錄中最新的模型檔案。"""
    files = [f for f in os.listdir(model_dir) if f.startswith(model_prefix) and f.endswith('.zip')]
    if not files:
        return None
    # 假設檔案名包含時間戳，按檔名排序即可找到最新的
    return os.path.join(model_dir, sorted(files)[-1])

def run_validation_backtest(symbol, xgb_model_path, ppo_model_path, validation_df):
    """對單一幣種執行標準化的驗證回測。"""
    print(f"\n--- 正在對 {symbol} 執行驗證回測 ---")

    initial_capital = 10000
    exchange = PaperExchange(validation_df)
    portfolio = Portfolio(initial_capital)
    context = Context(exchange, portfolio, initial_capital)

    strategy = XGBoostTrendStrategy(
        context,
        symbols=[symbol],
        timeframe='1m',
        use_ppo=True,
        ppo_model_path=ppo_model_path
    )

    # 使用新的 set_model 方法，動態載入要驗證的 XGB 模型
    model_set_successfully = strategy.set_model(symbol, xgb_model_path)
    if not model_set_successfully:
        raise RuntimeError(f"無法為 {symbol} 設定指定的 XGBoost 模型。")

    # 注意：回測引擎需要傳入原始數據，而不是已經計算好特徵的數據
    backtest_data = {symbol: validation_df}
    backtest = BacktestEngine(context, strategy, backtest_data)
    backtest.run()

    report = portfolio.generate_performance_report()
    return report

def validate_and_promote_models():
    """
    主函數：執行模型驗證，並在通過時將其提升為生產模型。
    """
    print("\n=======================================================")
    print(f"--- 開始執行模型自動化驗證與上線流程 ---")
    print(f"=======================================================")

    # 1. 載入當前的生產模型清單 (如果存在)
    try:
        with open(PRODUCTION_MODELS_FILE, 'r') as f:
            current_production_models = json.load(f)
    except FileNotFoundError:
        current_production_models = {}
        print("--- 找不到現有的生產模型清單，將創建新的。 ---")

    # 2. 找到最新的統一 PPO 模型
    latest_ppo_model_path = get_latest_model_path("ppo_models", "ppo_agent_UNIFIED")
    if not latest_ppo_model_path:
        print("🛑 錯誤：找不到任何已訓練的 PPO 模型。")
        return

    new_production_models = current_production_models.copy()
    all_symbols_passed = True

    # 3. 遍歷所有幣種，驗證其最新的 XGB 模型
    for symbol in SYMBOLS_TO_TRADE:
        symbol_str = symbol.replace('/', '')
        xgb_model_path = os.path.join("models", f"trend_model_{symbol_str}_1m_v{TREND_MODEL_VERSION}.json")

        if not os.path.exists(xgb_model_path):
            print(f"🛑 警告：找不到 {symbol} 的 XGBoost 模型，跳過驗證。")
            all_symbols_passed = False
            continue

        # 4. 準備驗證數據 (例如，過去 30 天的數據)
        # 這裡需要一個方法來獲取最新的數據作為樣本外數據
        # 暫時假設我們有一個完整的數據檔案，並從中切分出最後 30 天
        full_data_path = os.path.join(DATA_DIR, f"{symbol_str}_1m.csv")
        if not os.path.exists(full_data_path):
            print(f"🛑 警告：找不到 {symbol} 的數據檔案，無法進行驗證。")
            all_symbols_passed = False
            continue

        full_df = pd.read_csv(full_data_path, index_col='Date', parse_dates=True)
        validation_start_date = full_df.index.max() - timedelta(days=30)
        validation_df = full_df[full_df.index >= validation_start_date]

        if len(validation_df) < 100: # 確保有足夠的數據進行驗證
             print(f"🛑 警告：{symbol} 的驗證數據不足 (< 100 筆)，跳過。")
             all_symbols_passed = False
             continue

        # 5. 執行回測
        try:
            report = run_validation_backtest(symbol, xgb_model_path, latest_ppo_model_path, validation_df)
        except Exception as e:
            print(f"🛑 {symbol} 的回測過程中發生錯誤: {e}")
            all_symbols_passed = False
            continue

        # 6. 檢查是否滿足上線標準
        passed = True
        print(f"--- {symbol} 驗證結果 ---")
        for key, threshold in VALIDATION_CRITERIA.items():
            value = report.get(key, 0)
            check = "✅" if value >= threshold else "❌"
            if check == "❌":
                passed = False
            print(f"  - {key}: {value:.4f} (要求: >= {threshold}) {check}")

        if passed:
            print(f"✅ {symbol} 的新模型已通過品質檢定！")
            # 可以在這裡加入與舊模型比較，只有更好才更新的邏輯
            new_production_models[symbol] = {
                "xgb_model_path": xgb_model_path,
                "updated_at": datetime.now().isoformat()
            }
        else:
            print(f"❌ {symbol} 的新模型未能通過品質檢定，將不會上線。")
            all_symbols_passed = False

    # 7. 如果所有模型都驗證通過，則更新統一 PPO 模型
    if all_symbols_passed:
        print("\n✅ 所有幣種的 XGB 模型均已通過驗證。")
        new_production_models["UNIFIED_PPO"] = {
            "ppo_model_path": latest_ppo_model_path,
            "updated_at": datetime.now().isoformat()
        }
    else:
        print("\n❌ 由於部分 XGB 模型未能通過驗證，統一 PPO 模型本次將不會更新。")


    # 8. 寫入新的生產模型清單
    with open(PRODUCTION_MODELS_FILE, 'w') as f:
        json.dump(new_production_models, f, indent=4)

    print(f"\n✅ 模型驗證與上線流程完成！生產模型清單已更新：{PRODUCTION_MODELS_FILE}")

if __name__ == '__main__':
    validate_and_promote_models()
