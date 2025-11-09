# 檔案: strategies/xgboost_trend_strategy.py

import pandas as pd
import numpy as np
import xgboost as xgb

from strategies.base_strategy import BaseStrategy
from utils.common import create_features_trend
from settings import SYMBOLS_TO_TRADE, TREND_MODEL_VERSION, get_trend_model_path
from core.ppo_manager import PPOManager

class XGBoostTrendStrategy(BaseStrategy):
    def __init__(self, context, symbols=SYMBOLS_TO_TRADE, timeframe='1m', use_ppo=False, ppo_model_path=None):
        super().__init__(context)
        self.symbols = symbols
        self.timeframe = timeframe
        self.use_ppo = use_ppo
        self.models = {}
        self._load_models()

        if self.use_ppo:
            if not ppo_model_path:
                raise ValueError("使用 PPO 時，必須提供 PPO 模型路徑")
            self.ppo_managers = {
                symbol: PPOManager(
                    model_path=ppo_model_path,
                    symbol=symbol,
                    timeframe=self.timeframe,
                    version=TREND_MODEL_VERSION
                ) for symbol in self.symbols
            }
            # DEBUG: 打印初始化時的 ppo_managers 狀態
            for symbol, manager in self.ppo_managers.items():
                print(f"[DEBUG __init__] Symbol: {symbol}, Manager ID: {id(manager)}, Initialized: {manager.initialized}")

    def _load_models(self):
        print("--- 正在載入 XGBoost 趨勢模型... ---")
        for symbol in self.symbols:
            try:
                model_path = get_trend_model_path(symbol, self.timeframe, TREND_MODEL_VERSION)
                model = xgb.XGBClassifier()
                model.load_model(model_path)
                self.models[symbol] = model
                print(f"✅ {symbol} 的 XGBoost 模型載入成功！")
            except Exception as e:
                print(f"🛑 警告：無法載入 {symbol} 的模型。")
                pass

    def on_bar(self, dt, current_features):
        """
        每個時間 K 棒被呼叫一次。
        dt: 當前時間戳
        current_features: 一個字典，包含此時間戳下所有 symbol 的預計算特徵 (Pandas Series)
        """
        for symbol in self.symbols:
            if symbol not in self.models or symbol not in current_features:
                # print(f"--- ({symbol}) 缺少模型或當前數據，跳過 ---")
                continue

            # 獲取當前 K 棒的特徵數據
            features_for_symbol = current_features[symbol]

            if self.use_ppo:
                self._process_symbol_with_ppo(symbol, dt, features_for_symbol)
            else:
                self._process_symbol_with_rules(symbol, dt, features_for_symbol)

    def _get_xgb_prediction(self, symbol, features_series):
        """
        使用預先計算好的特徵 Series 來進行預測。
        """
        # XGBoost 模型的特徵順序必須與訓練時完全一致
        # 我們從模型內部獲取這個順序
        model_features = self.models[symbol].get_booster().feature_names

        # 準備模型需要的輸入 (一個 DataFrame，只有一行)
        # 確保特徵的順序是正確的
        input_df = pd.DataFrame([features_series[model_features]], columns=model_features)

        # 假設模型輸出為: 0 (做空), 1 (空手), 2 (做多)
        prediction = self.models[symbol].predict(input_df)[0]
        return int(prediction)

    def _process_symbol_with_ppo(self, symbol, dt, features_series):
        ppo_manager = self.ppo_managers[symbol]

        # DEBUG: 打印每一次 bar 的 ppo_manager 狀態
        if dt.minute == 0 and dt.second == 0: # 只在整點打印，避免洗版
            print(f"[DEBUG on_bar] Dt: {dt}, Symbol: {symbol}, Manager ID: {id(ppo_manager)}, Initialized: {ppo_manager.initialized}")

        # 增加穩健性檢查
        if not ppo_manager.initialized:
            print(f"警告：{symbol} 的 PPO 管理器未成功初始化，跳過 PPO 決策。")
            return

        # PPO 仍然需要一個小範圍的歷史數據來計算其內部狀態（例如，觀察空間）
        ohlcv = self.context.exchange.get_ohlcv(symbol, '5m', limit=200)
        if ohlcv.empty:
            return

        portfolio_state = {
            'position': self.context.portfolio.get_positions().get(symbol.split('/')[0], 0),
            'net_worth_ratio': self.context.portfolio.get_total_value() / self.context.initial_capital
        }

        # 將 XGBoost 訊號傳遞給 PPO
        xgb_prediction = self._get_xgb_prediction(symbol, features_series)
        action = ppo_manager.get_action(ohlcv, portfolio_state, xgb_prediction)

        # 後續邏輯保持不變...
        target_position = ppo_manager.action_map[action]
        current_position_value = self.context.portfolio.get_positions().get(symbol.split('/')[0], 0)

        # 根據 PPO 的目標倉位調整下單
        total_value = self.context.portfolio.get_total_value()
        current_price = ohlcv['Close'].iloc[-1]

        # 計算目標倉位價值
        target_position_value = total_value * target_position

        # 計算需要交易的數量
        amount_to_trade = (target_position_value - current_position_value * current_price) / current_price

        if amount_to_trade > 0:
            print(f"PPO 決策 for {symbol}: 執行做多 (Buy) {amount_to_trade:.4f}！")
            self.context.exchange.create_order(symbol, 'market', 'buy', amount_to_trade)
        elif amount_to_trade < 0:
            print(f"PPO 決策 for {symbol}: 執行做空/平倉 (Sell) {abs(amount_to_trade):.4f}！")
            self.context.exchange.create_order(symbol, 'market', 'sell', abs(amount_to_trade))
        elif target_position == 0 and current_position_value != 0:
            print(f"PPO 決策 for {symbol}: 執行平倉！")
            self.context.exchange.create_order(symbol, 'market', 'sell' if current_position_value > 0 else 'buy', abs(current_position_value))
        else:
            print(f"PPO 決策 for {symbol}: 持有 (Hold)。")

    def _process_symbol_with_rules(self, symbol, dt, features_series):
        """
        根據 XGBoost 模型的預測 (0=空手, 1=做多, 2=做空) 來執行交易。
        """
        prediction = self._get_xgb_prediction(symbol, features_series)

        # 獲取第一個字作為基礎貨幣 (例如 'ETH/USDT' -> 'ETH')
        base_currency = symbol.split('/')[0]
        current_position = self.context.portfolio.get_positions().get(base_currency, 0)

        # 獲取當前價格用於下單
        current_price = self.context.exchange.get_latest_price(symbol)
        if not current_price or current_price <= 0:
             # print(f"警告：無法獲取 {symbol} 的有效價格，跳過下單。")
             return

        # 倉位大小計算：每次交易總價值的 10%
        trade_size_usd = self.context.portfolio.get_total_value() * 0.1
        amount_to_trade = trade_size_usd / current_price

        # --- 全新的、更簡潔的交易邏輯 ---
        if prediction == 1:  # 目標: 做多
            if current_position < 0: # 如果是空倉
                amount_to_buy = abs(current_position) + amount_to_trade
                self.context.exchange.create_order(symbol, 'market', 'buy', amount_to_buy)
            elif current_position == 0: # 如果是空手
                self.context.exchange.create_order(symbol, 'market', 'buy', amount_to_trade)
            # 如果已是多倉，不動作

        elif prediction == 2: # 目標: 做空
            if current_position > 0: # 如果是多倉
                amount_to_sell = current_position + amount_to_trade
                self.context.exchange.create_order(symbol, 'market', 'sell', amount_to_sell)
            elif current_position == 0: # 如果是空手
                self.context.exchange.create_order(symbol, 'market', 'sell', amount_to_trade)
            # 如果已是空倉，不動作

        elif prediction == 0:  # 目標: 平倉
            if current_position > 0: # 如果是多倉
                self.context.exchange.create_order(symbol, 'market', 'sell', abs(current_position))
            elif current_position < 0: # 如果是空倉
                self.context.exchange.create_order(symbol, 'market', 'buy', abs(current_position))
            # 如果是空手，不動作
