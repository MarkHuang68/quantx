# 檔案: strategies/xgboost_trend_strategy.py

import pandas as pd
import numpy as np
import xgboost as xgb

from strategies.base_strategy import BaseStrategy
from utils.common import create_features_trend
from settings import SYMBOLS_TO_TRADE, TREND_MODEL_VERSION, get_trend_model_path
from core.ppo_manager import PPOManager
import settings
from utils.common import convert_symbol_to_ccxt

class XGBoostTrendStrategy(BaseStrategy):
    def __init__(self, context, symbols=SYMBOLS_TO_TRADE, timeframe='1m', use_ppo=False, ppo_model_path=None):
        super().__init__(context)
        self.symbols = symbols
        self.timeframe = timeframe
        self.fee_rate = settings.FEE_RATE
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

    async def on_bar(self, dt, current_features, historical_data=None):
        for symbol in self.symbols:
            if symbol not in self.models or symbol not in current_features:
                continue
            features_for_symbol = current_features[symbol]
            historical_data_for_symbol = historical_data.get(symbol) if historical_data else None

            if self.use_ppo:
                if historical_data_for_symbol is None or historical_data_for_symbol.empty:
                    print(f"警告：回測模式下 PPO 策略缺少 {symbol} 的歷史數據，跳過。")
                    continue
                await self._process_symbol_with_ppo(symbol, dt, features_for_symbol, historical_data_for_symbol)
            else:
                await self._process_symbol_with_rules(symbol, dt, features_for_symbol)

    def _get_xgb_prediction(self, symbol, features_series):
        model_features = self.models[symbol].get_booster().feature_names
        input_df = pd.DataFrame([features_series[model_features]], columns=model_features)
        raw_prediction = self.models[symbol].predict(input_df)[0]
        signal_map = {1: 1, 2: -1, 0: 0}
        return signal_map.get(int(raw_prediction), 0)

    async def _process_symbol_with_ppo(self, symbol, dt, features_series, historical_data):
        ccxt_symbol = convert_symbol_to_ccxt(symbol)
        ppo_manager = self.ppo_managers[symbol]
        if not ppo_manager.initialized:
            print(f"警告：{symbol} 的 PPO 管理器未成功初始化，跳過。")
            return

        # 在回測模式下，直接使用傳入的歷史數據
        # 在即時交易中，historical_data 會是 None，此時才需要從交易所獲取
        if historical_data is not None:
            ohlcv = historical_data.tail(200) # 取最近 200 筆
        else:
            ohlcv = await self.context.exchange.get_ohlcv(ccxt_symbol, self.timeframe, limit=200)

        if ohlcv is None or ohlcv.empty:
            print(f"警告：{symbol} 在 {dt} 沒有可用的 OHLCV 數據，跳過 PPO 處理。")
            return

        positions = self.context.portfolio.get_positions()
        symbol_positions = positions.get(ccxt_symbol, {'long': {'contracts': 0}, 'short': {'contracts': 0}})
        long_pos = symbol_positions['long']['contracts']
        short_pos = symbol_positions['short']['contracts']
        net_position = long_pos - short_pos

        portfolio_state = {'position': net_position, 'net_worth_ratio': self.context.portfolio.get_total_value() / self.context.initial_capital}
        xgb_prediction = self._get_xgb_prediction(symbol, features_series)
        action = ppo_manager.get_action(ohlcv, portfolio_state, xgb_prediction)
        target_position_ratio = ppo_manager.action_map[action]
        total_value = self.context.portfolio.get_total_value()
        current_price = ohlcv['Close'].iloc[-1]
        
        # 只有在目標倉位與現有倉位反向時，才先平倉
        if target_position_ratio < 0 and long_pos > 0:
            print(f"PPO({symbol}): [反向平多] {long_pos:.4f}")
            await self.context.exchange.create_order(ccxt_symbol, 'market', 'sell', long_pos, params={'position_idx': 1})

        if target_position_ratio > 0 and short_pos > 0:
            print(f"PPO({symbol}): [反向平空] {short_pos:.4f}")
            await self.context.exchange.create_order(ccxt_symbol, 'market', 'buy', short_pos, params={'position_idx': 2})

        # 開新倉
        if target_position_ratio > 0 and long_pos == 0:
            amount_to_trade = (total_value * target_position_ratio) / current_price
            if amount_to_trade * current_price > 10.0:
                print(f"PPO({symbol}): [開多] {amount_to_trade:.4f}")
                await self.context.exchange.create_order(ccxt_symbol, 'market', 'buy', amount_to_trade, params={'position_idx': 1})
        elif target_position_ratio < 0 and short_pos == 0:
            amount_to_trade = (total_value * abs(target_position_ratio)) / current_price
            if amount_to_trade * current_price > 10.0:
                print(f"PPO({symbol}): [開空] {amount_to_trade:.4f}")
                await self.context.exchange.create_order(ccxt_symbol, 'market', 'sell', amount_to_trade, params={'position_idx': 2})

    async def _process_symbol_with_rules(self, symbol, dt, features_series):
        ccxt_symbol = convert_symbol_to_ccxt(symbol)
        prediction = self._get_xgb_prediction(symbol, features_series)
        positions = self.context.portfolio.get_positions()
        symbol_positions = positions.get(ccxt_symbol, {'long': {'contracts': 0}, 'short': {'contracts': 0}})
        long_position = symbol_positions['long']['contracts']
        short_position = symbol_positions['short']['contracts']

        current_price = await self.context.exchange.get_latest_price(ccxt_symbol)
        if not current_price or current_price <= 0:
             return

        trade_size_usd = self.context.portfolio.get_total_value() * 0.1
        amount_to_trade = trade_size_usd / current_price

        if prediction == 1:
            if long_position == 0:
                print(f"訊號({symbol}): [開多] {amount_to_trade:.4f}")
                await self.context.exchange.create_order(ccxt_symbol, 'market', 'buy', amount_to_trade, params={'position_idx': 1})
            if short_position > 0:
                print(f"訊號({symbol}): [平空] {short_position:.4f}")
                await self.context.exchange.create_order(ccxt_symbol, 'market', 'buy', short_position, params={'position_idx': 2})
        elif prediction == -1:
            if short_position == 0:
                print(f"訊號({symbol}): [開空] {amount_to_trade:.4f}")
                await self.context.exchange.create_order(ccxt_symbol, 'market', 'sell', amount_to_trade, params={'position_idx': 2})
            if long_position > 0:
                print(f"訊號({symbol}): [平多] {long_position:.4f}")
                await self.context.exchange.create_order(ccxt_symbol, 'market', 'sell', long_position, params={'position_idx': 1})
        elif prediction == 0:
            # 訊號為 0，持倉不動
            print(f"訊號({symbol}): [持倉] (prediction=0)")
            pass
