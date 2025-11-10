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
        if self.use_ppo and ppo_model_path:
            self.ppo_managers = {s: PPOManager(model_path=ppo_model_path, symbol=s, timeframe=timeframe, version=TREND_MODEL_VERSION) for s in symbols}

    def _load_models(self):
        for symbol in self.symbols:
            try:
                model_path = get_trend_model_path(symbol, self.timeframe, TREND_MODEL_VERSION)
                model = xgb.XGBClassifier()
                model.load_model(model_path)
                self.models[symbol] = model
            except: self.models[symbol] = None

    def _get_xgb_prediction(self, symbol, features_series):
        if not self.models.get(symbol): return 0
        model_features = self.models[symbol].get_booster().feature_names
        input_df = pd.DataFrame([features_series[model_features]], columns=model_features)
        raw_prediction = self.models[symbol].predict(input_df)[0]
        return {1: 1, 2: -1, 0: 0}.get(int(raw_prediction), 0)

    def on_bar(self, dt, current_features):
        for symbol in self.symbols:
            if self.use_ppo: self._process_symbol_with_ppo(symbol, dt, current_features.get(symbol))
            else: self._process_symbol_with_rules(symbol, dt, current_features.get(symbol))

    def _process_symbol_with_ppo(self, symbol, dt, features_series):
        ppo_manager = self.ppo_managers[symbol]
        if not ppo_manager.initialized:
            return

        ohlcv = self.context.exchange.get_ohlcv(symbol, '5m', limit=200)
        if ohlcv.empty:
            return

        current_price = ohlcv['Close'].iloc[-1]
        current_prices = {symbol: current_price}

        portfolio_state = {
            'position': self.context.portfolio.get_positions().get(symbol.split('/')[0], 0),
            'net_worth_ratio': self.context.portfolio.get_total_value(current_prices) / self.context.initial_capital
        }

        xgb_prediction = self._get_xgb_prediction(symbol, features_series)
        action = ppo_manager.get_action(ohlcv, portfolio_state, xgb_prediction)
        target_position = ppo_manager.action_map[action]

        current_position_value = self.context.portfolio.get_positions().get(symbol.split('/')[0], 0)
        total_value = self.context.portfolio.get_total_value(current_prices)
        target_position_value = total_value * target_position
        amount_to_trade = (target_position_value - current_position_value * current_price) / current_price

        side = None
        if amount_to_trade > 1e-9: side = 'buy'
        elif amount_to_trade < -1e-9: side = 'sell'

        if side:
            try:
                amount = abs(amount_to_trade)
                trade_receipt = self.context.exchange.create_order(symbol=symbol, type='market', side=side, amount=amount)
                if trade_receipt:
                    signed_amount = amount if side == 'buy' else -amount
                    self.context.portfolio.update_position(symbol=trade_receipt['symbol'], amount=signed_amount, price=trade_receipt['price'])
                    self.context.portfolio.charge_fee(trade_receipt['fee'])
            except Exception as e:
                print(f"❌ 下單失敗: {e}")

    def _process_symbol_with_rules(self, symbol, dt, features_series):
        prediction = self._get_xgb_prediction(symbol, features_series)
        if prediction == 0: return

        current_price = self.context.exchange.get_latest_price(symbol)
        if not current_price: return

        trade_size_usd = self.context.portfolio.get_total_value({symbol: current_price}) * 0.1
        amount = trade_size_usd / current_price
        side = 'buy' if prediction == 1 else 'sell'

        try:
            trade_receipt = self.context.exchange.create_order(symbol=symbol, type='market', side=side, amount=amount)
            if trade_receipt:
                signed_amount = amount if side == 'buy' else -amount
                self.context.portfolio.update_position(symbol, signed_amount, trade_receipt['price'])
                self.context.portfolio.charge_fee(trade_receipt['fee'])
        except Exception as e:
            print(f"❌ 下單失敗: {e}")
