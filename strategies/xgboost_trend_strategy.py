# 檔案: strategies/xgboost_trend_strategy.py
# 【!!! 核心修正：改為使用 best_model_registry.json 動態載入模型 !!!】

import pandas as pd
import numpy as np
import xgboost as xgb
import os # 【!!! 修正 !!!】 導入 os 模組

from strategies.base_strategy import BaseStrategy
from utils.common import create_features_trend
# 【!!! 修正 !!!】 移除 get_trend_model_path, 新增 load_registry
from settings import SYMBOLS_TO_TRADE, TREND_MODEL_VERSION, load_registry 
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
        
        # 【!!! 修正 !!!】 在初始化時載入一次註冊表
        print("--- 正在載入 best_model_registry.json ... ---")
        self.registry = load_registry() 
        if not self.registry:
             print("🛑 警告：`best_model_registry.json` 為空或載入失敗。")

        self._load_models() # 呼叫下方修正過的 _load_models

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

    # --- 【!!! 核心修正：_load_models 已重寫 !!!】 ---
    def _load_models(self):
        print("--- 正在載入 XGBoost 趨勢模型 (使用 Registry)... ---")
        
        if not self.registry:
            print("🛑 錯誤：模型註冊表 (Registry) 未載入，無法繼續。")
            return

        for symbol in self.symbols:
            key = f"{symbol}_{self.timeframe}"
            
            # 1. 檢查註冊表中是否有紀錄
            if key not in self.registry:
                print(f"🛑 警告：{key} 在 best_model_registry.json 中沒有紀錄。")
                continue # 跳過此 symbol

            try:
                model_config = self.registry[key]
                model_path = model_config.get('model_file')
                
                # 2. 檢查紀錄是否完整
                if not model_path:
                    print(f"🛑 警告：{key} 在 registry 中的紀錄已損壞 (缺少 'model_file' 鍵)。")
                    continue
                    
                # 3. 檢查實體檔案是否存在 (關鍵)
                if not os.path.exists(model_path):
                    print(f"🛑 警告：{key} 的模型檔案 {model_path} 不存在。")
                    print(f"   (Registry 檔案可能與 models 資料夾不同步)")
                    continue

                # 4. 載入模型
                model = xgb.XGBClassifier()
                model.load_model(model_path)
                model.n_classes_ = 2 # (確保 n_classes_ 屬性被設置)
                self.models[symbol] = model
                print(f"✅ {symbol} ({self.timeframe}) 的 XGBoost 模型載入成功！({model_path})")
            
            except Exception as e:
                print(f"🛑 警告：載入 {symbol} 的模型 {model_path} 時發生嚴重錯誤: {e}")
                pass # 繼續嘗試載入下一個模型

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
        # 【!!! 核心修正：處理 Target 標籤 !!!】
        # 舊模型 (target=1,2,0) vs 新模型 (target=1,0)
        # 我們在這裡統一使用 predict_proba 來處理，更為穩健
        
        try:
            model = self.models[symbol]
            model_features = model.get_booster().feature_names
            input_df = pd.DataFrame([features_series[model_features]], columns=model_features)
            
            # 使用 predict_proba() 會返回 (機率_0, 機率_1)
            y_prob = model.predict_proba(input_df)
            prob_buy = y_prob[0][1]  # 類別 1 (做多) 的機率
            prob_sell = y_prob[0][0] # 類別 0 (做空) 的機率

            # 載入 registry 中的信心門檻
            key = f"{symbol}_{self.timeframe}"
            config = self.registry.get(key, {})
            conf_buy = config.get('reference_conf_buy', 0.51) # 預設 0.51
            conf_sell = config.get('reference_conf_sell', 0.51) # 預設 0.51

            if prob_buy > conf_buy:
                return 1 # 做多
            elif prob_sell > conf_sell:
                return -1 # 做空
            else:
                return 0 # 不動
                
        except Exception as e:
            print(f"🛑 錯誤：在 {symbol} 執行 _get_xgb_prediction 時失敗: {e}")
            return 0 # 出錯時返回 0 (不動)


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
        
        # 【!!! 修正 !!!】 使用新的 predict_proba 邏輯
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
        
        # 【!!!】 使用新的 predict_proba 邏輯
        prediction = self._get_xgb_prediction(symbol, features_series)
        
        positions = self.context.portfolio.get_positions()
        symbol_positions = positions.get(ccxt_symbol, {'long': {'contracts': 0}, 'short': {'contracts': 0}})
        long_position = symbol_positions['long']['contracts']
        short_position = symbol_positions['short']['contracts']

        # === 【!!!】 從 K 棒數據獲取價格 (已修正) ===
        current_price = features_series.get('Close')
        
        if not current_price or current_price <= 0:
             print(f"警告：{ccxt_symbol} 在 {dt} 的 K 棒中找不到 'Close' 價格，跳過。")
             return
        # === 【修正結束】 ===

        trade_size_usd = self.context.portfolio.get_total_value() * 0.1
        amount_to_trade = trade_size_usd / current_price

        if prediction == 1:
            if long_position == 0:
                print(f"訊號({symbol} @ {dt}): [開多] {amount_to_trade:.4f}")
                await self.context.exchange.create_order(ccxt_symbol, 'market', 'buy', amount_to_trade, price=current_price, params={'position_idx': 1})
            if short_position > 0:
                print(f"訊號({symbol} @ {dt}): [平空] {short_position:.4f}")
                await self.context.exchange.create_order(ccxt_symbol, 'market', 'buy', short_position, price=current_price, params={'position_idx': 2})
        
        elif prediction == -1:
            if short_position == 0:
                print(f"訊號({symbol} @ {dt}): [開空] {amount_to_trade:.4f}")
                await self.context.exchange.create_order(ccxt_symbol, 'market', 'sell', amount_to_trade, price=current_price, params={'position_idx': 2})
            if long_position > 0:
                print(f"訊號({symbol} @ {dt}): [平多] {long_position:.4f}")
                await self.context.exchange.create_order(ccxt_symbol, 'market', 'sell', long_position, price=current_price, params={'position_idx': 1})
        
        # === 【!!! 核心修正：訊號 0 = 平倉 !!!】 ===
        elif prediction == 0:
            if long_position > 0:
                print(f"訊號({symbol} @ {dt}): [訊號 0 - 平多] {long_position:.4f}")
                await self.context.exchange.create_order(ccxt_symbol, 'market', 'sell', long_position, price=current_price, params={'position_idx': 1})
            if short_position > 0:
                print(f"訊號({symbol} @ {dt}): [訊號 0 - 平空] {short_position:.4f}")
                await self.context.exchange.create_order(ccxt_symbol, 'market', 'buy', short_position, price=current_price, params={'position_idx': 2})
            # 如果 (long_position == 0 and short_position == 0)，則 prediction 0 保持空手，不做任何事。
        # === 【修正結束】 ===