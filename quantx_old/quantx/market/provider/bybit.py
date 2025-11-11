# 檔案: quantx/market/provider/bybit.py
# 版本: v32 (重構完成)
# 說明:
# - __init__ 和 _resolve_keys 已重構，以支援代理和標準化的環境變數。
# - submit_order 的介面被標準化，並能正確處理 postOnly 和 positionIdx。

from __future__ import annotations
import os, time, logging, random, asyncio
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List, Tuple, Callable
import pandas as pd
import ccxt
import ccxt.pro as ccxtpro
from .base import MarketProvider
from ...core.timeframe import parse_tf_minutes
from ccxt.base.errors import InsufficientFunds, InvalidOrder, ExchangeError, NetworkError, RequestTimeout, AuthenticationError, BadRequest, BadSymbol

log = logging.getLogger("quantx.bybit.ccxt")

class BybitProvider(MarketProvider):
    def __init__(self, mode: Optional[str] = None, test_run: bool = True):
        super().__init__()
        self.mode = (mode or os.environ.get("mode", "testnet")).lower()
        self.test_run = test_run
        log.info(f"BybitProvider 初始化，模式: {self.mode}, 紙上交易: {self.test_run}")

        api_key, api_secret = self._resolve_keys()
        
        common_config = {
            'apiKey': api_key,
            'secret': api_secret,
            'options': {'defaultType': 'swap', 'ws': {'pingInterval': 20000}},
            'timeout': 30000,
        }

        https_proxy = os.environ.get('https_proxy') or os.environ.get('HTTPS_PROXY')

        rest_config = {**common_config, 'enableRateLimit': True}
        ws_config = {**common_config} # 創建一個獨立的 ws_config 副本

        if https_proxy:
            log.info(f"偵測到代理伺服器: {https_proxy}，將應用於所有 CCXT 連線。")
            rest_config['https_proxy'] = https_proxy

            # --- [核心修正] ---
            # 為 WebSocket 連接明確設定代理
            ws_config['ws_proxy'] = https_proxy
            ws_config['wss_proxy'] = https_proxy

            # 某些舊版本的 ccxt 可能需要這種格式
            if 'options' not in rest_config: rest_config['options'] = {}
            rest_config['options']['proxy'] = https_proxy

        self.exchange = ccxt.bybit(rest_config)
        self.exchange_pro = ccxtpro.bybit(ws_config)

        if self.mode == 'testnet':
            self.exchange.set_sandbox_mode(True)
            self.exchange_pro.set_sandbox_mode(True)
            log.info("CCXT 和 CCXT.pro 皆已設定為 Testnet (沙盒) 模式")

        self.hedged = True
        try:
            # 根據用戶建議，嘗試在程式碼層面強制設定為雙向持倉模式
            self.exchange.set_position_mode(hedged=True)
            log.info("已嘗試發送請求將持倉模式設定為「雙向持倉 (Hedge Mode)」。")
        except Exception as e:
            log.warning(f"嘗試設定持倉模式時發生錯誤（這可能是正常的，如果帳戶權限不足或已是目標模式）: {e}")
        self._market_data: Dict[str, Any] = {}
        self.load_market_data()

    def _resolve_keys(self) -> tuple[str, str]:
        mode_upper = self.mode.upper()
        exchange_upper = 'BYBIT'
        key_var = f"{exchange_upper}_{mode_upper}_API_KEY"
        secret_var = f"{exchange_upper}_{mode_upper}_API_SECRET"
        key = os.environ.get(key_var, "").strip()
        sec = os.environ.get(secret_var, "").strip()
        if not (key and sec):
            old_key_var = f"{self.mode.lower()}_api_key"
            old_secret_var = f"{self.mode.lower()}_api_secret"
            key = os.environ.get(old_key_var, "").strip()
            sec = os.environ.get(old_secret_var, "").strip()
            if key and sec:
                 # 根據用戶要求，移除關於舊格式的警告訊息
                 pass
            else:
                 log.warning(f"在環境變數中未找到 {self.mode} 模式的 API 金鑰 ({key_var})。")
        return key, sec
    
    def load_market_data(self):
        try:
            if not self.exchange.markets or not self._market_data:
                self.exchange.load_markets() 
                self._market_data = self.exchange.markets
                log.info("CCXT 市場數據已成功載入並緩存。")
            return self._market_data
        except Exception as e:
            log.error(f"CCXT 載入市場數據失敗: {e}", exc_info=True)
            self._market_data = {}
            return {}

    def get_market_params(self, symbol: str) -> Dict[str, Any]:
        ccxt_symbol = self.exchange.symbol(symbol)
        if not self._market_data: self.load_market_data()
        market = self._market_data.get(ccxt_symbol)
        if market:
            return {
                'min_qty': market.get('limits', {}).get('amount', {}).get('min', 0.0),
                'min_notional': market.get('limits', {}).get('cost', {}).get('min', 0.0),
                'tick_size': market.get('precision', {}).get('price', 0.01),
            }
        return {'min_qty': 0.001, 'min_notional': 5.0, 'tick_size': 0.01}

    def round_qty(self, symbol: str, qty: float) -> float:
        ccxt_symbol = self.exchange.symbol(symbol)
        if not self.exchange.markets: self.load_market_data()
        if ccxt_symbol in self.exchange.markets:
            try:
                return self.exchange.amount_to_precision(ccxt_symbol, qty)
            except Exception as e:
                log.warning(f"[BybitProvider] 數量精度轉換失敗 for {symbol}/{qty}: {e}. 回傳原始數量。")
        return qty

    def fetch_balance(self, currency: str = 'USDT') -> float:
        try:
            balance = self.exchange.fetch_balance()
            return float(balance['total'].get(currency, 0.0))
        except Exception as e:
            log.error(f"查詢帳戶餘額失敗: {e}", exc_info=True)
            raise
            
    def get_positions(self) -> List[Dict]:
        try:
            positions = self.exchange.fetch_positions()
            return [p for p in positions if float(p.get('contracts', 0)) != 0]
        except Exception as e:
            log.error(f"查詢倉位失敗: {e}", exc_info=True)
            return []

    async def watch_ohlcv_stream(self, symbols_tfs: List[List[str]], callback: Callable):
        """
        [實現] 監聽多個 OHLCV 數據流。
        為每個 (symbol, tf) 組合創建一個獨立的監聽任務。
        """
        async def watch_loop(symbol, tf):
            while True:
                try:
                    ohlcv_list = await self.exchange_pro.watch_ohlcv(symbol, tf)
                    # ccxt.pro 返回的是一個包含新 K 線的列表
                    if ohlcv_list:
                        callback(symbol, tf, ohlcv_list)
                except Exception as e:
                    log.error(f"監聽 {symbol}-{tf} K 線時發生錯誤: {e}", exc_info=False)
                    await asyncio.sleep(10) # 發生錯誤後等待10秒重試

        tasks = [watch_loop(symbol, tf) for symbol, tf in symbols_tfs]
        await asyncio.gather(*tasks)

    async def watch_orderbook_stream(self, symbols: List[str], callback: Callable):
        """
        [實現] 監聽多個訂單簿數據流。
        """
        async def watch_loop(symbol):
            while True:
                try:
                    orderbook = await self.exchange_pro.watch_order_book(symbol)
                    callback(symbol, orderbook)
                except BadSymbol as e:
                    # [核心修正] 捕獲永久性錯誤，終止重試
                    log.error(f"監聽訂單簿失敗：交易對 '{symbol}' 無效或不存在。將停止對該交易對的監聽。錯誤: {e}")
                    break # 終止這個 symbol 的 while 循環
                except (NetworkError, RequestTimeout, ExchangeError) as e:
                    # 對於暫時性錯誤，保留重試邏輯
                    log.warning(f"監聽訂單簿 {symbol} 時發生暫時性錯誤，將在10秒後重試。錯誤: {e}")
                    await asyncio.sleep(10)
                except Exception as e:
                    # 對於其他未知錯誤，也進行重試，但記錄更詳細的日誌
                    log.error(f"監聽訂單簿 {symbol} 時發生未知錯誤，將在10秒後重試。錯誤: {e}", exc_info=True)
                    await asyncio.sleep(10)

        tasks = [watch_loop(symbol) for symbol in symbols]
        await asyncio.gather(*tasks)

    async def watch_orders_stream(self, callback: Callable):
        """
        [新增] 監聽私有訂單更新數據流。
        """
        while True:
            try:
                orders = await self.exchange_pro.watch_orders()
                if orders:
                    callback(orders)
            except Exception as e:
                log.error(f"監聽訂單更新時發生錯誤: {e}", exc_info=True)
                await asyncio.sleep(10)
    
    async def close_ws(self):
        if hasattr(self.exchange_pro, 'close'):
             await self.exchange_pro.close()

    def fetch_klines(self, symbol: str, start: datetime, end: datetime, tf: str = '1m') -> pd.DataFrame:
        limit = 1000
        start_ts_ms = int(start.timestamp() * 1000)
        end_ts_ms = int(end.timestamp() * 1000)
        all_klines = []
        log.info(f"開始抓取 K 線: {symbol}, TF: {tf}, 從 {start.isoformat()} 到 {end.isoformat()}")
        current_ts_ms = start_ts_ms
        while current_ts_ms < end_ts_ms:
            try:
                klines = self.exchange.fetch_ohlcv(symbol, timeframe=tf, since=current_ts_ms, limit=limit)
                if not klines: break
                all_klines.extend(klines)
                current_ts_ms = klines[-1][0] + 1
                time.sleep(self.exchange.rateLimit / 1000)
            except ExchangeError as e:
                if 'AuthenticationError' in str(e) or 'retCode":33004' in str(e):
                    log.error(f"❌ API 金鑰錯誤: {self.mode.upper()} API Key 似乎已過期或無效。請更新您的 .env 檔案。")
                    return pd.DataFrame()
                else:
                    log.error(f"抓取 K 線時發生交易所錯誤: {e}")
                    break
            except Exception as e:
                log.error(f"抓取 K 線時發生未知錯誤: {e}")
                break
        if not all_klines: return pd.DataFrame()
        df = pd.DataFrame(all_klines, columns=['ms_timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = (df['ms_timestamp'] / 1000).astype(int)
        df = df.drop(columns=['ms_timestamp'])[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
        return df

    def submit_order(self, symbol: str, order_type: str, side: str, qty: float, price: Optional[float] = None, params: Optional[Dict] = None) -> Dict[str, Any]:
        intent = params.get('intent', 'unknown') if params else 'unknown'
        log.info(f"準備訂單: {symbol}, 意圖: {intent.upper()}, {side.upper()} {qty} @ {order_type.upper()}")

        if self.test_run:
            log.info(f"[Paper Trading] 訂單已攔截，未發送至交易所。")
            return {"dry_run": True, "status": "accepted", "symbol": symbol, "side": side, "qty": qty, "type": order_type, "price": price, "params": params}

        try:
            ccxt_symbol = self.exchange.symbol(symbol)
            qty_str = self.round_qty(symbol, qty)
            price_str = self.exchange.price_to_precision(ccxt_symbol, price) if order_type == 'limit' and price else None

            final_params = params.copy() if params else {}
            final_params['category'] = 'linear'

            if self.hedged:
                if intent in ["open_long", "close_long"]:
                    final_params["positionIdx"] = 1
                elif intent in ["open_short", "close_short"]:
                    final_params["positionIdx"] = 2
                elif intent == "close":
                    final_params["positionIdx"] = 1 if side.lower() == "sell" else 2
                else:
                    final_params["positionIdx"] = 1 if side.lower() == "buy" else 2
            
            if final_params.pop('postOnly', False):
                final_params['timeInForce'] = 'PostOnly'

            # [核心改造] 將止損/止盈價格添加到 final_params
            if sl_price := final_params.pop('stopLoss', None):
                final_params['stopLoss'] = self.exchange.price_to_precision(ccxt_symbol, sl_price)
            if tp_price := final_params.pop('takeProfit', None):
                final_params['takeProfit'] = self.exchange.price_to_precision(ccxt_symbol, tp_price)

            order_result = self.exchange.create_order(
                symbol=ccxt_symbol, type=order_type, side=side,
                amount=qty_str, price=price_str, params=final_params
            )
            log.info(f"訂單已成功提交至 Bybit, 訂單 ID: {order_result.get('id')}")
            return {"dry_run": False, "status": "accepted", "resp": order_result}

        except InsufficientFunds as e:
            log.warning(f"⚠️ 【資金不足】下單失敗: {e}")
            return {"dry_run": False, "status": "rejected", "reason": "InsufficientFunds"}
        except InvalidOrder as e:
            log.warning(f"⚠️ 【訂單無效】下單失敗: {e}")
            return {"dry_run": False, "status": "rejected", "reason": "InvalidOrder"}
        except Exception as e:
            log.error(f"❌ CCXT 提交訂單時發生未知嚴重錯誤: {e}", exc_info=True)
            raise