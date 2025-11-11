# quantx/core/data/loader.py
# 檔案: quantx/core/data/loader.py
# 版本: v5 (最終修復：InterfaceError)
# 說明:
# - 修正了時間戳單位被重複轉換的錯誤。
# - 新增了對 inf 值的處理，將其替換為 NaN 以便能被 dropna 清除。

import sqlite3
import threading
import pandas as pd
import numpy as np # 引入 numpy 用於處理 inf
from pathlib import Path
from quantx.core.timeframe import parse_tf_seconds

class DataLoader:
    def _normalize_symbol(self, symbol: str) -> str:
        """將各種格式的 symbol (例如 'BTC/USDT:USDT', 'BTC/USDT') 標準化為 'BTCUSDT'。"""
        return symbol.replace('/', '').split(':')[0]

    def __init__(self, cache_dir: str, scope: str, provider):
        self.cache_dir = Path(cache_dir)
        self.scope = scope
        self.provider = provider
        self.db_path = self.cache_dir / scope / "data.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.lock = threading.Lock() # 初始化線程鎖
        self._init_schema()

    def _init_schema(self):
        q = """
        CREATE TABLE IF NOT EXISTS ohlcv (
            symbol TEXT, ts INTEGER, open REAL, high REAL, low REAL,
            close REAL, volume REAL, PRIMARY KEY(symbol, ts)
        )
        """
        self.conn.execute(q)
        self.conn.commit()

    def save(self, symbol: str, df: pd.DataFrame):
        # [核心修正] 使用鎖來保護整個保存過程，防止並發寫入衝突
        with self.lock:
            if df.empty: return

            normalized_symbol = self._normalize_symbol(symbol)
            df = df.copy()

            # --- [時間戳處理] ---
            if isinstance(df.index, pd.DatetimeIndex):
                df["ts"] = (df.index.astype('int64') // 10**9).astype(int)
            elif "timestamp" in df.columns:
                df["ts"] = df["timestamp"].astype(int)
            else:
                raise ValueError("DataFrame 缺少可用的時間戳來源 (DatetimeIndex 或 'timestamp' 欄位)")

            df["symbol"] = normalized_symbol
            cols_to_save = ["symbol", "ts", "open", "high", "low", "close", "volume"]

            if not all(c in df.columns for c in cols_to_save):
                raise ValueError("DataFrame 缺少必要的 OHLCV 或 ts 欄位")

            df_to_save = df[cols_to_save]

            # --- [健壯性強化] ---
            df_to_save = df_to_save.replace([np.inf, -np.inf], np.nan)

            for col in ["open", "high", "low", "close", "volume"]:
                df_to_save[col] = pd.to_numeric(df_to_save[col], errors='coerce')

            df_to_save = df_to_save.dropna()

            if df_to_save.empty:
                return

            df_to_save = df_to_save.astype({
                "symbol": str, "ts": int, "open": float, "high": float,
                "low": float, "close": float, "volume": float
            })

            try:
                self.conn.executemany(
                    "INSERT OR REPLACE INTO ohlcv (symbol, ts, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    df_to_save.values.tolist()
                )
                self.conn.commit()
            except Exception as e:
                self.log.error(f"在保存 {normalized_symbol} 到數據庫時出錯: {e}")
                # 即使出錯，也要確保不會影響其他線程，所以這裡不拋出異常，只記錄日誌

    def read(self, symbol: str, start_ts: int, end_ts: int) -> pd.DataFrame:
        normalized_symbol = self._normalize_symbol(symbol)
        q = "SELECT ts, open, high, low, close, volume FROM ohlcv WHERE symbol=? AND ts BETWEEN ? AND ? ORDER BY ts"
        df = pd.read_sql(q, self.conn, params=(normalized_symbol, start_ts, end_ts))
        if df.empty: return df

        # 將 ts 轉換為 DatetimeIndex
        df['timestamp'] = pd.to_datetime(df['ts'], unit='s', utc=True)
        df = df.set_index('timestamp')
        return df

    def load_ohlcv(self, symbol: str, tf: str, start, end) -> pd.DataFrame:
        if isinstance(start, str): start = pd.to_datetime(start, utc=True)
        if isinstance(end, str): end = pd.to_datetime(end, utc=True)

        start_ts = int(start.timestamp())
        end_ts = int(end.timestamp())

        # 讀取 1m 數據 (read 方法現在會回傳帶有 DatetimeIndex 的 df)
        df_1m = self.read(symbol, start_ts, end_ts)

        # 檢查是否需要從遠端 provider 補抓數據
        need_fetch = df_1m.empty
        if not df_1m.empty:
            min_ts_in_db = df_1m.index.min().timestamp()
            max_ts_in_db = df_1m.index.max().timestamp()
            # 需要補頭或補尾
            diff = parse_tf_seconds(tf)
            if min_ts_in_db - diff > start_ts or max_ts_in_db + diff < end_ts:
                need_fetch = True
        
        if need_fetch:
            # 這裡修正為傳入 '1m' 作為 Provider 的請求時間框
            raw_df_from_provider = self.provider.fetch_klines(symbol, start, end, tf='1m')
            
            # 🟢 修正：刪除舊的、錯誤的 'timestamp' 處理邏輯
            # if not raw_df_from_provider.empty:
            #     raw_df_from_provider["ts"] = raw_df_from_provider["timestamp"].astype(int) # <-- 這是導致 KeyError 的舊邏輯
            
            if not raw_df_from_provider.empty:
                self.save(symbol, raw_df_from_provider)
            # 重新從資料庫讀取完整的 1m 數據
            df_1m = self.read(symbol, start_ts, end_ts)

        if tf == "1m":
            return df_1m

        # 聚合成目標時間週期
        return self.aggregate_from_1m(df_1m, tf)

    def aggregate_from_1m(self, df: pd.DataFrame, dst_tf: str) -> pd.DataFrame:
        if df.empty: return df

        # df 傳入時已經有 DatetimeIndex
        if dst_tf.endswith("m"): rule = f"{int(dst_tf[:-1])}min"
        elif dst_tf.endswith("h"): rule = f"{int(dst_tf[:-1])}h"
        elif dst_tf.endswith("d"): rule = f"{int(dst_tf[:-1])}d"
        else: raise ValueError(f"不支援的時間週期: {dst_tf}")

        agg = df.resample(rule).agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum"
        }).dropna()
        
        # 回傳的 DataFrame 保持 DatetimeIndex
        return agg