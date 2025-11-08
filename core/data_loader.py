# 檔案: core/data_loader.py

import pandas as pd

def load_csv_data(filepath, symbol=None):
    """
    從 CSV 檔案載入 K 線數據。
    CSV 檔案應至少包含 'timestamp', 'open', 'high', 'low', 'close', 'volume' 這些欄位。
    """
    try:
        df = pd.read_csv(filepath)

        # 確保 'timestamp' 欄位存在且為日期時間格式
        if 'timestamp' not in df.columns:
            raise ValueError("CSV 檔案中缺少 'timestamp' 欄位")

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

        # 如果提供了 symbol，可以做一些特定的處理
        if symbol:
            print(f"已為 {symbol} 載入數據，共 {len(df)} 筆")

        return df

    except FileNotFoundError:
        print(f"錯誤：找不到檔案 {filepath}")
        return None
    except Exception as e:
        print(f"載入 CSV 檔案時發生錯誤: {e}")
        return None
