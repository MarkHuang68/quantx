import pandas as pd
import pandas_ta as ta
import numpy as np
import talib
import xgboost as xgb
import matplotlib.pyplot as plt

# --- 步驟 1: 參數設定 (必須與您的「冠軍模型」完全一致) ---

# 1a. 您的精英特徵列表 (請填入您 69% 準確率的那一組)
# (這只是範例，請務必換成您真實的列表)
FEATURES_LIST = [
        'HOUR',
        # 'D_OF_W',
        'MONTH',

        'EMA',
        'CLOSE_EMA',
        'ATR',
        'RSI',
        'MOM',
        'ADX',
        'ADX_hist',
        'ADX_hist_ema',
        'WILLR',
        'KDJ_K',
        'KDJ_D',
        'KDJ_J',

        'MACD', 'MACD_signal',
        'MACD_hist',
        
        'BB_Width',
        'BB_Percent',

        'OBV',
        'Volume',
        'VOLUME_CHANGE',

        # 'return',
        # 'return_lag_1',
        # 'return_lag_2',
        # 'return_lag_3',
        # 'return_lag_4',
        # 'return_lag_5',
    ]

# 1b. 您的預測週期 (Horizon)
# (您是用 5m 預測 12h，所以 horizon = 12 * 12 = 144)
HORIZON = 1 # 12 小時 * (60分鐘/5分鐘) = 144 根 K 棒

# 1c. 策略參數 (包含手續費)
THRESHOLD = 0.0001      # 您的進場門檻
FEE_RATE = 0.00055      # 您設定的 Bybit 手續費 (0.055%)

# --- 步驟 2: 特徵工程函數 (必須 100% 複製) ---
# (這一步最關鍵，我們把所有特徵計算包成一個函數)
def calculate_all_features(df):
    """
    在傳入的 DataFrame 上計算所有必要的特徵。
    注意：這個 df 必須包含足夠的「歷史」資料來計算指標 (例如 MA, RSI)。
    """
    print("開始計算特徵...")
    
    close_prices = df['Close'].values.astype(float)
    high_prices = df['High'].values.astype(float)
    low_prices = df['Low'].values.astype(float)
    volume = df['Volume'].values.astype(float)
    close_sma = talib.SMA(close_prices, 3)
    df['Close_SMA'] = close_sma

    # 您的指標 (WMA + 快速 MACD + 快速 BBANDS)
    ema = talib.EMA(close_prices, timeperiod=20)
    df['HOUR'] = df.index.hour
    df['D_OF_W'] = df.index.dayofweek
    df['MONTH'] = df.index.month
    df['EMA'] = ema
    df['CLOSE_EMA'] = (close_prices - ema) / ema
    df['ATR'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
    df['RSI'] = talib.RSI(close_prices, timeperiod=14)
    df['MOM'] = talib.MOM(close_prices, timeperiod=10)
    df['ADX'] = talib.ADX(high_prices, low_prices, close_prices)
    df['ADX_hist'] = talib.PLUS_DI(high_prices, low_prices, close_prices) - talib.MINUS_DI(high_prices, low_prices, close_prices)
    df['ADX_hist_ema'] = talib.EMA(df['ADX_hist'], 10)
    df['WILLR'] = talib.WILLR(high_prices, low_prices, close_prices, timeperiod=20)

    df['KDJ_K'], df['KDJ_D'] = talib.STOCH(
                                high_prices, 
                                low_prices, 
                                close_prices, 
                                fastk_period=9,
                                slowk_period=3,
                                slowk_matype=0, # 設為 0 使用 SMA
                                slowd_period=3,
                                slowd_matype=0  # 設為 0 使用 SMA
                                )
    df['KDJ_J'] = (3 * df['KDJ_K']) - (2 * df['KDJ_D'])
    
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(close_prices, 
                                                                fastperiod=12, 
                                                                slowperiod=26, 
                                                                signalperiod=9)

    upperband, middleband, lowerband = talib.BBANDS(close_prices, 
                                                    timeperiod=20, 
                                                    nbdevup=2, 
                                                    nbdevdn=2, 
                                                    matype=0)
    
    df['BB_Width'] = (upperband - lowerband) / (middleband + 1e-10)
    df['BB_Percent'] = (close_prices - lowerband) / (upperband - lowerband + 1e-10)

    df['OBV'] = talib.OBV(close_prices, volume)
    df['VOLUME_CHANGE'] = df['Volume'].pct_change()
    df['VOLUME_CHANGE'].replace([np.inf, -np.inf], np.nan, inplace=True)

    df['return'] = (df['Close'].shift(-1) - close_prices) / close_prices
    df['return_lag_1'] = df['return'].shift(1)
    df['return_lag_2'] = df['return'].shift(2)
    df['return_lag_3'] = df['return'].shift(3)
    df['return_lag_4'] = df['return'].shift(4)
    df['return_lag_5'] = df['return'].shift(5)
    
    print("特徵計算完成。")
    return df

# --- 步驟 3: 載入「全新」數據並進行測試 ---

# 3a. 載入模型
MODEL_FILENAME = 'ETH_USDT_5m.json' # (假設您已儲存)
print(f"正在載入模型: {MODEL_FILENAME}...")
model = xgb.XGBRegressor()
model.load_model(MODEL_FILENAME)
print("模型載入成功。")

# 3b. 載入「全新」的樣本外資料
# (您必須提供這份 CSV 檔案)
NEW_DATA_FILE = 'ETH_USDT_5m_data.csv' 
print(f"正在載入全新資料: {NEW_DATA_FILE}...")
# (請確保您的 CSV 欄位名稱是 'Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume')
try:
    forward_test_df = pd.read_csv(
        NEW_DATA_FILE, 
        parse_dates=['Timestamp'], 
        index_col='Timestamp'
    )
except FileNotFoundError:
    print(f"錯誤：找不到 '{NEW_DATA_FILE}'。請將您的新數據放在同一個資料夾。")
    # 這裡我們用一個假的 DataFrame 來讓腳本繼續，但您必須替換它
    forward_test_df = pd.DataFrame() # 替換我！

if not forward_test_df.empty:
    print(f"成功載入 {len(forward_test_df)} 筆新資料。")

    # 3c. 計算「全新資料」的特徵
    # (注意：為了計算 MA/RSI，真實的 'forward_test_df' 
    # 必須包含前面至少 100 根 K 棒的歷史資料)
    forward_test_df = calculate_all_features(forward_test_df)
    
    # 3d. 計算「全新資料」的目標 (y) - (僅供回測比較用)
    forward_test_df['Close_SMA'] = talib.SMA(forward_test_df['Close'], 3)
    forward_test_df['target'] = forward_test_df['Close'].shift(-HORIZON) / forward_test_df['Close_SMA'] - 1

    # 3e. 清理所有因特徵計算產生的 NaN
    forward_test_df.dropna(subset=FEATURES_LIST + ['target'], inplace=True)
    
    # --- 步驟 4: 在「全新資料」上產生預測 ---
    print("模型正在對全新資料進行預測...")
    X_forward = forward_test_df[FEATURES_LIST]
    y_forward_test = forward_test_df['target']
    
    forward_test_df['y_pred'] = model.predict(X_forward)

    # (可選) 檢查「全新資料」上的準確率
    actual_direction = np.sign(y_forward_test)
    predicted_direction = np.sign(forward_test_df['y_pred'])
    actual_direction[actual_direction == 0] = 1
    predicted_direction[predicted_direction == 0] = 1
    new_accuracy = (actual_direction == predicted_direction).mean()
    market_up = (y_forward_test > 0).mean()
    print(f"--- 全新資料測試結果 ---")
    print(f"方向準確率: {new_accuracy * 100:.2f}%")
    print(f"真實市場上漲比例: {market_up * 100:.2f}%")

    # --- 步驟 5: 在「全新資料」上執行「含手續費」的回測 ---
    print("正在執行含手續費的回測...")
    
    # 產生訊號
    forward_test_df['signal'] = 0
    forward_test_df.loc[forward_test_df['y_pred'] > THRESHOLD, 'signal'] = 1
    forward_test_df.loc[forward_test_df['y_pred'] < -THRESHOLD, 'signal'] = -1
    
    # 計算毛收益
    forward_test_df['strategy_return'] = forward_test_df['signal'].shift(1) * forward_test_df['target']
    
    # 計算手續費
    forward_test_df['trades'] = forward_test_df['signal'].diff().abs().fillna(0)
    forward_test_df['transaction_costs'] = forward_test_df['trades'] * FEE_RATE
    
    # 計算淨收益
    forward_test_df['strategy_net_return'] = forward_test_df['strategy_return'] - forward_test_df['transaction_costs']
    
    # 計算 Buy & Hold (比較基準)
    forward_test_df['buy_and_hold_return'] = forward_test_df['target']
    
    # 計算淨值曲線
    forward_test_df['strategy_net_equity'] = forward_test_df['strategy_net_return'].cumsum()
    forward_test_df['buy_and_hold_equity'] = forward_test_df['buy_and_hold_return'].cumsum()

    # --- 步驟 6: 繪製「全新資料」的最終結果 ---
    print("繪製「全新資料」回測圖...")
    plt.figure(figsize=(14, 7))
    forward_test_df['strategy_net_equity'].plot(label='模型策略 (扣費後)', color='blue')
    forward_test_df['buy_and_hold_equity'].plot(label='買入並持有 (Buy & Hold)', color='gray', linestyle='--')
    plt.title(f"「前向測試」回測 (已計入手續費 {FEE_RATE*100}%)")
    plt.ylabel('累計收益率 (Cumulative Returns)')
    plt.legend()
    plt.grid(True)
    plt.show()

else:
    print("腳本停止，因為沒有載入新的測試資料。")