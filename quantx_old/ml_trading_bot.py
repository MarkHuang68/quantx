# ml_trading_eth.py
import ccxt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
import time
import logging
import os

# ==================== 1. Bybit 設定 ====================
exchange = ccxt.bybit({
    'apiKey': 'amYOf9q40Iz7kKMWwD',      # 填你的實盤 Key
    'secret': 'MBcaFWRdhQoONpwp46M5t8maf8tYLSXfp551',
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'}
})

SYMBOL = 'ETHUSDT'          # ETH 專用
TIMEFRAME = '5m'
LIMIT = 1000
LEVERAGE = 2
POSITION_PCT = 0.4          # 40% 資金 = 160 USDT
MODEL_PATH = 'ml_eth_model.h5'

# ==================== 2. 模型與縮放器 ====================
scaler = MinMaxScaler()
model = None

# ==================== 3. 資料獲取 ====================
def get_data():
    ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=LIMIT)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# ==================== 4. 特徵工程 ====================
def create_features(df):
    df['return'] = df['close'].pct_change()
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    df['rsi'] = compute_rsi(df['close'])
    df['volatility'] = df['return'].rolling(20).std()
    df['bb_upper'] = df['ma20'] + 2 * df['close'].rolling(20).std()
    df['bb_lower'] = df['ma20'] - 2 * df['close'].rolling(20).std()
    df = df.dropna().reset_index(drop=True)
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# ==================== 5. 模型訓練 ====================
def train_model():
    global model, scaler
    df = get_data()
    df = create_features(df)
    
    features = ['return', 'ma5', 'ma20', 'rsi', 'volatility', 'bb_upper', 'bb_lower']
    X = df[features].values
    y = (df['close'].shift(-5) > df['close']).astype(int).values[:-5]
    X = X[:-5]
    
    X_scaled = scaler.fit_transform(X)
    X_seq = []
    for i in range(60, len(X_scaled)):
        X_seq.append(X_scaled[i-60:i])
    X_seq = np.array(X_seq)
    y_seq = y[60:]
    
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(60, len(features))),
        Dropout(0.2),
        LSTM(100),
        Dropout(0.2),
        Dense(50, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    logging.info("ETH 模型訓練中...")
    model.fit(X_seq, y_seq, epochs=15, batch_size=32, verbose=1)
    model.save(MODEL_PATH)
    logging.info("ETH 模型已儲存")

# ==================== 6. 預測 ====================
def predict():
    if not model:
        return None
    df = get_data()
    df = create_features(df)
    if len(df) < 60:
        return None
    
    features = ['return', 'ma5', 'ma20', 'rsi', 'volatility', 'bb_upper', 'bb_lower']
    X = df[features].values[-60:]
    X_scaled = scaler.transform(X)
    X_seq = X_scaled.reshape(1, 60, len(features))
    
    prob = model.predict(X_seq, verbose=0)[0][0]
    return 'BUY' if prob > 0.65 else 'SELL' if prob < 0.35 else None

# ==================== 7. 交易執行 ====================
current_position = None

def get_balance():
    balance = exchange.fetch_balance()
    return balance['total'].get('USDT', 0)

def get_price():
    return exchange.fetch_ticker(SYMBOL)['last']

def close_position():
    global current_position
    if not current_position:
        return
    try:
        side = 'sell' if current_position == 'BUY' else 'buy'
        size = abs(current_position)
        exchange.create_market_order(SYMBOL, side, size)
        logging.info(f"ETH 平倉 {current_position} {size:.5f} 張")
        current_position = None
    except Exception as e:
        logging.error(f"ETH 平倉失敗: {e}")

def open_position(signal):
    global current_position
    if current_position == signal:
        return
    close_position()
    
    usdt = get_balance() * POSITION_PCT
    price = get_price()
    size = round((usdt * LEVERAGE) / price, 5)
    if size < 0.01:
        return
    
    try:
        side = 'buy' if signal == 'BUY' else 'sell'
        exchange.create_market_order(SYMBOL, side, size)
        current_position = signal if signal == 'BUY' else -size
        logging.info(f"ETH 開倉 {signal} {size:.5f} 張 @ {price}")
    except Exception as e:
        logging.error(f"ETH 開倉失敗: {e}")

# ==================== 8. 主循環 ====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')

if __name__ == "__main__":
    try:
        exchange.set_leverage(LEVERAGE, SYMBOL)
        logging.info(f"ETH 槓桿設定: {LEVERAGE}x")
    except: pass
    
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        logging.info("ETH 模型已載入")
    else:
        train_model()
    
    logging.info("ETH 機器學習 Bot 實盤啟動！")
    
    while True:
        try:
            signal = predict()
            if signal:
                open_position(signal)
            time.sleep(60)
        except KeyboardInterrupt:
            close_position()
            logging.info("ETH Bot 已停止")
            break
        except Exception as e:
            logging.error(f"ETH 錯誤: {e}")
            time.sleep(10)