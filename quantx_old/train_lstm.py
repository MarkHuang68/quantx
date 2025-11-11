import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import ccxt
import matplotlib
matplotlib.use('Agg')  # 改非互動後端，避免plt.show()卡住
import matplotlib.pyplot as plt

# 設定設備
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(42)  # 固定隨機，MSE穩定

print(f"開始執行，使用設備: {device}")

# 生成合成數據
def generate_synthetic_data(trend='up', volatility=0.01, length=1000):
    time = np.arange(length)
    base = time * 0.01 if trend == 'up' else -time * 0.01
    noise = np.random.normal(0, volatility, length)
    prices = 100 + base + np.cumsum(noise)
    return pd.DataFrame({'price': prices})

# 準備數據 (加debug)
def prepare_data(data, time_step=60):
    try:
        print(f"  - 輸入數據形狀: {data.shape}")
        data_array = data.values.reshape(-1, 1)
        data_min, data_max = np.min(data_array), np.max(data_array)
        scaled_data = (data_array - data_min) / (data_max - data_min + 1e-8)
        X, y = [], []
        num_samples = len(scaled_data) - time_step
        print(f"  - 可用樣本數: {num_samples}")
        if num_samples <= 0:
            raise ValueError("數據太短，無法生成序列")
        for i in range(num_samples):
            X.append(scaled_data[i:i+time_step, 0])
            y.append(scaled_data[i+time_step, 0])
        X = np.array(X).reshape(len(X), time_step, 1)
        y = np.array(y)
        print(f"  - 輸出 X shape: {X.shape}, y length: {len(y)}")
        return X, y, (data_min, data_max)
    except Exception as e:
        print(f"  - prepare_data 錯誤: {e}")
        return np.array([]), np.array([]), (0, 0)

# LSTM模型
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 50, batch_first=True)
        self.fc = nn.Linear(50, 1)
    
    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), 50).to(device)
        c0 = torch.zeros(1, x.size(0), 50).to(device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

# 訓練模型 (加debug)
def train_model(X, y, epochs=100, lr=0.0005):
    try:
        print(f"  - 開始訓練，X shape: {X.shape}")
        if len(X) == 0:
            raise ValueError("無訓練數據")
        model = LSTMModel().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        X_tensor = torch.FloatTensor(X).to(device)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1).to(device)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            if epoch % 20 == 0:
                print(f"    Epoch {epoch}, Loss: {loss.item():.6f}")
        
        model.eval()
        with torch.no_grad():
            predicted = model(X_tensor).cpu().numpy().squeeze()
        mse = np.mean((y - predicted)**2)
        print(f"  - 訓練完成，MSE: {mse:.6f}")
        return model, mse, predicted
    except Exception as e:
        print(f"  - train_model 錯誤: {e}")
        return None, float('inf'), None

# 下載BTC數據 (加debug)
def fetch_btc_data():
    try:
        print("  - 連線Binance...")
        exchange = ccxt.binance({'timeout': 60000})
        ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1h', limit=1000)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['price'] = df['close']
        print(f"  - 下載成功: {len(df)} 筆數據")
        print(f"  - 價格範圍: {df['price'].min():.2f} ~ {df['price'].max():.2f}")
        print(f"  - 最後5筆: {df['price'].tail().tolist()}")
        return df[['price']]
    except Exception as e:
        print(f"  - 下載錯誤: {e}")
        raise ValueError(f"ccxt失敗: {e}")

# 畫圖 (存檔)
def plot_prediction(y_true, y_pred, title):
    try:
        if len(y_true) > 0 and y_pred is not None:
            plt.figure(figsize=(10, 5))
            plt.plot(y_true, label='Actual')
            plt.plot(y_pred, label='Predicted')
            plt.title(title)
            plt.legend()
            filename = f"{title.replace(' ', '_')}.png"
            plt.savefig(filename)
            plt.close()
            print(f"  - 圖表存檔: {filename}")
        else:
            print("  - 無法畫圖: 數據空")
    except Exception as e:
        print(f"  - 畫圖錯誤: {e}")

def main():
    # 步驟1: 上漲
    print("\n=== 步驟1: 上漲波動訓練 ===")
    data_up_low = generate_synthetic_data('up', 0.01)
    X_up_low, y_up_low, _ = prepare_data(data_up_low)
    model_up_low, mse_up_low, pred_up_low = train_model(X_up_low, y_up_low)
    print(f"上漲低波動 MSE: {mse_up_low:.6f}")
    plot_prediction(y_up_low, pred_up_low, "Up_Low_Prediction")
    
    data_up_high = generate_synthetic_data('up', 0.05)
    X_up_high, y_up_high, _ = prepare_data(data_up_high)
    model_up_high, mse_up_high, pred_up_high = train_model(X_up_high, y_up_high)
    print(f"上漲高波動 MSE: {mse_up_high:.6f}")
    plot_prediction(y_up_high, pred_up_high, "Up_High_Prediction")
    
    # 步驟2: 下跌
    print("\n=== 步驟2: 下跌波動訓練 ===")
    data_down_low = generate_synthetic_data('down', 0.01)
    X_down_low, y_down_low, _ = prepare_data(data_down_low)
    model_down_low, mse_down_low, pred_down_low = train_model(X_down_low, y_down_low)
    print(f"下跌低波動 MSE: {mse_down_low:.6f}")
    plot_prediction(y_down_low, pred_down_low, "Down_Low_Prediction")
    
    data_down_high = generate_synthetic_data('down', 0.05)
    X_down_high, y_down_high, _ = prepare_data(data_down_high)
    model_down_high, mse_down_high, pred_down_high = train_model(X_down_high, y_down_high)
    print(f"下跌高波動 MSE: {mse_down_high:.6f}")
    plot_prediction(y_down_high, pred_down_high, "Down_High_Prediction")
    
    # 步驟3: 實際數據
    print("\n=== 步驟3: 實際BTC數據回測 ===")
    btc_prices = fetch_btc_data()
    print("下載完畢，開始準備數據...")
    X_btc, y_btc, _ = prepare_data(btc_prices)
    print(f"準備完成，數據長度: {len(btc_prices)}, X shape: {X_btc.shape if len(X_btc)>0 else 'Empty'}")
    if len(X_btc) > 0:
        model_btc, mse_btc, pred_btc = train_model(X_btc, y_btc)
        print(f"BTC實際數據 MSE: {mse_btc:.6f}")
        plot_prediction(y_btc, pred_btc, "BTC_Prediction")
        
        # 預測範例
        last_seq = torch.FloatTensor(X_btc[-1:]).to(device)
        with torch.no_grad():
            pred = model_btc(last_seq).cpu().numpy().item()  # .item() 取scalar
        print(f"最後預測 (scaled): {pred:.4f} vs 實際: {y_btc[-1]:.4f}")
    else:
        print("數據不足（<61筆），無法訓練。檢查下載。")
    
    print("\n腳本結束！")

if __name__ == "__main__":
    main()