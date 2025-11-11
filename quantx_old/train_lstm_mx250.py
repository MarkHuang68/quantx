import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader  # 加分批
import ccxt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# GPU設定 (MX250優化)
torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用設備: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
if torch.cuda.is_available():
    print(f"GPU VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

np.random.seed(42)

# 生成合成數據 (同前)
def generate_synthetic_data(trend='up', volatility=0.01, length=1000):
    time = np.arange(length)
    base = time * 0.01 if trend == 'up' else -time * 0.01
    noise = np.random.normal(0, volatility, length)
    prices = 100 + base + np.cumsum(noise)
    return pd.DataFrame({'price': prices})

# 準備數據 (同前)
def prepare_data(data, time_step=60):
    data_array = data.values.reshape(-1, 1)
    data_min, data_max = np.min(data_array), np.max(data_array)
    scaled_data = (data_array - data_min) / (data_max - data_min + 1e-8)
    X, y = [], []
    for i in range(len(scaled_data) - time_step):
        X.append(scaled_data[i:i+time_step, 0])
        y.append(scaled_data[i+time_step, 0])
    X = np.array(X).reshape(len(X), time_step, 1)
    y = np.array(y)
    return X, y, (data_min, data_max)

# LSTM模型 (減hidden=32, MX250友好)
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 32, batch_first=True)  # 減32, 省VRAM
        self.fc = nn.Linear(32, 1)
    
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(1, batch_size, 32).to(x.device)
        c0 = torch.zeros(1, batch_size, 32).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

# 訓練模型 (加DataLoader分批)
def train_model(X, y, epochs=50, lr=0.0005, batch_size=64):
    model = LSTMModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 加DataLoader
    dataset = TensorDataset(torch.FloatTensor(X).to(device), torch.FloatTensor(y).to(device).reshape(-1, 1))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        if epoch % 10 == 0:
            print(f"    Epoch {epoch}, Avg Loss: {avg_loss:.6f}")
    
    model.eval()
    with torch.no_grad():
        X_full = torch.FloatTensor(X).to(device)
        predicted = model(X_full).cpu().numpy().squeeze()
    mse = np.mean((y - predicted)**2)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # 清快取
    return model, mse, predicted

# 下載BTC數據 (同前)
def fetch_btc_data():
    exchange = ccxt.binance({'timeout': 60000})
    ohlcv = exchange.fetch_ohlcv('BTCUSDT', '5m', limit=1000)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['price'] = df['close']
    return df[['price']]

# 畫圖 (同前)
def plot_prediction(y_true, y_pred, title):
    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title(title)
    plt.legend()
    filename = f"{title.replace(' ', '_')}.png"
    plt.savefig(filename)
    plt.close()
    print(f"  - 圖表存檔: {filename}")

def main():
    # 步驟1-3 同前 (上漲/下跌/BTC, 用train_model)
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
    
    print("\n=== 步驟3: 實際BTC數據回測 ===")
    btc_prices = fetch_btc_data()
    X_btc, y_btc, _ = prepare_data(btc_prices)
    model_btc, mse_btc, pred_btc = train_model(X_btc, y_btc)
    print(f"BTC實際數據 MSE: {mse_btc:.6f}")
    plot_prediction(y_btc, pred_btc, "BTC_Prediction")
    
    last_seq = torch.FloatTensor(X_btc[-1:]).to(device)
    with torch.no_grad():
        pred = model_btc(last_seq).cpu().numpy().item()
    print(f"最後預測 (scaled): {pred:.4f} vs 實際: {y_btc[-1]:.4f}")
    
    print("\nGPU訓練完成！監控nvidia-smi看利用率。")

if __name__ == "__main__":
    main()