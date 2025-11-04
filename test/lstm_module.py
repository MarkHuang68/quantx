import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

def prepare_data(data, time_step=60):
    """準備數據：標準化 + 序列生成"""
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

def train_lstm_model(X, y, epochs=100, lr=0.001, batch_size=64):
    """訓練LSTM模型"""
    model = LSTMModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
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
        torch.cuda.empty_cache()
    return model, mse, predicted

def predict_with_lstm(model, data, scaler=None):
    """用LSTM預測下一價"""
    X, _, _ = prepare_data(pd.DataFrame({'price': data}))
    if len(X) == 0:
        return 0
    X_tensor = torch.FloatTensor(X[-1:]).to(device)
    with torch.no_grad():
        pred_scaled = model(X_tensor).cpu().numpy().item()
    if scaler:
        pred_price = pred_scaled * (scaler[1] - scaler[0]) + scaler[0]
        return pred_price
    return pred_scaled

# 測試模組 (獨立跑)
if __name__ == "__main__":
    # 合成數據測試
    data = pd.DataFrame({'price': np.random.rand(1000) * 100})
    X, y, scaler = prepare_data(data)
    model, mse, pred = train_lstm_model(X, y)
    print(f"模組測試 MSE: {mse:.6f}")
    print("LSTM模組 OK！")