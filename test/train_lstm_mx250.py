import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import ccxt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings

# 忽略 Scikit-learn 中因預測全為 0/1 產生的警告
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# --- 基礎設定 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用設備: {device}")
np.random.seed(42)

# --- 模型定義 (修改為分類) ---
class LSTMClassifier(nn.Module):
    # input_size=2 (Price+Vol), hidden_size=64, num_layers=2
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, output_size=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.1)
        
        # *** 關鍵修改：輸出 2 個節點 (0=跌, 1=漲) ***
        self.fc = nn.Linear(hidden_size, output_size) 
    
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        # 輸出 [batch_size, 2] 的原始分數 (logits)
        return self.fc(out[:, -1, :])

# --- 輔助函數 ---

def fetch_btc_data(length=2500):
    print(f"  > 正在下載 {length} 筆 BTC 5分鐘 K線數據...")
    try:
        exchange = ccxt.binance({'timeout': 60000})
        ohlcv = exchange.fetch_ohlcv('BTCUSDT', '5m', limit=length)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        print("  > BTC 數據下載完成。")
        return df[['close', 'volume']] 
    except Exception as e:
        print(f"  > 數據下載失敗: {e}。")
        return None

def create_sequences_classification(scaled_features, targets, time_step):
    """
    創建分類任務的序列
    X: (N, time_step, num_features)
    y: (N,) - 標籤
    """
    X, y = [], []
    for i in range(len(scaled_features) - time_step):
        # X: 過去 60 筆的特徵 [price_diff, vol_diff]
        X.append(scaled_features[i : i + time_step, :])
        
        # y: 預測 "序列結尾" 的 "下一個" 時間點的方向
        # 我們在 main 函數中已經對齊了
        y.append(targets[i + time_step - 1]) 
    
    X = np.array(X)
    y = np.array(y)
    
    X_tensor = torch.FloatTensor(X)
    # *** 關鍵修改：分類標籤必須是 LongTensor (整數) ***
    y_tensor = torch.LongTensor(y) 
    return X_tensor, y_tensor

def run_training_loop(model, criterion, optimizer, X_train, y_train, epochs, batch_size):
    """(修改) 訓練函數，加入準確率計算"""
    model.train()
    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
    
    print(f"  > 開始 {epochs} 個 Epochs 的訓練...")
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        
        for seq, labels in train_loader:
            seq, labels = seq.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # y_pred 的 shape 是 [batch_size, 2]
            y_pred = model(seq) 
            
            # CrossEntropyLoss 會自動處理 softmax
            loss = criterion(y_pred, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # 計算準確率
            # torch.max(y_pred.data, 1) 返回 (values, indices)
            _, predicted_labels = torch.max(y_pred.data, 1)
            epoch_total += labels.size(0)
            epoch_correct += (predicted_labels == labels).sum().item()
            
        avg_loss = epoch_loss / len(train_loader)
        avg_acc = (epoch_correct / epoch_total) * 100
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:3d} / {epochs} | Loss: {avg_loss:.6f} | 訓練準確率: {avg_acc:.2f}%")
    print("  > 訓練完成。")

def evaluate_model_classification(model, X_test, y_test, batch_size):
    """(修改) 評估分類模型"""
    model.eval()
    test_data = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for seq, labels in test_loader:
            seq = seq.to(device)
            
            outputs = model(seq)
            _, predicted_labels = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted_labels.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return all_labels, all_predictions

# --- 主程式 (數據處理邏輯修改) ---

def main():
    # --- 參數設定 ---
    SEQ_LENGTH = 60
    BATCH_SIZE = 64
    EPOCHS = 150 # 分類任務可能需要更多時間收斂
    DATA_LENGTH = 2500
    TRAIN_SPLIT_RATIO = 0.8
    
    # === 步驟 1: 獲取並處理數據 ===
    print("\n=== 步驟 1: 獲取 BTC 數據並創建特徵與標籤 ===")
    btc_data = fetch_btc_data(length=DATA_LENGTH)
    if btc_data is None:
        print("無法執行，程式終止。")
        return

    # 1.1 創建 "特徵" (X)
    # 我們仍然使用 "變化量" 作為特徵
    features_df = pd.DataFrame(index=btc_data.index)
    features_df['price_diff'] = btc_data['close'].diff()
    features_df['volume_diff'] = btc_data['volume'].diff()
    
    # 1.2 創建 "標籤" (y)
    # 預測 "未來" 的方向
    # .shift(-1) 是看 "下一筆" 的價格
    targets_sr = (btc_data['close'].shift(-1) > btc_data['close']).astype(int)
    
    # 1.3 合併並清理
    # 將 'target' 合併到特徵 DataFrame 中
    full_data = features_df
    full_data['target'] = targets_sr
    
    # 由於 .diff() 和 .shift(-1) 會產生 NaN，我們全部移除
    full_data = full_data.dropna()
    
    # 分離 X (特徵) 和 y (標籤)
    y_data = full_data['target'].values
    X_data = full_data.drop('target', axis=1).values # Shape (N, 2)
    
    # 1.4 分割數據
    split_index = int(len(X_data) * TRAIN_SPLIT_RATIO)
    
    X_train_raw = X_data[:split_index]
    y_train_raw = y_data[:split_index]
    
    X_test_raw = X_data[split_index:]
    y_test_raw = y_data[split_index:]
    
    # 1.5 Scaler 處理 (只對 "特徵 X" 做)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_X_train = scaler.fit_transform(X_train_raw)
    scaled_X_test = scaler.transform(X_test_raw)
    print("  > Scaler 已在 (Price+Vol Diff) 訓練集上擬合。")

    # === 步驟 2: 創建序列並訓練 ===
    print("\n=== 步驟 2: 創建分類序列並訓練模型 ===")
    
    X_train, y_train = create_sequences_classification(scaled_X_train, y_train_raw, SEQ_LENGTH)
    X_test, y_test = create_sequences_classification(scaled_X_test, y_test_raw, SEQ_LENGTH)
    
    print(f"  > 訓練序列數量: {len(X_train)}")
    print(f"  > 測試序列數量: {len(X_test)}")

    # *** 關鍵修改：使用 LSTMClassifier 和 CrossEntropyLoss ***
    model = LSTMClassifier(input_size=2, hidden_size=64, num_layers=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    run_training_loop(model, criterion, optimizer, X_train, y_train, EPOCHS, BATCH_SIZE)
    
    # === 步驟 3: 評估與回測 ===
    print("\n=== 步驟 3: 評估分類結果 ===")
    
    # 3.1 評估 "訓練集"
    labels_train, preds_train = evaluate_model_classification(model, X_train, y_train, BATCH_SIZE)
    acc_train = accuracy_score(labels_train, preds_train) * 100
    print(f"  > 最終訓練準確率: {acc_train:.2f}%")
    
    # 3.2 評估 "測試集" (真正的回測)
    labels_test, preds_test = evaluate_model_classification(model, X_test, y_test, BATCH_SIZE)
    acc_test = accuracy_score(labels_test, preds_test) * 100
    print(f"  > **最終測試準確率: {acc_test:.2f}%**")

    # 3.3 顯示詳細報告
    print("\n--- 測試集詳細報告 ---")
    print(classification_report(labels_test, preds_test, target_names=['0_Down/Same', '1_Up']))
    
    print("\n--- 測試集混淆矩陣 (Confusion Matrix) ---")
    #     Predicted:
    #       0   1
    # Act.0 [TN, FP]
    # Act.1 [FN, TP]
    cm = confusion_matrix(labels_test, preds_test)
    print(cm)
    
    # 檢查模型是否只猜單邊
    if len(np.unique(preds_test)) == 1:
        print("\n*** 警告：模型在測試集上只預測了單一類別！***")
        print(f"*** 它只猜 {np.unique(preds_test)[0]}。這和『猜平均值』是同樣的問題。***")
    
    torch.save(model.state_dict(), 'classification_model.pt')
    print(f"\n分類回測完成！模型已保存為 'classification_model.pt'。")

if __name__ == "__main__":
    main()