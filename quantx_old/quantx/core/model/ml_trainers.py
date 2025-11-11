# 檔案: quantx/core/model/ml_trainers.py
# 版本: v6 (最終完美版)
# 說明:
# - 採納了 LabelEncoder().fit_transform() 的核心思想，從根本上解決 ValueError。
# - 優化了 predict_proba 後的機率還原邏輯，確保在任何標籤組合下都能正確映射。

from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Any
from sklearn.preprocessing import LabelEncoder

from .xgb_utils import build_xgb

def train_predict_xgb(X_is: pd.DataFrame, y_is: pd.Series, X_oos: pd.DataFrame, params: Dict[str, Any]) -> np.ndarray:
    """
    使用 XGBoost 進行訓練和預測 (最終完美版)。
    """
    # 1. 檢查訓練數據中的標籤種類是否足夠進行分類
    if y_is.nunique() < 2:
        # 若標籤單一，則回傳中性預測 (機率集中在類別 1)
        return np.array([[0.0, 1.0, 0.0]] * len(X_oos))

    # 2. 核心：使用 LabelEncoder 將當前數據的標籤從 0 開始編碼
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_is)

    # 3. 始終為三分類問題建立模型
    model = build_xgb(num_class=3, **params)
    
    # 4. 使用編碼後的標籤進行訓練
    model.fit(X_is, y_train_encoded)
    
    # 5. 預測機率
    probabilities = model.predict_proba(X_oos)
    
    # 6. 核心優化：建立一個全零的完整機率矩陣，並將預測出的機率還原到其「原始標籤」對應的位置
    #    例如，如果 y_is 只有 [1, 2]，le.classes_ 會是 [1, 2]，
    #    這段程式碼能確保 proba 的第一欄 (對應標籤1) 填入 full_proba 的第1欄，
    #    proba 的第二欄 (對應標籤2) 填入 full_proba 的第2欄。
    full_proba = np.zeros((len(X_oos), 3))
    original_classes_in_model = le.classes_
    for i, class_label in enumerate(original_classes_in_model):
        full_proba[:, class_label] = probabilities[:, i]

    return full_proba


# (後續的 LSTM 等函式保持不變)
class _SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=1, output_dim=3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def _create_sequences(X: pd.DataFrame, y: pd.Series, seq_len: int):
    X_seq, y_seq = [], []
    X_vals, y_vals = X.values, y.values
    for i in range(seq_len, len(X_vals) + 1):
        X_seq.append(X_vals[i-seq_len:i, :])
        y_seq.append(y_vals[i-1])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.int64)

def train_predict_lstm(X_is: pd.DataFrame, y_is: pd.Series, X_oos: pd.DataFrame, params: Dict[str, Any]) -> np.ndarray:
    seq_len = params.get("seq_len", 60)
    hidden_dim = params.get("hidden_dim", 32)
    epochs = params.get("epochs", 5)
    batch_size = params.get("batch_size", 64)
    lr = params.get("lr", 0.001)
    num_classes = 3
    X_train_seq, y_train_seq = _create_sequences(X_is, y_is, seq_len)
    if len(X_train_seq) == 0:
        return np.full((len(X_oos), num_classes), 1.0 / num_classes)
    train_dataset = TensorDataset(torch.from_numpy(X_train_seq), torch.from_numpy(y_train_seq))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    input_dim = X_is.shape[1]
    model = _SimpleLSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
    X_combined = pd.concat([X_is.tail(seq_len - 1), X_oos])
    X_predict_seq_list = []
    for i in range(len(X_oos)):
        window = X_combined.iloc[i : i + seq_len]
        if len(window) < seq_len:
            break
        X_predict_seq_list.append(window.values)
    if not X_predict_seq_list:
        return np.full((len(X_oos), num_classes), [0,1,0])
    X_predict_seq = np.array(X_predict_seq_list, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        outputs = model(torch.from_numpy(X_predict_seq))
        probabilities = torch.softmax(outputs, dim=1).numpy()
    if len(probabilities) < len(X_oos):
        missing_count = len(X_oos) - len(probabilities)
        padding = np.full((missing_count, num_classes), [0.0, 1.0, 0.0])
        probabilities = np.vstack([probabilities, padding])
    return probabilities

def train_predict_hybrid(X_is: pd.DataFrame, y_is: pd.Series, X_oos: pd.DataFrame, params: Dict[str, Any]) -> np.ndarray:
    print(f"[WFO-ML] 'hybrid' 模型目前尚未支援 WFO 訓練，跳過。")
    num_classes = 3
    return np.full((len(X_oos), num_classes), 1.0 / num_classes)