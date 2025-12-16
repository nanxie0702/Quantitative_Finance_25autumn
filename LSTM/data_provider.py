import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
from config import Config

class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_data():
    # 1. 读取数据
    df = pd.read_excel(Config.data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # 提取特征和标签数据
    data = df[Config.feature_columns].values
    
    # 2. 划分训练集和测试集
    train_size = int(len(data) * Config.train_split)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    # 3. 归一化 (只在训练集上fit)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_data)
    
    train_scaled = scaler.transform(train_data)
    test_scaled = scaler.transform(test_data)
    
    # 获取 close 列在特征中的索引，用于后续反归一化
    target_idx = Config.feature_columns.index(Config.target_column)
    
    return train_scaled, test_scaled, scaler, target_idx, df, train_size

def create_sequences(data, seq_length, target_idx):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i : i + seq_length]
        y = data[i + seq_length, target_idx] # 预测下一天的 close
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def get_dataloaders():
    train_scaled, test_scaled, scaler, target_idx, df, train_split_idx = get_data()
    
    X_train, y_train = create_sequences(train_scaled, Config.seq_length, target_idx)
    X_test, y_test = create_sequences(test_scaled, Config.seq_length, target_idx)
    
    train_dataset = StockDataset(X_train, y_train)
    test_dataset = StockDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    # 测试集不shuffle，保持时间顺序
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False)
    
    return train_loader, test_loader, scaler, target_idx, df, train_split_idx