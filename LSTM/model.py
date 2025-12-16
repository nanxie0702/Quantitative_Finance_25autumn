import torch
import torch.nn as nn
from config import Config

class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.hidden_size = Config.hidden_size
        self.num_layers = Config.num_layers
        
        self.lstm = nn.LSTM(
            input_size=Config.input_size,
            hidden_size=Config.hidden_size,
            num_layers=Config.num_layers,
            batch_first=True,
            dropout=Config.dropout
        )
        
        self.fc = nn.Linear(Config.hidden_size, Config.output_size)
        
    def forward(self, x):
        # 初始化隐藏状态 (h0, c0)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # 前向传播
        out, _ = self.lstm(x, (h0, c0))
        
        # 取最后一个时间步的输出
        out = out[:, -1, :]
        out = self.fc(out)
        return out