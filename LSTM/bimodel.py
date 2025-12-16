import torch
import torch.nn as nn
from config import Config

class BiLSTMModel(nn.Module):
    def __init__(self):
        super(BiLSTMModel, self).__init__()
        self.hidden_size = Config.hidden_size
        self.num_layers = Config.num_layers
        
        # 1. 设置 bidirectional=True
        self.lstm = nn.LSTM(
            input_size=Config.input_size,
            hidden_size=Config.hidden_size,
            num_layers=Config.num_layers,
            batch_first=True,
            dropout=Config.dropout,
            bidirectional=True  # 开启双向
        )
        
        # 2. 全连接层的输入维度变为 hidden_size * 2
        # 因为双向LSTM会将正向和反向的输出拼接在一起
        self.fc = nn.Linear(Config.hidden_size * 2, Config.output_size)
        
    def forward(self, x):
        # 3. 初始化隐藏状态，层数维度需要 * 2 (num_layers * 2)
        # 形状: (num_layers * num_directions, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        # 前向传播
        # out 的形状: (batch_size, seq_len, hidden_size * 2)
        out, _ = self.lstm(x, (h0, c0))
        
        # 取最后一个时间步的输出
        # out[:, -1, :] 包含了正向LSTM的最后时刻输出 和 反向LSTM的最后时刻输出（即序列开头的反向处理结果）
        out = out[:, -1, :]
        
        out = self.fc(out)
        return out