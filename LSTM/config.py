import torch

class Config:
    # 文件路径
    # data_path = 'SZ50day.xlsx'
    # data_path = 'KC50ETFday.xlsx'
    # data_path = 'ZZ1000ETFday.xlsx'
    data_path = 'HSKJETFday.xlsx'
    model_save_path = 'best_lstm_model.pth'
    
    # 数据参数
    feature_columns = ['open', 'high', 'low', 'close', 'turnover', 'volume', 'amount']
    target_column = 'close'
    seq_length = 20       # 用过去20天预测下一天
    train_split = 0.8     # 80% 训练，20% 测试
    
    # 模型超参数
    input_size = len(feature_columns)
    hidden_size = 64
    num_layers = 2
    output_size = 1
    dropout = 0.2
    
    # 训练参数
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 50
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 回测参数
    initial_capital = 1000000.0  # 初始资金 100万
    commission_rate = 0.0003     # 万三佣金
    buy_threshold = 0.003        # 买入的预测涨幅阈值（过滤噪音）
    sell_threshold = -0.001      # 卖出的预测涨幅阈值（过滤噪音）