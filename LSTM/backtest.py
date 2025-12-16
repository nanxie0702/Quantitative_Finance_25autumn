import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import Config
from model import LSTMModel
from bimodel import BiLSTMModel
from data_provider import get_dataloaders

def run_backtest():
    # 1. 准备数据和模型
    _, test_loader, scaler, target_idx, df, train_split_idx = get_dataloaders()
    
    # model = LSTMModel().to(Config.device)
    model = BiLSTMModel().to(Config.device)
    try:
        model.load_state_dict(torch.load(Config.model_save_path))
    except FileNotFoundError:
        print(f"错误：未找到模型文件 {Config.model_save_path}。请先运行 train.py 进行训练。")
        return
        
    model.eval()
    
    # 2. 模型预测
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(Config.device)
            outputs = model(X_batch)
            predictions.extend(outputs.cpu().numpy().flatten())
            actuals.extend(y_batch.numpy().flatten())
            
    # 3. 反归一化
    def inverse_transform_col(scaled_data, scaler, col_idx):
        dummy = np.zeros((len(scaled_data), len(Config.feature_columns)))
        dummy[:, col_idx] = scaled_data
        return scaler.inverse_transform(dummy)[:, col_idx]

    pred_prices = inverse_transform_col(predictions, scaler, target_idx)
    actual_prices = inverse_transform_col(actuals, scaler, target_idx)
    
    # 对齐日期 (本数据集的测试集长度为 N=223)
    test_dates = df['date'].iloc[train_split_idx + Config.seq_length:].reset_index(drop=True)
    
    # 4. 回测逻辑
    capital = Config.initial_capital
    position = 0
    portfolio_values = []
    
    trade_log = []
    buy_dates, buy_prices = [], []
    sell_dates, sell_prices = [], []
    
    # 循环从 i=0 到 N-2 (共 222 天的交易决策)
    for i in range(len(actual_prices) - 1): # i runs from 0 to 221
        curr_price = actual_prices[i]
        next_pred_price = pred_prices[i+1] # 预测明天的价格
        date = test_dates[i]
        
        expected_return = (next_pred_price - curr_price) / curr_price
        
        # ====== 交易决策 ======
        if position == 0:
            if expected_return > Config.buy_threshold:
                # 买入操作
                shares_to_buy = int((capital / (curr_price * (1 + Config.commission_rate))) // 100 * 100)
                
                if shares_to_buy > 0:
                    commission = shares_to_buy * curr_price * Config.commission_rate
                    cost = shares_to_buy * curr_price + commission
                    
                    capital -= cost
                    position += shares_to_buy
                    
                    trade_log.append({
                        'Date': date, 'Type': 'BUY', 'Price': curr_price, 'Shares': shares_to_buy,
                        'Cost/Revenue': -cost, 'Commission': commission, 'Capital After': capital,
                        'Position After': position
                    })
                    buy_dates.append(date)
                    buy_prices.append(curr_price)
        
        elif position > 0:
            if expected_return < Config.sell_threshold: # 预测下跌就清仓卖出
                # 卖出操作
                shares_to_sell = position
                commission = shares_to_sell * curr_price * Config.commission_rate
                revenue = shares_to_sell * curr_price - commission
                
                capital += revenue
                position = 0
                
                trade_log.append({
                    'Date': date, 'Type': 'SELL', 'Price': curr_price, 'Shares': shares_to_sell,
                    'Cost/Revenue': revenue, 'Commission': commission, 'Capital After': capital,
                    'Position After': position
                })
                sell_dates.append(date)
                sell_prices.append(curr_price)
        
        # 记录当日净值
        # portfolio_values[i] 对应 test_dates[i] 的净值
        current_value = capital + position * actual_prices[i]
        portfolio_values.append(current_value) # 循环结束时，长度为 222
    
    # ====== 补齐最后一个交易日的数据 (Day 222) ======
    
    # 1. 如果有持仓，强制在最后一个交易日收盘平仓
    last_date = test_dates.iloc[-1]
    last_price = actual_prices[-1]
    
    if position > 0:
        shares_to_sell = position
        commission = shares_to_sell * last_price * Config.commission_rate
        revenue = shares_to_sell * last_price - commission
        
        capital += revenue
        position = 0
        
        # 记录最终平仓交易
        trade_log.append({
            'Date': last_date, 'Type': 'SELL (EOD)', 'Price': last_price, 'Shares': shares_to_sell,
            'Cost/Revenue': revenue, 'Commission': commission, 'Capital After': capital,
            'Position After': 0
        })
    
    # 2. 补齐最后一个交易日的净值
    # 此时 position 应该为 0，净值等于最终现金
    final_value = capital + position * last_price
    portfolio_values.append(final_value) # 长度变为 223，与 test_dates 匹配
    
    # 5. 打印交易记录
    print("\n" + "="*50)
    print("详细交易记录:")
    print("="*50)
    
    trade_df = pd.DataFrame(trade_log)
    print(trade_df.to_string())
    
    # 6. 计算指标 (现在 test_dates 和 portfolio_values 长度都为 223，不再需要切片)
    portfolio_df = pd.DataFrame({'date': test_dates, 'value': portfolio_values})
    portfolio_df['daily_return'] = portfolio_df['value'].pct_change()
    
    final_capital = portfolio_values[-1]
    
    total_return = (final_capital - Config.initial_capital) / Config.initial_capital
    # 年化收益率 (假设每年 252 个交易日)
    annual_return = (1 + total_return) ** (252 / len(portfolio_df)) - 1 
    
    sharpe_ratio = (portfolio_df['daily_return'].mean() / portfolio_df['daily_return'].std()) * np.sqrt(252)
    max_drawdown = (portfolio_df['value'] / portfolio_df['value'].cummax() - 1).min()
    
    print("\n" + "="*30)
    print(f"回测结果 ({test_dates.iloc[0].date()} 至 {test_dates.iloc[-1].date()})")
    print("="*30)
    print(f"初始资金: {Config.initial_capital:,.2f}")
    print(f"最终资金: {final_capital:,.2f}")
    print(f"总收益率: {total_return*100:.2f}%")
    print(f"年化收益: {annual_return*100:.2f}%")
    print(f"夏普比率: {sharpe_ratio:.2f}")
    print(f"最大回撤: {max_drawdown*100:.2f}%")
    print(f"交易次数: {len(buy_dates)} 买入, {len(trade_df[trade_df['Type'].str.contains('SELL')])} 卖出")
    print(f"总佣金支出: {trade_df['Commission'].sum():,.2f}")
    
    # 7. 可视化 (所有数组现在长度都为 223)
    plt.figure(figsize=(14, 10))
    
    # 子图1: 股价与买卖点
    plt.subplot(2, 1, 1)
    plt.plot(test_dates, actual_prices, label='Actual Price', color='gray', alpha=0.6)
    plt.scatter(buy_dates, buy_prices, marker='^', color='red', s=100, label='Buy Signal', zorder=5)
    plt.scatter(sell_dates, sell_prices, marker='v', color='green', s=100, label='Sell Signal', zorder=5)
    plt.title('Stock Price & Trading Signals')
    plt.legend()
    plt.grid(True)
    
    # 子图2: 账户净值曲线
    plt.subplot(2, 1, 2)
    plt.plot(test_dates, portfolio_values, label='Strategy Equity', color='purple')
    plt.title('Strategy Equity Curve')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_backtest()