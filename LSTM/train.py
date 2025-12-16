import torch
import torch.nn as nn
import torch.optim as optim
from config import Config
from model import LSTMModel
from bimodel import BiLSTMModel
from data_provider import get_dataloaders
import matplotlib.pyplot as plt

def train():
    # 获取数据
    train_loader, test_loader, _, _, _, _ = get_dataloaders()
    
    # 初始化模型(LSTM 或 双向LSTM)
    # model = LSTMModel().to(Config.device)
    model = BiLSTMModel().to(Config.device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
    
    best_loss = float('inf')
    train_losses = []
    
    print("开始训练...")
    for epoch in range(Config.num_epochs):
        model.train()
        running_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(Config.device)
            y_batch = y_batch.to(Config.device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        print(f"Epoch [{epoch+1}/{Config.num_epochs}], Loss: {avg_loss:.6f}")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), Config.model_save_path)
            
    print(f"训练完成，最佳模型已保存至 {Config.model_save_path}")
    
    # 绘制Loss曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train()