import sys
import os
# 获取当前文件的父目录的父目录 (即项目根目录 Drought_Prediction_LSTM)
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)



import torch
import matplotlib.pyplot as plt
import numpy as np
from src.config import config
from src.model import LSTMModel
from src.data_loader import get_dataloaders
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

def evaluate():
    # 加载数据
    _, test_loader, scaler = get_dataloaders()
    
    # 加载模型
    model = LSTMModel().to(config.DEVICE)
    checkpoint_path = os.path.join(config.MODEL_SAVE_DIR, 'lstm_model.pth')
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print("Loaded trained model.")
    else:
        print("Warning: Model checkpoint not found, using random weights.")

    model.eval()
    predictions = []
    actuals = []
    
    # 预测
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(config.DEVICE)
            pred = model(X_batch).cpu().numpy()
            predictions.extend(pred)
            actuals.extend(y_batch.numpy())
            
    # 反归一化 (可选，这里为了简单展示趋势，直接对比归一化后的值)
    # 如果想反归一化，需要只对第0列(NDVI)进行 inverse_transform
    
    # 计算指标
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    print(f"Test MSE: {mse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    
    # 绘图 (取前200天展示，太长了看不清)
    plt.figure(figsize=(12, 6))
    plt.plot(actuals[:200], label='Actual NDVI', color='green')
    plt.plot(predictions[:200], label='Predicted NDVI', color='red', linestyle='--')
    plt.title('NDVI Prediction Result (Test Set)')
    plt.xlabel('Days')
    plt.ylabel('Normalized NDVI')
    plt.legend()
    plt.savefig(os.path.join(config.FIGURE_SAVE_DIR, 'prediction_comparison.png'))
    plt.show()

if __name__ == "__main__":
    evaluate()