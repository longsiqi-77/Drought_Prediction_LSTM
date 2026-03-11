# src/visualize_fixed.py
"""
修复版可视化脚本 - 只保存不显示
"""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import os
from src.model import LSTMModel
from src.data_loader import get_dataloaders
from src.config import config

def load_trained_model(model_path=None):
    """加载训练好的模型"""
    if model_path is None:
        model_path = os.path.join(config.MODEL_SAVE_DIR, "best_lstm_model.pth")
    
    print(f">>> Loading model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    model = LSTMModel(
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        output_size=1,
        prediction_steps=config.PREDICT_STEPS
    )
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model

def quick_finish_visualization():
    """快速完成剩余的可视化（不显示窗口）"""
    print("=" * 60)
    print("QUICK VISUALIZATION COMPLETION")
    print("=" * 60)
    
    # 1. 加载模型
    model = load_trained_model()
    
    # 2. 快速获取少量数据
    _, test_loader, scalers, _ = get_dataloaders()
    
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch_idx, (inputs, target_batch) in enumerate(test_loader):
            outputs = model(inputs)
            predictions.append(outputs.numpy())
            targets.append(target_batch.numpy())
            
            if batch_idx >= 2:  # 只取3个批次，加快速度
                break
    
    predictions = np.vstack(predictions)
    targets = np.vstack(targets)
    
    # 反归一化
    ndvi_scaler = scalers['NDVI']
    predictions = ndvi_scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(predictions.shape)
    targets = ndvi_scaler.inverse_transform(targets.reshape(-1, 1)).reshape(targets.shape)
    
    print(f">>> Data loaded: {predictions.shape}")
    
    # 3. 生成预测分布图（修复版，不显示）
    print(">>> Creating prediction distribution plot...")
    create_prediction_distribution(predictions, targets)
    
    # 4. 生成简化的分析报告
    print(">>> Creating analysis report...")
    create_simple_report(predictions, targets)
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)

def create_prediction_distribution(predictions, targets):
    """创建预测分布图（只保存不显示）"""
    fig, axes = plt.subplots(2, config.PREDICT_STEPS, figsize=(15, 8))
    
    for day in range(config.PREDICT_STEPS):
        pred_day = predictions[:, day]
        target_day = targets[:, day]
        errors = pred_day - target_day
        
        # 散点图
        ax_scatter = axes[0, day]
        ax_scatter.scatter(target_day, pred_day, alpha=0.6, s=20, color='steelblue')
        
        min_val = min(target_day.min(), pred_day.min())
        max_val = max(target_day.max(), pred_day.max())
        ax_scatter.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
        
        ax_scatter.set_xlabel('Actual NDVI', fontsize=10)
        ax_scatter.set_ylabel('Predicted NDVI', fontsize=10)
        ax_scatter.set_title(f'Day {day+1}', fontsize=11)
        ax_scatter.grid(True, alpha=0.3)
        
        # 误差直方图
        ax_hist = axes[1, day]
        ax_hist.hist(errors, bins=20, alpha=0.7, color='orange', edgecolor='black')
        ax_hist.axvline(x=0, color='red', linestyle='--', linewidth=1)
        ax_hist.set_xlabel('Prediction Error', fontsize=10)
        ax_hist.set_ylabel('Frequency', fontsize=10)
        ax_hist.set_title(f'Day {day+1} Error Distribution', fontsize=11)
        ax_hist.grid(True, alpha=0.3)
    
    plt.suptitle('Prediction Distribution Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(config.FIGURE_SAVE_DIR, 'prediction_distribution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"    Saved: {save_path}")
    
    plt.close()

def create_simple_report(predictions, targets):
    """创建简化的分析报告"""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    report_path = os.path.join(config.PREDICTION_SAVE_DIR, 'simple_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("Multi-Step Prediction Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Model: LSTM\n")
        f.write(f"Prediction Horizon: {config.PREDICT_STEPS} days\n")
        f.write(f"Samples Analyzed: {len(predictions)}\n\n")
        
        f.write("Performance by Day:\n")
        f.write("-" * 40 + "\n")
        f.write("Day\tMAE\t\tR²\n")
        
        for day in range(config.PREDICT_STEPS):
            pred_day = predictions[:, day]
            target_day = targets[:, day]
            
            mae = mean_absolute_error(target_day, pred_day)
            r2 = r2_score(target_day, pred_day)
            
            f.write(f"{day+1}\t{mae:.6f}\t{r2:.4f}\n")
        
        f.write("\nOverall Performance:\n")
        f.write("-" * 40 + "\n")
        
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()
        
        mae = mean_absolute_error(target_flat, pred_flat)
        rmse = np.sqrt(mean_squared_error(target_flat, pred_flat))
        r2 = r2_score(target_flat, pred_flat)
        
        f.write(f"MAE:  {mae:.6f}\n")
        f.write(f"RMSE: {rmse:.6f}\n")
        f.write(f"R²:   {r2:.4f}\n")
    
    print(f"    Report saved: {report_path}")

if __name__ == "__main__":
    quick_finish_visualization()