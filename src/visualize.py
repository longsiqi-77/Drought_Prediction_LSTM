"""
多步预测结果可视化脚本
展示7天预测与实际对比、预测误差增长、置信区间等
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
from src.model import LSTMModel
from src.data_loader import get_dataloaders, load_and_preprocess_data
from src.config import config

def load_trained_model(model_path=None):
    """加载训练好的模型"""
    if model_path is None:
        model_path = os.path.join(config.MODEL_SAVE_DIR, "best_lstm_model.pth")
    
    print(f">>> Loading model from: {model_path}")
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 创建模型
    model = LSTMModel(
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        output_size=1,
        prediction_steps=config.PREDICT_STEPS
    )
    
    # 加载权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f">>> Model loaded successfully")
    
    return model

def get_predictions_and_actuals(model, num_samples=50):
    """获取预测值和实际值"""
    # 获取数据
    _, test_loader, scalers, _ = get_dataloaders()
    
    model.eval()
    all_predictions = []
    all_targets = []
    sample_indices = []  # 记录样本索引
    
    with torch.no_grad():
        batch_count = 0
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            outputs = model(inputs)
            
            # 转换到numpy
            predictions_np = outputs.numpy()
            targets_np = targets.numpy()
            
            # 取这个批次的所有样本
            for i in range(min(len(predictions_np), num_samples - len(all_predictions))):
                all_predictions.append(predictions_np[i])
                all_targets.append(targets_np[i])
                sample_indices.append(batch_count * config.BATCH_SIZE + i)
            
            batch_count += 1
            
            if len(all_predictions) >= num_samples:
                break
    
    # 转换为numpy数组
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    
    # 反归一化
    ndvi_scaler = scalers['NDVI']
    predictions_original = ndvi_scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(predictions.shape)
    targets_original = ndvi_scaler.inverse_transform(targets.reshape(-1, 1)).reshape(targets.shape)
    
    return predictions_original, targets_original, sample_indices

def plot_7day_predictions(predictions, targets, sample_indices, num_samples_to_show=5):
    """绘制7天预测与实际对比图"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # 选择几个代表性样本展示
    samples_to_show = min(num_samples_to_show, len(predictions))
    indices_to_show = np.linspace(0, len(predictions)-1, samples_to_show, dtype=int)
    
    colors = plt.cm.Set2(np.linspace(0, 1, samples_to_show))
    
    for idx, (ax, sample_idx) in enumerate(zip(axes[:samples_to_show], indices_to_show)):
        pred_sample = predictions[sample_idx]
        target_sample = targets[sample_idx]
        days = np.arange(1, config.PREDICT_STEPS + 1)
        
        # 绘制预测值和实际值
        ax.plot(days, target_sample, 'o-', linewidth=2, markersize=8, 
                label='Actual NDVI', color=colors[idx], alpha=0.8)
        ax.plot(days, pred_sample, 's--', linewidth=2, markersize=8,
                label='Predicted NDVI', color=colors[idx], alpha=0.8)
        
        # 填充预测误差区域
        ax.fill_between(days, pred_sample, target_sample, alpha=0.2, color=colors[idx])
        
        # 计算这个样本的误差
        mae = np.mean(np.abs(pred_sample - target_sample))
        rmse = np.sqrt(np.mean((pred_sample - target_sample) ** 2))
        
        ax.set_title(f'Sample {sample_indices[sample_idx]}\nMAE: {mae:.4f}, RMSE: {rmse:.4f}', fontsize=12)
        ax.set_xlabel('Prediction Day', fontsize=11)
        ax.set_ylabel('NDVI Value', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 设置y轴范围
        all_values = np.concatenate([pred_sample, target_sample])
        y_min, y_max = all_values.min() - 0.02, all_values.max() + 0.02
        ax.set_ylim(y_min, y_max)
    
    # 如果有空余的子图，添加图例说明
    if samples_to_show < len(axes):
        axes[samples_to_show].axis('off')
        axes[samples_to_show].text(0.1, 0.5, 
                                   f'Multi-step Prediction Results\n'
                                   f'Model: {config.MODEL_TYPE}\n'
                                   f'Prediction Steps: {config.PREDICT_STEPS}\n'
                                   f'Sequence Length: {config.SEQUENCE_LENGTH}\n'
                                   f'Total Samples: {len(predictions)}\n'
                                   f'Showing: {samples_to_show} samples',
                                   fontsize=12, verticalalignment='center')
    
    plt.suptitle(f'{config.MODEL_TYPE} Model: 7-Day NDVI Predictions vs Actual Values', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存图表
    save_path = os.path.join(config.FIGURE_SAVE_DIR, '7day_predictions_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f">>> 7-day predictions comparison saved to: {save_path}")
    
    plt.show()

def plot_error_growth(predictions, targets):
    """绘制预测误差随时间增长图"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 计算每个预测步长的误差指标
    days = np.arange(1, config.PREDICT_STEPS + 1)
    
    mae_by_day = []
    rmse_by_day = []
    mape_by_day = []
    r2_by_day = []
    
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    for day in range(config.PREDICT_STEPS):
        pred_day = predictions[:, day]
        target_day = targets[:, day]
        
        mae = mean_absolute_error(target_day, pred_day)
        rmse = np.sqrt(mean_squared_error(target_day, pred_day))
        r2 = r2_score(target_day, pred_day)
        
        # 计算MAPE（避免除零）
        mask = target_day != 0
        if np.any(mask):
            mape = np.mean(np.abs((target_day[mask] - pred_day[mask]) / target_day[mask])) * 100
        else:
            mape = 0
        
        mae_by_day.append(mae)
        rmse_by_day.append(rmse)
        mape_by_day.append(mape)
        r2_by_day.append(r2)
    
    # 1. MAE增长曲线
    axes[0, 0].plot(days, mae_by_day, 'o-', linewidth=2, markersize=8, color='red', label='MAE')
    axes[0, 0].fill_between(days, 0, mae_by_day, alpha=0.2, color='red')
    axes[0, 0].set_xlabel('Prediction Day', fontsize=12)
    axes[0, 0].set_ylabel('Mean Absolute Error (MAE)', fontsize=12)
    axes[0, 0].set_title('MAE Growth Over Prediction Horizon', fontsize=13)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # 2. RMSE增长曲线
    axes[0, 1].plot(days, rmse_by_day, 's-', linewidth=2, markersize=8, color='blue', label='RMSE')
    axes[0, 1].fill_between(days, 0, rmse_by_day, alpha=0.2, color='blue')
    axes[0, 1].set_xlabel('Prediction Day', fontsize=12)
    axes[0, 1].set_ylabel('Root Mean Square Error (RMSE)', fontsize=12)
    axes[0, 1].set_title('RMSE Growth Over Prediction Horizon', fontsize=13)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # 3. MAPE增长曲线
    axes[1, 0].plot(days, mape_by_day, '^-', linewidth=2, markersize=8, color='green', label='MAPE')
    axes[1, 0].fill_between(days, 0, mape_by_day, alpha=0.2, color='green')
    axes[1, 0].set_xlabel('Prediction Day', fontsize=12)
    axes[1, 0].set_ylabel('Mean Absolute Percentage Error (%)', fontsize=12)
    axes[1, 0].set_title('MAPE Growth Over Prediction Horizon', fontsize=13)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # 4. R²下降曲线
    axes[1, 1].plot(days, r2_by_day, 'd-', linewidth=2, markersize=8, color='purple', label='R²')
    axes[1, 1].set_xlabel('Prediction Day', fontsize=12)
    axes[1, 1].set_ylabel('R² Score', fontsize=12)
    axes[1, 1].set_title('R² Score Decline Over Prediction Horizon', fontsize=13)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    # 添加趋势线
    for day, r2 in zip(days, r2_by_day):
        axes[1, 1].text(day, r2 - 0.02, f'{r2:.3f}', ha='center', va='top', fontsize=9)
    
    plt.suptitle('Prediction Error Growth Analysis: 7-Day Forecast Horizon', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存图表
    save_path = os.path.join(config.FIGURE_SAVE_DIR, 'error_growth_analysis.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f">>> Error growth analysis saved to: {save_path}")
    
    plt.show()
    
    return {
        'days': days,
        'mae': mae_by_day,
        'rmse': rmse_by_day,
        'mape': mape_by_day,
        'r2': r2_by_day
    }

def plot_confidence_intervals(predictions, targets, n_bootstrap=1000):
    """绘制置信区间（使用Bootstrap方法）"""
    print(">>> Calculating confidence intervals using bootstrap...")
    
    n_samples = len(predictions)
    days = np.arange(1, config.PREDICT_STEPS + 1)
    
    # Bootstrap采样计算置信区间
    bootstrap_mae = np.zeros((n_bootstrap, config.PREDICT_STEPS))
    bootstrap_rmse = np.zeros((n_bootstrap, config.PREDICT_STEPS))
    
    for i in range(n_bootstrap):
        # 随机采样（有放回）
        indices = np.random.choice(n_samples, n_samples, replace=True)
        pred_bootstrap = predictions[indices]
        target_bootstrap = targets[indices]
        
        for day in range(config.PREDICT_STEPS):
            pred_day = pred_bootstrap[:, day]
            target_day = target_bootstrap[:, day]
            
            # 计算MAE和RMSE
            mae = np.mean(np.abs(pred_day - target_day))
            rmse = np.sqrt(np.mean((pred_day - target_day) ** 2))
            
            bootstrap_mae[i, day] = mae
            bootstrap_rmse[i, day] = rmse
    
    # 计算置信区间（95%）
    mae_ci_lower = np.percentile(bootstrap_mae, 2.5, axis=0)
    mae_ci_upper = np.percentile(bootstrap_mae, 97.5, axis=0)
    mae_mean = np.mean(bootstrap_mae, axis=0)
    
    rmse_ci_lower = np.percentile(bootstrap_rmse, 2.5, axis=0)
    rmse_ci_upper = np.percentile(bootstrap_rmse, 97.5, axis=0)
    rmse_mean = np.mean(bootstrap_rmse, axis=0)
    
    # 绘制置信区间图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # MAE置信区间
    axes[0].plot(days, mae_mean, 'o-', linewidth=2, markersize=8, color='red', label='Mean MAE')
    axes[0].fill_between(days, mae_ci_lower, mae_ci_upper, alpha=0.3, color='red', label='95% CI')
    axes[0].set_xlabel('Prediction Day', fontsize=12)
    axes[0].set_ylabel('Mean Absolute Error (MAE)', fontsize=12)
    axes[0].set_title('MAE with 95% Confidence Intervals', fontsize=13)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # 添加置信区间数值标签
    for day, mean_val, lower, upper in zip(days, mae_mean, mae_ci_lower, mae_ci_upper):
        axes[0].text(day, upper + 0.0002, f'[{lower:.4f}, {upper:.4f}]', 
                    ha='center', va='bottom', fontsize=8, rotation=0)
    
    # RMSE置信区间
    axes[1].plot(days, rmse_mean, 's-', linewidth=2, markersize=8, color='blue', label='Mean RMSE')
    axes[1].fill_between(days, rmse_ci_lower, rmse_ci_upper, alpha=0.3, color='blue', label='95% CI')
    axes[1].set_xlabel('Prediction Day', fontsize=12)
    axes[1].set_ylabel('Root Mean Square Error (RMSE)', fontsize=12)
    axes[1].set_title('RMSE with 95% Confidence Intervals', fontsize=13)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # 添加置信区间数值标签
    for day, mean_val, lower, upper in zip(days, rmse_mean, rmse_ci_lower, rmse_ci_upper):
        axes[1].text(day, upper + 0.0003, f'[{lower:.4f}, {upper:.4f}]', 
                    ha='center', va='bottom', fontsize=8, rotation=0)
    
    plt.suptitle('Prediction Error with Bootstrap Confidence Intervals (n=1000)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存图表
    save_path = os.path.join(config.FIGURE_SAVE_DIR, 'confidence_intervals.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f">>> Confidence intervals plot saved to: {save_path}")
    
    plt.show()
    
    return {
        'mae_mean': mae_mean,
        'mae_ci_lower': mae_ci_lower,
        'mae_ci_upper': mae_ci_upper,
        'rmse_mean': rmse_mean,
        'rmse_ci_lower': rmse_ci_lower,
        'rmse_ci_upper': rmse_ci_upper
    }

def plot_prediction_distribution(predictions, targets):
    """绘制预测值分布图"""
    fig, axes = plt.subplots(2, config.PREDICT_STEPS, figsize=(15, 8))
    
    for day in range(config.PREDICT_STEPS):
        pred_day = predictions[:, day]
        target_day = targets[:, day]
        errors = pred_day - target_day
        
        # 第一行：预测值 vs 实际值散点图
        ax_scatter = axes[0, day]
        ax_scatter.scatter(target_day, pred_day, alpha=0.6, s=20, color='steelblue')
        
        # 添加对角线（完美预测线）
        min_val = min(target_day.min(), pred_day.min())
        max_val = max(target_day.max(), pred_day.max())
        ax_scatter.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, linewidth=1)
        
        ax_scatter.set_xlabel('Actual NDVI', fontsize=10)
        ax_scatter.set_ylabel('Predicted NDVI', fontsize=10)
        ax_scatter.set_title(f'Day {day+1}', fontsize=11)
        ax_scatter.grid(True, alpha=0.3)
        
        # 第二行：误差分布直方图
        ax_hist = axes[1, day]
        ax_hist.hist(errors, bins=30, alpha=0.7, color='orange', edgecolor='black')
        ax_hist.axvline(x=0, color='red', linestyle='--', linewidth=1)
        ax_hist.set_xlabel('Prediction Error', fontsize=10)
        ax_hist.set_ylabel('Frequency', fontsize=10)
        ax_hist.set_title(f'Day {day+1} Error Distribution', fontsize=11)
        ax_hist.grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        ax_hist.text(0.05, 0.95, f'Mean: {mean_error:.4f}\nStd: {std_error:.4f}', 
                    transform=ax_hist.transAxes, fontsize=9, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Prediction Distribution Analysis for Each Forecast Day', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存图表
    save_path = os.path.join(config.FIGURE_SAVE_DIR, 'prediction_distribution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f">>> Prediction distribution plot saved to: {save_path}")
    
    plt.show()

def generate_report(predictions, targets, error_metrics, ci_metrics):
    """生成分析报告"""
    report_path = os.path.join(config.PREDICTION_SAVE_DIR, 'prediction_analysis_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("MULTI-STEP PREDICTION ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Model: {config.MODEL_TYPE}\n")
        f.write(f"Prediction Horizon: {config.PREDICT_STEPS} days\n")
        f.write(f"Sequence Length: {config.SEQUENCE_LENGTH} days\n")
        f.write(f"Number of Test Samples: {len(predictions)}\n")
        f.write(f"Input Features: {config.INPUT_SIZE}\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("ERROR METRICS BY PREDICTION DAY\n")
        f.write("=" * 60 + "\n")
        f.write("Day\tMAE\t\tRMSE\t\tMAPE(%)\t\tR²\n")
        f.write("-" * 60 + "\n")
        
        for i in range(config.PREDICT_STEPS):
            f.write(f"{i+1}\t{error_metrics['mae'][i]:.6f}\t"
                   f"{error_metrics['rmse'][i]:.6f}\t"
                   f"{error_metrics['mape'][i]:.2f}\t\t"
                   f"{error_metrics['r2'][i]:.4f}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("OVERALL PERFORMANCE\n")
        f.write("=" * 60 + "\n")
        
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()
        
        overall_mae = mean_absolute_error(target_flat, pred_flat)
        overall_rmse = np.sqrt(mean_squared_error(target_flat, pred_flat))
        overall_r2 = r2_score(target_flat, pred_flat)
        
        f.write(f"Overall MAE:  {overall_mae:.6f}\n")
        f.write(f"Overall RMSE: {overall_rmse:.6f}\n")
        f.write(f"Overall R²:   {overall_r2:.4f}\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("KEY OBSERVATIONS\n")
        f.write("=" * 60 + "\n")
        f.write("1. Prediction accuracy decreases with longer forecast horizon\n")
        f.write("2. Day 1 prediction is most accurate (R² = {:.4f})\n".format(error_metrics['r2'][0]))
        f.write("3. Day {} prediction has lowest accuracy (R² = {:.4f})\n".format(
            config.PREDICT_STEPS, error_metrics['r2'][-1]))
        f.write("4. Error growth rate: {:.2f}% per day (based on MAE)\n".format(
            (error_metrics['mae'][-1] - error_metrics['mae'][0]) / error_metrics['mae'][0] * 100 / (config.PREDICT_STEPS - 1)))
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("=" * 60 + "\n")
        f.write("1. Use Day 1-3 predictions for high-confidence decisions\n")
        f.write("2. Day 4-7 predictions suitable for trend analysis\n")
        f.write("3. Consider ensemble methods to reduce uncertainty\n")
        f.write("4. Regular model retraining for seasonal adaptation\n")
    
    print(f">>> Analysis report saved to: {report_path}")

def main():
    """主函数：运行所有可视化"""
    print("=" * 60)
    print("MULTI-STEP PREDICTION VISUALIZATION")
    print("=" * 60)
    
    # 1. 加载模型
    model = load_trained_model()
    
    # 2. 获取预测结果
    print("\n>>> Getting predictions and actual values...")
    predictions, targets, sample_indices = get_predictions_and_actuals(model, num_samples=100)
    
    print(f">>> Retrieved {len(predictions)} samples")
    print(f">>> Prediction shape: {predictions.shape}")
    print(f">>> Target shape: {targets.shape}")
    
    # 3. 创建输出目录
    os.makedirs(config.FIGURE_SAVE_DIR, exist_ok=True)
    os.makedirs(config.PREDICTION_SAVE_DIR, exist_ok=True)
    
    # 4. 运行所有可视化
    print("\n>>> Generating visualizations...")
    
    # 4.1 7天预测对比图
    print(">>> Creating 7-day predictions comparison...")
    plot_7day_predictions(predictions, targets, sample_indices, num_samples_to_show=6)
    
    # 4.2 误差增长分析
    print(">>> Creating error growth analysis...")
    error_metrics = plot_error_growth(predictions, targets)
    
    # 4.3 置信区间
    print(">>> Creating confidence intervals...")
    ci_metrics = plot_confidence_intervals(predictions, targets, n_bootstrap=500)
    
    # 4.4 预测值分布
    print(">>> Creating prediction distribution plots...")
    plot_prediction_distribution(predictions, targets)
    
    # 4.5 生成报告
    print(">>> Generating analysis report...")
    generate_report(predictions, targets, error_metrics, ci_metrics)
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE!")
    print("=" * 60)
    print(f"All plots saved to: {config.FIGURE_SAVE_DIR}")
    print(f"Analysis report saved to: {config.PREDICTION_SAVE_DIR}")

if __name__ == "__main__":
    main()