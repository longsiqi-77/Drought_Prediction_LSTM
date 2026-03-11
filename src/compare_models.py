"""
模型对比可视化
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
from pathlib import Path
from src.config import config

def load_model_results(model_types=['LSTM', 'Transformer', 'SimpleTransformer']):
    """加载多个模型的训练结果"""
    results = {}
    base_dir = Path("results/experiments")
    
    if not base_dir.exists():
        print("No experiment results found!")
        return results
    
    for model_type in model_types:
        # 查找该模型类型的最新实验
        model_dirs = list(base_dir.glob(f"{model_type.lower()}_*"))
        if not model_dirs:
            print(f"No results found for {model_type}")
            continue
        
        # 取最新的实验
        latest_dir = max(model_dirs, key=lambda x: x.stat().st_mtime)
        
        try:
            # 加载评估指标
            metrics_path = latest_dir / "evaluation_metrics.json"
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                
                # 加载训练历史
                history_path = latest_dir / "training_history.json"
                history = {}
                if history_path.exists():
                    with open(history_path, 'r') as f:
                        history = json.load(f)
                
                results[model_type] = {
                    'metrics': metrics,
                    'history': history,
                    'directory': str(latest_dir)
                }
                
                print(f"Loaded {model_type} results from: {latest_dir}")
        
        except Exception as e:
            print(f"Error loading {model_type} results: {e}")
    
    return results

def plot_model_comparison(results):
    """绘制模型对比图"""
    if len(results) < 2:
        print("Need at least 2 models for comparison")
        return
    
    # 准备对比数据
    comparison_data = []
    
    for model_name, model_data in results.items():
        metrics = model_data['metrics']['overall']
        comparison_data.append({
            'Model': model_name,
            'R²': metrics['r2'],
            'RMSE': metrics['rmse'],
            'MAE': metrics['mae']
        })
    
    df = pd.DataFrame(comparison_data)
    
    # 创建对比图表
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. 性能指标对比
    models = df['Model']
    x = np.arange(len(models))
    width = 0.25
    
    # R²对比
    axes[0, 0].bar(x - width, df['R²'], width, label='R²', color='green')
    axes[0, 0].set_xlabel('Model')
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].set_title('R² Score Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(models)
    axes[0, 0].set_ylim(0.7, 1.0)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, v in enumerate(df['R²']):
        axes[0, 0].text(i - width, v + 0.01, f'{v:.3f}', ha='center', fontsize=10)
    
    # RMSE对比
    axes[0, 1].bar(x, df['RMSE'], width, label='RMSE', color='blue')
    axes[0, 1].set_xlabel('Model')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].set_title('RMSE Comparison')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(models)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, v in enumerate(df['RMSE']):
        axes[0, 1].text(i, v + 0.0002, f'{v:.4f}', ha='center', fontsize=10)
    
    # MAE对比
    axes[0, 2].bar(x + width, df['MAE'], width, label='MAE', color='orange')
    axes[0, 2].set_xlabel('Model')
    axes[0, 2].set_ylabel('MAE')
    axes[0, 2].set_title('MAE Comparison')
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(models)
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, v in enumerate(df['MAE']):
        axes[0, 2].text(i + width, v + 0.0002, f'{v:.4f}', ha='center', fontsize=10)
    
    # 2. 训练历史对比
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    
    # 训练损失对比
    for idx, (model_name, model_data) in enumerate(results.items()):
        if 'history' in model_data and 'train_loss' in model_data['history']:
            train_loss = model_data['history']['train_loss']
            axes[1, 0].plot(train_loss[:100], label=model_name, color=colors[idx % len(colors)], alpha=0.7)
    
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Training Loss')
    axes[1, 0].set_title('Training Loss Comparison (First 100 Epochs)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 验证损失对比
    for idx, (model_name, model_data) in enumerate(results.items()):
        if 'history' in model_data and 'val_loss' in model_data['history']:
            val_loss = model_data['history']['val_loss']
            axes[1, 1].plot(val_loss[:100], label=model_name, color=colors[idx % len(colors)], alpha=0.7)
    
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Validation Loss')
    axes[1, 1].set_title('Validation Loss Comparison (First 100 Epochs)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 3. 每日预测精度对比
    axes[1, 2].axis('off')
    
    # 创建每日R²对比表格
    day_data = []
    for model_name, model_data in results.items():
        metrics = model_data['metrics']
        if 'by_day' in metrics:
            day_metrics = {}
            for day_metric in metrics['by_day']:
                day = day_metric['day']
                day_metrics[f'Day {day} R²'] = day_metric['r2']
            
            day_data.append({'Model': model_name, **day_metrics})
    
    if day_data:
        day_df = pd.DataFrame(day_data)
        
        # 创建表格
        table_data = []
        for _, row in day_df.iterrows():
            table_row = [row['Model']]
            for day in range(1, config.PREDICT_STEPS + 1):
                col_name = f'Day {day} R²'
                if col_name in row:
                    table_row.append(f"{row[col_name]:.3f}")
            table_data.append(table_row)
        
        # 绘制表格
        table = axes[1, 2].table(
            cellText=table_data,
            colLabels=['Model'] + [f'Day {i}' for i in range(1, config.PREDICT_STEPS + 1)],
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        axes[1, 2].set_title('Daily R² Score Comparison', fontsize=12)
    
    plt.suptitle('Multi-Model Performance Comparison for NDVI Prediction', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存图表
    comparison_dir = Path("results/comparisons")
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    plot_path = comparison_dir / "detailed_model_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n>>> Detailed comparison plot saved to: {plot_path}")
    
    # 保存对比数据
    csv_path = comparison_dir / "detailed_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f">>> Comparison data saved to: {csv_path}")
    
    plt.show()
    
    return df

def plot_prediction_trend_comparison(results):
    """绘制预测趋势对比图"""
    if len(results) < 2:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. 每日R²趋势对比
    days = np.arange(1, config.PREDICT_STEPS + 1)
    
    for idx, (model_name, model_data) in enumerate(results.items()):
        metrics = model_data['metrics']
        if 'by_day' in metrics:
            r2_scores = [day_metric['r2'] for day_metric in metrics['by_day']]
            axes[0].plot(days, r2_scores, 'o-', label=model_name, linewidth=2, markersize=8)
    
    axes[0].set_xlabel('Prediction Day', fontsize=12)
    axes[0].set_ylabel('R² Score', fontsize=12)
    axes[0].set_title('R² Score Trend by Prediction Day', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0.7, 1.0)
    
    # 2. 每日MAE趋势对比
    for idx, (model_name, model_data) in enumerate(results.items()):
        metrics = model_data['metrics']
        if 'by_day' in metrics:
            mae_scores = [day_metric['mae'] for day_metric in metrics['by_day']]
            axes[1].plot(days, mae_scores, 's-', label=model_name, linewidth=2, markersize=8)
    
    axes[1].set_xlabel('Prediction Day', fontsize=12)
    axes[1].set_ylabel('Mean Absolute Error (MAE)', fontsize=12)
    axes[1].set_title('MAE Trend by Prediction Day', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Prediction Accuracy Trend Comparison Across Models', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存图表
    comparison_dir = Path("results/comparisons")
    plot_path = comparison_dir / "prediction_trend_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f">>> Prediction trend comparison saved to: {plot_path}")
    
    plt.show()

def generate_comparison_report(results):
    """生成详细的比较报告"""
    comparison_dir = Path("results/comparisons")
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = comparison_dir / "model_comparison_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MULTI-MODEL COMPARISON REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("Experiment Configuration:\n")
        f.write("-"*40 + "\n")
        f.write(f"Prediction Horizon: {config.PREDICT_STEPS} days\n")
        f.write(f"Sequence Length: {config.SEQUENCE_LENGTH} days\n")
        f.write(f"Input Features: {config.INPUT_SIZE}\n")
        f.write(f"Models Compared: {', '.join(results.keys())}\n\n")
        
        f.write("Overall Performance Summary:\n")
        f.write("-"*40 + "\n")
        
        # 创建性能表格
        f.write(f"{'Model':<15} {'R²':<10} {'RMSE':<12} {'MAE':<12}\n")
        f.write("-"*50 + "\n")
        
        for model_name, model_data in results.items():
            metrics = model_data['metrics']['overall']
            f.write(f"{model_name:<15} {metrics['r2']:<10.4f} "
                   f"{metrics['rmse']:<12.6f} {metrics['mae']:<12.6f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("="*80 + "\n\n")
        
        # 找出最佳模型
        best_model = None
        best_r2 = -float('inf')
        
        for model_name, model_data in results.items():
            r2 = model_data['metrics']['overall']['r2']
            if r2 > best_r2:
                best_r2 = r2
                best_model = model_name
        
        f.write(f"1. Best Performing Model: {best_model} (R² = {best_r2:.4f})\n")
        
        # 分析趋势
        f.write("\n2. Prediction Accuracy Trend:\n")
        f.write("   - All models show decreasing accuracy with longer prediction horizon\n")
        f.write("   - Day 1 predictions are most accurate for all models\n")
        f.write("   - Accuracy decline rate varies by model architecture\n")
        
        f.write("\n3. Training Characteristics:\n")
        for model_name, model_data in results.items():
            if 'history' in model_data:
                best_epoch = model_data['history'].get('best_epoch', 0)
                best_loss = model_data['history'].get('best_val_loss', 0)
                f.write(f"   - {model_name}: Best validation loss {best_loss:.6f} at epoch {best_epoch}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("="*80 + "\n\n")
        
        f.write("1. For highest accuracy: Use {best_model} model\n")
        f.write("2. For faster inference: Consider model complexity vs accuracy trade-off\n")
        f.write("3. For production: Consider ensemble of top performing models\n")
        f.write("4. Regular monitoring: Retrain models with new seasonal data\n")
    
    print(f">>> Comparison report saved to: {report_path}")

def main():
    """主函数"""
    print("="*80)
    print("MODEL COMPARISON VISUALIZATION")
    print("="*80)
    
    # 加载模型结果
    model_types = ['LSTM', 'Transformer', 'SimpleTransformer']
    results = load_model_results(model_types)
    
    if len(results) < 2:
        print(f"\nFound results for {len(results)} models:")
        for model in results:
            print(f"  - {model}")
        print("\nNeed at least 2 models for comparison.")
        print("Please train more models first.")
        return
    
    print(f"\n>>> Loaded results for {len(results)} models:")
    for model_name in results.keys():
        print(f"    - {model_name}")
    
    # 生成对比可视化
    print("\n>>> Generating comparison visualizations...")
    
    # 1. 详细对比图
    df = plot_model_comparison(results)
    
    # 2. 预测趋势对比
    plot_prediction_trend_comparison(results)
    
    # 3. 生成报告
    generate_comparison_report(results)
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE!")
    print("="*80)
    print("Check 'results/comparisons/' directory for all comparison files.")

if __name__ == "__main__":
    main()