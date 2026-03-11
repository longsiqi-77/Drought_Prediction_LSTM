"""
修复版多模型训练脚本
"""
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import time
import json
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 修复导入问题
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import get_dataloaders
from src.config import config

# 动态导入模型
try:
    # 尝试从models目录导入
    from src.models.lstm import LSTMModel
    HAS_MODELS = True
except ImportError:
    # 如果models目录不存在，从原位置导入
    from src.model import LSTMModel
    HAS_MODELS = False

def create_model(model_type, input_size, prediction_steps, **kwargs):
    """创建模型工厂函数"""
    model_type = model_type.lower()
    
    if model_type == 'lstm':
        return LSTMModel(
            input_size=input_size,
            hidden_size=kwargs.get('hidden_size', 64),
            num_layers=kwargs.get('num_layers', 2),
            output_size=1,
            prediction_steps=prediction_steps
        )
    
    elif model_type == 'transformer':
        print("⚠️  Transformer模型未实现，暂时使用LSTM")
        return LSTMModel(
            input_size=input_size,
            hidden_size=kwargs.get('d_model', 64),
            num_layers=kwargs.get('num_layers', 2),
            output_size=1,
            prediction_steps=prediction_steps
        )
    
    elif model_type == 'simpletransformer':
        print("⚠️  SimpleTransformer模型未实现，暂时使用LSTM")
        return LSTMModel(
            input_size=input_size,
            hidden_size=kwargs.get('d_model', 64),
            num_layers=kwargs.get('num_layers', 2),
            output_size=1,
            prediction_steps=prediction_steps
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

class MultiModelTrainer:
    """多模型训练器"""
    
    def __init__(self, model_type='LSTM', save_dir=None):
        self.model_type = model_type
        self.device = torch.device(config.DEVICE)
        
        # 创建保存目录
        if save_dir is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.save_dir = Path(f"results/experiments/{model_type}_{timestamp}")
        else:
            self.save_dir = Path(save_dir)
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练历史记录
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'best_val_loss': float('inf'),
            'best_epoch': 0
        }
        
        print(f"\n{'='*60}")
        print(f"TRAINING {model_type.upper()} MODEL")
        print(f"{'='*60}")
    
    def prepare_data(self):
        """准备数据"""
        print(">>> Preparing data...")
        
        # 获取数据加载器
        train_loader, val_loader, scalers, features = get_dataloaders(
            pred_steps=config.PREDICT_STEPS,
            model_type=self.model_type
        )
        
        # 获取输入维度
        sample_batch = next(iter(train_loader))
        input_size = sample_batch[0].shape[2]
        
        print(f">>> Data loaded:")
        print(f"    Input size: {input_size}")
        print(f"    Features: {features}")
        print(f"    Train batches: {len(train_loader)}")
        print(f"    Val batches: {len(val_loader)}")
        
        return train_loader, val_loader, scalers, input_size
    
    def create_model(self, input_size):
        """创建模型"""
        print(f">>> Creating {self.model_type} model...")
        
        # 根据模型类型设置参数
        if self.model_type.lower() == 'lstm':
            model_kwargs = {
                'hidden_size': config.HIDDEN_SIZE,
                'num_layers': config.NUM_LAYERS,
                'dropout': config.DROPOUT if hasattr(config, 'DROPOUT') else 0.2
            }
        elif self.model_type.lower() in ['transformer', 'simpletransformer']:
            model_kwargs = {
                'd_model': getattr(config, 'TRANSFORMER_D_MODEL', 64),
                'nhead': getattr(config, 'TRANSFORMER_NHEAD', 4),
                'num_layers': getattr(config, 'TRANSFORMER_NUM_LAYERS', 2)
            }
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # 创建模型
        model = create_model(
            model_type=self.model_type,
            input_size=input_size,
            prediction_steps=config.PREDICT_STEPS,
            **model_kwargs
        ).to(self.device)
        
        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f">>> Model created:")
        print(f"    Type: {self.model_type}")
        print(f"    Total parameters: {total_params:,}")
        print(f"    Trainable parameters: {trainable_params:,}")
        
        return model
    
    def train_epoch(self, model, train_loader, criterion, optimizer):
        """训练一个epoch"""
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # 前向传播
            outputs = model(inputs)
            
            # 确保形状匹配
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(1)
            if targets.dim() == 1:
                targets = targets.unsqueeze(1)
            
            # 计算损失
            loss = criterion(outputs, targets)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate(self, model, val_loader, criterion):
        """验证"""
        model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                
                # 形状处理
                if outputs.dim() == 1:
                    outputs = outputs.unsqueeze(1)
                if targets.dim() == 1:
                    targets = targets.unsqueeze(1)
                
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def evaluate_model(self, model, val_loader, scalers):
        """评估模型性能"""
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                
                if outputs.dim() == 1:
                    outputs = outputs.unsqueeze(1)
                
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(targets.numpy())
        
        # 合并批次
        predictions = np.vstack(all_predictions)
        targets = np.vstack(all_targets)
        
        # 反归一化
        ndvi_scaler = scalers['NDVI']
        predictions_original = ndvi_scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(predictions.shape)
        targets_original = ndvi_scaler.inverse_transform(targets.reshape(-1, 1)).reshape(targets.shape)
        
        # 计算指标
        metrics = {}
        predictions_flat = predictions_original.flatten()
        targets_flat = targets_original.flatten()
        
        metrics['mse'] = mean_squared_error(targets_flat, predictions_flat)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(targets_flat, predictions_flat)
        metrics['r2'] = r2_score(targets_flat, predictions_flat)
        
        # 按天计算指标
        day_metrics = []
        for day in range(config.PREDICT_STEPS):
            day_pred = predictions_original[:, day]
            day_target = targets_original[:, day]
            
            day_metrics.append({
                'day': day + 1,
                'mse': mean_squared_error(day_target, day_pred),
                'mae': mean_absolute_error(day_target, day_pred),
                'r2': r2_score(day_target, day_pred)
            })
        
        metrics['day_metrics'] = day_metrics
        
        return metrics, predictions_original, targets_original
    
    def train(self):
        """训练模型"""
        # 准备数据
        train_loader, val_loader, scalers, input_size = self.prepare_data()
        
        # 创建模型
        model = self.create_model(input_size)
        
        # 损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            model.parameters(), 
            lr=config.LEARNING_RATE,
            weight_decay=1e-5
        )
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        )
        
        # 早停
        best_val_loss = float('inf')
        best_model_state = None
        patience = 20
        patience_counter = 0
        
        # 训练开始时间
        start_time = time.time()
        
        print(f">>> Start training on {self.device}...")
        print("-" * 80)
        
        # 训练循环
        for epoch in range(config.EPOCHS):
            epoch_start = time.time()
            
            # 训练
            train_loss = self.train_epoch(model, train_loader, criterion, optimizer)
            
            # 验证
            val_loss = self.validate(model, val_loader, criterion)
            
            # 学习率调整
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                self.history['best_val_loss'] = best_val_loss
                self.history['best_epoch'] = epoch + 1
                patience_counter = 0
                
                # 保存最佳模型
                self.save_model(model, f"best_{self.model_type.lower()}_model.pth")
            else:
                patience_counter += 1
            
            # 打印进度
            epoch_time = time.time() - epoch_start
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:03d}/{config.EPOCHS:03d} | "
                      f"Time: {epoch_time:.1f}s | "
                      f"Train Loss: {train_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f} | "
                      f"LR: {current_lr:.2e} | "
                      f"Best Val: {best_val_loss:.6f}")
            
            # 早停检查
            if patience_counter >= patience:
                print(f">>> Early stopping triggered at epoch {epoch+1}")
                break
        
        # 训练完成
        total_time = time.time() - start_time
        
        # 加载最佳模型
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # 最终评估
        print(f"\n>>> Training completed in {total_time:.1f} seconds")
        print(f">>> Best validation loss: {best_val_loss:.6f} (epoch {self.history['best_epoch']})")
        
        # 保存最终模型
        self.save_model(model, f"{self.model_type.lower()}_model_final.pth")
        
        # 评估性能
        print(f"\n>>> Evaluating model performance...")
        metrics, predictions, targets = self.evaluate_model(model, val_loader, scalers)
        
        # 打印评估结果
        self.print_evaluation(metrics)
        
        # 保存训练历史
        self.save_history()
        self.save_metrics(metrics)
        
        # 生成训练图表
        self.plot_training_history()
        
        return {
            'model': model,
            'history': self.history,
            'metrics': metrics,
            'predictions': predictions,
            'targets': targets,
            'scalers': scalers
        }
    
    def save_model(self, model, filename):
        """保存模型"""
        model_path = self.save_dir / filename
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_type': self.model_type,
            'config': {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
        }, model_path)
        print(f">>> Model saved to: {model_path}")
    
    def save_history(self):
        """保存训练历史"""
        history_path = self.save_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def save_metrics(self, metrics):
        """保存评估指标"""
        metrics_path = self.save_dir / "evaluation_metrics.json"
        
        # 简化metrics以便保存
        save_metrics = {
            'overall': {
                'mse': float(metrics['mse']),
                'rmse': float(metrics['rmse']),
                'mae': float(metrics['mae']),
                'r2': float(metrics['r2'])
            },
            'by_day': [
                {
                    'day': m['day'],
                    'mse': float(m['mse']),
                    'mae': float(m['mae']),
                    'r2': float(m['r2'])
                }
                for m in metrics['day_metrics']
            ]
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(save_metrics, f, indent=2)
        
        # 也保存为txt便于阅读
        txt_path = self.save_dir / "evaluation_results.txt"
        with open(txt_path, 'w') as f:
            f.write(f"{self.model_type.upper()} Model Evaluation Results\n")
            f.write("="*50 + "\n\n")
            f.write("Overall Performance:\n")
            f.write(f"  MSE:  {metrics['mse']:.6f}\n")
            f.write(f"  RMSE: {metrics['rmse']:.6f}\n")
            f.write(f"  MAE:  {metrics['mae']:.6f}\n")
            f.write(f"  R²:   {metrics['r2']:.4f}\n\n")
            
            f.write("Performance by Prediction Day:\n")
            f.write("-"*40 + "\n")
            f.write("Day\tMSE\t\tMAE\t\tR²\n")
            for m in metrics['day_metrics']:
                f.write(f"{m['day']}\t{m['mse']:.6f}\t{m['mae']:.6f}\t{m['r2']:.4f}\n")
    
    def print_evaluation(self, metrics):
        """打印评估结果"""
        print("\n" + "="*60)
        print(f"{self.model_type.upper()} MODEL EVALUATION")
        print("="*60)
        
        print("\nOverall Performance:")
        print(f"  MSE:  {metrics['mse']:.6f}")
        print(f"  RMSE: {metrics['rmse']:.6f}")
        print(f"  MAE:  {metrics['mae']:.6f}")
        print(f"  R²:   {metrics['r2']:.4f}")
        
        print("\nPerformance by Prediction Day:")
        print("-"*50)
        print("Day\tMSE\t\tMAE\t\tR²")
        for m in metrics['day_metrics']:
            print(f"{m['day']}\t{m['mse']:.6f}\t{m['mae']:.6f}\t{m['r2']:.4f}")
        print("="*60)
    
    def plot_training_history(self):
        """绘制训练历史图表"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # 损失曲线
            axes[0].plot(self.history['train_loss'], label='Training Loss', alpha=0.8)
            axes[0].plot(self.history['val_loss'], label='Validation Loss', alpha=0.8)
            axes[0].axvline(x=self.history['best_epoch']-1, color='red', linestyle='--', alpha=0.5, label='Best Epoch')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title(f'{self.model_type} Training History')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # 学习率曲线
            axes[1].plot(self.history['learning_rate'], color='green', linewidth=2)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Learning Rate')
            axes[1].set_title('Learning Rate Schedule')
            axes[1].set_yscale('log')
            axes[1].grid(True, alpha=0.3)
            
            plt.suptitle(f'{self.model_type} Model Training Summary', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # 保存图表
            plot_path = self.save_dir / "training_history.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f">>> Training history plot saved to: {plot_path}")
            
            plt.close()
            
        except ImportError:
            print(">>> Matplotlib not available, skipping plots...")

def train_single_model(model_type='LSTM'):
    """训练单个模型"""
    trainer = MultiModelTrainer(model_type=model_type)
    results = trainer.train()
    return results

def main():
    """主函数"""
    print("="*80)
    print("MULTI-MODEL TRAINING SYSTEM")
    print("="*80)
    
    # 训练LSTM模型
    print("\n>>> Training LSTM model...")
    lstm_results = train_single_model('LSTM')
    
    print("\n" + "="*80)
    print("LSTM TRAINING COMPLETED")
    print("="*80)

if __name__ == "__main__":
    main()