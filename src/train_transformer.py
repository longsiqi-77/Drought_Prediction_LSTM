# src/train_transformer.py
"""
训练Transformer模型的简单脚本
"""
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 修复导入问题
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import get_dataloaders
from src.config import config
from src.models.transformer import SimpleTransformer  # 使用简化版Transformer

def train_transformer():
    """训练Transformer模型"""
    print("="*80)
    print("TRAINING TRANSFORMER MODEL")
    print("="*80)
    
    # 设置随机种子
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    
    print(">>> Preparing data...")
    
    # 获取数据 - Transformer可以使用更多特征
    train_loader, test_loader, scalers, features = get_dataloaders(
        pred_steps=config.PREDICT_STEPS,
        model_type='Transformer'  # 告诉data_loader使用更多特征
    )
    
    # 获取输入维度
    sample_batch = next(iter(train_loader))
    input_size = sample_batch[0].shape[2]
    
    print(f">>> Data loaded:")
    print(f"    Input size: {input_size}")
    print(f"    Features: {features}")
    print(f"    Train batches: {len(train_loader)}")
    print(f"    Test batches: {len(test_loader)}")
    
    # 创建Transformer模型
    model = SimpleTransformer(
        input_size=input_size,
        d_model=64,
        nhead=4,
        num_layers=2,
        prediction_steps=config.PREDICT_STEPS,
        dropout=0.1
    ).to(config.DEVICE)
    
    print(f">>> Transformer model created:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Total parameters: {total_params:,}")
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )
    
    # 早停
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    print(f">>> Start training on {config.DEVICE}...")
    print("-" * 80)
    
    start_time = time.time()
    
    # 训练循环
    for epoch in range(config.EPOCHS):
        epoch_start = time.time()
        
        # 训练
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(config.DEVICE), targets.to(config.DEVICE)
            outputs = model(inputs)
            
            # 形状处理
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(1)
            if targets.dim() == 1:
                targets = targets.unsqueeze(1)
            
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(config.DEVICE), targets.to(config.DEVICE)
                outputs = model(inputs)
                
                if outputs.dim() == 1:
                    outputs = outputs.unsqueeze(1)
                if targets.dim() == 1:
                    targets = targets.unsqueeze(1)
                
                val_loss += criterion(outputs, targets).item()
        
        avg_val_loss = val_loss / len(test_loader)
        
        # 学习率调整
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_epoch = epoch + 1
            
            # 保存最佳模型
            os.makedirs("results/models", exist_ok=True)
            torch.save(model.state_dict(), "results/models/best_transformer_model.pth")
        else:
            patience_counter += 1
        
        # 打印进度
        epoch_time = time.time() - epoch_start
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d}/{config.EPOCHS:03d} | "
                  f"Time: {epoch_time:.1f}s | "
                  f"Train Loss: {avg_train_loss:.6f} | "
                  f"Val Loss: {avg_val_loss:.6f} | "
                  f"LR: {current_lr:.2e} | "
                  f"Best Val: {best_val_loss:.6f}")
        
        # 早停触发
        if patience_counter >= patience:
            print(f">>> Early stopping at epoch {epoch+1}")
            break
    
    total_time = time.time() - start_time
    
    print(f"\n>>> Training completed in {total_time:.1f} seconds")
    print(f">>> Best validation loss: {best_val_loss:.6f}")
    
    # 保存最终模型
    torch.save(model.state_dict(), "results/models/transformer_model_final.pth")
    print(f">>> Models saved to results/models/")
    
    # 评估
    evaluate_model(model, test_loader, scalers)
    
    return model

def evaluate_model(model, test_loader, scalers):
    """评估模型"""
    print("\n>>> Evaluating Transformer model...")
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(config.DEVICE)
            outputs = model(inputs)
            
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(1)
            
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())
    
    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)
    
    # 反归一化
    ndvi_scaler = scalers['NDVI']
    predictions = ndvi_scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(predictions.shape)
    targets = ndvi_scaler.inverse_transform(targets.reshape(-1, 1)).reshape(targets.shape)
    
    # 计算指标
    predictions_flat = predictions.flatten()
    targets_flat = targets.flatten()
    
    mse = mean_squared_error(targets_flat, predictions_flat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets_flat, predictions_flat)
    r2 = r2_score(targets_flat, predictions_flat)
    
    print("\n" + "="*80)
    print("TRANSFORMER MODEL EVALUATION")
    print("="*80)
    print(f"\nOverall Performance:")
    print(f"  MSE:  {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  R²:   {r2:.4f}")
    
    # 按天计算
    print(f"\nPerformance by Prediction Day:")
    print("-"*60)
    for day in range(config.PREDICT_STEPS):
        pred_day = predictions[:, day]
        target_day = targets[:, day]
        r2_day = r2_score(target_day, pred_day)
        mae_day = mean_absolute_error(target_day, pred_day)
        print(f"  Day {day+1}: R²={r2_day:.4f}, MAE={mae_day:.6f}")
    
    print("="*80)

if __name__ == "__main__":
    train_transformer()