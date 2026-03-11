import re
import matplotlib.pyplot as plt
import numpy as np
import os

# 从你提供的训练日志中提取数据
log_text = """
Epoch 001/100 | Time: 1.1s | Train Loss: 0.048553 | Val Loss: 0.013943 | LR: 1.00e-03 | Best Val: 0.013943
Epoch 010/100 | Time: 0.7s | Train Loss: 0.005141 | Val Loss: 0.001291 | LR: 1.00e-03 | Best Val: 0.001291
Epoch 020/100 | Time: 0.7s | Train Loss: 0.003744 | Val Loss: 0.002018 | LR: 1.00e-03 | Best Val: 0.001190
Epoch 030/100 | Time: 0.7s | Train Loss: 0.002498 | Val Loss: 0.001391 | LR: 5.00e-04 | Best Val: 0.001094
Epoch 040/100 | Time: 0.7s | Train Loss: 0.002030 | Val Loss: 0.001461 | LR: 2.50e-04 | Best Val: 0.001094
>>> Early stopping at epoch 49
"""

# 提取数据
epochs = []
train_losses = []
val_losses = []
lrs = []

lines = log_text.strip().split('\n')
for line in lines:
    if 'Epoch' in line:
        # 提取epoch数字
        epoch_match = re.search(r'Epoch (\d+)/', line)
        if epoch_match:
            epoch = int(epoch_match.group(1))
            epochs.append(epoch)
        
        # 提取训练损失
        train_match = re.search(r'Train Loss: ([\d\.]+)', line)
        if train_match:
            train_losses.append(float(train_match.group(1)))
        
        # 提取验证损失
        val_match = re.search(r'Val Loss: ([\d\.]+)', line)
        if val_match:
            val_losses.append(float(val_match.group(1)))
        
        # 提取学习率
        lr_match = re.search(r'LR: ([\d\.e+-]+)', line)
        if lr_match:
            lr_str = lr_match.group(1)
            # 将科学计数法转换为浮点数
            if 'e' in lr_str:
                base, exp = lr_str.split('e')
                lr = float(base) * (10 ** int(exp))
            else:
                lr = float(lr_str)
            lrs.append(lr)

print(f"提取到的数据:")
print(f"Epochs: {epochs}")
print(f"Train Losses: {train_losses}")
print(f"Val Losses: {val_losses}")
print(f"Learning Rates: {lrs}")

# 生成完整的历史数据（通过插值）
full_epochs = list(range(1, 50))  # 1到49
full_train_losses = np.interp(full_epochs, epochs, train_losses)
full_val_losses = np.interp(full_epochs, epochs, val_losses)

# 学习率变化（假设每10个epoch变化）
full_lrs = []
for epoch in full_epochs:
    if epoch <= 20:
        full_lrs.append(0.001)  # 1.00e-03
    elif epoch <= 30:
        full_lrs.append(0.0005)  # 5.00e-04
    else:
        full_lrs.append(0.00025)  # 2.50e-04

# 创建图表
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 1. 训练和验证损失
axes[0, 0].plot(full_epochs, full_train_losses, 'b-', label='Train Loss', linewidth=2)
axes[0, 0].plot(full_epochs, full_val_losses, 'r-', label='Val Loss', linewidth=2)
axes[0, 0].axhline(y=0.001094, color='g', linestyle='--', label=f'Best Val: {0.001094:.6f}')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Transformer Training History')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_yscale('log')

# 2. 学习率计划
axes[0, 1].step(full_epochs, full_lrs, where='post', linewidth=2)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Learning Rate')
axes[0, 1].set_title('Learning Rate Schedule')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_yscale('log')

# 3. 损失下降率
train_loss_gradient = np.gradient(full_train_losses)
val_loss_gradient = np.gradient(full_val_losses)
axes[1, 0].plot(full_epochs[1:], train_loss_gradient[1:], 'b-', label='Train Loss Gradient', alpha=0.7)
axes[1, 0].plot(full_epochs[1:], val_loss_gradient[1:], 'r-', label='Val Loss Gradient', alpha=0.7)
axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Loss Gradient')
axes[1, 0].set_title('Loss Convergence Rate')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. 最终评估结果（从你提供的输出）
prediction_days = list(range(1, 8))
r2_scores = [0.9565, 0.9321, 0.8949, 0.8311, 0.7725, 0.6858, 0.5977]
mae_scores = [0.003075, 0.003781, 0.004686, 0.005999, 0.006906, 0.008053, 0.008910]

axes[1, 1].plot(prediction_days, r2_scores, 'g-o', linewidth=2, markersize=8, label='R² Score')
axes[1, 1].set_xlabel('Prediction Day')
axes[1, 1].set_ylabel('R² Score', color='g')
axes[1, 1].set_title('Prediction Performance by Day')
axes[1, 1].tick_params(axis='y', labelcolor='g')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend(loc='upper left')

# 添加第二个Y轴显示MAE
ax2 = axes[1, 1].twinx()
ax2.plot(prediction_days, mae_scores, 'b-s', linewidth=2, markersize=6, label='MAE', alpha=0.7)
ax2.set_ylabel('MAE', color='b')
ax2.tick_params(axis='y', labelcolor='b')
ax2.legend(loc='upper right')

plt.tight_layout()

# 保存图表
output_dir = 'results/experiments/transformer_20251209_173724'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'training_history.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ 图表已生成并保存到: {output_path}")

# 显示图表信息
print(f"\n图表包含:")
print("1. 训练和验证损失曲线")
print("2. 学习率调度计划") 
print("3. 损失收敛速率")
print("4. 按预测天数的性能分析")

plt.show()
