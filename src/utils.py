import matplotlib.pyplot as plt
import torch
import os
from src.config import config

def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training Process')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.savefig(os.path.join(config.FIGURE_SAVE_DIR, 'loss_curve.png'))
    plt.close()

def save_checkpoint(model, optimizer):
    path = os.path.join(config.MODEL_SAVE_DIR, 'lstm_model.pth')
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")