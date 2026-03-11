"""
多模型训练示例 - 支持 LSTM / Transformer / SimpleTransformer
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.models import create_model
from src.data_loader import get_dataloaders
from src.config import config

class MultiModelTrainer:
    def __init__(self, model_type='LSTM', save_dir=None):
        self.model_type = model_type
        self.device = torch.device(config.DEVICE)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.save_dir = Path(save_dir or f"results/experiments/{model_type}_{timestamp}")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = {'train_loss': [], 'val_loss': [], 'learning_rate': [],
                        'best_val_loss': float('inf'), 'best_epoch': 0}
        print(f"\n{'='*60}\nTRAINING {model_type.upper()} MODEL\n{'='*60}")

    def prepare_data(self):
        print(">>> Preparing data...")
        train_loader, val_loader, scalers, features = get_dataloaders(
            pred_steps=config.PREDICT_STEPS,
            model_type=self.model_type
        )
        sample_batch = next(iter(train_loader))
        input_size = sample_batch[0].shape[2]
        print(f">>> Data loaded: input_size={input_size}, features={features}, "
              f"train_batches={len(train_loader)}, val_batches={len(val_loader)}")
        return train_loader, val_loader, scalers, input_size

    def create_model(self, input_size):
        print(f">>> Creating {self.model_type} model...")
        model = create_model(
            model_type=self.model_type,
            input_size=input_size,
            prediction_steps=config.PREDICT_STEPS,
            hidden_size=getattr(config, 'HIDDEN_SIZE', 64),
            num_layers=getattr(config, 'NUM_LAYERS', 2),
            dropout=getattr(config, 'DROPOUT', 0.2),
            d_model=getattr(config, 'TRANSFORMER_D_MODEL', 64),
            nhead=getattr(config, 'TRANSFORMER_NHEAD', 4),
            num_encoder_layers=getattr(config, 'TRANSFORMER_NUM_LAYERS', 2),
            num_decoder_layers=getattr(config, 'TRANSFORMER_NUM_LAYERS', 2),
            dim_feedforward=256
        ).to(self.device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f">>> Model created: total_params={total_params:,}, trainable_params={trainable_params:,}")
        return model

    def train_epoch(self, model, loader, criterion, optimizer):
        model.train()
        total_loss = 0
        for inputs, targets in loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = model(inputs)
            if outputs.dim() == 1: outputs = outputs.unsqueeze(1)
            if targets.dim() == 1: targets = targets.unsqueeze(1)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def validate(self, model, loader, criterion):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                if outputs.dim() == 1: outputs = outputs.unsqueeze(1)
                if targets.dim() == 1: targets = targets.unsqueeze(1)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
        return total_loss / len(loader)

    def train(self, num_epochs=None):
        num_epochs = num_epochs or config.EPOCHS
        train_loader, val_loader, scalers, input_size = self.prepare_data()
        model = self.create_model(input_size)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
        
        best_val_loss = float('inf')
        best_model_state = None
        patience = 20
        patience_counter = 0
        start_time = time.time()

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(model, train_loader, criterion, optimizer)
            val_loss = self.validate(model, val_loader, criterion)
            scheduler.step(val_loss)
            lr = optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(lr)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                self.history['best_val_loss'] = best_val_loss
                self.history['best_epoch'] = epoch+1
                patience_counter = 0
                torch.save(model.state_dict(), self.save_dir / f"best_{self.model_type.lower()}.pth")
            else:
                patience_counter += 1

            if (epoch+1) % 10 == 0 or epoch==0:
                print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {lr:.2e}")

            if patience_counter >= patience:
                print(f">>> Early stopping at epoch {epoch+1}")
                break

        if best_model_state:
            model.load_state_dict(best_model_state)
        torch.save(model.state_dict(), self.save_dir / f"{self.model_type.lower()}_final.pth")
        total_time = time.time() - start_time
        print(f">>> Training completed in {total_time:.1f}s | Best val loss: {best_val_loss:.6f}")
        return model, train_loader, val_loader, scalers

def main():
    for model_type in ["lstm", "transformer", "simpletransformer"]:
        print("\n" + "="*50)
        print(f">>> Start training {model_type.upper()}")
        trainer = MultiModelTrainer(model_type=model_type)
        model, train_loader, val_loader, scalers = trainer.train()
        print(f">>> Finished training {model_type.upper()}")

if __name__ == "__main__":
    main()