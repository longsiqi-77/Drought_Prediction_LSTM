"""
交互式参数调节训练界面
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入项目模块
try:
    from src.data.data_loader import load_and_preprocess_data, create_sequences
    from src.models import create_model
    from src.utils.data_utils import StandardScaler
except ImportError:
    st.error("❌ 无法导入项目模块，请确保在项目根目录运行")
    st.stop()

# 页面配置
st.set_page_config(
    page_title="交互式参数调节",
    page_icon="⚙️",
    layout="wide"
)

class InteractiveTraining:
    def __init__(self):
        self.data = None
        self.scalers = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_data(self):
        """加载数据"""
        try:
            self.data, self.scalers = load_and_preprocess_data()
            return True
        except Exception as e:
            st.error(f"数据加载失败: {e}")
            return False
    
    def create_sequences_from_data(self, data, seq_length, prediction_steps):
        """从数据创建序列"""
        X, y = create_sequences(
            data.values,
            seq_length=seq_length,
            prediction_steps=prediction_steps
        )
        return X, y
    
    def prepare_dataloaders(self, X, y, batch_size, train_ratio=0.8):
        """准备数据加载器"""
        # 划分训练集和测试集
        train_size = int(len(X) * train_ratio)
        
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # 转换为Tensor
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test)
        
        # 创建Dataset和DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader
    
    def train_model(self, model, train_loader, test_loader, epochs, learning_rate, patience=10):
        """训练模型"""
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                        patience=patience//2, factor=0.5)
        
        model.to(self.device)
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_times': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        loss_chart = st.empty()
        
        # 创建损失图表
        fig = go.Figure()
        fig.update_layout(
            title="训练过程实时监控",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            height=400
        )
        loss_chart.plotly_chart(fig, use_container_width=True)
        
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            epoch_start_time = datetime.now()
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            
            # 验证阶段
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(test_loader)
            epoch_time = (datetime.now() - epoch_start_time).total_seconds()
            
            # 更新学习率
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # 保存历史
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['learning_rate'].append(current_lr)
            history['epoch_times'].append(epoch_time)
            
            # 更新进度
            progress = (epoch + 1) / epochs
            progress_bar.progress(progress)
            
            status_text.text(f"Epoch {epoch+1}/{epochs} - "
                           f"Train Loss: {avg_train_loss:.6f}, "
                           f"Val Loss: {avg_val_loss:.6f}, "
                           f"LR: {current_lr:.6f}, "
                           f"Time: {epoch_time:.2f}s")
            
            # 更新图表
            fig.data = []
            fig.add_trace(go.Scatter(
                x=list(range(1, epoch+2)),
                y=history['train_loss'],
                mode='lines+markers',
                name='Train Loss',
                line=dict(color='blue', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=list(range(1, epoch+2)),
                y=history['val_loss'],
                mode='lines+markers',
                name='Val Loss',
                line=dict(color='red', width=2)
            ))
            loss_chart.plotly_chart(fig, use_container_width=True)
            
            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(model.state_dict(), 'best_model_temp.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    status_text.text(f"🎯 早停在 epoch {epoch+1}, 最佳验证损失: {best_val_loss:.6f}")
                    break
        
        progress_bar.empty()
        status_text.empty()
        
        # 加载最佳模型
        model.load_state_dict(torch.load('best_model_temp.pth'))
        os.remove('best_model_temp.pth')
        
        return model, history
    
    def evaluate_model(self, model, test_loader):
        """评估模型"""
        model.eval()
        criterion = nn.MSELoss()
        
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = model(batch_X)
                
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()
                
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())
        
        avg_loss = total_loss / len(test_loader)
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # 计算指标
        mse = avg_loss
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))
        
        # 计算R²
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2)
        }
        
        return metrics, predictions, targets
    
    def create_parameter_sidebar(self):
        """创建参数调节侧边栏"""
        with st.sidebar:
            st.markdown("## ⚙️ 模型参数调节")
            
            # 模型选择
            model_type = st.selectbox(
                "选择模型类型",
                ["LSTM", "Transformer", "SimpleTransformer"],
                index=1
            )
            
            # 通用参数
            st.markdown("### 📊 数据参数")
            seq_length = st.slider("时间窗口长度", 10, 60, 30, 5,
                                 help="输入序列的时间步长")
            prediction_steps = st.slider("预测步数", 1, 14, 7, 1,
                                       help="需要预测的未来时间步数")
            batch_size = st.selectbox("批大小", [16, 32, 64, 128], index=1)
            
            st.markdown("### 🎯 训练参数")
            epochs = st.slider("训练轮数", 10, 200, 100, 10)
            learning_rate = st.number_input("学习率", 1e-5, 1e-1, 1e-3, format="%.5f")
            patience = st.slider("早停耐心值", 5, 50, 10, 5)
            
            # 模型特定参数
            st.markdown("### 🔧 模型特定参数")
            
            if model_type == "LSTM":
                hidden_size = st.slider("LSTM隐藏层大小", 16, 256, 64, 16)
                num_layers = st.slider("LSTM层数", 1, 4, 2, 1)
                dropout = st.slider("Dropout率", 0.0, 0.5, 0.2, 0.05)
                
                model_params = {
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'dropout': dropout
                }
                
            elif model_type in ["Transformer", "SimpleTransformer"]:
                d_model = st.slider("模型维度", 32, 256, 64, 32)
                nhead = st.slider("注意力头数", 2, 16, 4, 2)
                
                if model_type == "Transformer":
                    num_encoder_layers = st.slider("编码器层数", 1, 6, 2, 1)
                    num_decoder_layers = st.slider("解码器层数", 1, 6, 2, 1)
                    dim_feedforward = st.slider("前馈网络维度", 128, 1024, 256, 128)
                    
                    model_params = {
                        'd_model': d_model,
                        'nhead': nhead,
                        'num_encoder_layers': num_encoder_layers,
                        'num_decoder_layers': num_decoder_layers,
                        'dim_feedforward': dim_feedforward,
                        'dropout': 0.1
                    }
                else:
                    num_layers = st.slider("编码器层数", 1, 6, 2, 1)
                    
                    model_params = {
                        'd_model': d_model,
                        'nhead': nhead,
                        'num_layers': num_layers,
                        'dropout': 0.1
                    }
            
            # 训练按钮
            st.markdown("---")
            train_button = st.button("🚀 开始训练", type="primary", use_container_width=True)
            
            return {
                'model_type': model_type.lower(),
                'seq_length': seq_length,
                'prediction_steps': prediction_steps,
                'batch_size': batch_size,
                'epochs': epochs,
                'learning_rate': learning_rate,
                'patience': patience,
                'model_params': model_params,
                'train_button': train_button
            }
    
    def display_training_results(self, history, metrics, predictions, targets):
        """显示训练结果"""
        st.markdown("## 📊 训练结果")
        
        # 指标卡片
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("R²分数", f"{metrics['r2']:.4f}")
        with col2:
            st.metric("MAE", f"{metrics['mae']:.6f}")
        with col3:
            st.metric("MSE", f"{metrics['mse']:.6f}")
        with col4:
            st.metric("RMSE", f"{metrics['rmse']:.6f}")
        
        # 训练历史图表
        st.markdown("### 📈 训练历史")
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('损失曲线', '学习率变化'),
            horizontal_spacing=0.15
        )
        
        epochs = list(range(1, len(history['train_loss']) + 1))
        
        fig.add_trace(
            go.Scatter(x=epochs, y=history['train_loss'],
                      name='训练损失', line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=epochs, y=history['val_loss'],
                      name='验证损失', line=dict(color='red', width=2)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=epochs, y=history['learning_rate'],
                      name='学习率', line=dict(color='green', width=2)),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=True)
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Learning Rate", row=1, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 预测结果可视化
        st.markdown("### 🔮 预测结果示例")
        
        # 显示前5个样本的预测
        num_samples = min(5, len(predictions))
        
        for i in range(num_samples):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                fig_sample = go.Figure()
                
                # 实际值
                fig_sample.add_trace(go.Scatter(
                    x=list(range(1, len(targets[i]) + 1)),
                    y=targets[i],
                    mode='lines+markers',
                    name='实际值',
                    line=dict(color='blue', width=3),
                    marker=dict(size=8)
                ))
                
                # 预测值
                fig_sample.add_trace(go.Scatter(
                    x=list(range(1, len(predictions[i]) + 1)),
                    y=predictions[i],
                    mode='lines+markers',
                    name='预测值',
                    line=dict(color='red', width=3, dash='dash'),
                    marker=dict(size=8)
                ))
                
                fig_sample.update_layout(
                    title=f"样本 {i+1} 的预测结果",
                    xaxis_title="预测天数",
                    yaxis_title="NDVI值",
                    height=300,
                    showlegend=True
                )
                
                st.plotly_chart(fig_sample, use_container_width=True)
            
            with col2:
                # 计算该样本的误差
                sample_mae = np.mean(np.abs(predictions[i] - targets[i]))
                sample_mse = np.mean((predictions[i] - targets[i]) ** 2)
                
                st.metric("样本MAE", f"{sample_mae:.6f}")
                st.metric("样本MSE", f"{sample_mse:.6f}")
    
    def run(self):
        """运行交互式训练界面"""
        st.title("⚙️ 交互式参数调节训练")
        st.markdown("调整模型参数并实时观察训练效果")
        
        # 加载数据
        if self.data is None:
            with st.spinner("正在加载数据..."):
                if not self.load_data():
                    st.stop()
        
        # 创建参数侧边栏
        params = self.create_parameter_sidebar()
        
        # 如果点击训练按钮
        if params['train_button']:
            try:
                # 准备数据
                with st.spinner("准备数据..."):
                    # 这里需要根据你的数据格式调整
                    # 假设self.data是一个DataFrame
                    X, y = self.create_sequences_from_data(
                        self.data,
                        seq_length=params['seq_length'],
                        prediction_steps=params['prediction_steps']
                    )
                    
                    train_loader, test_loader = self.prepare_dataloaders(
                        X, y,
                        batch_size=params['batch_size']
                    )
                
                # 创建模型
                with st.spinner("创建模型..."):
                    model = create_model(
                        params['model_type'],
                        input_size=X.shape[2],  # 特征数量
                        prediction_steps=params['prediction_steps'],
                        **params['model_params']
                    )
                    
                    st.success(f"✅ 模型创建成功！参数量: {sum(p.numel() for p in model.parameters()):,}")
                
                # 训练模型
                with st.spinner("开始训练..."):
                    model, history = self.train_model(
                        model, train_loader, test_loader,
                        epochs=params['epochs'],
                        learning_rate=params['learning_rate'],
                        patience=params['patience']
                    )
                
                # 评估模型
                with st.spinner("评估模型..."):
                    metrics, predictions, targets = self.evaluate_model(model, test_loader)
                
                # 显示结果
                self.display_training_results(history, metrics, predictions, targets)
                
                # 保存结果选项
                st.markdown("---")
                save_exp = st.checkbox("�� 保存本次实验")
                
                if save_exp:
                    exp_name = st.text_input("实验名称", value=f"{params['model_type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                    
                    if st.button("保存实验"):
                        # 创建实验目录
                        exp_dir = f"results/experiments/{exp_name}"
                        os.makedirs(exp_dir, exist_ok=True)
                        
                        # 保存模型
                        model_path = os.path.join(exp_dir, f"{params['model_type']}_model.pth")
                        torch.save(model.state_dict(), model_path)
                        
                        # 保存训练历史
                        history_path = os.path.join(exp_dir, "training_history.json")
                        with open(history_path, 'w') as f:
                            json.dump(history, f)
                        
                        # 保存指标
                        metrics_path = os.path.join(exp_dir, "metrics.json")
                        with open(metrics_path, 'w') as f:
                            json.dump(metrics, f)
                        
                        # 保存参数
                        params_path = os.path.join(exp_dir, "params.json")
                        with open(params_path, 'w') as f:
                            json.dump(params, f, default=str)
                        
                        st.success(f"✅ 实验已保存到: {exp_dir}")
                        
            except Exception as e:
                st.error(f"训练过程中出现错误: {str(e)}")
                st.exception(e)

def main():
    trainer = InteractiveTraining()
    trainer.run()

if __name__ == "__main__":
    main()
