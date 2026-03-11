"""
干旱预测结果展示仪表盘
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
import json
import os
import sys
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 页面配置
st.set_page_config(
    page_title="干旱预测系统",
    page_icon="🌵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .model-card {
        background-color: #e9f7fe;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #b3e0ff;
        margin-bottom: 1rem;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class DroughtPredictionDashboard:
    def __init__(self):
        self.results_dir = "results"
        self.models_dir = "results/models"
        self.experiments_dir = "results/experiments"
        
    def load_experiment_data(self, model_type):
        """加载实验数据"""
        exp_dirs = [d for d in os.listdir(self.experiments_dir) 
                   if d.lower().startswith(model_type.lower())]
        
        if not exp_dirs:
            return None
        
        latest_exp = max(exp_dirs, key=lambda x: os.path.getmtime(
            os.path.join(self.experiments_dir, x)))
        exp_path = os.path.join(self.experiments_dir, latest_exp)
        
        data = {
            'experiment_path': exp_path,
            'model_type': model_type,
            'timestamp': datetime.fromtimestamp(os.path.getmtime(exp_path))
        }
        
        # 加载训练历史
        history_files = [f for f in os.listdir(exp_path) if 'history' in f and f.endswith('.json')]
        if history_files:
            with open(os.path.join(exp_path, history_files[0]), 'r') as f:
                data['history'] = json.load(f)
        
        # 加载评估结果
        eval_files = [f for f in os.listdir(exp_path) if 'eval' in f or 'metrics' in f]
        for file in eval_files:
            file_path = os.path.join(exp_path, file)
            if file.endswith('.json'):
                with open(file_path, 'r') as f:
                    data['metrics'] = json.load(f)
            elif file.endswith('.txt'):
                with open(file_path, 'r') as f:
                    data['eval_text'] = f.read()
        
        return data
    
    def create_metrics_display(self, lstm_data, transformer_data):
        """创建指标对比显示"""
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.markdown("### 📈 模型对比")
            
            if lstm_data and 'metrics' in lstm_data:
                lstm_metrics = lstm_data['metrics']
                st.metric("LSTM R²", f"{lstm_metrics.get('r2', 0):.3f}")
                st.metric("LSTM MAE", f"{lstm_metrics.get('mae', 0):.6f}")
            
            if transformer_data and 'metrics' in transformer_data:
                transformer_metrics = transformer_data['metrics']
                st.metric("Transformer R²", f"{transformer_metrics.get('r2', 0):.3f}")
                st.metric("Transformer MAE", f"{transformer_metrics.get('mae', 0):.6f}")
        
        with col2:
            st.markdown("### �� 最佳性能")
            
            if lstm_data and transformer_data:
                lstm_r2 = lstm_data['metrics'].get('r2', 0)
                transformer_r2 = transformer_data['metrics'].get('r2', 0)
                
                if lstm_r2 > transformer_r2:
                    best_model = "LSTM"
                    best_score = lstm_r2
                else:
                    best_model = "Transformer"
                    best_score = transformer_r2
                
                st.metric("最佳模型", best_model)
                st.metric("最佳R²", f"{best_score:.3f}")
        
        with col3:
            st.markdown("### 📊 训练信息")
            
            if lstm_data:
                st.metric("LSTM训练时长", 
                         f"{lstm_data['history'].get('training_time', 0):.1f}s" 
                         if 'history' in lstm_data else "N/A")
            
            if transformer_data:
                st.metric("Transformer训练时长", 
                         f"{transformer_data['history'].get('training_time', 0):.1f}s" 
                         if 'history' in transformer_data else "N/A")
    
    def create_training_history_plot(self, lstm_data, transformer_data):
        """创建训练历史图表"""
        st.markdown("### 📈 训练历史对比")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('训练损失', '验证损失', '学习率变化', 'R²分数演变'),
            vertical_spacing=0.15,
            horizontal_spacing=0.15
        )
        
        # 训练损失
        if lstm_data and 'history' in lstm_data:
            lstm_history = lstm_data['history']
            epochs = range(1, len(lstm_history.get('train_loss', [])) + 1)
            fig.add_trace(
                go.Scatter(x=list(epochs), y=lstm_history.get('train_loss', []),
                          name='LSTM Train', line=dict(color='blue', width=2)),
                row=1, col=1
            )
        
        if transformer_data and 'history' in transformer_data:
            transformer_history = transformer_data['history']
            epochs = range(1, len(transformer_history.get('train_loss', [])) + 1)
            fig.add_trace(
                go.Scatter(x=list(epochs), y=transformer_history.get('train_loss', []),
                          name='Transformer Train', line=dict(color='red', width=2)),
                row=1, col=1
            )
        
        # 验证损失
        if lstm_data and 'history' in lstm_data:
            lstm_history = lstm_data['history']
            epochs = range(1, len(lstm_history.get('val_loss', [])) + 1)
            fig.add_trace(
                go.Scatter(x=list(epochs), y=lstm_history.get('val_loss', []),
                          name='LSTM Val', line=dict(color='blue', width=2, dash='dash')),
                row=1, col=2
            )
        
        if transformer_data and 'history' in transformer_data:
            transformer_history = transformer_data['history']
            epochs = range(1, len(transformer_history.get('val_loss', [])) + 1)
            fig.add_trace(
                go.Scatter(x=list(epochs), y=transformer_history.get('val_loss', []),
                          name='Transformer Val', line=dict(color='red', width=2, dash='dash')),
                row=1, col=2
            )
        
        # 学习率
        if transformer_data and 'history' in transformer_data:
            transformer_history = transformer_data['history']
            if 'learning_rate' in transformer_history:
                epochs = range(1, len(transformer_history['learning_rate']) + 1)
                fig.add_trace(
                    go.Scatter(x=list(epochs), y=transformer_history['learning_rate'],
                              name='Transformer LR', line=dict(color='green', width=2)),
                    row=2, col=1
                )
        
        # R²分数
        if transformer_data and 'history' in transformer_data:
            transformer_history = transformer_data['history']
            if 'val_r2' in transformer_history:
                epochs = range(1, len(transformer_history['val_r2']) + 1)
                fig.add_trace(
                    go.Scatter(x=list(epochs), y=transformer_history['val_r2'],
                              name='Transformer R²', line=dict(color='purple', width=2)),
                    row=2, col=2
                )
        
        fig.update_layout(height=600, showlegend=True, title_text="训练过程对比分析")
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_xaxes(title_text="Epoch", row=2, col=1)
        fig.update_xaxes(title_text="Epoch", row=2, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def create_prediction_performance_plot(self, transformer_data):
        """创建预测性能图表"""
        if not transformer_data or 'metrics' not in transformer_data:
            return
        
        metrics = transformer_data['metrics']
        
        st.markdown("### 📊 多步预测性能分析")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # R²分数随预测天数的变化
            if 'day_r2' in metrics:
                days = list(range(1, len(metrics['day_r2']) + 1))
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(
                    x=days, y=metrics['day_r2'],
                    mode='lines+markers',
                    name='R² Score',
                    line=dict(color='green', width=3),
                    marker=dict(size=8)
                ))
                fig1.update_layout(
                    title='R²分数随预测天数变化',
                    xaxis_title='预测天数',
                    yaxis_title='R²分数',
                    height=400
                )
                st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # MAE随预测天数的变化
            if 'day_mae' in metrics:
                days = list(range(1, len(metrics['day_mae']) + 1))
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=days, y=metrics['day_mae'],
                    mode='lines+markers',
                    name='MAE',
                    line=dict(color='red', width=3),
                    marker=dict(size=8)
                ))
                fig2.update_layout(
                    title='MAE随预测天数变化',
                    xaxis_title='预测天数',
                    yaxis_title='MAE',
                    height=400
                )
                st.plotly_chart(fig2, use_container_width=True)
    
    def create_model_comparison(self, lstm_data, transformer_data):
        """创建模型对比分析"""
        st.markdown("### �� 模型详细对比")
        
        if not lstm_data or not transformer_data:
            st.warning("缺少模型数据，无法进行对比")
            return
        
        # 创建对比表格
        comparison_data = []
        
        metrics_to_compare = ['r2', 'mae', 'mse', 'rmse']
        
        for metric in metrics_to_compare:
            lstm_value = lstm_data['metrics'].get(metric, 'N/A')
            transformer_value = transformer_data['metrics'].get(metric, 'N/A')
            
            if metric == 'r2':
                better = "LSTM" if lstm_value > transformer_value else "Transformer"
            else:
                better = "LSTM" if lstm_value < transformer_value else "Transformer"
            
            comparison_data.append({
                '指标': metric.upper(),
                'LSTM': f"{lstm_value:.6f}" if isinstance(lstm_value, (int, float)) else lstm_value,
                'Transformer': f"{transformer_value:.6f}" if isinstance(transformer_value, (int, float)) else transformer_value,
                '更优模型': better
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True)
        
        # 雷达图对比
        st.markdown("#### 📊 模型性能雷达图")
        
        metrics_for_radar = ['r2', 'mae', 'mse']
        categories = [m.upper() for m in metrics_for_radar]
        
        fig = go.Figure()
        
        # LSTM数据
        lstm_values = []
        for metric in metrics_for_radar:
            value = lstm_data['metrics'].get(metric, 0)
            if metric == 'r2':
                lstm_values.append(value)
            else:
                # 对损失类指标进行归一化（值越小越好）
                lstm_values.append(1 - min(value, 1))
        
        fig.add_trace(go.Scatterpolar(
            r=lstm_values + [lstm_values[0]],  # 闭合图形
            theta=categories + [categories[0]],
            fill='toself',
            name='LSTM',
            line_color='blue'
        ))
        
        # Transformer数据
        transformer_values = []
        for metric in metrics_for_radar:
            value = transformer_data['metrics'].get(metric, 0)
            if metric == 'r2':
                transformer_values.append(value)
            else:
                transformer_values.append(1 - min(value, 1))
        
        fig.add_trace(go.Scatterpolar(
            r=transformer_values + [transformer_values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name='Transformer',
            line_color='red'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="模型性能雷达图对比",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def run(self):
        """运行仪表盘"""
        # 标题
        st.markdown('<h1 class="main-header">🌵 干旱预测系统 - 结果展示仪表盘</h1>', 
                   unsafe_allow_html=True)
        
        # 侧边栏
        with st.sidebar:
            st.markdown("### ⚙️ 控制面板")
            
            # 模型选择
            st.markdown("#### 选择模型")
            show_lstm = st.checkbox("显示LSTM结果", value=True)
            show_transformer = st.checkbox("显示Transformer结果", value=True)
            
            # 图表选项
            st.markdown("#### 图表选项")
            show_training_history = st.checkbox("显示训练历史", value=True)
            show_prediction_performance = st.checkbox("显示预测性能", value=True)
            show_model_comparison = st.checkbox("显示模型对比", value=True)
            
            # 刷新按钮
            if st.button("🔄 刷新数据"):
                st.rerun()
        
        # 加载数据
        lstm_data = self.load_experiment_data('lstm') if show_lstm else None
        transformer_data = self.load_experiment_data('transformer') if show_transformer else None
        
        # 指标显示
        self.create_metrics_display(lstm_data, transformer_data)
        
        # 训练历史图表
        if show_training_history:
            self.create_training_history_plot(lstm_data, transformer_data)
        
        # 预测性能图表
        if show_prediction_performance and transformer_data:
            self.create_prediction_performance_plot(transformer_data)
        
        # 模型对比
        if show_model_comparison and lstm_data and transformer_data:
            self.create_model_comparison(lstm_data, transformer_data)
        
        # 底部信息
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col2:
            st.markdown("""
            <div style='text-align: center; color: #666;'>
            <p>🌵 干旱预测系统 v1.0</p>
            <p>基于深度学习的多步时序预测</p>
            <p>© 2024 所有数据均为模拟数据，仅供参考</p>
            </div>
            """, unsafe_allow_html=True)

def main():
    # 检查必要的目录
    if not os.path.exists("results/experiments"):
        st.error("❌ 未找到实验结果目录，请先训练模型")
        if st.button("开始训练模型"):
            # 这里可以添加训练模型的代码
            pass
        return
    
    dashboard = DroughtPredictionDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
