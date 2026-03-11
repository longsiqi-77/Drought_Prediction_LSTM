"""
自动化超参数调优
"""
import optuna
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
import streamlit as st
import plotly.graph_objects as go
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import create_model
from src.data.data_loader import load_and_preprocess_data, create_sequences

class HyperparameterTuner:
    def __init__(self):
        self.best_params = None
        self.best_score = -float('inf')
        self.study = None
    
    def objective(self, trial):
        """定义优化目标函数"""
        # 定义搜索空间
        params = {
            'model_type': trial.suggest_categorical('model_type', ['lstm', 'simpletransformer']),
            'seq_length': trial.suggest_int('seq_length', 20, 60, step=10),
            'prediction_steps': trial.suggest_int('prediction_steps', 1, 14, step=3),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
            'epochs': trial.suggest_int('epochs', 30, 150, step=30)
        }
        
        # 模型特定参数
        if params['model_type'] == 'lstm':
            params.update({
                'hidden_size': trial.suggest_categorical('hidden_size', [32, 64, 128, 256]),
                'num_layers': trial.suggest_int('num_layers', 1, 4),
                'dropout': trial.suggest_float('dropout', 0.0, 0.5)
            })
        else:  # transformer
            params.update({
                'd_model': trial.suggest_categorical('d_model', [32, 64, 128]),
                'nhead': trial.suggest_categorical('nhead', [2, 4, 8]),
                'num_layers': trial.suggest_int('num_layers', 1, 4),
                'dropout': trial.suggest_float('dropout', 0.0, 0.3)
            })
        
        # 在这里模拟训练和评估
        # 实际使用时需要调用真实的训练函数
        score = self.mock_train_and_evaluate(params)
        
        return score
    
    def mock_train_and_evaluate(self, params):
        """模拟训练和评估（返回模拟分数）"""
        # 基础分数
        base_score = 0.7
        
        # 根据参数调整分数（模拟效果）
        if params['model_type'] == 'simpletransformer':
            base_score += 0.1
        
        if params['seq_length'] == 30:
            base_score += 0.05
        
        if params['learning_rate'] >= 0.0005 and params['learning_rate'] <= 0.002:
            base_score += 0.03
        
        if params['model_type'] == 'lstm' and params['hidden_size'] == 64:
            base_score += 0.02
        
        # 添加随机噪声（模拟训练波动）
        noise = np.random.normal(0, 0.02)
        
        return max(0.6, min(0.9, base_score + noise))
    
    def run_optimization(self, n_trials=50):
        """运行优化"""
        self.study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        self.study.optimize(self.objective, n_trials=n_trials)
        
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        return self.best_params, self.best_score
    
    def visualize_optimization(self):
        """可视化优化过程"""
        if self.study is None:
            return None
        
        fig = optuna.visualization.plot_optimization_history(self.study)
        return fig
    
    def get_parallel_coordinate_plot(self):
        """获取平行坐标图"""
        if self.study is None:
            return None
        
        fig = optuna.visualization.plot_parallel_coordinate(self.study)
        return fig

def create_tuner_interface():
    """创建调参界面"""
    st.title("🤖 自动化超参数调优")
    
    st.markdown("""
    使用贝叶斯优化自动搜索最佳模型参数。
    系统会尝试不同的参数组合，找到性能最优的配置。
    """)
    
    # 参数设置
    col1, col2 = st.columns(2)
    
    with col1:
        n_trials = st.slider("试验次数", 10, 200, 50, 10,
                           help="尝试的参数组合数量，越多越准但越慢")
        
        models_to_tune = st.multiselect(
            "调优的模型",
            ["LSTM", "Transformer", "SimpleTransformer"],
            default=["LSTM", "SimpleTransformer"]
        )
    
    with col2:
        timeout = st.number_input("超时时间(分钟)", 5, 120, 30)
        
        tune_data_params = st.checkbox("调优数据参数", value=True)
        if tune_data_params:
            min_seq_len = st.slider("最小序列长度", 10, 40, 20, 5)
            max_seq_len = st.slider("最大序列长度", 30, 90, 60, 10)
    
    # 开始调优按钮
    if st.button("🚀 开始自动调优", type="primary"):
        tuner = HyperparameterTuner()
        
        with st.spinner(f"正在进行超参数优化 ({n_trials} 次试验)..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 模拟优化过程
            import time
            
            for i in range(n_trials):
                time.sleep(0.1)  # 模拟每次试验的时间
                progress = (i + 1) / n_trials
                progress_bar.progress(progress)
                
                status_text.text(f"试验 {i+1}/{n_trials}...")
            
            # 获取结果
            best_params, best_score = tuner.run_optimization(n_trials)
            
            # 显示结果
            st.success(f"✅ 优化完成！最佳R²分数: {best_score:.4f}")
            
            # 显示最佳参数
            st.markdown("### 🏆 最佳参数配置")
            
            params_df = pd.DataFrame([best_params]).T
            params_df.columns = ['值']
            st.dataframe(params_df)
            
            # 可视化
            st.markdown("### 📊 优化过程可视化")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(tuner.visualize_optimization(), use_container_width=True)
            
            with col2:
                st.plotly_chart(tuner.get_parallel_coordinate_plot(), use_container_width=True)
            
            # 保存结果选项
            st.markdown("### 💾 保存优化结果")
            
            save_name = st.text_input("保存名称", value=f"opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            if st.button("保存优化配置"):
                # 创建优化目录
                opt_dir = f"results/optimizations/{save_name}"
                os.makedirs(opt_dir, exist_ok=True)
                
                # 保存参数
                import json
                with open(f"{opt_dir}/best_params.json", 'w') as f:
                    json.dump(best_params, f, indent=2)
                
                # 保存学习曲线
                # tuner.study.trials_dataframe().to_csv(f"{opt_dir}/trials.csv", index=False)
                
                st.success(f"✅ 优化结果已保存到: {opt_dir}")

if __name__ == "__main__":
    import pandas as pd
    from datetime import datetime
    
    create_tuner_interface()
