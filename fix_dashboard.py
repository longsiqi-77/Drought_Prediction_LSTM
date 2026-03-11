import re

# 读取文件
with open('src/dashboard.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 找到 create_prediction_performance_plot 方法的位置
pattern = r'def create_prediction_performance_plot\(self, transformer_data\):(.*?)(?=\n    def |\n\n|\Z)'
match = re.search(pattern, content, re.DOTALL)

if match:
    # 在这个方法后面添加 create_model_comparison 方法
    insert_pos = match.end()
    
    new_method = '''
    def create_model_comparison(self, lstm_data, transformer_data):
        """创建模型对比分析"""
        st.markdown("### 🔍 模型详细对比")
        
        if not lstm_data or not transformer_data:
            st.warning("缺少模型数据，无法进行对比")
            return
        
        # 创建对比表格
        comparison_data = []
        
        metrics_to_compare = ['r2', 'mae', 'mse', 'rmse']
        
        for metric in metrics_to_compare:
            lstm_value = lstm_data.get('metrics', {}).get(metric, 'N/A')
            transformer_value = transformer_data.get('metrics', {}).get(metric, 'N/A')
            
            if isinstance(lstm_value, (int, float)) and isinstance(transformer_value, (int, float)):
                if metric == 'r2':
                    better = "LSTM" if lstm_value > transformer_value else "Transformer"
                else:
                    better = "LSTM" if lstm_value < transformer_value else "Transformer"
            else:
                better = "N/A"
            
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
            value = lstm_data.get('metrics', {}).get(metric, 0)
            if isinstance(value, (int, float)):
                if metric == 'r2':
                    lstm_values.append(value)
                else:
                    # 对损失类指标进行归一化（值越小越好）
                    lstm_values.append(1 - min(value, 1))
            else:
                lstm_values.append(0)
        
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
            value = transformer_data.get('metrics', {}).get(metric, 0)
            if isinstance(value, (int, float)):
                if metric == 'r2':
                    transformer_values.append(value)
                else:
                    transformer_values.append(1 - min(value, 1))
            else:
                transformer_values.append(0)
        
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
'''
    
    # 插入新方法
    new_content = content[:insert_pos] + new_method + content[insert_pos:]
    
    with open('src/dashboard.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("✅ 已修复 dashboard.py，添加了 create_model_comparison 方法")
else:
    print("❌ 找不到插入位置，手动修复...")
