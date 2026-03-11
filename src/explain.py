"""
模型解释性分析
"""
import shap
import matplotlib.pyplot as plt

def explain_model(model, sample_data):
    """使用SHAP解释模型预测"""
    explainer = shap.DeepExplainer(model, background_data)
    shap_values = explainer.shap_values(sample_data)
    
    # 可视化
    shap.summary_plot(shap_values, sample_data)
    return shap_values
