"""
自动化超参数调优
"""
import optuna
from sklearn.model_selection import cross_val_score

def optimize_hyperparameters(trial):
    """定义要优化的参数空间"""
    params = {
        'learning_rate': trial.suggest_float('lr', 1e-5, 1e-1, log=True),
        'hidden_size': trial.suggest_categorical('hidden', [32, 64, 128, 256]),
        'num_layers': trial.suggest_int('layers', 1, 4),
        'dropout': trial.suggest_float('dropout', 0.0, 0.5),
        'batch_size': trial.suggest_categorical('batch', [16, 32, 64, 128])
    }
    
    # 训练模型并返回验证分数
    score = train_and_evaluate(params)
    return score
