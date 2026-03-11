import os
import torch

class Config:
    # ================= 路径配置 =================
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
    PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
    MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, 'results', 'models')
    FIGURE_SAVE_DIR = os.path.join(PROJECT_ROOT, 'results', 'figures')
    PREDICTION_SAVE_DIR = os.path.join(PROJECT_ROOT, 'results', 'predictions')
    
    # 确保文件夹存在
    for dir_path in [PROCESSED_DATA_DIR, MODEL_SAVE_DIR, FIGURE_SAVE_DIR, PREDICTION_SAVE_DIR]:
        os.makedirs(dir_path, exist_ok=True)

    # ================= 文件名与列名 =================
    WEATHER_FILE = 'NOAA_Weather.csv'
    NDVI_FILE = 'MOD13Q1_NDVI.csv'
    
    COL_NDVI = 'MOD13Q1_061__250m_16_days_NDVI'
    COL_DATE_NDVI = 'Date'
    
    COL_DATE_WEATHER = 'DATE'
    COL_PRCP = 'PRCP'   # 降水
    COL_TAVG = 'TAVG'   # 平均气温
    COL_TMAX = 'TMAX'
    COL_TMIN = 'TMIN'

    # ================= 模型超参数 =================
    MODEL_TYPE = "LSTM"  # 新增：模型类型选择
    SEQUENCE_LENGTH = 30    # 回看过去30天
    PREDICT_STEPS = 7       # 预测未来7天
    INPUT_SIZE = 3          # 特征数: [NDVI, 降水, 气温]
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    DROPOUT = 0.2
    
    # Transformer特定参数
    TRANSFORMER_D_MODEL = 64
    TRANSFORMER_NHEAD = 4
    TRANSFORMER_NUM_LAYERS = 2
    
    # ================= 训练参数 =================
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 100
    TRAIN_SPLIT = 0.8
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SEED = 42
    
    # ================= 模型保存配置 =================
    MODEL_NAMES = ['LSTM', 'Transformer', 'GRU', 'BiLSTM']

config = Config()