import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import os
from src.config import config

class DroughtDataset(Dataset):
    def __init__(self, X, y):
        # 确保y是一维的（如果是单步预测）或二维的（如果是多步预测）
        self.X = torch.FloatTensor(X)
        
        # 处理y的形状
        y_array = np.array(y)
        if len(y_array.shape) == 1:
            # 一维数组，单步预测
            self.y = torch.FloatTensor(y_array)
        elif len(y_array.shape) == 2 and y_array.shape[1] == 1:
            # 二维数组但只有一列，压缩成一维
            self.y = torch.FloatTensor(y_array.squeeze(1))
        else:
            # 多步预测或多输出，保持原状
            self.y = torch.FloatTensor(y_array)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_and_preprocess_data():
    print(">>> Loading raw data...")
    
    # 1. 加载气象数据
    weather_path = os.path.join(config.RAW_DATA_DIR, config.WEATHER_FILE)
    df_weather = pd.read_csv(weather_path)
    df_weather[config.COL_DATE_WEATHER] = pd.to_datetime(df_weather[config.COL_DATE_WEATHER])
    df_weather = df_weather.sort_values(config.COL_DATE_WEATHER)
    
    # 处理气温
    if config.COL_TAVG in df_weather.columns and df_weather[config.COL_TAVG].notna().sum() > 0:
        df_weather['Temp'] = df_weather[config.COL_TAVG]
    else:
        print("TAVG missing or empty, calculating from TMAX/TMIN")
        df_weather['Temp'] = (df_weather[config.COL_TMAX] + df_weather[config.COL_TMIN]) / 2
        
    # 选取需要的列
    df_weather = df_weather.set_index(config.COL_DATE_WEATHER)[[config.COL_PRCP, 'Temp']]
    df_weather = df_weather.rename(columns={config.COL_PRCP: 'Precipitation'})
    
    # 修复警告1：使用新的ffill/bfill方法
    df_weather = df_weather.ffill().bfill().fillna(0)

    # 2. 加载 NDVI 数据
    ndvi_path = os.path.join(config.RAW_DATA_DIR, config.NDVI_FILE)
    df_ndvi = pd.read_csv(ndvi_path)
    df_ndvi[config.COL_DATE_NDVI] = pd.to_datetime(df_ndvi[config.COL_DATE_NDVI])
    df_ndvi = df_ndvi.sort_values(config.COL_DATE_NDVI)
    
    df_ndvi['NDVI'] = df_ndvi[config.COL_NDVI]
    df_ndvi = df_ndvi.set_index(config.COL_DATE_NDVI)[['NDVI']]

    # 3. 数据融合
    start_date = max(df_weather.index.min(), df_ndvi.index.min())
    end_date = min(df_weather.index.max(), df_ndvi.index.max())
    full_idx = pd.date_range(start=start_date, end=end_date, freq='D')
    
    df_merged = pd.DataFrame(index=full_idx)
    df_merged = df_merged.join(df_weather).join(df_ndvi)
    
    # 4. 插值处理
    df_merged['NDVI'] = df_merged['NDVI'].interpolate(method='linear')
    
    # 修复警告2：使用新的ffill/bfill方法
    df_merged = df_merged.bfill().ffill()
    
    # 5. 添加时间特征
    df_merged['DayOfYear'] = df_merged.index.dayofyear
    df_merged['Month'] = df_merged.index.month
    df_merged['Season'] = (df_merged.index.month % 12 + 3) // 3  # 1:春, 2:夏, 3:秋, 4:冬
    
    print(f">>> Data merged. Shape: {df_merged.shape}")
    print(f">>> Date range: {df_merged.index.min()} to {df_merged.index.max()}")
    
    # 保存处理后的数据
    processed_dir = config.PROCESSED_DATA_DIR
    os.makedirs(processed_dir, exist_ok=True)
    df_merged.to_csv(os.path.join(processed_dir, 'merged_data.csv'))
    
    return df_merged

def create_sliding_windows(data, seq_length, pred_steps=1):
    """创建多步预测的滑窗"""
    X, y = [], []
    for i in range(len(data) - seq_length - pred_steps + 1):
        X.append(data[i:i+seq_length])
        # 预测未来pred_steps天的NDVI
        y.append(data[i+seq_length:i+seq_length+pred_steps, 0])  # 第0列是NDVI
    return np.array(X), np.array(y)

def get_dataloaders(pred_steps=None, model_type=None):
    """获取数据加载器，支持多步预测"""
    # 设置默认值
    if pred_steps is None:
        pred_steps = config.PREDICT_STEPS  # 使用你的config中的命名
    if model_type is None:
        # 安全获取MODEL_TYPE，如果没有则默认为LSTM
        model_type = getattr(config, 'MODEL_TYPE', 'LSTM')
    
    df = load_and_preprocess_data()
    
    # 选择特征
    if model_type == 'Transformer':
        # Transformer可以使用更多特征
        features = ['NDVI', 'Precipitation', 'Temp', 'DayOfYear', 'Month']
    else:
        features = ['NDVI', 'Precipitation', 'Temp']
    
    data_values = df[features].values
    
    # 归一化
    scalers = {}
    data_scaled = np.zeros_like(data_values)
    for i in range(data_values.shape[1]):
        scaler = MinMaxScaler()
        data_scaled[:, i] = scaler.fit_transform(data_values[:, i].reshape(-1, 1)).flatten()
        scalers[features[i]] = scaler
    
    # 制作滑窗
    X, y = create_sliding_windows(data_scaled, config.SEQUENCE_LENGTH, pred_steps)
    
    # 调整y的形状
    if pred_steps == 1:
        # 单步预测，y应该是(batch_size,)
        y = y.reshape(-1)
    else:
        # 多步预测，y是(batch_size, pred_steps)
        y = y.reshape(-1, pred_steps)
    
    # 划分训练集/测试集
    train_size = int(len(X) * config.TRAIN_SPLIT)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f">>> Dataset Info:")
    print(f"    Total samples: {len(X)}")
    print(f"    Train samples: {len(X_train)}")
    print(f"    Test samples: {len(X_test)}")
    print(f"    Sequence length: {config.SEQUENCE_LENGTH}")
    print(f"    Prediction steps: {pred_steps}")  # 新增：显示预测步数
    print(f"    Model type: {model_type}")  # 新增：显示模型类型
    print(f"    Input shape: {X_train.shape}")
    print(f"    Output shape: {y_train.shape}")
    
    # 创建DataLoader
    train_dataset = DroughtDataset(X_train, y_train)
    test_dataset = DroughtDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader, scalers, features

def get_feature_names(model_type=None):
    """获取特征名称"""
    if model_type is None:
        model_type = getattr(config, 'MODEL_TYPE', 'LSTM')
    
    if model_type == 'Transformer':
        return ['NDVI', 'Precipitation', 'Temp', 'DayOfYear', 'Month']
    else:
        return ['NDVI', 'Precipitation', 'Temp']