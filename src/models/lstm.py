"""
LSTM模型（移动到models目录）
"""
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, 
                 output_size=1, prediction_steps=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.prediction_steps = prediction_steps
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 全连接层 - 输出多个时间步
        self.fc = nn.Linear(hidden_size, output_size * prediction_steps)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        self.output_size = output_size
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # LSTM前向传播
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # 只取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        
        # Dropout
        last_output = self.dropout(last_output)
        
        # 全连接层
        output = self.fc(last_output)
        
        # 重塑为 (batch_size, prediction_steps, output_size)
        output = output.view(batch_size, self.prediction_steps, self.output_size)
        
        # 如果output_size=1，压缩最后一维
        if self.output_size == 1:
            output = output.squeeze(-1)  # (batch_size, prediction_steps)
        
        return output