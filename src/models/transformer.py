"""
Transformer模型用于多步时序预测
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

class TimeSeriesTransformer(nn.Module):
    """
    Transformer模型用于多步时序预测
    
    Args:
        input_size: 输入特征维度
        d_model: Transformer内部维度
        nhead: 注意力头数
        num_encoder_layers: 编码器层数
        num_decoder_layers: 解码器层数
        dim_feedforward: 前馈网络维度
        prediction_steps: 预测步数
        dropout: Dropout率
        activation: 激活函数
    """
    def __init__(self, input_size, d_model=64, nhead=4, 
                 num_encoder_layers=2, num_decoder_layers=2,
                 dim_feedforward=256, prediction_steps=7,
                 dropout=0.1, activation='relu'):
        super().__init__()
        
        self.d_model = d_model
        self.prediction_steps = prediction_steps
        
        # 输入投影层
        self.input_projection = nn.Linear(input_size, d_model)
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=False
        )
        
        # 输出投影层
        self.output_projection = nn.Linear(d_model, prediction_steps)
        
        # 初始化参数
        self._init_parameters()
    
    def _init_parameters(self):
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_square_subsequent_mask(self, sz):
        """生成因果掩码（用于解码器）"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, src, tgt_mask=None):
        """
        Args:
            src: 输入序列 (batch_size, seq_len, input_size)
            tgt_mask: 目标掩码 (prediction_steps, prediction_steps)
        
        Returns:
            output: 预测结果 (batch_size, prediction_steps)
        """
        batch_size = src.size(0)
        
        # 输入投影
        src = self.input_projection(src) * math.sqrt(self.d_model)
        src = self.positional_encoding(src)
        
        # 创建目标序列（全零，用于解码器）
        tgt = torch.zeros(batch_size, self.prediction_steps, self.d_model).to(src.device)
        tgt = self.positional_encoding(tgt)
        
        # 生成因果掩码
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(self.prediction_steps).to(src.device)
        
        # Transformer前向传播
        output = self.transformer(
            src=src,
            tgt=tgt,
            tgt_mask=tgt_mask
        )
        
        # 输出投影
        output = self.output_projection(output)
        
        # 确保输出形状为 (batch_size, prediction_steps)
        if output.dim() == 3:
            output = output.mean(dim=-1)  # 或者 output[:, :, 0]


        return output  # (batch_size, prediction_steps)

class SimpleTransformer(nn.Module):
    """
    简化版Transformer（更快训练）
    """
    def __init__(self, input_size, d_model=64, nhead=4, 
                 num_layers=2, prediction_steps=7, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.prediction_steps = prediction_steps
        
        # 输入投影
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, prediction_steps)
        )
    
    def forward(self, src):
        # src: (batch_size, seq_len, input_size)
        
        # 输入投影和位置编码
        src = self.input_projection(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        # Transformer编码器
        memory = self.transformer_encoder(src)
        
        # 取最后一个时间步
        last_output = memory[:, -1, :]
        
        # 解码
        output = self.decoder(last_output)
        
        return output  # (batch_size, prediction_steps)

# 测试代码
if __name__ == "__main__":
    # 测试Transformer模型
    batch_size = 32
    seq_len = 30
    input_size = 3
    pred_steps = 7
    
    model = TimeSeriesTransformer(
        input_size=input_size,
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=256,
        prediction_steps=pred_steps,
        dropout=0.1
    )
    
    # 测试输入
    x = torch.randn(batch_size, seq_len, input_size)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")