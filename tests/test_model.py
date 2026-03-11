import torch
import unittest
import sys
import os

# 将项目根目录加入 path 才能导入 src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import LSTMModel
from src.config import config

class TestLSTMModel(unittest.TestCase):
    def test_model_shape(self):
        # 模拟一个 Batch 的数据: (Batch=32, Seq=30, Feat=3)
        dummy_input = torch.randn(32, config.SEQUENCE_LENGTH, config.INPUT_SIZE)
        
        model = LSTMModel()
        output = model(dummy_input)
        
        # 检查输出形状，应该是 (32) 或者 (32, 1) squeeze后是 (32)
        print(f"\nModel Output Shape: {output.shape}")
        
        self.assertEqual(output.shape[0], 32)
        self.assertEqual(len(output.shape), 1) # 应该是一维向量

if __name__ == '__main__':
    unittest.main()