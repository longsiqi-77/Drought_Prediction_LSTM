import torch
import sys
sys.path.insert(0, '.')

from src.models import SimpleTransformer, TimeSeriesTransformer

print("=== 测试 SimpleTransformer ===")
try:
    model1 = SimpleTransformer(input_size=3, prediction_steps=7)
    test_input = torch.randn(32, 30, 3)
    output1 = model1(test_input)
    print(f"✓ SimpleTransformer 模型创建成功")
    print(f"  输入形状: {test_input.shape}")
    print(f"  输出形状: {output1.shape}")
    print(f"  期望输出: (32, 7)")
    
    if output1.shape == torch.Size([32, 7]):
        print("✓ 输出形状正确！")
    else:
        print(f"✗ 输出形状错误！应该是 (32, 7)")
        print(f"  实际形状: {output1.shape}")
except Exception as e:
    print(f"✗ 错误: {e}")
    import traceback
    traceback.print_exc()

print("\n=== 测试 TimeSeriesTransformer ===")
try:
    model2 = TimeSeriesTransformer(input_size=3, prediction_steps=7)
    output2 = model2(test_input)
    print(f"✓ TimeSeriesTransformer 模型创建成功")
    print(f"  输入形状: {test_input.shape}")
    print(f"  输出形状: {output2.shape}")
    print(f"  期望输出: (32, 7)")
    
    if output2.shape == torch.Size([32, 7]):
        print("✓ 输出形状正确！")
    else:
        print(f"✗ 输出形状错误！应该是 (32, 7)")
        print(f"  实际形状: {output2.shape}")
except Exception as e:
    print(f"✗ 错误: {e}")
    import traceback
    traceback.print_exc()

# 检查训练脚本使用的模型类型
print("\n=== 检查训练配置 ===")
