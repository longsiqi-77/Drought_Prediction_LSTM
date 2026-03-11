import torch
import sys
sys.path.insert(0, '.')

print("测试修复后的 TimeSeriesTransformer...")
try:
    # 重新导入以确保使用修改后的代码
    import importlib
    import src.models.transformer
    importlib.reload(src.models.transformer)
    
    from src.models.transformer import TimeSeriesTransformer
    
    model = TimeSeriesTransformer(input_size=3, prediction_steps=7)
    test_input = torch.randn(32, 30, 3)
    output = model(test_input)
    
    print(f"✓ 模型创建成功")
    print(f"  输入形状: {test_input.shape}")
    print(f"  输出形状: {output.shape}")
    print(f"  期望输出: (32, 7)")
    
    if output.shape == torch.Size([32, 7]):
        print("✓ 输出形状正确！可以开始训练了。")
    else:
        print(f"✗ 输出形状仍然错误: {output.shape}")
        
except Exception as e:
    print(f"✗ 错误: {e}")
    import traceback
    traceback.print_exc()
