import torch
import os
import sys
import argparse
from thop import profile
from torchinfo import summary

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.core import YAMLConfig

def calculate_model_complexity(config_path):
    """计算配置文件定义的完整模型的计算量"""
    
    # 加载配置但不加载权重
    cfg = YAMLConfig(config_path, resume=None)
    
    # 创建完整模型，包括backbone、encoder、decoder等所有组件
    class CompleteModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model
            self.postprocessor = cfg.postprocessor
        
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs
    
    # 实例化模型
    model = CompleteModel()
    model.eval()
    
    # 创建测试输入
    input_size = 640  # 常用输入大小
    dummy_input = torch.randn(1, 3, input_size, input_size)
    dummy_sizes = torch.tensor([[input_size, input_size]])
    
    # 计算FLOPs和参数量
    flops, params = profile(model, inputs=(dummy_input, dummy_sizes))
    
    # 打印结果
    print(f"{'='*50}")
    print(f"配置文件: {config_path}")
    print(f"{'='*50}")
    print(f"模型总参数量: {params / 1e6:.2f} M")
    print(f"模型总计算量: {flops / 1e9:.2f} G FLOPs")
    print(f"{'='*50}")
    
    # 打印更详细的模型结构和每层计算量
    print("\n详细的模型结构分析:")
    summary(model, [(1, 3, input_size, input_size), (1, 2)], depth=3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="计算RT-DETR模型复杂度")
    parser.add_argument('-c', '--config', type=str, required=True,
                       help="配置文件路径，例如configs/rtdetr/rtdetr_r50vd_6x_coco.yml")
    args = parser.parse_args()
    
    calculate_model_complexity(args.config)