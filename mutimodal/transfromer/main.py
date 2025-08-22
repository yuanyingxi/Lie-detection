# 模态融合
import torch
import torch.nn as nn
import torch.nn.functional as F




if __name__ == "__main__":
    # 定义参数
    batch_size = 2
    seq_len = 10
    d_model = 16

    # 创建 LayerNorm 层
    # 对于形状为 [batch_size, seq_len, d_model] 的输入
    # 我们通常希望在特征维度 d_model 上进行归一化
    layer_norm = nn.LayerNorm(d_model)
    # 创建随机输入数据
    input_tensor = torch.randn(batch_size, seq_len, d_model)
    print("输入张量形状:", input_tensor.shape)
    # 应用层归一化
    output = layer_norm(input_tensor)
    print("输出张量形状:", output.shape)
    # 查看可学习参数
    print("γ参数形状:", layer_norm.weight.shape)  # PyTorch 中使用 weight 而不是 gamma
    print("β参数形状:", layer_norm.bias.shape)  # PyTorch 中使用 bias 而不是 beta