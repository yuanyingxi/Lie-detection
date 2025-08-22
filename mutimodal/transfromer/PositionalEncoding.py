import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    位置编码层，为输入序列添加位置信息
    参数:
        d_model (int): 输入特征的维度，即词向量的维度
        max_len (int): 支持的最大序列长度，默认为5000

    输入:
        x: 形状为 [batch_size, seq_len, d_model] 的张量

    输出:
        形状为 [batch_size, seq_len, d_model] 的张量，已添加位置编码
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # 初始化一个位置编码矩阵，形状为 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        # 创建位置索引，形状为 [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 计算除数项: 10000^(2i/d_model)
        # 使用对数空间计算以避免数值问题
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )
        # 对偶数索引应用正弦函数
        pe[:, 0::2] = torch.sin(position * div_term)
        # 对奇数索引应用余弦函数
        pe[:, 1::2] = torch.cos(position * div_term)

        # 添加一个维度，使pe的形状变为 [1, max_len, d_model]
        # 这样可以在批处理时自动广播
        pe = pe.unsqueeze(0)

        # 将pe注册为缓冲区，它不是模型参数，但会被保存到状态字典中
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        参数:
            x: 输入张量，形状为 [batch_size, seq_len, d_model]

        返回:
            添加了位置编码的张量，形状为 [batch_size, seq_len, d_model]
        """
        # 获取输入序列的实际长度
        seq_len = x.size(1)

        # 将位置编码添加到输入中
        # 使用切片操作确保不会超出预计算的位置编码范围
        # x的形状: [batch_size, seq_len, d_model]
        # self.pe的形状: [1, max_len, d_model] → 切片后: [1, seq_len, d_model]
        # 通过广播机制，每个批次中的样本都会添加相同的位置编码
        x = x + self.pe[:, :seq_len, :] # [batch_size, seq_len, d_model]
        return x