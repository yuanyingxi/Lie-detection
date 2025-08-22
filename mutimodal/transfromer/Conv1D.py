from torch import nn


class MyConv1D(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        self.output_dim = output_dim
        # 使用延迟初始化（Lazy初始化）
        '''
        输出目标的通道数(默认是512)
        使用1*1大小的卷积核
        步长为1
        填充为0
        偏置为True
        '''
        self.conv = nn.LazyConv1d(
            out_channels=output_dim,  # 输出通道数（即目标维度512）
            kernel_size=1,  # 1x1卷积核
            stride=1,
            padding=0,
            bias=True
        )

    def forward(self, x):
        """
        输入: [b, t, d] -> 输出: [b, t, 512]
        步骤:
        1. 添加批次维度: [1, t, d]
        2. 调整维度顺序: [1, d, t] (Conv1d要求通道维度在中间)
        3. 应用1x1卷积: [1, 512, t]
        4. 恢复维度顺序: [1, t, 512]
        5. 移除批次维度: [t, 512]
        """
        x = x.permute(0, 2, 1)  # [b, d, t] (Conv1d要求通道维度在中间)
        x = self.conv(x)  # [b, 512, t]
        x = x.permute(0, 2, 1)  # [b, t, 512]
        return x


