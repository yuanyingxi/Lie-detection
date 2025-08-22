import torch.nn as nn
class XtoYMultiHeadAttention(nn.Module):
    """
    自己实现的MultiHeadAttention
    将x->y的注意力机制
    参数：
        input_size: 特征维度
        num_heads: 多头注意力的头数
    返回:
        增强后的x特征
    [batch_size, t_x, d_model] & [batch_size, t_y, d_model] -> [batch_size, t_x, d_model]
    """
    def __init__(self, d_model, num_heads,dropout=0.0):
        super(XtoYMultiHeadAttention, self).__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        # Norm层
        self.norm_x = nn.LayerNorm(d_model)
        self.norm_y = nn.LayerNorm(d_model)
        # 注意力层
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

    def forward(self,x,y):
        """
        参数:
            x: 输入的x特征[batch_size, t_x, d_model]
            y: 输入的y特征[batch_size, t_y, d_model]
        返回:
            增强后的x特征
        """
        x = self.norm_x(x) # [batch_size, t_x, d_model]
        y = self.norm_y(y) # [batch_size, t_y, d_model]
        # 注意力层
        output, _ = self.cross_attention(
            query=x,
            key=y,
            value=y,
        ) # [batch_size, t_x, d_model]
        output += x # [batch_size, t_x, d_model]
        return output # [batch_size, t_x, d_model]

class CrossModelAttentionLayer(nn.Module):
    """
    Cross-Modal Attention Layer
    将X对应的特征嵌入Y
    参数:
        output_size: 输出维度
    [batch_size, t_x, d_model] & [batch_size, t_y, d_model] -> [batch_size, t_x, d_model]
    """
    def __init__(self,output_size=512,num_heads=8,dropout=0.0):
        super().__init__()
        # 多头注意力层
        self.multihead_attn = XtoYMultiHeadAttention(d_model=output_size, num_heads=8,dropout=dropout)
        # Norm层
        self.norm = nn.LayerNorm(output_size)
        # FFN层
        self.ffn = nn.Sequential(
            nn.Linear(output_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, output_size)
        )

    def forward(self,x,y):
        """
        参数:
            x: 输入的x特征 [batch_size, t_x, d_model]
            y: 输入的y特征 [batch_size, t_y, d_model]
        返回:
            增强后的x特征 [batch_size, t_x, d_model]
        """
        x=self.multihead_attn(x,y)
        output=self.norm(x)
        output=self.ffn(output)
        output+=x
        return output # [batch_size, t_x, d_model]
