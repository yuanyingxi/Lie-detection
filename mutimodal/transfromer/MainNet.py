import torch
import torch.nn as nn
from CrossModalAttentionLayer import CrossModelAttentionLayer


class MainNet3(nn.Module):
    """
    主要网络结构(三模态)
    eg: 以x为主, 进行y1->x, y2->e, 再经过transformer
    """

    def __init__(self, d_model: int = 512):
        super(MainNet3, self).__init__()
        self.d_model = d_model
        # y1->x
        self.y1_to_x = CrossModelAttentionLayer(d_model)
        # y2->x
        self.y2_to_x = CrossModelAttentionLayer(d_model)
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model * 2,
                nhead=8,
                batch_first=True,
            ),
            num_layers=4
        )

    def forward(self, x, y1, y2):
        """
        将x,y1,y2输入到网络中，将y1->x,y2->x,再经过transformer
        [batch_size, t_x, d_model]
        [batch_size, t_y1, d_model]
        [batch_size, t_y2, d_model]
        """
        # y1->x
        y1_to_x = self.y1_to_x(x, y1)  # [batch_size, t_x, d_model]
        # y2->x
        y2_to_x = self.y2_to_x(x, y1)  # [batch_size, t_x, d_model]
        # 合并,将d_model拼接
        x = torch.cat([y1_to_x, y2_to_x], dim=2)  # [batch_size, t_x, 2*d_model]
        # transformer
        x = self.transformer(x)  # [batch_size, t_x, d_model*2]
        return x


class MainNet2(nn.Module):
    """
    主要网络结构(二模态)
    """
    def __init__(self, d_model: int = 512):
        self.d_model = d_model
        self.y_to_x = CrossModelAttentionLayer(d_model)
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                batch_first=True,
            ),
            num_layers=4
        )

    def forward(self, x, y):
        x = self.y_to_x(x, y) # [batch_size, t_x, d_model]
        x = self.transformer(x) # [batch_size, t_x, d_model]
        return x
