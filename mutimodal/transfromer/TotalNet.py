import torch
import torch.nn as nn
#
from PositionalEncoding import PositionalEncoding
from mutimodal.transfromer.Conv1D import MyConv1D
from mutimodal.transfromer.MainNet import MainNet


class TotalNet(nn.Module):
    """
    transformer版本的总网络结构
    包括：
        - 维度对齐 (Conv1D)
        - 位置编码 (PositionalEncoding)
        - 跨模态注意力 (CrossModalAttentionLayerAttention)
        - 自注意力Transformer
    """
    def __init__(self):
        super(TotalNet, self).__init__()
        # 初始化网络结构
        self.d_model = 512 # 维度对齐后的维度

        # Conv1
        self.conv1_e = MyConv1D(self.d_model) # Eeg
        self.conv1_f = MyConv1D(self.d_model) # Face
        self.conv1_c = MyConv1D(self.d_model) # Ecg
        # Positional Encoding,
        self.pos = PositionalEncoding(self.d_model)
        # MainNet for Eeg, Face, Ecg
        self.main_e = MainNet()
        self.main_f = MainNet()
        self.main_c = MainNet()
        # Linear and Softmax
        self.f = nn.Sequential(
            nn.Linear(self.d_model * 6, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self,d_eeg,d_face,d_ecg):
        # 维度对齐
        d_eeg = self.conv1_e(d_eeg) # (batch, t, d_model)
        d_face = self.conv1_f(d_face) # (batch, t, d_model)
        d_ecg = self.conv1_c(d_ecg) # (batch, t, d_model)
        # 位置编码
        d_eeg = self.pos(d_eeg) # (batch, t, d_model)
        d_face = self.pos(d_face) # (batch, t, d_model)
        d_ecg = self.pos(d_ecg) # (batch, t, d_model)
        # 跨模态注意力
        d_eeg = self.main_e(d_eeg,d_face,d_ecg) # (batch, t, d_model*2)
        d_face = self.main_f(d_face,d_eeg,d_ecg) # (batch, t, d_model*2)
        d_ecg = self.main_c(d_ecg,d_eeg,d_face) # (batch, t, d_model*2)
        # 拼接
        h = torch.cat((d_eeg[:,-1,:],d_face[:,-1,:],d_ecg[:,-1,:]),dim=1) # (batch, d_model*6)
        output = self.f(h) # (batch, 1)
        return output