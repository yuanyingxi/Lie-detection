import torch.nn as nn
import torch.nn.functional as F


class CNNmodel(nn.Module):
    def __init__(self):
        super(CNNmodel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=256, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(256)
        self.pool1 = nn.MaxPool1d(2)
        self.drop1 = nn.Dropout(0.25)

        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        self.drop2 = nn.Dropout(0.25)

        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(2)
        self.drop3 = nn.Dropout(0.25)

    def forward(self, x):  # x:[b,1,770]
        # x = x.permute(0, 2, 1)
        x = self.drop1(self.pool1(self.bn1(F.relu(self.conv1(x)))))
        x = self.drop2(self.pool2(self.bn2(F.relu(self.conv2(x)))))
        x = self.drop3(self.pool3(self.bn3(F.relu(self.conv3(x)))))
        return x


"""
input:[b,1,770]
output:[b,1]
"""
"""
rnn中一般需要将维度变换成[770,b,1]
"""


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.rnn1 = nn.LSTM(64, 128, 2, batch_first=True)
        self.rnn2 = nn.LSTM(128, 64, 1, batch_first=True)
        self.drop1 = nn.Dropout(0.25)
        self.drop2 = nn.Dropout(0.25)
        self.cnn = CNNmodel()
        self.f = nn.Sequential(
            nn.Linear(64 * 64, 64 * 16),
            nn.ReLU(),
            # nn.Dropout(0.25),
            nn.Linear(64 * 16, 256 * 2),
            nn.ReLU(),
            # nn.Dropout(0.25),
            nn.Linear(256 * 2, 128),
            nn.ReLU(),
            # nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.ReLU(),
            # nn.Dropout(0.25),
            nn.Linear(64, 32),
            nn.ReLU(),
            # nn.Dropout(0.25),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.conv_layers = nn.Sequential(
            nn.Conv1d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(64)  # 强制输出长度64
        )

    def forward(self, x):
        # print(x.shape)
        x = self.extra_feature(x)
        x = x.permute(0, 2, 1)
        x = self.conv_layers(x)
        x = x.reshape(x.size(0), -1)  # 将batch之后的维度展平
        output = self.f(x)
        return output

    # 可用于多模态调用
    def extra_feature(self, x):
        x = self.cnn(x)  # 输入的是[b,1,770]
        x = x.permute(0, 2, 1)
        x, _ = self.rnn1(x)
        x = self.drop1(x)
        x, _ = self.rnn2(x)
        x = self.drop2(x)
        return x


if __name__ == '__main__':
    print('hh')