import torch
from torch import nn
import torch.nn.functional as F


class CNNmodel(nn.Module):
    def __init__(self, dim):
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

        in_dim = self.calculate_fc1_input(dim, 3)
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        # x.shape = (batch_size, 1, dim)
        x = self.drop1(self.pool1(self.bn1(F.relu(self.conv1(x)))))
        x = self.drop2(self.pool2(self.bn2(F.relu(self.conv2(x)))))
        x = self.drop3(self.pool3(self.bn3(F.relu(self.conv3(x)))))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        output = torch.sigmoid(x)
        return output

    def calculate_fc1_input(self, dim, layer_num):
        L = dim
        for i in range(layer_num):
            L = L - 2
            L = (L - 2) // 2 + 1
        return 64 * L