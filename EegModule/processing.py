import os

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import pywt
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from .model import *


# 加载并合并数据
def load_and_concatenate_data(path):
    data_frames = []
    for filename in os.listdir(path):
        if filename.endswith('.csv'):
            df = pd.read_csv(os.path.join(path, filename))
            data_frames.append(df)
    concatenated_df = pd.concat(data_frames, ignore_index=True)
    return concatenated_df


# 滑动窗口
def segment_data(df, window_size=128, overlap=64):
    segments = []
    for start in range(0, len(df) - window_size, overlap):
        segment = df.iloc[start:start + window_size, :]
        segments.append(segment)
    return np.array(segments)


def extract_dwt_features(segments, wavelet='db4', level=4):
    features = []
    for segment in segments:
        segment_features = []
        for channel in range(segment.shape[1]):
            coeffs = pywt.wavedec(segment[:, channel], wavelet, level=level)  # 对一个窗口中的一个通道进行小波变换
            coeffs_flat = np.hstack(coeffs)  # 展平得到的低频与高频系数
            segment_features.append(coeffs_flat)
        features.append(np.hstack(segment_features))  # 合并通道
    return np.array(features)


def train_model(model, train_loader, test_loader, criterion, optimizer, epochs=15):
    save_path = r"best_model.pt"
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            preds = (outputs > 0.5).float()
            correct_train += (preds == y_batch).sum().item()
            total_train += len(y_batch)

        train_loss /= len(train_loader)
        train_acc = correct_train / total_train

        # 验证
        model.eval()
        val_loss = 0
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

                preds = (outputs > 0.5).float()
                correct_test += (preds == y_batch).sum().item()
                total_test += len(y_batch)

            val_loss /= len(test_loader)
            test_acc = correct_test / total_test

        print(
            f"Epochs [{epoch}/{epochs}], training loss: {train_loss:.4f}, training accurary: {train_acc * 100:.2f}%, testing loss: {val_loss:.4f}, testing accurary: {test_acc * 100:.2f}%")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    # 数据的地址
    truth_path = '../datasets/LieWaves/Truth_Sessions/Raw/'
    lie_path = '../datasets/LieWaves/Lie_Sessions/Raw/'

    # 加载并合并数据
    truth_data = load_and_concatenate_data(truth_path)
    lie_data = load_and_concatenate_data(lie_path)

    # 分割数据
    window_size = 384
    overlap = 128
    truth_segments = segment_data(truth_data, window_size, overlap)
    lie_segments = segment_data(lie_data, window_size, overlap)

    # 提取小波变换特征
    truth_features = extract_dwt_features(truth_segments)
    lie_features = extract_dwt_features(lie_segments)

    # 创建标签: 1 truth, 0 lie
    truth_labels = np.ones(truth_features.shape[0])
    lie_labels = np.zeros(lie_features.shape[0])

    # 合并两种数据
    X = np.vstack((truth_features, lie_features))
    y = np.hstack((truth_labels, lie_labels))

    # stratify=y, 确保分割数据集后的标签占比相同
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 归一化
    # scaler = StandardScaler()
    scaler = joblib.load('data_scaler.pkl')
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # 保存归一化参数
    # joblib.dump(scaler, 'data_scaler.pkl')

    X_train_cnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    print(X_train_cnn.shape)
    X_test_cnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # 转换成 PyTorch tensors
    X_train_tensor = torch.tensor(X_train_cnn, dtype=torch.float32).squeeze().unsqueeze(1)
    X_test_tensor = torch.tensor(X_test_cnn, dtype=torch.float32).squeeze().unsqueeze(1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # 创建 DataLoader
    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    test_data = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # 训练模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    model = CNNmodel(X_train_cnn.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    train_model(model, train_loader, test_loader, criterion, optimizer, epochs=10)

