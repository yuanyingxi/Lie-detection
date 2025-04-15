import os
import sys

import joblib
import numpy as np
import pandas as pd
import pywt
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from EegModule.model import CNNmodel


class Preprocessing:
    def __init__(self, file_path):
        self.path = file_path

    # 加载并合并数据
    @staticmethod
    def load_data(path):
        df = pd.read_csv(path)
        return df

    # 滑动窗口
    @staticmethod
    def segment_data(df, window_size=128, overlap=64):
        segments = []
        for start in range(0, len(df) - window_size, overlap):
            segment = df.iloc[start:start + window_size, :]
            segments.append(segment)
        return np.array(segments)

    # 小波变换
    @staticmethod
    def extract_dwt_features(data, wavelet='db4', level=4):
        features = []
        for channel in range(data.shape[1]):
            coeffs = pywt.wavedec(data[:, channel], wavelet, level=level)  # 对一个窗口中的一个通道进行小波变换
            coeffs_flat = np.hstack(coeffs)  # 展平得到的低频与高频系数
            features.append(coeffs_flat)
        features = np.hstack(features)  # 合并通道
        return np.array(features)

class EegFileProcessor:
    def __init__(self):
        self.scaler = joblib.load('EegModule/data_scaler.pkl')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def preprocessing(self, file):
        features = np.array(Preprocessing.load_data(file))[:384, :]
        features = Preprocessing.extract_dwt_features(features)
        features = self.scaler.transform(features.reshape(1, -1))
        features = features.reshape(1, 1, -1)  # (samples, channels, features)
        features = torch.tensor(features, device=self.device, dtype=torch.float32)
        return features

    def model_setup(self):
        self.model = CNNmodel(self.dim).to(self.device)
        self.model.load_state_dict(torch.load('EegModule/best_model.pt'))

    def predict(self, file):
        self.features = self.preprocessing(file)
        self.dim = self.features.shape[-1]
        self.model_setup()
        with torch.no_grad():
            output = self.model(self.features)

        return output.item()