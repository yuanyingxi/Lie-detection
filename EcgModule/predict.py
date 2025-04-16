import argparse
import os
import sys
import warnings

import numpy as np
import torch
import pandas as pd
from scipy import signal
from sklearn.preprocessing import StandardScaler

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from main import LieDetectionModel, process_ecg_signal
from wavelet_denoising import remove_baseline_wander, wavelet_noising

# 禁用警告
warnings.filterwarnings("ignore")


class EcgFileProcessor:
    def __init__(self, model_path='EcgModule/lie_detection_model.pth'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = LieDetectionModel().to(self.device)
        self.scaler = StandardScaler()
        self.sampling_rate = 250  # 模型训练时的采样率

        # 加载预训练模型
        if os.path.exists(model_path):
            self.load_model(model_path)
        else:
            raise FileNotFoundError(f"模型文件 {model_path} 未找到")

    def load_model(self, model_path):
        """加载预训练模型和标准化器参数"""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler.mean_ = checkpoint['scaler_mean']
        self.scaler.scale_ = checkpoint['scaler_scale']
        self.model.eval()
        print(f"成功加载模型: {model_path}")

    def preprocess_signal(self, ecg_signal, original_rate):
        """
        完整预处理流程:
        1. 重采样到250Hz
        2. 去除基线漂移
        3. 小波去噪
        """
        # 重采样到250Hz
        if original_rate != self.sampling_rate:
            num_samples = int(len(ecg_signal) * self.sampling_rate / original_rate)
            ecg_signal = signal.resample(ecg_signal, num_samples)

        # 去除基线漂移
        signal_no_baseline = remove_baseline_wander(ecg_signal, self.sampling_rate)

        # 小波去噪
        denoised_signal = wavelet_noising(signal_no_baseline)

        return denoised_signal

    def extract_features(self, ecg_signal, original_rate):
        """提取CWT和RR特征"""
        # 预处理信号
        processed_signal = self.preprocess_signal(ecg_signal, original_rate)

        # 提取特征
        cwt_feature, rr_features = process_ecg_signal(processed_signal, sampling_rate=self.sampling_rate)

        if cwt_feature is None or rr_features is None:
            raise ValueError("特征提取失败 - 信号可能太短或没有检测到R峰")

        # 标准化RR特征
        rr_features = self.scaler.transform(rr_features.reshape(1, -1))

        return cwt_feature, rr_features

    def predict_proba(self, ecg_signal, original_rate):
        """
        预测说谎概率
        返回:
            float: 说谎概率 (0-0.5表示诚实，0.5-1表示说谎)
        """
        # 提取特征
        cwt_feature, rr_features = self.extract_features(ecg_signal, original_rate)

        # 准备输入张量
        cwt_tensor = torch.from_numpy(np.expand_dims(np.expand_dims(cwt_feature, 0), 0)).float().to(self.device)
        rr_tensor = torch.from_numpy(rr_features).float().to(self.device)

        # 预测
        with torch.no_grad():
            outputs = self.model(cwt_tensor, rr_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            lie_prob = probabilities[0, 1].item()  # 获取说谎类别的概率

        Response_data = {
            "modality": "ecg",
            "confidence": 1 - abs(lie_prob - round(lie_prob)),
            "result": "诚实" if lie_prob < 0.5 else "说谎"
        }

        return Response_data


def load_ecg_from_csv(csv_path):
    """从CSV文件加载ECG数据"""
    df = pd.read_csv(csv_path)

    # 检测ECG信号列
    if 'ECG Signal' not in df.columns:
        possible_ecg_cols = [col for col in df.columns if 'ecg' in col.lower() or 'signal' in col.lower()]
        if not possible_ecg_cols:
            raise ValueError("在CSV文件中找不到ECG信号列")
        ecg_col = possible_ecg_cols[0]
    else:
        ecg_col = 'ECG Signal'

    # 提取ECG信号
    ecg_signal = df[ecg_col].values

    # 从时间列计算采样率（如果可用）
    sampling_rate = 250  # 默认值
    if 'Time (s)' in df.columns:
        time_diff = np.diff(df['Time (s)'].values)
        sampling_rate = 1.0 / np.mean(time_diff)

    return ecg_signal, sampling_rate

if __name__ == "__main__":
    # 设置参数解析器
    parser = argparse.ArgumentParser(description='ECG说谎检测预测')
    parser.add_argument('input_csv', type=str, help='输入ECG信号的CSV文件路径')
    parser.add_argument('--model_path', type=str, default='lie_detection_model.pth',
                        help='预训练模型路径 (默认为 lie_detection_model.pth)')
    args = parser.parse_args()

    # 检查输入文件
    if not os.path.exists(args.input_csv):
        print(f"错误: 输入文件 {args.input_csv} 不存在")

    # 初始化检测器
    try:
        detector = EcgFileProcessor(args.model_path)
    except Exception as e:
        print(f"加载模型失败: {str(e)}")

    # 加载并预测ECG数据
    try:
        ecg_signal, sampling_rate = load_ecg_from_csv(args.input_csv)
        print(f"成功加载ECG信号: {len(ecg_signal)} 个样本, 采样率: {sampling_rate:.1f}Hz")

        # 进行预测
        probability = detector.predict_proba(ecg_signal, sampling_rate)

        # 计算置信度
        confidence = 80 + 20 * (abs(probability - 0.5) / 0.5)

        # 打印结果
        print("\n预测结果:")
        print(f"说谎概率: {probability:.4f}")
        print(f"结论: {'可能说谎' if probability >= 0.5 else '可能诚实'}")
        print(f"置信度: {min(100, confidence):.1f}%")

    except Exception as e:
        print(f"预测过程中出错: {str(e)}")



