import argparse
import os
import sys
import warnings
import ast

import bioread
import numpy as np
import torch
from scipy import signal
from sklearn.preprocessing import StandardScaler

from detector.views.Error import CustomAPIException
from .main import LieDetectionModel, process_ecg_signal
from .wavelet_denoising import remove_baseline_wander, wavelet_noising

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

        return lie_prob


def load_ecg_data(input_data, sampling_rate=None):
    """
    加载ECG数据，支持两种输入格式:
    1. ACQ文件路径
    2. 直接的ECG信号数据数组

    参数:
        input_data: 可以是ACQ文件路径(str)或ECG信号数据(np.ndarray/list)
        sampling_rate: 当input_data是数组时需要提供采样率

    返回:
        tuple: (ecg_signal, sampling_rate)
    """
    try:
        if isinstance(input_data, str):
            # ACQ文件路径输入
            if input_data.lower().endswith('.acq'):
                return load_ecg_from_acq(input_data)
            else:
                raise ValueError("不支持的文件格式，仅支持.acq文件")
        else:  # 数据的输入
            try:
                data = bioread.read(input_data)
            except Exception as e:
                raise CustomAPIException(f"无法读取该 ACQ 文件，请检查文件格式")
            if not data.channels:
                raise CustomAPIException(f"该 ACQ 文件没有信号通道")
            first_data = data.channels[0]
            ecg_signal = first_data.data
            sampling_rate = first_data.samples_per_second

            return ecg_signal, sampling_rate
    except Exception as e:
        raise CustomAPIException(f"无法读取ECG数据: {str(e)}")


def load_ecg_from_acq(acq_path):
    """从ACQ文件加载ECG数据（直接读取第一个通道）"""
    try:
        import bioread
    except ImportError:
        raise ImportError("读取ACQ文件需要bioread库，请先安装: pip install bioread")

    # 检查文件是否存在
    if not os.path.exists(acq_path):
        raise FileNotFoundError(f"ACQ文件 {acq_path} 未找到")

    # 读取ACQ文件
    data = bioread.read_file(acq_path)

    # 检查是否有通道数据
    if not data.channels:
        raise ValueError("ACQ文件中没有通道数据")

    # 获取第一个通道的数据和采样率
    first_channel = data.channels[0]
    sampling_rate = first_channel.samples_per_second
    channel_data = first_channel.data

    return channel_data, sampling_rate


if __name__ == "__main__":
    # 设置参数解析器
    parser = argparse.ArgumentParser(description='ECG说谎检测预测')
    parser.add_argument('--input_type', type=str, required=True,
                        choices=['acq', 'array'],
                        help='输入类型: acq 或 array')
    parser.add_argument('--input_data', type=str, required=True,
                        help='输入数据: ACQ文件路径或ECG数组字符串')
    parser.add_argument('--model_path', type=str, default='lie_detection_model.pth',
                        help='预训练模型路径 (默认为 lie_detection_model.pth)')
    parser.add_argument('--sampling_rate', type=float,
                        help='当输入类型为array时需要提供采样率')
    args = parser.parse_args()

    # 初始化检测器
    try:
        detector = EcgFileProcessor(args.model_path)
    except Exception as e:
        print(f"加载模型失败: {str(e)}")
        sys.exit(1)

    # 加载并预测ECG数据
    try:
        if args.input_type == "array":
            if args.sampling_rate is None:
                print("错误: 直接传入数组数据时需要提供--sampling_rate参数")
                sys.exit(1)

            # 安全地将字符串形式的数组转换为numpy数组
            try:
                # 先尝试直接eval（适用于简单的数组字符串）
                ecg_signal = np.array(ast.literal_eval(args.input_data), dtype=np.float64)
            except:
                # 如果失败，尝试从文件加载（兼容旧方式）
                if os.path.exists(args.input_data):
                    ecg_signal = np.load(args.input_data)
                else:
                    raise ValueError("无法解析输入的数组数据，且指定的文件不存在")

            sampling_rate = args.sampling_rate
        else:
            # 从ACQ文件读取数据
            ecg_signal, sampling_rate = load_ecg_data(args.input_data)

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
        sys.exit(1)